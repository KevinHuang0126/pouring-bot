
#!/usr/bin/env python3
"""Experimental one-way Drake -> SPlisHSPlasH trajectory replay.

Requires:
    pysplishsplash Python bindings
    drake_vessel_trajectory.csv
    sph_pouring_trial/scene_static.json

This is an integration prototype, not yet validated fluid physics.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

import pysplishsplash as sph


def load_trajectory(path):
    times = []
    positions = []
    quaternions = []

    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["body"] != "source":
                continue

            times.append(float(row["time_s"]))
            positions.append([
                float(row[k]) for k in ("x_m", "y_m", "z_m")
            ])

            # CSV: wxyz; SciPy: xyzw
            quaternions.append([
                float(row[k]) for k in ("qx", "qy", "qz", "qw")
            ])

    times = np.asarray(times)
    positions = np.asarray(positions)
    quaternions = np.asarray(quaternions)

    if len(times) < 2 or not np.all(np.diff(times) > 0):
        raise ValueError("Trajectory needs increasing timestamps")

    rotations = Slerp(
        times, Rotation.from_quat(quaternions)
    )

    return times, positions, rotations


class DrakeReplay:
    def __init__(self, trajectory):
        self.times, self.positions, self.rotations = (
            load_trajectory(trajectory)
        )
        self.previous_time = None

    def pose(self, t):
        t = float(np.clip(t, self.times[0], self.times[-1]))

        p = np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ])

        R = self.rotations([t])[0].as_matrix()
        return p, R

    def velocity(self, t):
        eps = 1e-4
        t0 = max(self.times[0], t - eps)
        t1 = min(self.times[-1], t + eps)

        if t1 <= t0:
            return np.zeros(3), np.zeros(3)

        p0, R0 = self.pose(t0)
        p1, R1 = self.pose(t1)

        linear = (p1 - p0) / (t1 - t0)

        relative = Rotation.from_matrix(R1 @ R0.T)
        angular = relative.as_rotvec() / (t1 - t0)

        return linear, angular

    def apply(self, rigid_body, t):
        p, R = self.pose(t)
        v, omega = self.velocity(t)

        # SPlisHSPlasH expects a float32 quaternion [x, y, z, w].
        q = Rotation.from_matrix(R).as_quat().astype(np.float32)

        rigid_body.setPosition(np.asarray(p, dtype=np.float32))
        rigid_body.setRotation(q)
        rigid_body.setVelocity(np.asarray(v, dtype=np.float32))
        rigid_body.setAngularVelocity(
            np.asarray(omega, dtype=np.float32)
        )
        rigid_body.updateMeshTransformation()


def prepare_animated_scene(static_scene, output_scene):
    with open(static_scene) as f:
        scene = json.load(f)

    bodies = scene.get("RigidBodies", [])

    if len(bodies) < 2:
        raise ValueError(
            "Expected source and receiver rigid bodies"
        )

    # Generated scene order: source, receiver.
    bodies[0]["isDynamic"] = False
    bodies[0]["isAnimated"] = True

    bodies[1]["isDynamic"] = False
    bodies[1]["isAnimated"] = False

    with open(output_scene, "w") as f:
        json.dump(scene, f, indent=2)

    return output_scene


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trajectory",
        default="drake_vessel_trajectory.csv",
    )
    parser.add_argument(
        "--scene",
        default="sph_pouring_trial/scene_static.json",
    )
    parser.add_argument(
        "--duration", type=float, default=3.0
    )
    parser.add_argument(
        "--output",
        default="sph_pouring_trial/animated_output",
    )
    args = parser.parse_args()

    scene_path = Path(args.scene).resolve()
    animated_scene = scene_path.with_name(
        "scene_animated.json"
    )
    prepare_animated_scene(scene_path, animated_scene)

    replay = DrakeReplay(args.trajectory)

    simulator = sph.Exec.SimulatorBase()

    argv = [
        "SPHSimulator",
        str(animated_scene),
        "--no-gui",
        "--no-initial-pause",
        "--stopAt",
        str(args.duration),
        "--output-dir",
        str(Path(args.output).resolve()),
    ]

    # The exact pybind11 initialization signature can vary
    # between SPlisHSPlasH builds.
    try:
        simulator.init(argv, "SPHSimulator")
    except TypeError as exc:
        raise RuntimeError(
            "SimulatorBase.init() has a different signature "
            "in this build. Its pybind11 signature is:\n"
            f"{sph.Exec.SimulatorBase.init.__doc__}"
        ) from exc

    simulator.initSimulation()

    simulation = sph.Simulation.getCurrent()

    print("Before boundary initialization:",
          simulation.numberOfBoundaryModels())

    boundary_simulator = simulator.getBoundarySimulator()
    boundary_simulator.initBoundaryData()

    print("After boundary initialization:",
          simulation.numberOfBoundaryModels())

    print("\n=== SPH initialization diagnostics ===")

    print("Boundary model count:",
          simulation.numberOfBoundaryModels())

    print("Boundary simulator:",
          type(simulator.getBoundarySimulator()).__name__)

    print("Simulation time:",
          sph.TimeManager.getCurrent().getTime())

    for i in range(simulation.numberOfBoundaryModels()):
        boundary = simulation.getBoundaryModel(i)
        body = boundary.getRigidBodyObject()

        print(f"\nBoundary {i}:")
        print("  Type:", type(boundary).__name__)
        print("  Body type:", type(body).__name__)
        print("  Position:", body.getPosition())
        print("  Rotation:", body.getRotation())

    print("=== End diagnostics ===\n")

    if simulation.numberOfBoundaryModels() < 2:
        raise RuntimeError("Expected two boundary models")

    source_boundary = simulation.getBoundaryModel(1)
    source_body = source_boundary.getRigidBodyObject()

    source_body.setIsAnimated(True)

    print("Starting Drake-driven SPH replay")

    step_count = 0

    while sph.TimeManager.getCurrent().getTime() < args.duration:
        tm = sph.TimeManager.getCurrent()
        t = tm.getTime()

        replay.apply(source_body, t)

        # Update the moving boundary representation before
        # advancing the fluid solver.
        #
        # This API's argument signature and behavior must be
        # verified against the installed build.
        simulator.updateBoundaryParticles(True)

        simulator.step()
        step_count += 1

        if step_count % 100 == 0:
            print(
                f"t={tm.getTime():.3f} s "
                f"steps={step_count}"
            )

    print("Replay completed")
    print("Steps:", step_count)
    print("Output:", args.output)

    simulator.cleanup()


if __name__ == "__main__":
    main()
