
#!/usr/bin/env python3
"""Experimental one-way Drake -> SPlisHSPlasH trajectory replay.

Requires:
    pysplishsplash Python bindings
    drake_vessel_trajectory.csv
    sph_pouring_trial/scene_static.json

The Drake trajectory prescribes the source vessel pose and velocity.
"""

import argparse
import csv
import json
from tempfile import NamedTemporaryFile
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

    if len(times) < 2 or not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
        raise ValueError("Trajectory needs finite, increasing timestamps")
    if not np.isfinite(positions).all() or not np.isfinite(quaternions).all():
        raise ValueError("Trajectory poses must be finite")

    rotations = Slerp(
        times, Rotation.from_quat(quaternions)
    )

    return times, positions, rotations


class DrakeReplay:
    def __init__(self, trajectory, *, hold_source=False):
        self.hold_source = hold_source
        self.times, self.positions, self.rotations = (
            load_trajectory(trajectory)
        )

    def pose(self, t):
        t = self.times[0] if self.hold_source else np.clip(t, self.times[0], self.times[-1])

        p = np.array([
            np.interp(t, self.times, self.positions[:, i])
            for i in range(3)
        ])

        R = self.rotations([t])[0].as_matrix()
        return p, R

    def velocity(self, t):
        if self.hold_source or t < self.times[0] or t > self.times[-1]:
            return np.zeros(3), np.zeros(3)

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

    def apply(self, boundary, t, *, stationary=False):
        body = boundary.getRigidBodyObject()
        p, R = self.pose(t)
        v, omega = (np.zeros(3), np.zeros(3)) if stationary else self.velocity(t)

        # The binding uses Eigen quaternion coefficient order (x, y, z, w).
        body.setPosition(p)
        body.setRotation(Rotation.from_matrix(R).as_quat())
        body.setVelocity(v)
        body.setAngularVelocity(omega)
        body.updateMeshTransformation()

        # CFL takes the square root of the maximum velocity accumulator.
        speed_bound = np.linalg.norm(v) + boundary.getMaxDist() * np.linalg.norm(omega)
        boundary.setMaxVel(speed_bound ** 2)


def prepare_animated_scene(static_scene, output_scene):
    with open(static_scene) as f:
        scene = json.load(f)

    bodies = scene.get("RigidBodies", [])

    if len(bodies) != 2:
        raise ValueError("Expected two scene bodies in source, receiver order")

    bodies[0].update(isDynamic=False, isAnimated=True)
    bodies[1].update(isDynamic=False, isAnimated=False)

    with open(output_scene, "w") as f:
        json.dump(scene, f, indent=2)


def replay_scene(scene_path, replay, duration, output, *, settle_duration=0.0, on_step=None):
    """Return solver time, last applied trajectory time, and all fluid centers.

    The optional step callback receives the simulation and the start and
    end times on the solver clock, including any settling time.
    """
    if not np.isfinite(settle_duration) or settle_duration < 0:
        raise ValueError("Settling duration must be finite and nonnegative")
    simulator = sph.Exec.SimulatorBase()
    simulator.init(
        sceneFile=str(scene_path), outputDir=str(output),
        useGui=False, initialPause=False,
    )
    simulator.initSimulation()

    try:
        simulation = sph.Simulation.getCurrent()
        if simulation.getBoundaryHandlingMethod() != sph.BoundaryHandlingMethods.Bender2019:
            raise ValueError("Replay requires Bender2019 volume-map boundaries")

        boundary_simulator = simulator.getBoundarySimulator()
        if not isinstance(boundary_simulator, sph.Exec.StaticBoundarySimulator):
            raise RuntimeError("Replay requires a StaticBoundarySimulator")

        # The binding exposes deferred initialization through runSimulation().
        # Start past a temporary stop time so it initializes without stepping.
        tm = sph.TimeManager.getCurrent()
        initial_time = tm.getTime()
        simulator.setValueFloat(simulator.STOP_AT, 1.0)
        tm.setTime(2.0)
        simulator.runSimulation()
        tm.setTime(initial_time)
        simulator.setValueFloat(simulator.STOP_AT, -1.0)

        boundaries = [
            simulation.getBoundaryModel(i)
            for i in range(simulation.numberOfBoundaryModels())
        ]
        sources = [bm for bm in boundaries if bm.getRigidBodyObject().isAnimated()]
        if len(boundaries) != 2 or len(sources) != 1:
            raise RuntimeError("Expected an animated source and a static receiver")

        source = sources[0]
        body = source.getRigidBodyObject()
        offsets = np.asarray(body.getVertices()) - body.getPosition()
        source.setMaxDist(np.linalg.norm(offsets, axis=1).max())

        trajectory_start = None if settle_duration else initial_time
        pose_time = 0.0
        step_count = 0
        fluid_frames = []
        output = Path(output)
        vtk_output = output / "vtk"
        previous_frames = {
            path.name: path.stat().st_mtime_ns
            for path in vtk_output.glob("ParticleData_Fluid_*.vtk")
        }
        if settle_duration:
            print(f"Settling fluid for {settle_duration:g} s with the source fixed", flush=True)
        else:
            print("Starting Drake-driven SPH replay", flush=True)
        while True:
            step_start = tm.getTime()
            if trajectory_start is None and step_start - initial_time >= settle_duration:
                trajectory_start = step_start
                print(f"Starting Drake replay at solver t={step_start:.6f} s", flush=True)
            settling = trajectory_start is None
            trajectory_time = 0.0 if settling else step_start - trajectory_start
            if not settling and trajectory_time >= duration:
                break
            pose_time = trajectory_time
            replay.apply(source, pose_time, stationary=settling)
            # The solver samples the maps at the updated pose and uses the
            # rigid body's point velocities. step() alone only exports frames.
            if not simulator.timeStepNoGUI():
                raise RuntimeError("SPH stopped before the requested duration")
            current_time = tm.getTime()
            if not np.isfinite(current_time) or current_time <= step_start:
                raise RuntimeError("SPH simulation time stopped advancing")
            step_count += 1
            # VTK has no timestamp. Record the step that produced each frame
            # and the vessel pose sampled by that step.
            frame = len(fluid_frames) + 1
            vtk_path = vtk_output / f"ParticleData_Fluid_{frame}.vtk"
            if (vtk_path.is_file()
                    and vtk_path.stat().st_mtime_ns != previous_frames.get(vtk_path.name)):
                fluid_frames.append((frame, current_time, pose_time))
            if on_step is not None:
                on_step(simulation, step_start, current_time)
            if step_count % 100 == 0:
                print(f"t={tm.getTime():.3f} s steps={step_count}", flush=True)

        if fluid_frames:
            with (output / "fluid_frames.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(("frame", "time_s", "source_pose_time_s"))
                writer.writerows(fluid_frames)

        print(f"Replay completed at t={tm.getTime():.3f} s after {step_count} steps")
        print(f"Output directory {output}")
        positions = np.array([
            model.getPosition(i)
            for model in (simulation.getFluidModel(j)
                          for j in range(simulation.numberOfFluidModels()))
            for i in range(model.numActiveParticles())
        ]).reshape(-1, 3)
        return tm.getTime(), pose_time, positions
    finally:
        simulator.cleanup()


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
    parser.add_argument("--settle-duration", type=float, default=0.0, metavar="SECONDS")
    args = parser.parse_args()

    if not np.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be positive and finite")

    if not np.isfinite(args.settle_duration) or args.settle_duration < 0:
        parser.error("--settle-duration must be finite and nonnegative")

    scene_path = Path(args.scene).resolve()
    replay = DrakeReplay(args.trajectory)
    # Keep relative scene assets and the volume-map cache resolvable.
    with NamedTemporaryFile(
        dir=scene_path.parent, prefix="scene_animated_", suffix=".json"
    ) as animated_scene:
        prepare_animated_scene(scene_path, animated_scene.name)
        replay_scene(
            animated_scene.name, replay, args.duration, Path(args.output).resolve(),
            settle_duration=args.settle_duration,
        )


if __name__ == "__main__":
    main()
