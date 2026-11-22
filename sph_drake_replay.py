
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
from scipy.spatial.transform import Rotation

from drake_trajectory_export import PoseInterpolation

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

    return times, PoseInterpolation(times, positions, quaternions)


class DrakeReplay:
    def __init__(self, trajectory, *, hold_source=False):
        self.hold_source = hold_source
        self.times, self.interpolation = load_trajectory(trajectory)

    def pose(self, t):
        return self.interpolation.pose(self.times[0] if self.hold_source else t)

    def velocity(self, t):
        if self.hold_source or t < self.times[0] or t > self.times[-1]:
            return np.zeros(3), np.zeros(3)
        return self.interpolation.velocity(t)

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


class StepDiagnostics:
    """Trace the fastest changes by particle ID across neighbor sorting."""

    def __init__(self, path):
        self.path = path
        self.previous = {}
        self.rows = []

    def sample(self, simulation, source_time, start, end, source, receiver):
        dt = end - start
        snapshots = {}
        speed = []
        workspace_speed = []
        receiver_speed = []
        density = []
        strongest = (-1.,)
        impact = None
        max_projection = 0.
        for j in range(simulation.numberOfFluidModels()):
            model = simulation.getFluidModel(j)
            if not model.numActiveParticles():
                continue
            indices = range(model.numActiveParticles())
            ids = [(j, model.getParticleId(i)) for i in indices]
            p = np.array([model.getPosition(i) for i in indices])
            v = np.array([model.getVelocity(i) for i in indices])
            a = np.array([model.getAcceleration(i) for i in indices])
            rho = np.array([model.getDensity(i) for i in indices])
            speeds = np.linalg.norm(v, axis=1)
            speed.extend(speeds)
            density.extend(rho)
            workspace_speed.extend(speeds[p[:, 2] >= 0.])
            if receiver is not None:
                origin, contains = receiver
                receiver_speed.extend(speeds[contains(p - origin)])
            prior = np.array([self.previous.get(key, (pos - dt * vel, np.zeros(3)))
                              for key, pos, vel in zip(ids, p, v)])
            changes = np.linalg.norm(v - prior[:, 1], axis=1)
            projection = np.linalg.norm(p - prior[:, 0] - dt * v, axis=1)
            max_projection = max(max_projection, float(projection.max()))
            k = int(changes.argmax())
            if changes[k] > strongest[0]:
                impact = (model, j, k, prior[k, 0])
                strongest = (
                    changes[k], ids[k][1], np.linalg.norm(dt * a[k]),
                    np.linalg.norm(v[k] - prior[k, 1] - dt * a[k]),
                    rho[k], projection[k], *p[k],
                )
            snapshots.update(zip(ids, zip(p, v)))
        self.previous = snapshots
        if impact is None:
            raise RuntimeError("No active fluid particles to diagnose")
        model, fluid_index, particle_index, prior_position = impact
        wall_samples = {}
        for i in range(simulation.numberOfBoundaryModels()):
            boundary = simulation.getBoundaryModel(i)
            rigid_body = boundary.getRigidBodyObject()
            label = "source" if rigid_body.isAnimated() else "receiver"
            local = Rotation.from_quat(rigid_body.getRotation()).inv().apply(
                prior_position - rigid_body.getPosition()
            )
            distance = sph.Utilities.SDFFunctions.distance(boundary.getMap(), local, 0.)
            if distance == np.finfo(float).max:
                distance = None  # Outside the volume-map domain.
            volume = boundary.getBoundaryVolume(fluid_index, particle_index)
            sample = boundary.getBoundaryXj(fluid_index, particle_index)
            contribution = model.getDensity0() * volume * simulation.W(prior_position - sample)
            wall_samples[label] = (distance, contribution)
        body = source.getRigidBodyObject()
        self.rows.append((
            start, end, source_time, len(speed), *np.percentile(speed, [50, 95, 99]),
            max(speed), max(workspace_speed, default=0.), len(receiver_speed), max(receiver_speed, default=0.),
            max(density), *strongest, max_projection,
            *wall_samples["source"], *wall_samples["receiver"],
            *body.getVelocity(), *body.getAngularVelocity(),
        ))

    def save(self):
        with self.path.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow((
                "step_start_s", "time_s", "source_pose_time_s", "particles",
                "speed_p50_m_s", "speed_p95_m_s", "speed_p99_m_s", "speed_max_m_s",
                "workspace_max_speed_m_s",
                "receiver_particles", "receiver_max_speed_m_s", "density_max_kg_m3",
                "max_delta_v_m_s", "max_delta_v_particle_id", "nonpressure_delta_v_m_s",
                "residual_delta_v_m_s", "max_delta_v_density_kg_m3",
                "max_delta_v_position_correction_m", "max_delta_v_x_m", "max_delta_v_y_m",
                "max_delta_v_z_m", "max_position_correction_m",
                "source_sdf_m", "source_boundary_density_kg_m3",
                "receiver_sdf_m", "receiver_boundary_density_kg_m3",
                "source_vx_m_s", "source_vy_m_s", "source_vz_m_s",
                "source_wx_rad_s", "source_wy_rad_s", "source_wz_rad_s",
            ))
            writer.writerows(self.rows)


def replay_scene(scene_path, replay, duration, output, *, settle_duration=0.0, on_step=None,
                 diagnostics=False, diagnostic_receiver=None):
    """Return solver time, last applied trajectory time, and all fluid centers.

    The optional step callback receives the simulation and the start and
    end times on the solver clock, including any settling time.
    """
    if not np.isfinite(settle_duration) or settle_duration < 0:
        raise ValueError("Settling duration must be finite and nonnegative")
    trace = StepDiagnostics(Path(output) / "step_diagnostics.csv") if diagnostics else None
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
            if trace is not None:
                trace.sample(simulation, pose_time, step_start, current_time, source, diagnostic_receiver)
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
        if trace is not None:
            trace.save()


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
