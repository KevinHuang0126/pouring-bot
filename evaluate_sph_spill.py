"""Evaluate a Drake source-vessel trajectory with the generated SPH cup scene."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
from tempfile import mkdtemp
from threading import Lock

import numpy as np

from drake_trajectory_export import TrajectoryRecorder
from prepare_sph_pouring import (
    CUP_BOTTOM,
    DEFAULT_CONFIG,
    PouringConfig,
    cylindrical_profile,
    prepare_scene,
)
from sph_start import RestartThresholds, choose_start


DEFAULT_TEMPLATE = Path(__file__).with_name("DoubleDamBreak_export.json")
_REPLAY_LOCK = Lock()


def inside_cup(points, cup, *, bottom=CUP_BOTTOM):
    """Test particle centers in the faceted cavity in cup-local coordinates."""
    return cylindrical_profile((*cup[:3], bottom)).contains(points)


def classify(positions, source_origin, source_rotation, *, config=DEFAULT_CONFIG):
    if len(positions) == 0 or not np.isfinite(positions).all():
        raise RuntimeError("SPH returned empty or non-finite particle positions")
    source_local = (positions - source_origin) @ source_rotation
    receiver_local = (positions - config.receiver_origin) @ config.receiver_rotation().as_matrix()
    in_source = (config.source_profile.contains(source_local) if config.source_profile
                 else inside_cup(source_local, config.source_cup[:3], bottom=config.source_cup[3]))
    in_receiver = (config.receiver_profile.contains(receiver_local) if config.receiver_profile
                   else inside_cup(receiver_local, config.receiver_cup[:3], bottom=config.receiver_cup[3]))
    if np.any(in_source & in_receiver):
        raise RuntimeError("Source and receiver cavity classifications overlap")
    total = len(positions)
    source_count = int(in_source.sum())
    receiver_count = int(in_receiver.sum())
    outside_count = total - source_count - receiver_count
    return {
        "total_particles": total,
        "in_source": source_count,
        "in_receiver": receiver_count,
        "outside_both": outside_count,
        "collected_fraction": receiver_count / total,
        "remaining_fraction": source_count / total,
        "outside_fraction": outside_count / total,
        "below_world_z_zero": int((positions[:, 2] < 0).sum()),
    }


def evaluate(
    trajectory: str | Path | TrajectoryRecorder,
    output: str | Path | None = None,
    scene_dir: str | Path | None = None,
    template: str | Path = DEFAULT_TEMPLATE,
    duration: float | None = None,
    *,
    hold_source: bool = False,
    settle_duration: float | None = None,
    config: PouringConfig = DEFAULT_CONFIG,
    diagnostics: bool = False,
    export_fps: float = 30.,
    cfl_factor: float | None = None,
    sph_start_time: float | str = 0,
    sph_prepour_seconds: float = 2.,
    particle_spacing: float | None = None,
    restart_thresholds: RestartThresholds = RestartThresholds(),
    end_drake_time: float | None = None,
    final_hold_duration: float = 0.,
) -> dict:
    """Replay a CSV or an in-memory Drake recorder and return the JSON report.

    Each run owns its output and scene. Pass the same scene_dir to reuse
    volume maps across serial runs. When output is omitted, a fresh temporary
    directory is created and its path is returned in the report.
    settle_duration holds the selected pose before replay. Stationary restarts
    settle for 0.5 s by default. Other profile fills and tilted starts use 0.25 s.
    """
    from sph_drake_replay import DrakeReplay, prepare_animated_scene, replay_scene

    if particle_spacing is not None:
        if config.fill_fraction is None:
            raise ValueError("Changing particle spacing requires a saved profile fill fraction")
        config = replace(config, particle_radius=float(particle_spacing) / 2)

    output = Path(mkdtemp(prefix="sph-evaluation-")) if output is None else Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("output must be a new or empty directory")
    scene_path = output / "scene"
    scene_path.mkdir(parents=True)
    if isinstance(trajectory, TrajectoryRecorder):
        trajectory = trajectory.save(output / "input_trajectory.csv")

    replay = DrakeReplay(trajectory, hold_source=hold_source)
    selection = choose_start(replay, config, sph_start_time, sph_prepour_seconds, restart_thresholds)
    start = selection.elapsed_s
    time_origin = replay.input_time_origin_s
    initial_pose = replay.pose(start)
    if settle_duration is None:
        tilted = not np.allclose(initial_pose[1][:, 2], [0., 0., 1.], atol=1e-6)
        settle_duration = 0.25 if tilted or config.fill_fraction is not None else 0.
        if start:
            settle_duration = 0.5
    settle_duration = float(settle_duration)
    if not np.isfinite(settle_duration) or settle_duration < 0:
        raise ValueError("Settling duration must be finite and nonnegative")
    if start and settle_duration == 0:
        raise ValueError("A stationary restart requires a positive settling duration")

    interval = selection.interval_s
    checks = selection.checks or {}
    start_info = {
        "requested": sph_start_time,
        "method": "stationary-pose restart" if start else "full trajectory",
        "selected_elapsed_s": start,
        "selected_drake_time_s": time_origin + start,
        "onset_drake_time_s": time_origin + selection.onset_s if selection.onset_s is not None else None,
        "requested_prepour_s": sph_prepour_seconds,
        "stationary_interval": {
            "start_drake_time_s": time_origin + interval[0],
            "end_drake_time_s": time_origin + interval[1],
            "duration_s": interval[1] - interval[0],
            "max_translation_speed_m_s": checks["translation_speed_m_s"]["measured"],
            "max_angular_speed_deg_s": checks["angular_speed_deg_s"]["measured"],
        } if interval is not None else None,
        "restart_safety_checks": checks,
        "settling_duration_s": settle_duration,
        "initial_zero_fluid_velocity_assumption_used": bool(start),
        "fallback_reason": selection.fallback_reason,
    }
    available = replay.times[-1] - start
    if end_drake_time is not None:
        if duration is not None:
            raise ValueError("Specify either duration or an end Drake timestamp")
        duration = float(end_drake_time) - time_origin - start
    duration = float(available if duration is None else duration)
    final_hold_duration = float(final_hold_duration)
    if not np.isfinite(final_hold_duration) or final_hold_duration < 0:
        raise ValueError("Final hold duration must be finite and nonnegative")
    if not np.isfinite(duration) or not 0 < duration <= available:
        raise ValueError("Duration must be positive and within the exported trajectory")

    scene = prepare_scene(trajectory, scene_path, template, settings=config,
                          export_fps=export_fps, cfl_factor=cfl_factor,
                          initial_pose=initial_pose if start else None)
    animated_scene = scene_path / "scene_animated.json"
    prepare_animated_scene(scene, animated_scene)
    if scene_dir is not None and Path(scene_dir).resolve() != scene_path:
        cache = Path(scene_dir).resolve() / "Cache"
        cache.mkdir(parents=True, exist_ok=True)
        (scene_path / "Cache").symlink_to(cache, target_is_directory=True)

    summary = []
    if selection.onset_s is not None:
        summary.append(f"Detected pouring onset: {start_info['onset_drake_time_s']:.3f} s")
    if interval is not None:
        summary.append(f"Stationary interval: {time_origin + interval[0]:.3f}–"
                       f"{time_origin + interval[1]:.3f} s")
    if selection.fallback_reason:
        summary.append(f"SPH start fallback: {selection.fallback_reason}")
    summary.extend([
        f"Selected SPH start: {start_info['selected_drake_time_s']:.3f} s",
        f"Initialization: {start_info['method']}",
        "Fluid initialization: zero velocity",
        f"Settling duration: {settle_duration:g} s",
        f"Skipped Drake motion: {start:.3f} s",
    ])
    print("\n".join(summary), flush=True)

    # The native simulator owns process-global state.
    options = {"headless": True} if export_fps == 0 else {}
    if start:
        options["start_time"] = start
    if final_hold_duration:
        options["final_hold_duration"] = final_hold_duration
    if diagnostics:
        receiver = config.receiver_profile or cylindrical_profile(config.receiver_cup)
        receiver_rotation = config.receiver_rotation().as_matrix()
        options.update(diagnostics=True, diagnostic_receiver=(
            config.receiver_origin, lambda local: receiver.contains(local @ receiver_rotation),
        ))
    with _REPLAY_LOCK:
        time, pose_time, positions = replay_scene(
            animated_scene, replay, duration, output, settle_duration=settle_duration, **options,
        )

    origin, rotation = replay.pose(pose_time)
    report = {
        "schema_version": 1,
        "classification": "final_particle_centers",
        "spill_measurement_validated": False,
        "output_dir": str(output),
        "requested_duration_s": duration,
        "requested_settle_duration_s": settle_duration,
        "requested_final_hold_duration_s": final_hold_duration,
        "requested_end_drake_time_s": time_origin + start + duration,
        "simulation_time_s": time,
        "input_time_origin_s": replay.input_time_origin_s,
        "sph_start": start_info,
        "particle_spacing_m": 2 * config.particle_radius,
        "applied_drake_time_s": replay.input_time_origin_s + pose_time,
        "source_held_fixed": hold_source,
        "applied_source_pose_time_s": 0.0 if hold_source else pose_time,
        **classify(positions, origin, rotation, config=config),
    }
    if config.fill_fraction is not None:
        report["initial_fill"] = json.loads(scene.read_text())["initialFill"]
    else:
        report["initial_fill_metadata_status"] = "Legacy box: original planner fill fraction unavailable"
    (output / "spill_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def add_replay_start_arguments(parser, *, start_default="0", spacing_default=None):
    parser.add_argument("--sph-start-time", default=start_default,
                        help="Original CSV timestamp, 0 for full replay, or auto")
    parser.add_argument("--sph-prepour-seconds", type=float, default=2.,
                        help="Target lead-in within the final safe stationary interval")
    parser.add_argument("--sph-restart-thresholds", type=Path,
                        help="JSON overrides for the stationary restart thresholds")
    parser.add_argument("--settle-duration", type=float, metavar="SECONDS",
                        help="Fixed-pose dwell (default: 0.5 s for stationary restarts, "
                             "0.25 s for other profile fills or tilted starts)")
    parser.add_argument("--particle-spacing", type=float, default=spacing_default,
                        help="Particle center spacing in meters")


def read_restart_thresholds(path):
    return RestartThresholds(**json.loads(path.read_text())) if path else RestartThresholds()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--trajectory", type=Path)
    inputs.add_argument(
        "--station-export", type=Path,
        help="Directory exported by visualize_station.py, including its cup configuration",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-dir", type=Path,
        help="Reusable volume-map cache directory for serial runs",
    )
    parser.add_argument(
        "--template", type=Path,
        default=DEFAULT_TEMPLATE,
    )
    horizon = parser.add_mutually_exclusive_group()
    horizon.add_argument("--duration", type=float, help="Replay seconds after settling")
    horizon.add_argument("--end-drake-time", type=float, help="Stop replay at this original CSV timestamp")
    parser.add_argument("--final-hold-duration", type=float, default=0.,
                        help="Hold the last replay pose fixed for this many solver seconds")
    add_replay_start_arguments(parser)
    parser.add_argument("--hold-source", action="store_true", help="Run a fixed-vessel control")
    export = parser.add_mutually_exclusive_group()
    export.add_argument("--export-fps", type=float, default=30., help="VTK frames per second")
    export.add_argument("--headless", action="store_true", help="Skip VTK output and keep the spill report")
    parser.add_argument("--cfl-factor", type=float, help="Override the template CFL factor")
    parser.add_argument("--diagnostics", action="store_true", help="Write per-step particle diagnostics")
    parser.add_argument("--config", type=Path, help="JSON cup and fill overrides")
    args = parser.parse_args()
    if args.station_export is not None and args.config is not None:
        parser.error("--station-export uses the cup configuration from Drake")
    trajectory = args.trajectory
    config_path = args.config
    if args.station_export is not None:
        trajectory = args.station_export / "source_trajectory.csv"
        config_path = args.station_export / "cups.json"
    try:
        config = DEFAULT_CONFIG if config_path is None else PouringConfig(
            **json.loads(config_path.read_text())
        )
        report = evaluate(
            trajectory, args.output, args.scene_dir, args.template, args.duration,
            hold_source=args.hold_source, settle_duration=args.settle_duration, config=config,
            diagnostics=args.diagnostics, export_fps=0. if args.headless else args.export_fps,
            cfl_factor=args.cfl_factor,
            sph_start_time=args.sph_start_time, sph_prepour_seconds=args.sph_prepour_seconds,
            particle_spacing=args.particle_spacing,
            restart_thresholds=read_restart_thresholds(args.sph_restart_thresholds),
            end_drake_time=args.end_drake_time, final_hold_duration=args.final_hold_duration,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
