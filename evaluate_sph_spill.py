"""Evaluate a Drake source-vessel trajectory with the generated SPH cup scene."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from prepare_sph_pouring import (
    CUP_BOTTOM,
    CUP_SEGMENTS,
    RECEIVER_CUP,
    RECEIVER_ORIGIN,
    SOURCE_CUP,
    prepare_scene,
)


def inside_cup(points, cup):
    """Test particle centers in the faceted cavity in cup-local coordinates."""
    radius, _, height = cup
    half_segment = np.pi / CUP_SEGMENTS
    # Distance along the normal to the closest side of the regular polygon.
    x, y, z = points.T
    angle = np.arctan2(y, x) % (2 * half_segment) - half_segment
    distance = np.hypot(x, y) * np.cos(angle)
    return (
        (distance < radius * np.cos(half_segment))
        & (z > CUP_BOTTOM)
        & (z < height)
    )


def classify(positions, source_origin, source_rotation):
    if len(positions) == 0 or not np.isfinite(positions).all():
        raise RuntimeError("SPH returned empty or non-finite particle positions")
    in_source = inside_cup((positions - source_origin) @ source_rotation, SOURCE_CUP)
    in_receiver = inside_cup(positions - RECEIVER_ORIGIN, RECEIVER_CUP)
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
        "outside_fraction": outside_count / total,
        "below_world_z_zero": int((positions[:, 2] < 0).sum()),
    }


def evaluate(trajectory, output, scene_dir, template, duration=None, *, hold_source=False):
    from sph_drake_replay import DrakeReplay, prepare_animated_scene, replay_scene

    replay = DrakeReplay(trajectory, hold_source=hold_source)
    if replay.times[0] != 0:
        raise ValueError("The Drake source trajectory must start at t=0")
    _, initial_rotation = replay.pose(0)
    if not np.allclose(initial_rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError("The generated fill requires an identity initial source rotation")
    if duration is None:
        duration = float(replay.times[-1])
    if not np.isfinite(duration) or not 0 < duration <= replay.times[-1]:
        raise ValueError("Duration must be positive and within the exported trajectory")

    scene = prepare_scene(trajectory, scene_dir, template)
    output.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        dir=scene.parent, prefix="scene_animated_", suffix=".json"
    ) as animated_scene:
        prepare_animated_scene(scene, animated_scene.name)
        time, pose_time, positions = replay_scene(
            animated_scene.name, replay, duration, output
        )

    origin, rotation = replay.pose(pose_time)
    report = {
        "schema_version": 1,
        "classification": "final_particle_centers",
        "spill_measurement_validated": False,
        "requested_duration_s": duration,
        "simulation_time_s": time,
        "source_held_fixed": hold_source,
        "applied_source_pose_time_s": 0.0 if hold_source else pose_time,
        **classify(positions, origin, rotation),
    }
    (output / "spill_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--scene-dir", type=Path,
        help="Reusable generated-scene and volume-map cache directory",
    )
    parser.add_argument(
        "--template", type=Path,
        default=Path(__file__).with_name("DoubleDamBreak_export.json"),
    )
    parser.add_argument("--duration", type=float, help="Default is the trajectory end time")
    parser.add_argument("--hold-source", action="store_true", help="Run a fixed-vessel control")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty directory")

    output = args.output.resolve()
    report = evaluate(
        args.trajectory, output, args.scene_dir or output / "scene",
        args.template, args.duration, hold_source=args.hold_source,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
