"""Compare ML and fixed-height grasps on paired, reproducible Drake/SPH runs."""

import argparse
import csv
from dataclasses import asdict, dataclass
import io
import json
from math import isfinite
from pathlib import Path
import statistics
import subprocess
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp

import numpy as np
from scipy.spatial.transform import Rotation

from drake_trajectory_export import load_trajectory
from run_ml_sph_pour import (
    SHASHA_COMMIT,
    hollow_profile,
    load_shape_modules,
    select_particle_spacing,
)

ROOT = Path(__file__).resolve().parent
METHODS = ("ml", "manual")
FINAL_HOLD_S = 1.
COUNTS = ("total_particles", "in_receiver", "in_source", "outside_both", "below_world_z_zero")
RETIREMENT_OPTIONS = ("retire_escaped", "escape_plane_z")
RETIREMENT_FIELDS = ("active_particles", "retired_particles", "retired_volume_ml", "active_outside_both")
SCORING = (
    "Provisional, resolution-dependent particle-center endpoint score: "
    "100 * in_receiver / total_particles. Failures score zero."
)


@dataclass(frozen=True)
class Settings:
    trials: int = 10
    seed: int = 84211
    fixed_grasp_height: float = .09
    particle_spacing: float = .0035
    fill_fraction_min: float = .25
    fill_fraction_max: float = .65
    retire_escaped: bool = True
    escape_plane_z: float | None = None

    def __post_init__(self):
        if self.trials < 1 or self.seed < 0:
            raise ValueError("Trials must be positive and the seed must be nonnegative")
        if not 0 < self.fixed_grasp_height < .16:
            raise ValueError("Fixed grasp height must be positive and below 0.16 m")
        if not isfinite(self.particle_spacing) or self.particle_spacing <= 0:
            raise ValueError("Particle spacing must be finite and positive")
        if not 0 < self.fill_fraction_min <= self.fill_fraction_max <= 1:
            raise ValueError("Fill bounds must satisfy 0 < min <= max <= 1")
        if self.escape_plane_z is not None and (not self.retire_escaped or not isfinite(self.escape_plane_z)):
            raise ValueError("A finite escape plane requires escaped-particle retirement")


def load_profiles():
    with TemporaryDirectory(prefix="grasp-benchmark-profiles-") as snapshot:
        generator, _, _, transfer = load_shape_modules(Path(snapshot))
        return {**generator.PROFILES, **transfer.NOVEL_PROFILES}


def sample_trials(settings, profiles):
    # Use the upstream dataset's geometry ranges and the rectangle between its
    # two validated source positions. Both strategies consume one profile snapshot.
    wall, floor = .003, .004
    children = np.random.SeedSequence(settings.seed).spawn(settings.trials)
    families = sorted(profiles)
    for number, child in enumerate(children, 1):
        seed = int(child.generate_state(1, dtype=np.uint64)[0])
        rng = np.random.default_rng(seed)
        for _ in range(200):
            family = str(rng.choice(families))
            height = float(rng.uniform(.14, .16))
            mass = float(rng.uniform(.050, .070))
            scale = float(rng.uniform(.94, 1.06))
            source_x = float(rng.uniform(.498, .508))
            source_y = float(rng.uniform(-.057, -.047))
            fill = float(rng.uniform(settings.fill_fraction_min, settings.fill_fraction_max))
            if height <= settings.fixed_grasp_height:
                continue
            profile = [[z, r * scale] for z, r in profiles[family]]
            hollow = hollow_profile(profile, height, wall, floor)
            try:
                select_particle_spacing(hollow, fill, settings.particle_spacing)
            except ValueError:
                continue
            yield {
                "trial": number, "seed": seed, "family": family, "height_m": height,
                "mass_kg": mass, "radius_scale": scale, "source_x_m": source_x,
                "source_y_m": source_y, "fill_fraction": fill,
                "wall_m": wall, "floor_m": floor, "particle_spacing_m": settings.particle_spacing,
                "normalized_height_radius_m": profile,
                "methods": {name: {"status": "pending"} for name in METHODS},
            }
            break
        else:
            raise ValueError(f"Could not sample a seedable geometry for trial {number}. "
                             "Check the particle spacing, fill bounds, and fixed grasp height.")


def run_logged(script, args, log):
    with log.open("w") as stream:
        result = subprocess.run(
            [sys.executable, "-u", str(ROOT / script), *map(str, args)],
            stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT,
        )
    if result.returncode:
        raise RuntimeError(f"{script} exited with code {result.returncode}. See {log.name}.")


def run_drake(trial, method, settings, profile, attempt):
    args = [
        "--profile-json", profile, "--wall", trial["wall_m"], "--floor", trial["floor_m"],
        "--source-x", trial["source_x_m"], "--source-y", trial["source_y_m"],
        "--fill-fraction", trial["fill_fraction"], "--particle-spacing", trial["particle_spacing_m"],
        "--grasp-mode", method, "--output", attempt / "run", "--drake-only",
    ]
    if method == "manual":
        args.extend(("--grasp-height", settings.fixed_grasp_height))
    run_logged("run_ml_sph_pour.py", args, attempt / "drake.log")


def run_sph(package, end_time, settings, attempt, cache):
    args = [
        "--station-export", package, "--sph-start-time", "auto",
        "--particle-spacing", settings.particle_spacing, "--end-drake-time", end_time,
        "--final-hold-duration", FINAL_HOLD_S, "--headless", "--scene-dir", cache,
        "--output", attempt / "run",
    ]
    if settings.retire_escaped:
        args.append("--retire-escaped")
        if settings.escape_plane_z is not None:
            args.extend(("--escape-plane-z", settings.escape_plane_z))
    run_logged("evaluate_sph_spill.py", args, attempt / "sph.log")


def read_json(path):
    return json.loads(path.read_text())


def inspect_drake(package, trial, result):
    status = read_json(package / "station_motion.json")
    result["drake_status"] = status
    geometry = read_json(package / "vessel_geometry.json")
    result["grasp_height_m"] = geometry["grasp_height_m"]
    if not status["pickup_success"]:
        raise ValueError("Drake did not complete pickup")
    end = status.get("pour_end")
    if not end:
        raise ValueError("Drake did not complete the active pour before recovery")
    config = read_json(package / "cups.json")
    if (config["fill_fraction"] != trial["fill_fraction"]
            or 2 * config["particle_radius"] != trial["particle_spacing_m"]):
        raise ValueError("Drake export did not preserve the paired fill and particle spacing")

    samples = load_trajectory(package / "source_trajectory.csv").get("source", [])
    stamp = float(end["drake_time_s"])
    if len(samples) < 2 or stamp <= samples[0][0]:
        raise ValueError("No recorded motion through the active-pour endpoint")
    pose = next((pose for time, pose in samples if time == stamp), None)
    if pose is None:
        raise ValueError("The active-pour endpoint has no matching measured source pose")
    axis = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()[:, 2]
    mouth = pose[:3] + trial["height_m"] * axis
    mouth_error = float(np.linalg.norm(mouth - end["target_mouth_xyz_m"]))
    axis_error = float(np.rad2deg(np.arccos(np.clip(axis @ end["target_axis_world"], -1, 1))))
    tilt = float(np.rad2deg(np.arccos(np.clip(axis[2], -1, 1))))
    if not np.isfinite([mouth_error, axis_error, tilt]).all():
        raise ValueError("Nonfinite measured pour-pose errors")
    result["measured_pour_endpoint"] = {
        "drake_time_s": stamp, "mouth_xyz_m": mouth.tolist(), "mouth_error_m": mouth_error,
        "axis_error_deg": axis_error, "tilt_deg": tilt,
    }
    # Use the planner's successful tilt/hold marker and independently reject a
    # source that is upright or far from the commanded mouth/axis at that sample.
    if mouth_error > .075 or axis_error > 28. or tilt < 60.:
        raise ValueError(f"Recorded pour pose missed target: tilt {tilt:.1f} deg, "
                         f"mouth error {mouth_error:.3f} m, axis error {axis_error:.1f} deg")
    return stamp


def atomic_text(path, value):
    with NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        stream.write(value)
    Path(stream.name).replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    return {
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "population_stddev": statistics.pstdev(values) if values else None,
    }


def write_results(output, state):
    atomic_json(output / "trials.json", state)
    rows = []
    differences = []
    columns = ("trial", "seed", "family", "height_m", "mass_kg", "radius_scale",
               "source_x_m", "source_y_m", "wall_m", "floor_m", "fill_fraction",
               "particle_spacing_m")
    fields = (*columns, "grasp_mode", "grasp_height_m", "status", "drake_success", "sph_success",
              "transfer_percent", *COUNTS, *RETIREMENT_FIELDS, "pour_end_drake_time_s", "failure_reason",
              "source_trajectory", "sph_report")
    for trial in state["trials"]:
        for method, result in trial["methods"].items():
            rows.append({**{k: trial[k] for k in columns}, "grasp_mode": method,
                         **{k: result.get(k) for k in fields[len(columns) + 1:]}})
        if all(trial["methods"][method]["status"] == "complete" for method in METHODS):
            ml, manual = (trial["methods"][method]["transfer_percent"] for method in METHODS)
            differences.append({"trial": trial["trial"], "ml_minus_baseline_pp": ml - manual})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fields)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(output / "trials.csv", stream.getvalue())

    by_method = {}
    for method in METHODS:
        completed = [row for row in rows if row["grasp_mode"] == method and row["status"] == "complete"]
        count = len(completed)
        drake = sum(row["drake_success"] for row in completed)
        sph = sum(row["sph_success"] for row in completed)
        by_method[method] = {
            "completed_trials": count, "transfer_percent": stats([row["transfer_percent"] for row in completed]),
            "drake_success_rate": drake / count if count else None,
            "sph_success_rate": sph / count if count else None,
            "sph_success_rate_given_drake_success": sph / drake if drake else None,
        }
    atomic_json(output / "summary.json", {
        "settings": state["settings"], "scoring": SCORING, "final_hold_s": FINAL_HOLD_S,
        "methods": by_method,
        "paired_ml_minus_baseline_pp": {
            "by_trial": differences, **stats([row["ml_minus_baseline_pp"] for row in differences]),
        },
    })


def benchmark(output, overrides=None, *, resume=False):
    """Run both Drake attempts before starting either SPH attempt in each pair."""
    overrides = overrides or {}
    manifest = output / "trials.json"
    if resume:
        state = read_json(manifest)
        if state["schema_version"] != 1 or state["shape_generator_commit"] != SHASHA_COMMIT:
            raise ValueError("Benchmark manifest version or pinned generator differs")
        # Legacy manifests predate retirement. Keep their original physics
        # setting for any remaining evaluations in the same benchmark.
        settings = Settings(**{"retire_escaped": False, **state["settings"]})
        if any(getattr(settings, key) != value for key, value in overrides.items()):
            raise ValueError("Resume options must match the saved benchmark settings")
    else:
        settings = Settings(**overrides)
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Choose a new or empty output directory, or use --resume")
        state = {
            "schema_version": 1, "shape_generator_commit": SHASHA_COMMIT,
            "settings": asdict(settings), "trials": list(sample_trials(settings, load_profiles())),
        }
    pending = any(result["status"] != "complete" for trial in state["trials"]
                  for result in trial["methods"].values())
    if resume and not pending and (output / "trials.csv").is_file():
        try:
            summary = read_json(output / "summary.json")
            if all(summary["methods"][method]["completed_trials"] == len(state["trials"])
                   for method in METHODS):
                return state
        except (OSError, ValueError, KeyError, TypeError):
            # The manifest may have been saved before the derived outputs.
            pass
    if settings.retire_escaped and pending:
        check_retirement_binding()
    write_results(output, state)

    for trial in state["trials"]:
        pair = output / f"trial_{trial['trial']:03d}"
        if all(result["status"] == "complete" for result in trial["methods"].values()):
            continue
        pair.mkdir(exist_ok=True)
        profile = pair / "profile.json"
        if not profile.exists():
            atomic_json(profile, {key: trial[key] for key in (
                "family", "height_m", "mass_kg", "normalized_height_radius_m",
            )})
        for stage in ("drake", "sph"):
            for method, result in trial["methods"].items():
                if result["status"] != ("pending" if stage == "drake" else "drake_complete"):
                    continue
                run_stage(output, state, trial, method, settings, profile, pair / method, stage)
    return state


def run_stage(output, state, trial, method, settings, profile, method_dir, stage):
    result = trial["methods"][method]
    method_dir.mkdir(exist_ok=True)
    attempt_key = f"{stage}_attempt"
    attempt = output / result[attempt_key] if attempt_key in result else None
    suffix = "run/drake" if stage == "drake" else "run"
    checkpoints = ("station_motion.json", "vessel_geometry.json") if stage == "drake" else ("spill_report.json",)
    reuse = False
    if attempt is not None:
        try:
            reuse = all(isinstance(read_json(attempt / suffix / key), dict) for key in checkpoints)
        except (OSError, ValueError):
            # Interrupted subprocesses can leave a partially written report.
            pass
    if not reuse:
        attempt = Path(mkdtemp(prefix=f"{stage}-", dir=method_dir))
        result[attempt_key] = str(attempt.relative_to(output))
    folder = attempt / suffix
    if stage == "drake":
        result.update(
            source_trajectory=str((folder / "source_trajectory.csv").relative_to(output)),
            station_motion=str((folder / "station_motion.json").relative_to(output)),
            vessel_geometry=str((folder / "vessel_geometry.json").relative_to(output)),
            cups_config=str((folder / "cups.json").relative_to(output)),
            drake_log=str((attempt / "drake.log").relative_to(output)),
        )
    else:
        result.update(sph_report=str((folder / "spill_report.json").relative_to(output)),
                      sph_log=str((attempt / "sph.log").relative_to(output)))
    atomic_json(output / "trials.json", state)
    print(f"Trial {trial['trial']}/{settings.trials} {method} {stage}", flush=True)
    try:
        if stage == "drake":
            if not reuse:
                run_drake(trial, method, settings, profile, attempt)
            end_time = inspect_drake(folder, trial, result)
            result.update(status="drake_complete", drake_success=True, pour_end_drake_time_s=end_time)
        else:
            if not reuse:
                run_sph((output / result["source_trajectory"]).parent, result["pour_end_drake_time_s"],
                        settings, attempt, output / "station_cache")
            report = read_json(folder / "spill_report.json")
            if (report["particle_spacing_m"] != trial["particle_spacing_m"]
                    or report["initial_fill"]["requested_fill_fraction"] != trial["fill_fraction"]
                    or abs(report["applied_drake_time_s"] - result["pour_end_drake_time_s"]) > 1e-8
                    or report["requested_final_hold_duration_s"] != FINAL_HOLD_S):
                raise ValueError("SPH report does not match the paired fill, spacing, or scoring endpoint")
            counts = {key: report[key] for key in COUNTS}
            total = counts["total_particles"]
            if (any(type(value) is not int or not 0 <= value <= total for value in counts.values())
                    or not total or total != sum(counts[key] for key in ("in_source", "in_receiver", "outside_both"))):
                raise ValueError("Invalid SPH particle accounting")
            active = report.get("active_particles", total)
            retired = report.get("retired_particles", 0)
            active_outside = report.get("active_outside_both", counts["outside_both"] - retired)
            retired_volume = report.get("retired_volume_ml", 0.)
            if (report.get("escape_retirement", {}).get("enabled", False) != settings.retire_escaped
                    or not all(type(n) is int and n >= 0 for n in (active, retired, active_outside))
                    or active + retired != total or active_outside + retired != counts["outside_both"]
                    or not isfinite(retired_volume) or retired_volume < 0):
                raise ValueError("Invalid SPH retirement accounting")
            result.update(counts, status="complete", sph_success=True, retired_particles=retired,
                          retired_volume_ml=retired_volume, active_particles=active,
                          active_outside_both=active_outside,
                          transfer_percent=100 * counts["in_receiver"] / total)
    except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
        result.update(status="complete", sph_success=False, transfer_percent=0.,
                      failure_stage=stage, failure_reason=str(exc))
        if stage == "drake":
            result["drake_success"] = False
    write_results(output, state)


def check_retirement_binding():
    try:
        from sph_drake_replay import require_retirement_binding
    except ImportError as exc:
        raise RuntimeError("Benchmark retirement requires the patched SPH binding. Run scripts/setup_sph.sh.") from exc
    require_retirement_binding()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", action="store_true", help="Continue using the saved trial settings")
    defaults = Settings()
    for name, value in asdict(defaults).items():
        if name not in RETIREMENT_OPTIONS:
            parser.add_argument("--" + name.replace("_", "-"), type=type(value), help=f"Default: {value}")
    parser.add_argument("--retire-escaped", action=argparse.BooleanOptionalAction,
                        help="Retire escaped particles (enabled for new benchmarks)")
    parser.add_argument("--escape-plane-z", type=float, help="Override the safe world-Z cutoff in meters")
    args = vars(parser.parse_args())
    output, resume = args.pop("output").resolve(), args.pop("resume")
    try:
        benchmark(output, {key: value for key, value in args.items() if value is not None}, resume=resume)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    print(f"{SCORING}\nResults: {output / 'summary.json'}")


if __name__ == "__main__":
    main()
