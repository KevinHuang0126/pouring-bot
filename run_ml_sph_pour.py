"""Run a generated-vessel Drake experiment and evaluate its exported SPH package."""

import argparse
from hashlib import sha256
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np

from evaluate_sph_spill import add_replay_start_arguments, read_restart_thresholds
from fluid_model import DEFAULT_FILL_FRACTION
from prepare_sph_pouring import VesselProfile, initial_profile_fill

ROOT = Path(__file__).resolve().parent
SHASHA_COMMIT = "2fadb4b0dbf88d2c626bdfb9371d5978ccd474a3"
TRAINING_DIR = "grasp_dataset/shape_families/diverse_n10_p1_20261122_103654_seed84210"


def upstream_file(relative):
    result = subprocess.run(["git", "-C", str(ROOT), "show", f"{SHASHA_COMMIT}:{relative}"],
                            capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Fetch origin shasha/ml-grasp-dataset to obtain the pinned upstream inputs")
    return result.stdout


def hollow_profile(profile, height, wall, floor):
    """Offset the radial silhouette inward, preserving every physical height knot."""
    if not np.isfinite([height, wall, floor]).all() or not (height > floor > 0 and wall > 0):
        raise ValueError("Require finite positive wall and floor thickness below the rim height")
    if len(profile) < 2 or profile[0][0] != 0 or profile[-1][0] != 1:
        raise ValueError("Normalized profile must span z=0 through z=1")
    outer = tuple((z * height, r) for z, r in profile)
    if any(b[0] <= a[0] for a, b in zip(outer, outer[1:])):
        raise ValueError("Profile heights must be strictly increasing")
    floor_radius = float(np.interp(floor, *np.asarray(outer).T))
    inner = ((floor, floor_radius - wall), *((z, r - wall) for z, r in outer if z > floor))
    return VesselProfile(outer, inner, segments=48)


def select_grasp(training, trainer, collector, transfer, geometry, profile, com):
    import pandas as pd

    for filename in ("trials.csv", "settings.json"):
        (training / filename).write_bytes(upstream_file(f"{TRAINING_DIR}/{filename}"))
    # A supplied snapshot may come from the training set. Leave out those
    # geometry groups when predicting for the same numerical silhouette.
    settings = json.loads((training / "settings.json").read_text())
    duplicate_ids = [
        key for key, candidate in settings["geometry_profiles"].items()
        if len(candidate) == len(profile) and np.allclose(candidate, profile, atol=1e-8, rtol=0)
    ]
    if duplicate_ids:
        trials = pd.read_csv(training / "trials.csv")
        trials.loc[~trials.geometry_id.isin(duplicate_ids)].to_csv(training / "trials.csv", index=False)
    model, _, usable = transfer.load_train(training / "trials.csv", 84210)
    geometry_id = geometry["geometry_id"]
    height = geometry["height_m"]
    rows = []
    for fraction in transfer.FRACTIONS:
        radius, slope = collector.local_geometry(profile, height, fraction)
        if 2 * radius > collector.MAX_LOCAL_DIAMETER_M:
            continue
        rows.append(dict(geometry_id=geometry_id, shape_family=geometry["family"], radius_scale=1.,
                         height_m=height, mass_kg=geometry["mass_kg"], com_height_m=com,
                         max_diameter_m=2 * max(r for _, r in profile), position_index=0,
                         grasp_fraction=fraction, grasp_height_m=height * fraction,
                         local_radius_m=radius, local_slope_dr_dz=slope))
    if not rows:
        raise ValueError("No width-feasible grasp candidates")
    features = trainer.add_features(pd.DataFrame(rows), {"profiles": {}, "geometry_profiles": {
        geometry_id: profile,
    }})
    features["score"] = model.predict_proba(features[trainer.CANDIDATE_FEATURES])[:, 1]
    chosen = trainer.choose_one_per_scenario(features, "score").iloc[0]
    geometry.update(training_dataset=TRAINING_DIR, training_rows_used=usable,
                    model_score=float(chosen["score"]))
    return float(chosen["grasp_height_m"])


def load_shape_modules(upstream):
    modules = ("generate_shape_families", "collect_shape_grasp_dataset", "train_shape_grasp_model",
               "test_novel_shape_transfer")
    for module in modules:
        (upstream / f"{module}.py").write_bytes(upstream_file(f"{module}.py"))
    sys.path.insert(0, str(upstream))
    try:
        return tuple(map(import_module, modules))
    finally:
        sys.path.remove(str(upstream))


def generate(args, output, upstream):
    generator, collector, trainer, transfer = load_shape_modules(upstream)
    profiles = {**generator.PROFILES, **transfer.NOVEL_PROFILES}
    if args.profile_json:
        snapshot = json.loads(args.profile_json.read_text())
        profile = snapshot["normalized_height_radius_m"]
        height, mass = snapshot["height_m"], snapshot["mass_kg"]
        family = snapshot["family"]
    else:
        if args.family not in profiles:
            raise ValueError(f"Unknown family. Choose from {', '.join(profiles)}")
        family, height, mass = args.family, args.height, args.mass
        profile = [(z, r * args.radius_scale) for z, r in profiles[family]]
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("Vessel mass must be positive and finite")
    hollow = hollow_profile(profile, height, args.wall, args.floor)
    digest = sha256(json.dumps([family, height, mass, profile]).encode()).hexdigest()[:12]
    name = f"generated_{digest}"
    com, _, _ = generator.calc_inertia(profile, height, mass)
    geometry = dict(vessel_name=name, geometry_id=digest, family=family, height_m=height,
                    mass_kg=mass, com_height_m=com, normalized_height_radius_m=profile,
                    outer_height_radius_m=hollow.outer, inner_height_radius_m=hollow.inner,
                    radial_wall_thickness_m=args.wall, floor_thickness_m=args.floor,
                    shasha_commit=SHASHA_COMMIT, grasp_mode=args.grasp_mode)
    grasp = args.grasp_height
    if args.grasp_mode == "ml":
        grasp = select_grasp(upstream, trainer, collector, transfer, geometry, profile, com)
    if grasp is None or not 0 < grasp < height:
        raise ValueError("Specify a grasp height strictly within the vessel height")
    geometry["grasp_height_m"] = grasp
    geometry["integration_base_commit"] = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()

    generator.OUTPUT_DIR = output / "geometry"
    generator.PROFILES[family] = profile
    generator.generate(family, name, height, mass, 1.)
    model = generator.OUTPUT_DIR / f"{name}.sdf"
    model.write_text(model.read_text().replace("package://pouring_bot/generated/",
                                               generator.OUTPUT_DIR.as_uri() + "/"))
    return hollow, geometry, generator.OUTPUT_DIR



def select_particle_spacing(profile, fraction, requested):
    """Prefer 3.5 mm, preserving fill for shapes that need the old lattice."""
    spacing = .0035 if requested is None else requested
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("Particle spacing must be positive and finite")
    try:
        initial_profile_fill(profile, fraction, spacing / 2)
    except ValueError as exc:
        if requested is not None:
            raise
        initial_profile_fill(profile, fraction, .0015)
        return .003, str(exc)
    return spacing, None



def run_drake(args, output, hollow, geometry, models_dir):
    import fluid_model
    import pour_planner
    from simple_pour_trajectory import VESSELS, VesselSpec
    from station_sph_export import StationSphExport
    import visualize_station as station

    name = geometry["vessel_name"]
    h = geometry["height_m"]
    mouth = hollow.outer[-1][1]
    VESSELS[name] = VesselSpec(name, h, mouth, .7 * mouth)
    station.VESSELS = (*station.VESSELS, name)
    station.GRASP_HEIGHTS[name] = geometry["grasp_height_m"]
    station.GRIP_FORCE_LIMITS[name] = 60.
    pour_planner.VESSEL_TILT_ANGLES[name] = 95.
    pour_planner.VESSEL_TILT_DURATIONS[name] = 10.
    fluid_model.EFFECTIVE_AREA_FRACTIONS[name] = .05
    drake_output = output / "drake"
    capture = StationSphExport(drake_output, ROOT / "models", name, source_profile=hollow,
                               source_models_dir=models_dir, metadata=geometry, fill_fraction=args.fill_fraction,
                               particle_spacing=geometry["particle_spacing_m"])
    station.main(["--vessel", name, "--export-dir", str(drake_output), "--export-partial"],
                 capture=capture, source_xy=(args.source_x, args.source_y))
    return capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--family")
    inputs.add_argument("--profile-json", type=Path, help="Exact normalized profile snapshot")
    parser.add_argument("--height", type=float, default=.15)
    parser.add_argument("--mass", type=float, default=.12)
    parser.add_argument("--radius-scale", type=float, default=1.)
    parser.add_argument("--wall", type=float, default=.003)
    parser.add_argument("--floor", type=float, default=.004)
    parser.add_argument("--grasp-mode", choices=("manual", "ml"), required=True)
    parser.add_argument("--grasp-height", type=float)
    parser.add_argument("--fill-fraction", type=float, default=DEFAULT_FILL_FRACTION)
    parser.add_argument("--source-x", type=float, default=.508)
    parser.add_argument("--source-y", type=float, default=-.047)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-dir", type=Path, default=Path("sph_pouring_trial/station_cache"))
    parser.add_argument("--duration", type=float, help="Limit SPH replay for a smoke test")
    parser.add_argument("--export-fps", type=float, default=6.)
    add_replay_start_arguments(parser, start_default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--drake-only", action="store_true")
    args = parser.parse_args()
    if not 0 < args.fill_fraction <= 1:
        parser.error("--fill-fraction must be greater than zero and at most one")
    if args.grasp_height is not None and args.grasp_mode != "manual":
        parser.error("--grasp-height requires manual mode")
    output = args.output.resolve()
    try:
        output.mkdir(parents=True, exist_ok=True)
        if any(output.iterdir()):
            raise ValueError("Choose a new or empty output directory")
        thresholds = read_restart_thresholds(args.sph_restart_thresholds)
        with TemporaryDirectory(prefix="pouring-upstream-") as snapshot:
            hollow, geometry, models_dir = generate(args, output, Path(snapshot))
        spacing, fallback = select_particle_spacing(hollow, args.fill_fraction, args.particle_spacing)
        geometry.update(particle_spacing_m=spacing, particle_spacing_fallback_reason=fallback)
        if fallback:
            print(f"Using 3 mm spacing to preserve the requested fill. {fallback}", flush=True)
        capture = run_drake(args, output, hollow, geometry, models_dir)
        if not args.drake_only and len(capture.recorder.rows) < 2:
            print("Drake produced only the initial pose. See drake/station_motion.json.")
            return
        if not args.drake_only:
            from evaluate_sph_spill import evaluate

            report = evaluate(capture.output / "source_trajectory.csv", output / "sph",
                              args.scene_dir, duration=args.duration, config=capture.config,
                              export_fps=0. if args.headless else args.export_fps,
                              sph_start_time=args.sph_start_time,
                              sph_prepour_seconds=args.sph_prepour_seconds,
                              settle_duration=args.settle_duration, restart_thresholds=thresholds)
            print(json.dumps(report, indent=2))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
