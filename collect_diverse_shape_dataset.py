#!/usr/bin/env python3
"""Collect Drake grasp trials with NUMERICALLY varied vessel silhouettes.

Requires the existing generate_shape_families.py and collect_shape_grasp_dataset.py.
No changes to the original eight validated PROFILES are made on disk.

Example:
  python collect_diverse_shape_dataset.py --scales 3
  python collect_diverse_shape_dataset.py --scales 10
"""
import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import generate_shape_families as shapes
import collect_shape_grasp_dataset as old

ROOT = Path(__file__).resolve().parent
FRACTIONS = (0.33, 0.47, 0.60, 0.73, 0.80)


def varied_profile(base, rng, severity):
    """Change knot positions and radius plateaus, not just the vessel scale.

    Consecutive equal-radius knots share the same scale, retaining genuinely
    straight gripping sections. Preserve z=0,1 and positive ordered segments.
    """
    knots = [float(z) for z, _ in base]
    # Distinct internal transition locations, conservatively bounded.
    shifts = [0.0] * len(knots)
    for i in range(1, len(knots) - 1):
        shifts[i] = rng.uniform(-severity, severity)
    new_z = [0.0]
    min_gap = 0.035
    for i in range(1, len(knots) - 1):
        hi = 1.0 - (len(knots) - i - 1) * min_gap
        zi = max(new_z[-1] + min_gap, min(hi, knots[i] + shifts[i]))
        new_z.append(round(zi, 6))
    new_z.append(1.0)
    # Equal original radii form one *plateau*; perturb together.
    adjusted = []
    previous_original = None
    plateau_scale = 1.0
    global_scale = rng.uniform(0.96, 1.04)
    for z, radius in base:
        if previous_original is None or abs(radius - previous_original) > 1e-8:
            plateau_scale = rng.uniform(1.0 - severity * 1.5,
                                        1.0 + severity * 1.5)
        adjusted.append(float(radius) * plateau_scale * global_scale)
        previous_original = radius
    # Constrain overall diameter to approximately the working WSG grasp range;
    # this does not guarantee grasp feasibility for every candidate height.
    if max(adjusted) > 0.038:
        factor = 0.038 / max(adjusted)
        adjusted = [r * factor for r in adjusted]
    return [(z, round(max(0.010, r), 7)) for z, r in zip(new_z, adjusted)]


def run_trial(family, profile, name, height, mass, gh, x, y, log_file, timeout):
    # Child process imports the geometry module and overrides the selected
    # in-memory profile before calling both generation and the Drake station.
    # It never changes generate_shape_families.py on disk.
    code = (
        'import builtins\n'
        'import generate_shape_families as g\n'
        "builtins.input=lambda *a,**k:''\n"
        f'g.PROFILES[{family!r}]={profile!r}\n'
        f'g.generate({family!r},{name!r},{height!r},{mass!r},1.0)\n'
        f'g.run_station({name!r},{height!r},{profile[-1][1]!r},'
        f'{gh!r},{x!r},{y!r})\n'
    )
    t0 = time.monotonic()
    try:
        with log_file.open('w') as out:
            proc = subprocess.run([sys.executable, '-u', '-c', code],
                                  cwd=ROOT, stdout=out,
                                  stderr=subprocess.STDOUT, timeout=timeout)
        exit_code = proc.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    return (old.outcome(log_file.read_text(errors='replace'), exit_code),
            exit_code, time.monotonic() - t0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scales', type=int, default=3,
                        help='Distinct geometries per family (3 for pilot, 10+ for main dataset)')
    parser.add_argument('--seed', type=int, default=84210)
    parser.add_argument('--severity', type=float, default=0.08,
                        help='Max fractional-height breakpoint shift; default 0.08')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--positions', type=int, choices=[1,2], default=1)
    parser.add_argument('--families', nargs='+', choices=sorted(shapes.PROFILES),
                        default=list(shapes.PROFILES))
    args = parser.parse_args()
    if args.scales < 1 or not (0 < args.severity <= 0.12):
        parser.error('Use scales >= 1 and 0 < severity <= 0.12')

    rng = random.Random(args.seed)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = (ROOT / 'grasp_dataset' / 'shape_families' /
               f'diverse_n{args.scales}_p{args.positions}_{timestamp}_seed{args.seed}')
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / 'logs'
    logs.mkdir()
    geometry_profiles = {}
    rows = []
    counts = {}
    positions = [(0.508, -0.047), (0.498, -0.057)][:args.positions]
    total = len(args.families)*args.scales*len(positions)*len(FRACTIONS)
    tid = 0
    csv_path = run_dir / 'trials.csv'
    with csv_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=old.FIELDS)
        writer.writeheader()
        for family in args.families:
            baseline = shapes.PROFILES[family]
            for gi in range(args.scales):
                # The baseline vessel is retained only as a reference. All
                # experimental geometries have perturbed profile breakpoints.
                profile = varied_profile(baseline, rng, args.severity)
                height = rng.uniform(0.14, 0.16)
                mass = rng.uniform(0.050, 0.070)
                gh_payload = json.dumps([family, profile, height, mass], sort_keys=True)
                digest = hashlib.sha256(gh_payload.encode()).hexdigest()[:12]
                name = f'div_{family}_{digest}_{gi:03d}'
                geometry_profiles[name] = profile
                com, _, _ = shapes.calc_inertia(profile, height, mass)
                max_diameter = 2*max(r for _, r in profile)
                for pi, (x,y) in enumerate(positions):
                    for frac in FRACTIONS:
                        tid += 1
                        gh = height * frac
                        radius, slope = old.local_geometry(profile, height, frac)
                        feasible = 2*radius <= old.MAX_LOCAL_DIAMETER_M
                        log_file = logs / f'trial_{tid:04d}.log'
                        print(f'[{tid}/{total}] {family} geometry={gi} '
                              f'h={gh*1000:.1f}mm', flush=True)
                        if feasible:
                            result, ec, elapsed = run_trial(
                                family, profile, name, height, mass, gh,
                                x, y, log_file, args.timeout)
                        else:
                            result = dict(pickup_success=None, upright_grasp=None,
                                          pickup_tilt_deg=None,
                                          closure_horizontal_displacement_m=None,
                                          pour_success=None, stable_grasp_success=False,
                                          grasp_outcome='width_infeasible',
                                          failure_reason='width_infeasible')
                            ec, elapsed = '', 0
                        counts[result['failure_reason']] = counts.get(result['failure_reason'], 0) + 1
                        row = dict(trial_id=tid, dataset_run=run_dir.name,
                                   geometry_id=name, profile_hash=digest,
                                   shape_family=family, geometry_index=gi,
                                   position_index=pi, height_m=height, mass_kg=mass,
                                   radius_scale=1.0, com_height_m=com,
                                   source_x_m=x, source_y_m=y,
                                   grasp_fraction=frac, grasp_height_m=gh,
                                   local_radius_m=radius, local_slope_dr_dz=slope,
                                   max_diameter_m=max_diameter, width_feasible=feasible,
                                   exit_code=ec, wall_time_s=round(elapsed, 2),
                                   log_file=str(log_file.relative_to(ROOT)) if feasible else '',
                                   **result)
                        writer.writerow(row)
                        f.flush()
                        print(f'    {result["failure_reason"]} | tilt={result["pickup_tilt_deg"]}',
                              flush=True)
    settings = {
        'dataset_kind': 'diverse_shape_profiles_v1',
        'seed': args.seed,
        'profiles': {k: shapes.PROFILES[k] for k in args.families},
        'geometry_profiles': geometry_profiles,
        'severity': args.severity,
        'grasp_fractions': list(FRACTIONS),
        'note': 'Numerical profile saved per geometry_id; generated as solid convex-frustum segments',
    }
    (run_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    print('\nRESULTS')
    for name, count in sorted(counts.items()):
        print(f'  {name}: {count}')
    print(f'CSV: {csv_path.relative_to(ROOT)}')
    print('Training must read settings["geometry_profiles"][geometry_id].')


if __name__ == '__main__':
    main()
