"""Collect physics-based grasp labels for all eight Drake vessel families.

Put next to generate_shape_families.py and run with the Drake virtualenv.

Smoke test: python collect_shape_grasp_dataset_v2.py --smoke-test
First dataset: python collect_shape_grasp_dataset_v2.py --scales 3

The script imports YOUR CURRENT PROFILES dynamically; it does not override them.
A stable grasp is a completed physics-based pour, not merely a successful pickup.
"""
import argparse
import csv
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from generate_shape_families import PROFILES, calc_inertia, generate

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'grasp_dataset' / 'shape_families'
MAX_LOCAL_DIAMETER_M = 0.076  # Approximate width filter, NOT a guarantee of gripper clearance.

# Validated, *single geometry / single position* configurations at 0.15 m, 60 g.
BASELINE_GRASP_M = {
    'cylinder': 0.09,
    'conical': 0.07,
    'round_bottom': 0.11,
    'shouldered': 0.09,
    'inverted_taper': 0.09,
    'funnel': 0.09,
    'goblet': 0.09,
    'flared_rim': 0.11,
}

# Fractions of the candidate vessel's height. The range spans successful
# and unsuccessful grasps observed with the known 150-mm vessels.
GRASP_FRACTIONS = (0.33, 0.47, 0.60, 0.73, 0.80)

FIELDS = [
    'trial_id', 'dataset_run', 'geometry_id', 'profile_hash', 'shape_family',
    'geometry_index', 'position_index', 'height_m', 'mass_kg', 'radius_scale',
    'com_height_m', 'source_x_m', 'source_y_m', 'grasp_fraction',
    'grasp_height_m', 'local_radius_m', 'local_slope_dr_dz',
    'max_diameter_m', 'width_feasible',
    'pickup_success', 'upright_grasp', 'pickup_tilt_deg',
    'closure_horizontal_displacement_m', 'pour_success',
    'stable_grasp_success', 'grasp_outcome', 'failure_reason',
    'exit_code', 'wall_time_s', 'log_file',
]


def parse_boolean(log, label):
    vals = re.findall(r'^' + re.escape(label) + r':\s*(True|False)\s*$', log, re.M)
    return None if not vals else vals[-1] == 'True'


def parse_float(log, label):
    vals = re.findall(r'^' + re.escape(label) + r':\s*([+-]?(?:\d+\.?\d*|\.\d+))', log, re.M)
    return None if not vals else float(vals[-1])


def parse_text(log, label):
    vals = re.findall(r'^' + re.escape(label) + r':\s*(\S+)', log, re.M)
    return None if not vals else vals[-1]


def local_geometry(profile, height, fraction):
    for (z0, r0), (z1, r1) in zip(profile, profile[1:]):
        if fraction <= z1 + 1e-9:
            alpha = min(1.0, max(0.0, (fraction-z0)/(z1-z0)))
            return r0 + (r1-r0)*alpha, (r1-r0)/(height*(z1-z0))
    raise ValueError(f'Grasp fraction {fraction} outside profile')


def outcome(log, exit_code):
    pickup = parse_boolean(log, 'Pickup success')
    upright = parse_boolean(log, 'Upright grasp')
    pour = parse_boolean(log, 'Pouring motion completed')
    tilt = parse_float(log, 'Pickup tilt')
    displacement = parse_float(log, 'Closure horizontal displacement')
    reported = parse_text(log, 'Grasp outcome')
    if exit_code == 124:
        reason = 'timeout'
    elif exit_code != 0:
        reason = 'simulation_error'
    elif pickup is None or pour is None:
        reason = 'missing_summary'
    elif pickup is False:
        reason = 'pickup_failure'
    elif upright is False:
        reason = 'unstable_pickup'
    elif upright is None:
        reason = 'missing_upright_metric'
    elif 'WARNING: Vessel slipped relative to gripper.' in log:
        reason = 'grasp_slip'
    elif 'WARNING: Desired Cartesian pose was not reached.' in log:
        reason = 'ik_tracking_failure'
    elif pour is False:
        reason = 'other_pour_failure'
    else:
        reason = 'success'
    stable = (reason == 'success' and pickup is True and
              upright is True and pour is True)
    return dict(pickup_success=pickup, upright_grasp=upright,
                pickup_tilt_deg=tilt,
                closure_horizontal_displacement_m=displacement,
                pour_success=pour, stable_grasp_success=stable,
                grasp_outcome=reported or reason, failure_reason=reason)


def run_trial(family, name, height, mass, scale, grasp_height,
              x, y, log_path, timeout):
    argv = ['generate_shape_families.py', '--family', family, '--name', name,
            '--height', str(height), '--mass', str(mass),
            '--radius-scale', str(scale), '--grasp-height', str(grasp_height),
            '--source-x', str(x), '--source-y', str(y), '--run']
    # Bypass interactive Meshcat 'Press Enter' prompt; keep simulation physics.
    code = ('import builtins,runpy,sys\n'
            "builtins.input=lambda *args,**kwargs:''\n"
            f'sys.argv={argv!r}\n'
            "runpy.run_module('generate_shape_families',run_name='__main__')\n")
    start = time.monotonic()
    try:
        with log_path.open('w') as output:
            completed = subprocess.run([sys.executable, '-u', '-c', code],
                                       cwd=ROOT, stdout=output,
                                       stderr=subprocess.STDOUT,
                                       timeout=timeout, check=False)
        code_value = completed.returncode
    except subprocess.TimeoutExpired:
        code_value = 124
    elapsed = time.monotonic() - start
    return outcome(log_path.read_text(errors='replace'), code_value), code_value, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-test', action='store_true',
                        help='Eight trials using known good shape/height/mass combinations')
    parser.add_argument('--scales', type=int, default=3,
                        help='Random geometries per family (default: 3)')
    parser.add_argument('--positions', type=int, choices=[1, 2], default=1,
                        help='Test one or two nearby source positions (default 1)')
    parser.add_argument('--seed', type=int, default=4210)
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--families', nargs='+', choices=list(PROFILES), default=list(PROFILES))
    args = parser.parse_args()
    if args.scales < 1 or args.timeout < 1:
        parser.error('--scales and --timeout must be positive')
    randomizer = random.Random(args.seed)
    # Prevent mixing data from old versions of vessel profiles with new ones.
    canonical = json.dumps({k: PROFILES[k] for k in args.families}, sort_keys=True)
    profile_hash = hashlib.sha256(canonical.encode()).hexdigest()[:10]
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'{stamp}_{profile_hash}_seed{args.seed}'
    run_name = 'smoke' if args.smoke_test else f'n{args.scales}_p{args.positions}'
    run_dir = OUTPUT / f'{run_name}_{tag}'
    run_dir.mkdir(parents=True, exist_ok=False)
    logs = run_dir / 'logs'
    logs.mkdir()
    csv_path = run_dir / 'trials.csv'
    settings = dict(seed=args.seed, profile_hash=profile_hash,
                    profiles={k: PROFILES[k] for k in args.families},
                    mass_range_kg=[.050, .070], height_range_m=[.14, .16],
                    radius_scale_range=[.94, 1.06],
                    grasp_fractions=list(GRASP_FRACTIONS),
                    source_positions=[[.508,-.047], [.498,-.057]][:args.positions],
                    smoke_test=args.smoke_test)
    (run_dir / 'settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    fractions = [None] if args.smoke_test else GRASP_FRACTIONS
    num_geom = 1 if args.smoke_test else args.scales
    positions = [(.508, -.047), (.498, -.057)][:1 if args.smoke_test else args.positions]
    total = len(args.families)*num_geom*len(positions)*len(fractions)
    counts = {}
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        trial_id = 0
        for family in args.families:
            for gi in range(num_geom):
                if args.smoke_test:
                    height, mass, scale = .15, .06, 1.0
                else:
                    height = randomizer.uniform(.14, .16)
                    mass = randomizer.uniform(.050, .070)
                    scale = randomizer.uniform(.94, 1.06)
                profile = [(z, r*scale) for z,r in PROFILES[family]]
                com_height, _, _ = calc_inertia(profile, height, mass)
                max_diameter = 2*max(r for _,r in profile)
                # Include profile version in vessel name to avoid confusing
                # experimental geometries from different script revisions.
                name = f'ml_{family}_{profile_hash}_{args.seed}_{gi:03d}'
                generate(family, name, height, mass, scale)
                for pi, (x,y) in enumerate(positions):
                    for fraction in fractions:
                        trial_id += 1
                        frac = BASELINE_GRASP_M[family]/.15 if fraction is None else fraction
                        gh = frac*height
                        radius, slope = local_geometry(profile, height, frac)
                        feasible = 2*radius <= MAX_LOCAL_DIAMETER_M
                        log_file = logs / f'trial_{trial_id:04d}_{family}_{gi}_p{pi}_h{frac:.3f}.log'
                        print(f'[{trial_id}/{total}] {family} geometry={gi} '
                              f'grasp={gh*1000:.1f} mm', flush=True)
                        if feasible:
                            values, exit_code, elapsed = run_trial(
                                family, name, height, mass, scale, gh, x, y,
                                log_file, args.timeout)
                        else:
                            values = dict(pickup_success=None, upright_grasp=None,
                                          pickup_tilt_deg=None,
                                          closure_horizontal_displacement_m=None,
                                          pour_success=None, stable_grasp_success=False,
                                          grasp_outcome='width_infeasible',
                                          failure_reason='width_infeasible')
                            exit_code, elapsed = '', 0.0
                        reason = values['failure_reason']
                        counts[reason] = counts.get(reason,0)+1
                        writer.writerow(dict(
                            trial_id=trial_id, dataset_run=run_dir.name,
                            geometry_id=name, profile_hash=profile_hash,
                            shape_family=family, geometry_index=gi,
                            position_index=pi, height_m=height, mass_kg=mass,
                            radius_scale=scale, com_height_m=com_height,
                            source_x_m=x, source_y_m=y,
                            grasp_fraction=frac, grasp_height_m=gh,
                            local_radius_m=radius, local_slope_dr_dz=slope,
                            max_diameter_m=max_diameter, width_feasible=feasible,
                            exit_code=exit_code, wall_time_s=round(elapsed,2),
                            log_file=str(log_file.relative_to(ROOT)) if feasible else '',
                            **values))
                        handle.flush()
                        print(f'    {reason} | pickup_tilt={values["pickup_tilt_deg"]}',
                              flush=True)
    print('\nRESULTS')
    for key in sorted(counts):
        print(f'  {key}: {counts[key]}')
    print(f'CSV: {csv_path.relative_to(ROOT)}')
    if args.smoke_test and counts.get('success',0) != total:
        print('NOTE: Smoke test did not reproduce all eight baseline successes. '
              'Check local PROFILES, friction, mass, and generated trial logs.')


if __name__ == '__main__':
    main()
