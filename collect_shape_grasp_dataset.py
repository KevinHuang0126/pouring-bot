"""Collect grasp outcome labels for the 8 parametric Drake vessel families.

Put beside generate_shape_families.py and run in the pouring-bot virtualenv:
  ~/projects/pouring-bot/.venv/bin/python collect_shape_grasp_dataset.py --smoke-test
  ~/projects/pouring-bot/.venv/bin/python collect_shape_grasp_dataset.py --scales 3

This is a controlled physics-simulation dataset, not SPH validation.
"""
import argparse
import csv
import math
import random
import re
import subprocess
import sys
import time
from pathlib import Path

from generate_shape_families import PROFILES, generate

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'grasp_dataset' / 'shape_families'
MAX_DIAMETER_M = 0.076  # Geometry-only clearance screen, not contact guarantee.
FIELDS = [
    'trial_id', 'geometry_id', 'shape_family', 'scale_id',
    'height_m', 'mass_kg', 'radius_scale', 'source_x_m', 'source_y_m',
    'grasp_fraction', 'grasp_height_m', 'local_radius_m', 'local_slope_dr_dz',
    'max_diameter_m', 'width_feasible', 'pickup_success', 'upright_grasp',
    'pickup_tilt_deg', 'closure_horizontal_displacement_m',
    'pour_success', 'stable_grasp_success', 'grasp_outcome',
    'failure_reason', 'exit_code', 'wall_time_s', 'log_file',
]


def match_bool(log, label):
    m = re.search(r'^' + re.escape(label) + r':\s*(True|False)\s*$', log, re.M)
    return None if m is None else m.group(1) == 'True'


def match_float(log, label):
    m = re.search(r'^' + re.escape(label) + r':\s*([+-]?(?:\d+\.?\d*|\.\d+))', log, re.M)
    return None if m is None else float(m.group(1))


def match_text(log, label):
    m = re.search(r'^' + re.escape(label) + r':\s*(\S+)', log, re.M)
    return None if m is None else m.group(1)


def local_geometry(profile, height, fraction):
    for (z0, r0), (z1, r1) in zip(profile, profile[1:]):
        if fraction <= z1 + 1e-9:
            t = min(1., max(0., (fraction - z0) / (z1-z0)))
            radius = r0 + (r1-r0)*t
            slope = (r1-r0)/(height*(z1-z0))
            return radius, slope
    raise ValueError('grasp height outside profile')


def read_outcome(log, exit_code):
    pickup = match_bool(log, 'Pickup success')
    upright = match_bool(log, 'Upright grasp')
    pour = match_bool(log, 'Pouring motion completed')
    tilt = match_float(log, 'Pickup tilt')
    displacement = match_float(log, 'Closure horizontal displacement')
    reported = match_text(log, 'Grasp outcome')
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
    elif not pour:
        reason = 'other_pour_failure'
    else:
        reason = 'success'
    # 'pickup_success' means lifted only; don't substitute it for stable grasp.
    stable = (pickup is True and upright is True and pour is True
              and reason == 'success')
    return dict(pickup_success=pickup, upright_grasp=upright,
                pickup_tilt_deg=tilt, closure_horizontal_displacement_m=displacement,
                pour_success=pour, stable_grasp_success=stable,
                grasp_outcome=reported or reason, failure_reason=reason)


def run_trial(family, name, height, mass, scale, grasp_h, x, y, log_path, timeout):
    argv = ['generate_shape_families.py', '--family', family, '--name', name,
            '--height', str(height), '--mass', str(mass), '--radius-scale', str(scale),
            '--grasp-height', str(grasp_h), '--source-x', str(x), '--source-y', str(y), '--run']
    # In a child process, disable the terminal's final input() prompt.
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
        exit_code = completed.returncode
    except subprocess.TimeoutExpired:
        exit_code = 124
    log = log_path.read_text(errors='replace')
    return read_outcome(log, exit_code), exit_code, time.monotonic()-start


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-test', action='store_true',
                   help='One geometry and one grasp per family (8 trials)')
    p.add_argument('--scales', type=int, default=2,
                   help='Number of random geometries per family (default 2)')
    p.add_argument('--seed', type=int, default=4210)
    p.add_argument('--timeout', type=int, default=180)
    p.add_argument('--families', nargs='+', choices=list(PROFILES),
                   default=list(PROFILES))
    args = p.parse_args()
    if args.scales < 1 or args.timeout < 1:
        p.error('--scales and --timeout must be positive')
    randomizer = random.Random(args.seed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logs = OUTPUT / 'logs'
    logs.mkdir(exist_ok=True)
    grasp_fracs = [0.25, 0.50, 0.75]
    if args.smoke_test:
        grasp_fracs = [0.50]
    n_scales = 1 if args.smoke_test else args.scales
    output_name = f"shape_trials_seed{args.seed}_{'smoke' if args.smoke_test else 'n'+str(n_scales)}.csv"
    csv_file = OUTPUT / output_name
    total = len(args.families) * n_scales * len(grasp_fracs)
    with csv_file.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        trial = 0
        for family in args.families:
            for gi in range(n_scales):
                # Reproducible independent geometry, shared by all grasp heights.
                if args.smoke_test:
                    height, mass, scale = .15, .12, 1.0
                else:
                    height = randomizer.uniform(.12, .18)
                    mass = randomizer.uniform(.09, .16)
                    scale = randomizer.uniform(.82, 1.08)
                x, y = .508, -.047  # Fix scene position initially to isolate shape effects.
                name = f'ml_shape_{family}_{args.seed}_{gi:03d}'
                profile = [(z, r*scale) for z, r in PROFILES[family]]
                max_diameter = 2*max(r for _, r in profile)
                # Generate once and keep geometry fixed for trials at different heights.
                generate(family, name, height, mass, scale)
                for frac in grasp_fracs:
                    trial += 1
                    gh = frac*height
                    radius, slope = local_geometry(profile, height, frac)
                    # A wide top does not automatically prevent a lower grasp:
                    # screen LOCAL diameter only, and let Drake test approach collisions.
                    feasible = 2*radius <= MAX_DIAMETER_M
                    log_file = logs / f'trial_{trial:04d}_{name}_g{int(frac*100)}.log'
                    print(f'[{trial}/{total}] {family} #{gi} grasp={frac:.0%} '
                          f'local_diameter={2*radius:.3f} m', flush=True)
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
                    row = dict(trial_id=trial, geometry_id=name, shape_family=family,
                               scale_id=gi, height_m=height, mass_kg=mass,
                               radius_scale=scale, source_x_m=x, source_y_m=y,
                               grasp_fraction=frac, grasp_height_m=gh,
                               local_radius_m=radius, local_slope_dr_dz=slope,
                               max_diameter_m=max_diameter, width_feasible=feasible,
                               exit_code=exit_code, wall_time_s=round(elapsed, 2),
                               log_file=str(log_file.relative_to(ROOT)) if feasible else '',
                               **values)
                    writer.writerow(row)
                    handle.flush()
                    print(f"  -> {values['failure_reason']} (tilt={values['pickup_tilt_deg']})", flush=True)
    print(f'CSV saved: {csv_file.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
