#!/usr/bin/env python3
"""Frozen numerical-ML transfer test on four NEW axisymmetric shape families.

Run from pouring-bot's folder, with generate_shape_families.py,
collect_shape_grasp_dataset.py, train_shape_grasp_model.py installed.

  python test_novel_shape_transfer.py --per-family 1  # 20-trial pilot
  python test_novel_shape_transfer.py --per-family 5  # 100-trial test

The model trains ONLY on the original, diverse 8-family CSV. It predicts all
candidate grasps for the 4 new families and writes predictions to disk BEFORE
executing any Drake simulation. It never uses new-family labels to train.
"""
import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import generate_shape_families as shapes
import collect_shape_grasp_dataset as collector
from train_shape_grasp_model import CANDIDATE_FEATURES, add_features, make_model, choose_one_per_scenario

ROOT = Path(__file__).resolve().parent
FRACTIONS = [0.33, 0.47, 0.60, 0.73, 0.80]
INVALID = {'simulation_error', 'timeout', 'missing_summary', 'missing_upright_metric'}

# Knot (normalized height, radius in m) representations. Each adjacent
# pair makes one convex frustum; no topology modification to the Drake plant.
# New types are deliberately NOT in the original eight training families.
NOVEL_PROFILES = {
    'hourglass': [(0.00,.027),(.16,.027),(.38,.020),(.60,.020),(.84,.028),(1.00,.028)],
    'stepped': [(0.00,.021),(.25,.021),(.28,.028),(.57,.028),(.60,.022),(1.00,.022)],
    'double_bulge': [(0.00,.021),(.17,.029),(.34,.029),(.48,.021),(.61,.021),(.80,.030),(1.00,.026)],
    'collared': [(0.00,.022),(.38,.022),(.43,.030),(.62,.030),(.67,.022),(1.00,.022)],
}


def varying_profile(base, rng, severity):
    z_values = [float(z) for z,_ in base]
    # Change knot heights while keeping all segments non-degenerate.
    new_z = [0.0]
    for i in range(1,len(base)-1):
        min_z = new_z[-1]+0.025
        max_z = 1.0-(len(base)-1-i)*0.025
        new_z.append(round(min(max(z_values[i]+rng.uniform(-severity,severity),min_z),max_z),6))
    new_z.append(1.0)
    radii=[]
    scale=rng.uniform(.96,1.04)
    prev=None
    plateau=1.0
    for _,r in base:
        if prev is None or abs(r-prev)>1e-9:
            plateau=rng.uniform(.95,1.05)
        radii.append(round(r*scale*plateau,7))
        prev=r
    if max(radii) > .036:
        radii=[round(r*.036/max(radii),7) for r in radii]
    return list(zip(new_z,radii))


def load_train(path, seed):
    settings=json.loads((path.parent/'settings.json').read_text())
    train=pd.read_csv(path)
    train=train[~train.failure_reason.isin(INVALID)].copy()
    train=add_features(train,settings)
    train['target']=train.stable_grasp_success.astype(str).str.lower().eq('true').astype(int)
    for c in CANDIDATE_FEATURES:
        train[c]=pd.to_numeric(train[c],errors='coerce')
    train=train[np.isfinite(train[CANDIDATE_FEATURES].to_numpy(dtype=float)).all(axis=1)].copy()
    if train.target.nunique()!=2:
        raise SystemExit('Training CSV requires both successes and failures')
    model=make_model(seed)
    model.fit(train[CANDIDATE_FEATURES],train.target)
    frac=float(train.groupby('grasp_fraction').target.mean().idxmax())
    return model,frac,len(train)


def simulate_trial(row, profile, log_path, timeout):
    name=row['geometry_id']; family=row['shape_family']
    height=float(row['height_m']); mass=float(row['mass_kg'])
    grasp=float(row['grasp_height_m'])
    code=(
        'import builtins\n'
        'import generate_shape_families as g\n'
        "builtins.input=lambda *a,**k:''\n"
        f'g.PROFILES[{family!r}]={profile!r}\n'
        f'g.generate({family!r},{name!r},{height!r},{mass!r},1.0)\n'
        f'g.run_station({name!r},{height!r},{profile[-1][1]!r},'
        f'{grasp!r},0.508,-0.047)\n'
    )
    start=time.monotonic()
    try:
        with log_path.open('w') as f:
            result=subprocess.run([sys.executable,'-u','-c',code],cwd=ROOT,
                                  stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
        ec=result.returncode
    except subprocess.TimeoutExpired:
        ec=124
    return collector.outcome(log_path.read_text(errors='replace'),ec),ec,time.monotonic()-start


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-csv',type=Path,default=Path('grasp_dataset/shape_families/diverse_n3_p1_20261122_001031_seed84210/trials.csv'))
    p.add_argument('--per-family',type=int,default=1,help='New vessels per family: 1 pilot or 5 full test')
    p.add_argument('--seed',type=int,default=93717)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--prepare-only',action='store_true',help='Write frozen predictions and geometry without running Drake')
    p.add_argument('--resume',type=Path,help='Resume simulation for a previously prepared run directory')
    args=p.parse_args()
    if args.per_family<1 or args.timeout<1:
        p.error('--per-family and --timeout must be positive')
    training_path=args.train_csv.resolve()
    if not training_path.is_file() or not (training_path.parent/'settings.json').is_file():
        p.error('Training CSV and its settings.json must exist')

    if args.resume:
        outdir=args.resume.resolve()
        prepared=outdir/'candidate_predictions.csv'
        if not prepared.exists():
            p.error('Resume directory requires candidate_predictions.csv')
        candidates=pd.read_csv(prepared)
        settings=json.loads((outdir/'settings.json').read_text())
        profiles=settings['geometry_profiles']
    else:
        model,fixed_frac,train_count=load_train(training_path,args.seed)
        rng=random.Random(args.seed)
        stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
        outdir=ROOT/'grasp_dataset'/'novel_shape_tests'/f'novel_n{args.per_family}_{stamp}_seed{args.seed}'
        outdir.mkdir(parents=True,exist_ok=False)
        profiles={}; rows=[]
        for family,base in NOVEL_PROFILES.items():
            for gi in range(args.per_family):
                profile=varying_profile(base,rng,.045)
                height=rng.uniform(.14,.16)
                mass=rng.uniform(.05,.07)
                sig=json.dumps([family,profile,height,mass],sort_keys=True)
                digest=hashlib.sha256(sig.encode()).hexdigest()[:12]
                name=f'novel_{family}_{digest}_{gi:03d}'
                profiles[name]=profile
                com,_,_=shapes.calc_inertia(profile,height,mass)
                max_d=2*max(r for _,r in profile)
                for fraction in FRACTIONS:
                    radius,slope=collector.local_geometry(profile,height,fraction)
                    rows.append(dict(geometry_id=name,shape_family=family,
                        geometry_index=gi,position_index=0,
                        height_m=height,mass_kg=mass,radius_scale=1.0,
                        com_height_m=com,grasp_fraction=fraction,
                        grasp_height_m=fraction*height,local_radius_m=radius,
                        local_slope_dr_dz=slope,max_diameter_m=max_d,
                        width_feasible=2*radius<=.076,
                        source_x_m=.508,source_y_m=-.047))
        settings=dict(dataset_kind='novel_family_frozen_test_v1',seed=args.seed,
                      training_csv=str(training_path.relative_to(ROOT)) if training_path.is_relative_to(ROOT) else str(training_path),
                      profiles=NOVEL_PROFILES,geometry_profiles=profiles,
                      grasp_fractions=FRACTIONS,fixed_fraction=fixed_frac,
                      train_rows=train_count,
                      methodology='Model fit only on training_csv, predictions committed before new Drake trials')
        (outdir/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        candidates=pd.DataFrame(rows)
        enhanced=add_features(candidates,settings)
        # Width-infeasible options cannot be selected.
        scores=model.predict_proba(enhanced[CANDIDATE_FEATURES])[:,1]
        candidates['ml_score']=np.where(candidates.width_feasible,scores,-1.0)
        candidates['fixed_score']=np.where(candidates.width_feasible,
                             -abs(candidates.grasp_fraction-fixed_frac),-999.0)
        ml=choose_one_per_scenario(candidates,'ml_score')
        fx=choose_one_per_scenario(candidates,'fixed_score')
        candidates['selected_ml']=candidates.index.isin(ml.index)
        candidates['selected_fixed']=candidates.index.isin(fx.index)
        candidates.to_csv(outdir/'candidate_predictions.csv',index=False)
        print(f'Predictions frozen before simulation: {outdir / "candidate_predictions.csv"}',flush=True)
        print(f'Model trained on {train_count} trials from the ORIGINAL eight families.',flush=True)
        print(f'Fixed-fraction baseline learned from training: {fixed_frac:.2f}',flush=True)

    if args.prepare_only:
        print('Preparation complete. To simulate later, run:')
        print(f'  python {Path(__file__).name} --resume {outdir} --train-csv {training_path}')
        return

    logdir=outdir/'logs'; logdir.mkdir(exist_ok=True)
    output=outdir/'trial_results.csv'
    existing={}
    if output.exists():
        for row in csv.DictReader(output.open(newline='')):
            existing[(row['geometry_id'],row['grasp_fraction'])]=row
    fieldnames=list(candidates.columns)+['pickup_success','upright_grasp','pickup_tilt_deg',
                  'closure_horizontal_displacement_m','pour_success','stable_grasp_success',
                  'grasp_outcome','failure_reason','exit_code','wall_time_s','log_file']
    all_results=list(existing.values())
    for i,row in candidates.iterrows():
        key=(str(row.geometry_id),str(row.grasp_fraction))
        if key in existing:
            continue
        log=logdir/f'trial_{i+1:04d}.log'
        print(f'[{i+1}/{len(candidates)}] {row.shape_family} vessel={row.geometry_index} '
              f'h={row.grasp_height_m*1000:.1f}mm',flush=True)
        if bool(row.width_feasible):
            result,ec,elapsed=simulate_trial(row.to_dict(),profiles[row.geometry_id],log,args.timeout)
        else:
            result=dict(pickup_success=None,upright_grasp=None,pickup_tilt_deg=None,
                       closure_horizontal_displacement_m=None,pour_success=None,
                       stable_grasp_success=False,grasp_outcome='width_infeasible',
                       failure_reason='width_infeasible')
            ec,elapsed='',0.0
        record={**row.to_dict(),**result,'exit_code':ec,'wall_time_s':round(elapsed,2),
                'log_file':str(log.relative_to(ROOT)) if ec!='' else ''}
        all_results.append(record)
        with output.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fieldnames)
            writer.writeheader();writer.writerows(all_results)
        print(f'  {result["failure_reason"]} | pickup tilt={result["pickup_tilt_deg"]}',flush=True)

    df=pd.read_csv(output)
    if len(df)<len(candidates):
        print('Incomplete data collection; resume to finish.');return
    invalid=df.failure_reason.isin(INVALID)
    print('\nNOVEL SHAPE-FAMILY TRANSFER TEST')
    if invalid.any():
        print(f'WARNING: {invalid.sum()} simulator-error candidate rows; treating those scenarios as unevaluable.')
    for method,column in [('Numerical ML','selected_ml'),('Fixed baseline','selected_fixed')]:
        chosen=df[df[column].astype(str).str.lower().eq('true')]
        evaluable=chosen[~chosen.failure_reason.isin(INVALID)]
        hits=evaluable.stable_grasp_success.astype(str).str.lower().eq('true').sum()
        print(f'  {method}: {hits}/{len(evaluable)} ({100*hits/max(len(evaluable),1):.1f}%)')
    geom=df.groupby('geometry_id',sort=False)
    oracle=int(geom.apply(lambda x: x.stable_grasp_success.astype(str).str.lower().eq('true').any(),include_groups=False).sum())
    print(f'  Sampled-height oracle: {oracle}/{df.geometry_id.nunique()}')
    for family,g in df.groupby('shape_family'):
        chosen=g[g.selected_ml.astype(str).str.lower().eq('true')]
        successes=int(chosen.stable_grasp_success.astype(str).str.lower().eq('true').sum())
        print(f'  {family}: ML {successes}/{len(chosen)}')
    print(f'Results: {output.relative_to(ROOT)}')

if __name__=='__main__':
    main()
