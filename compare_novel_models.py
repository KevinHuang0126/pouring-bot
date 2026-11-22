#!/usr/bin/env python3
"""Compare the 120- and 450-trial numerical grasp models on one existing novel-vessel set.

No Drake calls are made. Reuses the candidate outcomes recorded in trial_results.csv.
The novel set must be disjoint from both training sets. Both random forests use
identical features, hyperparameters, and random seed.

Usage (from pouring-bot repository):
 python compare_novel_models.py \
   --small-train grasp_dataset/shape_families/diverse_n3_p1_20261122_001031_seed84210/trials.csv \
   --large-train grasp_dataset/shape_families/diverse_n10_p1_20261122_103654_seed84210/trials.csv \
   --novel grasp_dataset/novel_shape_tests/novel_n5_20261122_101555_seed93717/trial_results.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_shape_grasp_model import CANDIDATE_FEATURES, add_features, make_model, choose_one_per_scenario

INVALID = {'simulation_error', 'timeout', 'missing_summary', 'missing_upright_metric'}


def load_training(csv_file):
    csv_file = csv_file.resolve()
    settings_file = csv_file.parent / 'settings.json'
    if not csv_file.is_file() or not settings_file.is_file():
        raise ValueError(f'Missing training CSV or settings.json: {csv_file}')
    raw = pd.read_csv(csv_file)
    if raw.failure_reason.isin(INVALID).any():
        raw = raw.loc[~raw.failure_reason.isin(INVALID)].copy()
    settings = json.loads(settings_file.read_text())
    df = add_features(raw, settings)
    df['target'] = df.stable_grasp_success.astype(str).str.lower().eq('true').astype(int)
    for name in CANDIDATE_FEATURES:
        df[name] = pd.to_numeric(df[name], errors='coerce')
    valid = np.isfinite(df[CANDIDATE_FEATURES].to_numpy(dtype=float)).all(axis=1)
    df = df.loc[valid].copy()
    if df.empty or df.target.nunique() != 2:
        raise ValueError('Training data missing usable positive/negative labels')
    return df


def selected(frame, score_column):
    """Select one candidate per geometry + starting position."""
    return choose_one_per_scenario(frame, score_column).copy()


def outcomes(chosen):
    return chosen.stable_grasp_success.astype(str).str.lower().eq('true').astype(int)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--small-train', type=Path, required=True)
    p.add_argument('--large-train', type=Path, required=True)
    p.add_argument('--novel', type=Path, required=True, help='Existing novel trial_results.csv')
    p.add_argument('--seed', type=int, default=93717, help='Use same seed as original frozen novel-shape run')
    args = p.parse_args()

    small = load_training(args.small_train)
    large = load_training(args.large_train)
    if not args.novel.is_file() or not (args.novel.parent/'settings.json').is_file():
        p.error('Novel trial_results.csv and matching settings.json must exist')
    novel_raw = pd.read_csv(args.novel)
    novel_settings = json.loads((args.novel.parent/'settings.json').read_text())

    if set(novel_raw.geometry_id) & (set(small.geometry_id) | set(large.geometry_id)):
        p.error('Detected overlapping geometry IDs between train and novel tests')
    if novel_raw.failure_reason.isin(INVALID).any():
        p.error('Novel trial results contain simulator errors; resolve them before comparing')
    if novel_raw.duplicated(['geometry_id','position_index','grasp_fraction']).any():
        p.error('Novel CSV contains duplicate candidate rows')
    if novel_raw.empty:
        p.error('Novel CSV contains no trials')

    novel = add_features(novel_raw, novel_settings)
    for col in CANDIDATE_FEATURES:
        novel[col] = pd.to_numeric(novel[col], errors='coerce')
    if not np.isfinite(novel[CANDIDATE_FEATURES].to_numpy(dtype=float)).all():
        p.error('Nonfinite numerical feature in novel test set')
    novel['target'] = novel.stable_grasp_success.astype(str).str.lower().eq('true').astype(int)
    novel['width_feasible'] = novel.width_feasible.astype(str).str.lower().eq('true')

    for name, frame in [('small',small),('large',large)]:
        model=make_model(args.seed)
        model.fit(frame[CANDIDATE_FEATURES], frame.target)
        novel[f'{name}_ml_score'] = np.where(
            novel.width_feasible,
            model.predict_proba(novel[CANDIDATE_FEATURES])[:,1],
            -1.0)
        best_frac=float(frame.groupby('grasp_fraction').target.mean().idxmax())
        novel[f'{name}_fixed_fraction'] = best_frac
        novel[f'{name}_fixed_score'] = np.where(
            novel.width_feasible,
            -(novel.grasp_fraction-best_frac).abs(),
            -999.0)
        print(f'{name.upper()} TRAINING: {len(frame)} trials, '
              f'{frame.geometry_id.nunique()} geometries, '
              f'{frame.shape_family.nunique()} shape families, fixed fraction={best_frac:.2f}')

    scenarios=novel[['geometry_id','position_index']].drop_duplicates()
    count=len(scenarios)
    if 'selected_ml' in novel:
        previous=novel[novel.selected_ml.astype(str).str.lower().eq('true')]
        if len(previous)!=count:
            p.error('Original frozen model selection does not have exactly one choice per vessel')
        prior_hit=int(outcomes(previous).sum())
        print(f'Original frozen novel-test ML predictions: {prior_hit}/{count}')

    selections={}
    for key in ['small_ml','large_ml','small_fixed','large_fixed']:
        score=f'{key}_score'
        chosen=selected(novel, score)
        selections[key]=chosen
        if len(chosen)!=count:
            p.error(f'{key}: got {len(chosen)} chosen grasps; expected {count}')
        hits=int(outcomes(chosen).sum())
        print(f'{key:>13}: {hits}/{count} ({100*hits/count:.1f}%)')

    oracle=int(novel.groupby(['geometry_id','position_index']).target.max().sum())
    print(f'{"oracle":>13}: {oracle}/{count} ({100*oracle/count:.1f}%)')

    # One row per held-out vessel; include selected heights and whether each worked.
    summary=scenarios.copy()
    meta=novel[['geometry_id','position_index','shape_family']].drop_duplicates()
    summary=summary.merge(meta,on=['geometry_id','position_index'],how='left',validate='one_to_one')
    for key, chosen in selections.items():
        subset=chosen[['geometry_id','position_index','grasp_height_m','grasp_fraction','target']].copy()
        subset=subset.rename(columns={'grasp_height_m':f'{key}_height_m',
                                      'grasp_fraction':f'{key}_fraction',
                                      'target':f'{key}_success'})
        summary=summary.merge(subset,on=['geometry_id','position_index'],validate='one_to_one')
    summary['oracle_success']=summary.apply(
        lambda r: int(novel.loc[(novel.geometry_id==r.geometry_id) &
                                (novel.position_index==r.position_index), 'target'].max()),axis=1)
    if 'selected_ml' in novel:
        previous=novel[novel.selected_ml.astype(str).str.lower().eq('true')]
        previous=previous[['geometry_id','position_index','grasp_height_m','target']].rename(columns={
            'grasp_height_m':'original_frozen_ml_height_m','target':'original_frozen_ml_success'})
        summary=summary.merge(previous,on=['geometry_id','position_index'],validate='one_to_one')

    print('\nRESULTS BY NOVEL FAMILY')
    for fam, group in summary.groupby('shape_family'):
        n=len(group)
        print(f'  {fam:<14} small={int(group.small_ml_success.sum())}/{n}, '
              f'large={int(group.large_ml_success.sum())}/{n}, '
              f'fixed_large={int(group.large_fixed_success.sum())}/{n}, '
              f'oracle={int(group.oracle_success.sum())}/{n}')

    output=args.novel.parent/'comparison_120_vs_450.csv'
    summary.to_csv(output,index=False)
    print(f'\nSaved same-vessel comparison: {output}')
    print('Caution: this previously inspected novel set is a development benchmark, '
          'not a fresh untouched final test.')

if __name__=='__main__':
    main()
