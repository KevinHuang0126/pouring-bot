#!/usr/bin/env python3
"""Evaluate numerical-only grasp selection when one entire vessel family is unseen.

Usage:
  python evaluate_unseen_shape_families.py path/to/trials.csv

Place next to train_shape_grasp_model.py. Uses the same numerical features and
random-forest configuration. Vessel names identify held-out groups only; they
are never passed to the classifier.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

from train_shape_grasp_model import CANDIDATE_FEATURES, add_features, make_model, choose_one_per_scenario


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('csv_path', type=Path)
    ap.add_argument('--seed', type=int, default=4210)
    ap.add_argument('--output-dir', type=Path, default=None)
    args = ap.parse_args()

    path = args.csv_path.expanduser().resolve()
    settings_path = path.parent / 'settings.json'
    if not path.is_file() or not settings_path.is_file():
        ap.error('trials.csv and settings.json must exist in the same folder')

    df = pd.read_csv(path)
    settings = json.loads(settings_path.read_text())
    required = {'shape_family', 'geometry_id', 'position_index', 'grasp_fraction',
                'stable_grasp_success', 'failure_reason', 'radius_scale'} | set(CANDIDATE_FEATURES[:8])
    missing = sorted(required - set(df.columns))
    if missing:
        ap.error(f'Missing columns: {missing}')

    invalid = {'simulation_error', 'timeout', 'missing_summary', 'missing_upright_metric'}
    excluded = int(df.failure_reason.isin(invalid).sum())
    df = df.loc[~df.failure_reason.isin(invalid)].copy()
    df = add_features(df, settings)
    df['target'] = df.stable_grasp_success.astype(str).str.lower().eq('true').astype(int)
    for feature in CANDIDATE_FEATURES:
        df[feature] = pd.to_numeric(df[feature], errors='coerce')
    bad = ~np.isfinite(df[CANDIDATE_FEATURES].to_numpy(dtype=float)).all(axis=1)
    if bad.any():
        print(f'Excluding {bad.sum()} rows with invalid numerical features')
        df = df.loc[~bad].copy()

    families = sorted(df.shape_family.unique())
    if len(families) < 3:
        ap.error('At least 3 shape families are needed')
    outdir = args.output_dir or path.parent / 'model_evaluation' / 'unseen_families'
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows, predictions = [], []

    print(f'{len(df)} valid trials, {df.geometry_id.nunique()} geometries, '
          f'{len(families)} families; excluded {excluded} simulator-error rows')
    print('Numerical features only; shape-family names are used only to define holdout folds.\n')

    for fold, family in enumerate(families):
        test = df.loc[df.shape_family == family].copy()
        train = df.loc[df.shape_family != family].copy()
        assert not set(train.geometry_id).intersection(test.geometry_id), 'Geometry ID leakage across folds'
        if train.target.nunique() < 2:
            print(f'Skipping {family}: training portion contains only one target class')
            continue

        model = make_model(args.seed + fold)
        model.fit(train[CANDIDATE_FEATURES], train.target)
        test['predicted_success_probability'] = model.predict_proba(test[CANDIDATE_FEATURES])[:, 1]
        test['predicted_label'] = (test.predicted_success_probability >= 0.5).astype(int)

        # The baseline is chosen using ONLY training families, with no test leakage.
        best_fraction = float(train.groupby('grasp_fraction').target.mean().idxmax())
        test['fixed_baseline_score'] = -(test.grasp_fraction - best_fraction).abs()
        picked_ml = choose_one_per_scenario(test, 'predicted_success_probability')
        picked_fixed = choose_one_per_scenario(test, 'fixed_baseline_score')
        upper_bound = test.groupby(['geometry_id', 'position_index']).target.max()
        n = len(upper_bound)
        record = {
            'heldout_family': family,
            'geometries': int(test.geometry_id.nunique()),
            'scenarios': n,
            'ml_success': int(picked_ml.target.sum()),
            'fixed_success': int(picked_fixed.target.sum()),
            'oracle_success': int(upper_bound.sum()),
            'training_selected_fraction': best_fraction,
            'classification_accuracy': float(accuracy_score(test.target, test.predicted_label)),
            'balanced_accuracy': float(balanced_accuracy_score(test.target, test.predicted_label)) if test.target.nunique() == 2 else None,
            'roc_auc': float(roc_auc_score(test.target, test.predicted_success_probability)) if test.target.nunique() == 2 else None,
        }
        summary_rows.append(record)
        test['heldout_family'] = family
        test['selected_by_ml'] = test.index.isin(picked_ml.index)
        test['selected_by_fixed'] = test.index.isin(picked_fixed.index)
        predictions.append(test)
        print(f'{family:17s} ML {record["ml_success"]}/{n} | '
              f'fixed {record["fixed_success"]}/{n} | '
              f'oracle {record["oracle_success"]}/{n} | '
              f'train-selected fixed fraction {best_fraction:.2f}')

    if not summary_rows:
        ap.error('No folds could be trained')
    scores = pd.DataFrame(summary_rows)
    scores.to_csv(outdir / 'family_scores.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(outdir / 'heldout_predictions.csv', index=False)
    total = int(scores.scenarios.sum())
    aggregate = {
        'scenarios': total,
        'ml_success': int(scores.ml_success.sum()),
        'fixed_success': int(scores.fixed_success.sum()),
        'oracle_success': int(scores.oracle_success.sum()),
        'family_count': len(summary_rows),
        'features': CANDIDATE_FEATURES,
        'evaluation': 'Leave-one-shape-family-out, same sampled candidate heights',
        'caution': 'Only 3 geometries per family; these are offline selections scored using existing Drake trials, not independently rerun robot rollouts.',
    }
    (outdir / 'summary.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    print('\nUNSEEN SHAPE-FAMILY GRASP SELECTION')
    for label, field in [('Numerical random forest', 'ml_success'),
                         ('Training-selected fixed height fraction', 'fixed_success'),
                         ('Sampled-height oracle', 'oracle_success')]:
        count = aggregate[field]
        print(f'  {label}: {count}/{total} ({count / total:.1%})')
    print(f'\nSaved evaluation to {outdir}')


if __name__ == '__main__':
    main()
