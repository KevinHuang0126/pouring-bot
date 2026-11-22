# Adaptive Robotic Pouring — ML Grasp Selection

**Project:** MIT Robotics Manipulation (6.4210/2)  
**Status:** Physics-based grasp/pour simulation operational; supervised numerical grasp-selection proof of concept evaluated.

## Goal

Select a stable gripper grasp height for a previously unseen vessel geometry, pick it up using Drake contact dynamics, and complete a controlled pour into a receiving beaker. The learned policy should generalize from **numerical geometry** rather than shape-family names.

## Current simulation pipeline

1. **Generate geometry.** `generate_shape_families.py` constructs solid, axisymmetric surrogate SDF vessels using piecewise radius-versus-height profiles. The validated library includes cylinder, conical, round-bottom, shouldered, inverted-taper, funnel, goblet, and flared-rim shapes.
2. **Execute candidate grasps.** `visualize_station.py` uses Drake IK and a Schunk WSG gripper with physical contact (no rigid grasp attachment), lifts, transports, and attempts a roughly 95° pouring motion. `pour_planner.py` handles the pouring trajectory; `simple_pour_trajectory.py` provides a simplified predicted liquid stream, not a validated SPH simulation.
3. **Record outcomes.** Dataset scripts vary geometric parameters and grasp heights and log full-pour success, grasp slip, and unstable pickup, together with pickup tilt and contact/motion diagnostics. Exact generated vessel profiles are saved in dataset settings.
4. **Train numerical model.** `train_shape_grasp_model.py` fits a random forest predicting the probability of *full grasp-and-pour success* for candidate heights. Inputs include vessel dimensions, mass, COM height, candidate grasp height, local radius and slope, and nearby radial geometry. **Shape names, geometry IDs, and post-grasp measurements are excluded from model inputs.**
5. **Evaluate generalization.** `evaluate_unseen_shape_families.py` performs leave-one-family-out cross-validation: train on seven families and test on three vessels from the eighth; repeat for all eight. Geometry IDs are used to group samples, not as predictive features.

## Key datasets and outcomes

| Dataset / evaluation | Total | Outcome |
|---|---:|---|
| Original shape dataset (`seed4210`) | 120 trials / 24 geometries | 54 full successes, 23 grasp slips, 43 unstable pickups |
| Original held-out geometry evaluation | 24 vessels | Random forest **19/24 (79.2%)**; fixed-height **14/24 (58.3%)**; sampled oracle **22/24 (91.7%)** |
| Original leave-one-family-out evaluation | 24 vessels | Random forest **13/24 (54.2%)**; fixed-height **12/24 (50.0%)**; oracle **22/24 (91.7%)** |
| Diverse profile dataset (`seed84210`) | 120 trials / 24 geometries | **58** full successes, **23** grasp slips, **39** unstable pickups |
| Diverse leave-one-family-out evaluation | 24 vessels | Random forest **21/24 (87.5%)**; fixed-height **13/24 (54.2%)**; oracle **23/24 (95.8%)** |

**Sampled-height oracle:** For each vessel, it picks any successful grasp among the five *already simulated* candidates. It uses hindsight and is **not deployable**; it is an upper bound within the sampled heights.

**Interpretation:** The diverse-profile experiment gives encouraging unseen-family performance, but results come from different test geometries than the original experiment. The change from 54.2% to 87.5% is **not a controlled causal estimate** of the effect of added diversity. Each family currently has only three evaluation geometries; larger independent tests are needed.

## Main active files

| File | Role |
|---|---|
| `visualize_station.py` | Drake robot station, pickup checks, simulation logging |
| `pour_planner.py` | Jacobian / IK pouring trajectory |
| `simple_pour_trajectory.py` | Simplified liquid stream model |
| `drake_trajectory_export.py` | Required import for trajectory recording |
| `generate_shape_families.py` | Validated eight-family SDF geometry generation |
| `collect_shape_grasp_dataset.py` | Original shape dataset/smoke tests |
| `collect_diverse_shape_dataset.py` | Varies numerical vessel *profile proportions* |
| `train_shape_grasp_model.py` | Numerical random-forest training and held-out-geometry evaluation |
| `evaluate_unseen_shape_families.py` | Leave-one-family-out evaluation |
| `enable_diverse_profiles.py` | One-time compatibility migration for numerical profile settings; retain in `scripts/` or archive after verifying changes |

## Reproducibility and immediate next steps

- Keep all generated CSVs, vessel profiles/settings files, and logs outside the Git commit unless intentionally publishing a small representative dataset. Never delete them during cleanup.
- Record Git commit hash, random seed, geometry profile version, and exact CSV path for each experiment.
- Audit `git status`, tracked imports, and `.gitignore` before staging; **stage selected source files**, not `git add .`.
- Commit the numerical training and diversity collection pipeline, then push the existing feature branch to GitHub.
- Next scientific step: a larger **independent**, geometry-disjoint dataset with varied profile transition heights and dimensions; compare policies on the **same untouched evaluation set**.

**Current branch:** `shasha/ml-grasp-dataset`  
**Current diverse dataset:** `grasp_dataset/shape_families/diverse_n3_p1_20261122_001031_seed84210/trials.csv`
