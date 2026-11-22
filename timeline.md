# Project timeline

MIT 6.4210/2 pouring-bot. Dates are commit dates in US Eastern time. Hashes are the commits that introduced the work. This file is a record of what landed in the repository, for use when writing the paper. It is not a claim that every step was evaluated the same way.

People: Shasha Shankaramani (scene, manipulation, grasp data, and the grasp-selection model), Wyatt Vick (physics, lift control, SPH), Kevin Huang (repository setup and the shared plant models).

## How the history is actually shaped

`origin/main` only contains the work through 11 November 2026 (pull request #3). Everything after that is still on feature branches. Three lines diverge, and they do not contain each other.

- Shared trunk, now `origin/main` at `00e6bb4`: scene, collision models, and the simplified pour stream.
- Manipulation line: randomized grasping, Jacobian lift, then Jacobian pouring. Tip of that shared segment is `shasha/integrated-pouring` at `fe28983` (12 November).
- From that pouring commit, two branches split.
  - Camera localization, then the reduced-order flow model, then Drake-to-SPlisHSPlasH. Tip is `wyatt/drake-sph-integration` at `be251e3` (21 November).
  - Grasp-height learning. Tip is `shasha/ml-grasp-dataset` at `5dfc029` (22 November). This line does not include the camera, the flow model, or SPH.

The learned grasp selector has not been run on the SPH pour. The SPH demo replays a recorded vessel pose. It does not resimulate the grasp contact.

## 2 November 2026 — repository and a static scene

`8581f24`, Wyatt Vick. Initial commit.

`808cf0b`, Kevin Huang, branch `wyatt/physics-simple`. Team split and branch rules. Shasha owns the station and the arm. Wyatt owns physics. Kevin owns repository organization and the first ML experiments. `main` stays the shared line. Task branches are reviewed before merge. At this point the README still said there was no runnable simulation, and the physics scope and the ML objective were open.

`fbf94b1`, Shasha Shankaramani, branch `shasha/scene-setup`. First Drake station in `visualize_station.py`. A KUKA iiwa with a Schunk WSG, from the Manipulation course models, stands at a table. Source vessels are revolved glass profiles drawn in Meshcat, chosen at random. They are visuals, not bodies in the multibody plant. A beaker is the receiving vessel.

## 7 November 2026 — scene and labware merged

`db74b44`, Kevin Huang. Pull request #1 merges `shasha/scene-setup` into `main`.

`8ade98e`, Kevin Huang, branch `kevin/plant-labware`. The table, source holder, beaker, boiling flask, Erlenmeyer, and test tube become SDF models with collision. `841a963` merges this as pull request #2. This is the last commit on local `main` and on `kevin/ml-experiments`. That ML branch has no commits of its own.

## 9–10 November 2026 — grasp and lift

`ae14e4e` and `8b32371`, Shasha, branch `shasha/randomized-grasping`. Vessels are placed at randomized table positions. `grasp_planner.py` solves inverse kinematics for a pre-grasp and a lift. The station closes the WSG on the vessel through contact and checks whether the vessel rose. The grasp is not welded to the gripper.

`2a333fe` and `17b77bb`, Wyatt, branch `wyatt/randomized-grasping-lift`. The lift becomes a Cartesian motion from the gripper Jacobian. The second commit turns random vessel selection back on.

## 11 November 2026 — a pour stream, then a pour motion

`850d768`, Shasha. `simple_pour_trajectory.py` draws a planned liquid stream from the current vessel pose. A fill-dependent tilt threshold opens the pour. One low point on the rim emits a ballistic line. The file says this is a visual and planning surrogate: no free surface, no splash, no wall contact, no volume loss. `tests/test_simple_pour_trajectory.py` covers the math without Drake. `SETUP.md` and `WSL_SETUP.md` record how to run it. Pull request #3 (`00e6bb4`) merges this to `origin/main`. Nothing later has been merged.

The same day, Shasha starts `shasha/integrated-pouring` by merging `main` (`3f3c670`), adds a first pouring planner (`dff6c2e`), and connects Wyatt's Jacobian lift to it (`e30ee8b`).

## 12 November 2026 — closed-loop pouring on known glassware

`fe28983`, Shasha, tip of `shasha/integrated-pouring`. `pour_planner.py` commands a Jacobian pour for the test tube, Erlenmeyer, and boiling flask. The station monitors the grasp during the motion and stops if the vessel slips or the Cartesian target is missed. This commit is the fork point for both the perception/physics line and the grasp-learning line.

## 13 November 2026 — camera, then a volume model

`a831359`, `74eaf1c`, `b092534`, Shasha, branch `shasha/camera-localization`. `perception.py` adds a fixed overhead RGB-D camera (640×480, about 1.15 m above the table). It estimates the vessel's initial x and y from Drake segmentation. The estimator does not read the ground-truth pose. Geometry is assumed known and the vessel starts upright. The branch reports the camera-guided grasp and pour working for the three vessel geometries, including randomized source positions.

`c765f59`, Shasha, branch `shasha/fluid-flow-model`. `fluid_model.py` replaces the purely visual stream with a mass-conserving surrogate. It tracks initial volume, volume remaining, flow rate, volume collected, and volume spilled, in milliliters. Nominal capacities are modeling assumptions, not measured glassware: test tube 35 mL, Erlenmeyer 350 mL, boiling flask 350 mL. The pour can adapt using that tracked volume. The module says it is not a fluid dynamics simulation.

## 20–21 November 2026 — particle fluid on a recorded pour

Wyatt, branch `wyatt/drake-sph-integration`, built on top of the fluid-flow branch.

- `191b8bc`. Work in progress coupling Drake to SPlisHSPlasH.
- `33f1849`. Replay of the Drake vessel motion with Bender et al. 2019 boundary handling.
- `2069a80`. Spill evaluation on those trajectories.
- `976a770`. A configurable Drake-to-SPH evaluator.
- `535dc8d`. Setup script and Meshcat playback of the particles with the arm and cups.
- `be251e3` (21 November, 01:14). DFSPH timestep patch so the solver timestep stays consistent, plus a more stable start to the pour.

The pinned solver is SPlisHSPlasH 2.19.1, commit `dba6b26b`, single precision. `evaluate_sph_spill.py` takes a vessel-pose CSV from `drake_trajectory_export.py`, writes VTK particles and a `spill_report.json`. The viewer solves arm joints from the cup pose for display. The README on that branch says playback does not reconstruct the station's joint commands, contact forces, or grasp dynamics.

## 21–22 November 2026 — grasp height as a learning problem

Shasha, branch `shasha/ml-grasp-dataset`, forked from the 12 November pouring commit. This is the line checked out when this timeline was written (`5dfc029`).

What changed, in order:

- `cc87795` (21 November, 13:35). Baseline pour tilt set to about 95 degrees over 10 seconds.
- `2853908`. The real vessel pose during the pour is recorded, which is the same export the SPH side consumes.
- `9f8833f`. Parametric solid vessels. A radius-versus-height profile is revolved into convex frustum collision pieces. Early generator, not yet the eight-family set.
- `38e83cb`. A pickup counts as upright only if the vessel axis stays within 15 degrees of vertical after a lift of more than 6 cm. Otherwise the label is an unstable pickup even if the cup rose.
- `3a3c8ca`. Shape-diverse generation and the first ML training scripts.
- `e7e3ee4` and `f9c636a`. Grasp geometry settled for an inverse Erlenmeyer, then checked across eight families: cylinder, conical, round bottom, shouldered, inverted taper, funnel, goblet, flared rim.
- `6190608` (22 November, 00:42). Numerical random forest, leave-one-family-out evaluation, and `docs/ML_GRASP_PROJECT_OVERVIEW.md`.
- `5dfc029` (22 November, 12:16). Training set grown to 450 trials, a ninth family (waisted bottle), generated meshes committed, and a transfer test on four new families.

### What one trial is

`collect_shape_grasp_dataset.py` and `collect_diverse_shape_dataset.py` build a solid, axisymmetric cup, then try five grasp heights at 33, 47, 60, 73, and 80 percent of the vessel height. Height is about 14–16 cm and mass about 50–70 g. The diverse collector also moves the profile's corners, so two cups in one family are not the same silhouette. The gripper force limit is 60 N. Contact holds the cup. A trial is a full success only if the pickup rises, stays upright, the pour motion finishes, and the log shows no slip and no missed Cartesian target. Anything else is grasp slip, unstable pickup, pickup failure, an infeasible width, or a simulator failure. Simulator failures are dropped before training. The 450-trial file had none.

The forest in `train_shape_grasp_model.py` predicts the probability of full success. Features are numbers only: size, mass, center of mass, grasp height, local radius and slope, and the radius just above and below the fingers. Family name is not a feature. For each cup the model picks the candidate with the highest probability. The comparison policies are a single height fraction chosen on the training rows, and an oracle that is allowed to pick any of the five heights that already succeeded. The oracle is an upper bound on these five samples. It is not a controller.

The fitted forest was not saved. The trial tables and the scorecards were.

### Datasets and scores

All selection rates below are "the chosen height was a full pour success," one choice per vessel. Counts were taken from the CSVs in the repo.

Original eight families, template shapes scaled a little, seed 4210, 120 trials, 24 cups (`n3_p1_20261121_231300_e9203cbf96_seed4210`). Outcomes: 54 success, 23 grasp slip, 43 unstable pickup.

| Test | Forest | Fixed height | Oracle |
| --- | ---: | ---: | ---: |
| Held-out geometries | 19/24 (79.2%) | 14/24 (58.3%) | 22/24 (91.7%) |
| Held-out family | 13/24 (54.2%) | 12/24 (50.0%) | 22/24 (91.7%) |

Same eight families, profile corners perturbed, seed 84210, 120 trials, 24 cups (`diverse_n3_p1_20261122_001031_seed84210`). Outcomes: 58 success, 23 grasp slip, 39 unstable pickup.

| Test | Forest | Fixed height | Oracle |
| --- | ---: | ---: | ---: |
| Held-out geometries | 21/24 (87.5%) | 12/24 (50.0%) | 23/24 (95.8%) |
| Held-out family | 21/24 (87.5%) | 13/24 (54.2%) | 23/24 (95.8%) |

The jump from 54.2% to 87.5% on the family holdout is not a controlled measurement of "adding diversity." The test cups are different. The overview in the repo says this explicitly.

Expanded set, seed 84210, 450 trials, 90 cups, nine families including waisted bottle (`diverse_n10_p1_20261122_103654_seed84210`). Outcomes: 203 success, 74 grasp slip, 169 unstable pickup, 4 pickup failure.

| Test | Forest | Fixed height | Oracle |
| --- | ---: | ---: | ---: |
| Held-out geometries | 68/90 (75.6%) | 54/90 (60.0%) | 86/90 (95.6%) |
| Held-out family | 60/90 (66.7%) | 50/90 (55.6%) | 86/90 (95.6%) |

On the family holdout the forest was 10/10 on inverted taper, 9/10 on conical and cylinder, 8/10 on goblet, 7/10 on flared rim and round bottom, 4/10 on waisted bottle, and 3/10 on funnel and shouldered. Of the 30 family-holdout misses, 26 already had a successful height among the five. Only 4 cups had no working sample. Shouldered misses all chose 47 percent, on the wide body. The working grasp was often the same diameter, higher up, closer to the neck. Funnel misses usually chose the wide mouth. Waisted misses did not share one height.

Novel families, not in training: hourglass, stepped, double bulge, collared. 20 cups, 100 trials (`novel_n5_20261122_101555_seed93717`). Trial outcomes: 46 success, 23 grasp slip, 31 unstable pickup. Selection on those saved outcomes (`comparison_120_vs_450.csv`):

| Chooser | Successes |
| --- | ---: |
| Forest trained on the 120-trial diverse set | 13/20 (65%) |
| Forest trained on the 450-trial set | 17/20 (85%) |
| Fixed height from the 450-trial set | 12/20 (60%) |
| Oracle | 19/20 (95%) |

The 85% number is a development result. Those novel cups had already been inspected, and the 450-trial training set includes a waisted bottle, which is related to some of the new shapes. The overview says a fresh held-out set is still needed, and that the predicted height has not been wired back into the live Drake pour.

## What is still open for the paper

- `origin/main` does not contain grasping, pouring control, camera localization, either fluid model, or the grasp selector.
- The grasp model and the SPH evaluator have not been run as one system. One picks a finger height in Drake contact. The other pours particles along a pose trajectory.
- Most of the selector's errors are ranking errors on cups that already had a good sampled height. Funnel and shouldered, and to a lesser extent the waisted bottle, are where the family holdout falls down.
- Four cups in the 90-vessel family holdout, and one of the 20 novel cups, had no success at any of the five heights. A better ranker cannot fix those.
- No trained weights are stored. Rerunning `train_shape_grasp_model.py` on the saved CSVs refits the reported forest.
