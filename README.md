# Pouring Bot

A collaborative project simulating a KUKA iiwa robotic arm for pouring, using Python and `pydrake`.

## Quick Start — Fluid Pouring Demo

The demo replays a recorded Drake vessel trajectory through SPlisHSPlasH, then shows the iiwa, cups, and simulated liquid together in a browser. Start from a GitHub checkout of `wyatt/drake-sph-integration`. The trajectory CSV and scene template are included.

### One-time setup on Ubuntu 24.04 or WSL

Use Python 3.12–3.14. If the development tools and libraries are missing, install them once. The setup script keeps Python packages, the pinned SPlisHSPlasH checkout, and its build inside the project's `.venv`.

```bash
sudo apt-get update
sudo apt-get install -y git build-essential cmake pkg-config python3-dev python3-venv \
  xorg-dev libgl1-mesa-dev libglu1-mesa-dev
bash scripts/setup_sph.sh
```

Setup needs network access to GitHub and PyPI. The first setup builds SPlisHSPlasH from source and applies the repository’s DFSPH timestep and implicit-viscosity initial-guess patches. Rerun the setup script to update an existing demo environment. Set `PYTHON` to select a compatible interpreter or `BUILD_JOBS` to control compile parallelism. Drake downloads its pinned public robot models on first use.

### Run and view the demo

```bash
OMP_NUM_THREADS=1 .venv/bin/python evaluate_sph_spill.py \
  --trajectory drake_vessel_trajectory.csv \
  --scene-dir sph_pouring_trial/eval_scene \
  --output sph_pouring_trial/demo

.venv/bin/python visualize_pour.py \
  --trajectory drake_vessel_trajectory.csv \
  --fluid-output sph_pouring_trial/demo
```

Open the Meshcat URL printed by the viewer. Playback loops until you press Ctrl+C. The evaluator writes the scene meshes, `vtk/` particle frames, `fluid_frames.csv` timestamps, and `spill_report.json` under the output directory. Give each evaluation a new output directory and reuse `--scene-dir` for the volume-map cache. Add `--duration 0.8` to the first command for a short smoke run.

The viewer reads the actual VTK particle positions in world coordinates and uses the applied source pose recorded alongside each frame. The robot uses the iiwa7 model and wrist mounting from the existing grasping and pouring stations. Its joint poses are solved kinematically from the cup pose. The playback does not reconstruct the original station's joint commands, contact forces, or grasp dynamics. If a custom motion is outside the demo robot's reach, use `--no-robot` for cup and fluid playback. For older VTK exports, frame times are estimated from the scene's export FPS. Use `--scene path/to/scene_animated.json` if the original scene sits outside the run directory.

### Run the robot planner and replay its measured vessel motion

After the SPH setup above, install the Drake station helper package. The station and viewer start local Meshcat servers.

```bash
.venv/bin/python -m pip install 'manipulation==2026.10.17'
vessel=test_tube
run_dir="sph_pouring_trial/${vessel}_run_01"
OMP_NUM_THREADS=1 .venv/bin/python visualize_station.py \
  --vessel "$vessel" --export-dir "$run_dir"

PYTHONPATH="$PWD/.venv/splishsplash/build/lib" OMP_NUM_THREADS=1 \
  .venv/bin/python evaluate_sph_spill.py \
  --station-export "$run_dir" --scene-dir sph_pouring_trial/station_cache \
  --output "${run_dir}_sph"

.venv/bin/python visualize_pour.py --fluid-output "${run_dir}_sph" --no-robot
```

Set `vessel=erlenmeyer` or `vessel=boiling_flask` for a different source. Drake writes a full station trajectory after the grasp and motion checks pass. Add `--export-partial` to keep the measured prefix for experiments when the planner aborts. Check `station_motion.json` for the completion flag. Add `--randomize-position` to vary the default source XY of `[0.42, -0.16]` m. Running without `--export-dir` uses the normal camera-guided workflow. Open the viewer's printed Meshcat URL. Use new directories for each station export and evaluation. The evaluator copies the trajectory into the output, where the viewer finds it automatically.

The recorder samples the free vessel's measured world pose after simulator advances, beginning at zero before approach with the initial vessel upright. The pose origin is the outer bottom center. SPH initializes its own fill at rest before approach and uses a 1 ms timestep cap for station exports. Add `--settle-duration 0.2` to hold the first pose before replay. Add `--duration 0.4` for a startup check. Omit it to include the entire motion.

The original test-tube replay reversed at 77.2° after the reduced-order planner declared the tube empty. At peak tilt, the highest SPH particle was 3.2 mm below the low edge of the lip. Station export now completes the tilt and hold independently of that empty signal, with a small mouth offset for wrist clearance. Transfer depends on the executed motion, fill, and vessel.

Drake displays the same hollow mesh SPH uses. Rigid contact uses the station's existing solid proxies. The Erlenmeyer hollow mesh is a conical-body, straight-neck approximation to the station model. The boiling flask uses a faceted spherical bulb, a small flat base and a straight neck. The two flask meshes model the interior and neck explicitly. The station beaker is welded at world XYZ `[0.72, 0.14, 0.20]` m. The viewer command shows cups and particles. The optional demo robot uses inverse kinematics and can differ from the executed joint motion.

### Run the short outflow control

```bash
control=sph_pouring_trial/control_120deg_01
.venv/bin/python station_sph_export.py --controlled "$control" --vessel test_tube
PYTHONPATH="$PWD/.venv/splishsplash/build/lib" OMP_NUM_THREADS=1 \
  .venv/bin/python evaluate_sph_spill.py --station-export "$control" \
  --scene-dir sph_pouring_trial/station_cache --output "${control}_sph"
.venv/bin/python visualize_pour.py --fluid-output "${control}_sph" --no-robot
```

The controlled trajectory rotates the source around its mouth to 120° and holds it there. It bypasses robot grasping to isolate fluid outflow. The 3-second tube replay produced visible outflow and finished with 332 of 448 particles in the receiver, 116 outside both vessels, and zero in the source. These are provisional particle-center counts. `spill_report.json` marks spill measurement as unvalidated.

### Run Shasha’s generated vessels with manual or ML grasps

Use the existing SPH build. Fetch Shasha’s branch to make the pinned generator, feature code, and dataset available locally. The command reads those files through Git and leaves your checkout on Wyatt’s branch.

```bash
git fetch origin shasha/ml-grasp-dataset
.venv/bin/python -m pip install 'manipulation==2026.10.17' 'pandas==3.1.1' 'scikit-learn==1.9.1'
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1

.venv/bin/python run_ml_sph_pour.py \
  --family cylinder --grasp-mode ml \
  --output sph_pouring_trial/generated_cylinder_01 --export-fps 12

.venv/bin/python run_ml_sph_pour.py \
  --family hourglass --grasp-mode manual --grasp-height 0.08 \
  --output sph_pouring_trial/generated_hourglass_01 --duration 0.25
```

The generator supports `cylinder`, `conical`, `round_bottom`, `shouldered`, `inverted_taper`, `funnel`, `goblet`, `flared_rim`, `waisted_bottle`, `hourglass`, `stepped`, `double_bulge`, and `collared`. Use `--profile-json PATH` in place of `--family` to supply a snapshot with `family`, `height_m`, `mass_kg`, and `normalized_height_radius_m`. The profile is a list of pairs `[fractional_height, radius_in_meters]`, ordered strictly from height 0 to 1. Shasha’s stepped family uses short tapered transitions. The adapter preserves the exterior knots and 48 angular facets, closes the bottom, and offsets the inner profile radially by `--wall`. Use `--floor` to change the base thickness. Profiles that leave no cavity are rejected.

ML mode uses the current 450-trial diversified dataset `diverse_n10_p1_20261122_103654_seed84210` and Shasha’s random-forest implementation at the pinned upstream commit. It reconstructs numerical features from each exact profile, trains on the archived trials, filters candidates by the upstream width limit, and chooses the highest-scoring candidate. Exact matching training silhouettes are excluded for custom profile snapshots. This score chooses a grasp height. Drake then executes the motion, and SPlisHSPlasH replays the measured poses.

Each run writes `drake/source_trajectory.csv`, `drake/vessel_geometry.json`, `drake/cups.json`, and `drake/station_motion.json` under the output directory. Check the latter for pickup, completed motion, and replayability. A slip or failed pickup can still produce a measured prefix. SPH results are under `sph/`. Add `--drake-only` to export without fluid evaluation, `--headless` for a report without frames, or `--duration` for a short prefix. An external trajectory never requires rerunning Drake:

```bash
run=sph_pouring_trial/generated_cylinder_01
.venv/bin/python evaluate_sph_spill.py --station-export "$run/drake" \
  --scene-dir sph_pouring_trial/station_cache --export-fps 12 --output "${run}_replay"
.venv/bin/python visualize_pour.py --fluid-output "${run}_replay" --no-robot
```

**Recorded frames and tilted starts.** Shasha’s original recorder samples `EvalBodyPoseInWorld` after pickup, so the pose is `X_WB` and the earlier fluid state is unknown. The generated link origin is the outer bottom center with local +Z toward the rim. The fresh workflow records from before approach through the measured endpoint. For either recorder, match geometry and receiver metadata to the actual run. The evaluator places the hollow source at the initial measured pose and rotates the source-local fill into world coordinates. A tilted start holds that same pose for 0.25 s by default to let gravity redistribute the freshly seeded particles. Use `--settle-duration` to adjust it. The dwell adds solver time and preserves all recorded poses. For old post-pickup CSVs, this initializes a new fill at the beginning of the recorded segment and cannot recover any earlier spill. Use the replay tool for tilted scene JSON, since it applies the local fill transform to the sampled particles before solving.

The default generated cylinder with the ML-selected 0.09 m grasp completed a 58 s measured-motion replay in about 251 s with cached maps and 12 fps export. The final counts were 26 in source, 328 in receiver and 62 outside of 416.

The saved `shared_trajectories/shasha/source_trajectory.csv` has a valid tilted first pose but no confirmed matching geometry snapshot. The backup contains multiple plausible vessels. Use the external CSV command with a `cups.json` from the verified source profile and receiver setup if that pairing is recovered. Outside-both counts include particles still in flight at the endpoint and remain provisional.

### Paired ML versus fixed-height transfer benchmark

Run ten paired trials with the same generated vessel and fill for both grasp strategies. The benchmark samples from the pinned shape families using seed `84211` by default. It uses the upstream training ranges for height (0.14–0.16 m), mass (0.05–0.07 kg), and radius scale (0.94–1.06), and samples the source position within the rectangle between the two upstream positions. By default it samples 25–65% fill and uses 3.5 mm particles. It resamples candidates that cannot fit the requested fill at the selected spacing or are too short for the fixed grasp. The fixed-height baseline defaults to 0.09 m.

```bash
cd ~/pouring-bot
git fetch origin shasha/ml-grasp-dataset
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
bench=sph_pouring_trial/grasp_transfer_10
.venv/bin/python benchmark_grasp_transfer.py --trials 10 --seed 84211 \
  --fixed-grasp-height 0.09 --particle-spacing 0.0035 \
  --fill-fraction-min 0.25 --fill-fraction-max 0.65 --output "$bench"
```

For a new benchmark, escaped-particle retirement is enabled by default. Rebuild an existing binding with `bash scripts/setup_sph.sh` before the first retirement run. Use `--no-retire-escaped` for a control benchmark or `--escape-plane-z -0.1` to request a lower cutoff in world meters. The evaluator rejects a plane that lies too close to either vessel. Older manifests resume with retirement disabled, and new manifests keep their saved setting.

Resume with the same output directory. Completed methods and completed Drake stages are reused. Incomplete attempts remain on disk, and each new attempt gets its own folder. Repeating an explicit option on resume requires it to match the saved settings.

```bash
.venv/bin/python benchmark_grasp_transfer.py --output "$bench" --resume
```

A lower-resolution one-pair smoke benchmark uses the same scoring horizon.

```bash
.venv/bin/python benchmark_grasp_transfer.py --trials 1 --seed 84212 \
  --particle-spacing 0.006 --fill-fraction-min 0.12 --fill-fraction-max 0.18 \
  --output sph_pouring_trial/grasp_transfer_smoke
```

Both Drake attempts run before SPH for each pair. To pass, a method must complete pickup and the active pour, and the recorded pose at the end of the pour must be tilted at least 60° with its mouth within 75 mm and its axis within 28° of the commanded pose. Recovery starts after the scoring endpoint, so an abort during recovery is recorded separately from active-pour success. An unsuccessful Drake attempt scores zero and skips SPH. Successful attempts replay serially and headlessly using the saved fill and automatic stationary start selection, including its full-replay fallback. The evaluator holds the final active-pour pose fixed for another second of fluid simulation before scoring, with Drake time held at the recorded endpoint.

The score is `100 * in_receiver / total_particles`. Particle volumes are uniform within a run, and only centers inside the receiver contribute. Particles still in flight count outside both vessels. The endpoint score is provisional and resolution-dependent. The mean, median, and population standard deviation include zeros for methods that failed in Drake or SPH. SPH success rates describe completed solver evaluations, reported both across all completed methods and conditional on Drake success. Paired differences are ML minus manual in percentage points.

`trials.csv` contains one row per method in each pair. `summary.json` contains aggregate and paired statistics. `trials.json` stores seeds, sampled geometry, stage checkpoints, errors, and paths relative to the output directory. Each pair has a shared profile snapshot and separate `ml` and `manual` directories containing Drake trajectories, logs, and SPH reports where available.

#### Escaped-particle retirement

The escape plane is below a conservative bound on the lowest receiver or swept source geometry across the replay. It leaves at least three SPH kernel support radii of clearance. The automatic plane adds another 50 mm and is at or below world Z = 0. A particle is eligible only below that plane with downward velocity. A group still connected to retained fluid within kernel support stays active. The pouring scene uses downward gravity and has no solid geometry below the padded boundary. This cutoff treats detached falling groups as irreversible misses.

The native DFSPH binding compacts the active particle set and its neighbor search while preserving survivor IDs, mass, and solver warm starts. Retired particles stop participating in force calculations and CFL. Reports distinguish `active_outside_both` from `retired_particles`, with both included in `outside_both`. `retired_volume_ml` accumulates the retired native particle mass divided by rest density. `total_particles` still counts the original fill. `below_world_z_zero` counts active centers below zero at the endpoint and retired centers that were below zero when retired. The sampled step log records active and retired counts and peak active speed.

Standalone evaluation defaults to retirement off. To compare the same station export with and without retirement, point `package` at a successful method's `run/drake` directory and use its recorded active-pour endpoint.

```bash
package=path/to/method/drake-attempt/run/drake
end=$(.venv/bin/python -c 'import json, sys; print(json.load(open(sys.argv[1]))["pour_end"]["drake_time_s"])' "$package/station_motion.json")
for mode in no-retire-escaped retire-escaped; do
  .venv/bin/python evaluate_sph_spill.py --station-export "$package" \
    --sph-start-time auto --end-drake-time "$end" --final-hold-duration 1 --headless \
    --scene-dir sph_pouring_trial/station_cache --"$mode" \
    --output "sph_pouring_trial/retirement_comparison_$mode"
done
```

In a matched headless replay of the smoke benchmark's ML goblet, both runs started at Drake 26.325 s, ended at Drake 44 s, and held for one second. Both initialized 143 particles at 6 mm spacing and used cached maps with one OpenMP thread.

| Retirement | Final transfer | Retired | Solver steps | Wall time |
| --- | ---: | ---: | ---: | ---: |
| Off | 123/143 (86.0%) | 0 | 99,283 | 34.93 s |
| On | 119/143 (83.2%) | 1 (0.216 mL) | 32,773 | 12.95 s |

Adaptive timesteps can diverge after particles retire, and the endpoint transfer can change. The measured timing applies to this replay.

### Match the planner’s initial fill in WSL

Use the existing patched SPH build and cache. Generated runs default to 75% fill. This example runs the goblet’s manual grasp and checks its SPH startup.

```bash
cd ~/pouring-bot
git fetch origin shasha/ml-grasp-dataset
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1
run=sph_pouring_trial/goblet_fill75
.venv/bin/python run_ml_sph_pour.py --family goblet --grasp-mode manual \
  --grasp-height 0.08 --fill-fraction 0.75 --duration 0.05 --export-fps 12 --output "$run"
.venv/bin/python visualize_pour.py --fluid-output "$run/sph" --no-robot
```

Use `--grasp-mode ml` without `--grasp-height` for model-selected grasps. Fill is separate from grasp selection. Omit `--duration` for the full replay or use `--headless` to skip frames. Check the startup particle count in the Drake log or the `initial_fill` section of the exported geometry metadata before committing to a long run.

The planner and SPH use the same capacity, integrated over the actual polygonal cross-sections of the hollow mesh. The analytic free-surface height comes from inverting that integral. SPH fills horizontal layers of a wall-clipped lattice. Each particle represents 0.042875 mL at the generated workflow’s default 3.5 mm spacing, or 0.027 mL at 3 mm. Rounding to the nearest particle gives at most half a particle volume of error for a supported fill. Wall and bottom clearance leave fewer particle centers per layer. To preserve the requested volume, the seeded top layer can sit above the analytic free-surface height. Both heights are recorded. A 0.25 s fixed-pose dwell lets gravity redistribute the fill, including for tilted starts. The replay retains the measured trajectory, and `--settle-duration` controls the dwell. Requests that cannot fit at this spacing and clearance are rejected with the maximum volume the lattice can seed.

The station package records the requested fraction in `cups.json` and writes the capacity, target volume, particle count, realized fraction and signed volume error under `initial_fill` in the metadata and evaluation report. Older bundles with explicit fill boxes still replay as recorded. Add `--fill-fraction` to `visualize_station.py --export-dir ...` to opt an original vessel into profile fill.

| Generated shape | Particles at 25% | At 50% | At 75% |
| --- | ---: | ---: | ---: |
| Cylinder | 2,240 | 4,480 | 6,721 |
| Goblet | 1,829 | 3,657 | 5,486 |
| Waisted bottle | 2,691 | 5,381 | 8,072 |
| Hourglass | 1,918 | 3,836 | 5,754 |

The 75%-filled goblet targets 148.127 mL and initializes 148.122 mL with 5,486 particles. It retained all particles through settling and the 0.05 s replay. On the same cached, headless 0.06 s cylinder replay, the old 416-particle block took 0.26 s and the new 6,721-particle fill took 2.50 s. The [fill benchmark](benchmarks/sph_fill.csv) records the volume error, heights, particle counts and startup runtime across all 12 cases. Use a short replay to measure runtime before evaluating an entire take.

The compact-cylinder verification used `--family cylinder --height 0.12 --radius-scale 0.65 --grasp-mode manual --grasp-height 0.07 --fill-fraction 0.25 --headless`. It seeded 650 particles representing 17.550 mL for a 17.5524 mL target. The cached full 58 s replay took 385 s, including its 0.25 s settling phase. All 650 particles were conserved and ended with 277 in the source, 275 in the receiver, and 98 outside both.

### Optimized goblet replay in WSL

Use the existing patched SPH build. The first replay at a new spacing builds new volume maps. Later runs reuse that spacing’s maps.

```bash
cd ~/pouring-bot
git fetch origin shasha/ml-grasp-dataset
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1
run=sph_pouring_trial/goblet_optimized
.venv/bin/python run_ml_sph_pour.py --family goblet --grasp-mode ml \
  --output "$run" --export-fps 6
.venv/bin/python visualize_pour.py --fluid-output "$run/sph" --no-robot
```

Replay an existing package with the same optimization settings without repeating Drake or grasp selection.

```bash
.venv/bin/python evaluate_sph_spill.py --station-export "$run/drake" \
  --sph-start-time auto --sph-prepour-seconds 2.0 \
  --export-fps 6 --scene-dir sph_pouring_trial/station_cache --output "${run}_replay"
.venv/bin/python visualize_pour.py --fluid-output "${run}_replay" --no-robot
```

Auto-start detects a sustained rise in tilt toward a pouring angle. It uses measured Drake positions and quaternions to estimate linear and angular velocities over a 0.15 s window, then differentiates those rates for acceleration. It looks for the final sustained quiet interval ending within 1.25 s of the detected onset. The default limits are 16 mm/s, 7°/s, 0.2 m/s², and 80°/s². The vessel must stay within 10° of upright and retain at least one particle spacing of estimated rim clearance after allowing for tilt and motion. The selected timestamp targets a two-second lead-in and leaves at least 0.75 s of quiet recorded motion before the interval ends. Use `--sph-prepour-seconds` to change the target lead-in.

If the checks fail or saved fill metadata is unavailable, auto-start replays from the first pose. A nonzero explicit `--sph-start-time` uses the original CSV timestamp and also requires a safe quiet interval with enough time after that timestamp. Use `--sph-start-time 0` for full replay. Both replay commands accept `--sph-restart-thresholds` with a JSON file of overrides, for example `{"min_pause_s": 1.0}`. The other keys are `max_onset_gap_s`, `smoothing_window_s`, `max_speed_m_s`, `max_angular_speed_deg_s`, `max_acceleration_m_s2`, `max_angular_acceleration_deg_s2`, and `max_tilt_deg`.

A restart seeds the saved fill volume at the measured vessel pose with zero particle velocity. The quiet interval is evidence that slosh could decay, and restarting from rest is an approximation. It discards residual slosh and can restore liquid lost during skipped motion. Use full replay when earlier fluid motion or loss matters. The restart defaults to a 0.5 s SPH settling dwell, configurable with `--settle-duration`. The dwell advances the solver clock while holding Drake time fixed, then replay resumes from the selected timestamp.

The original CSV is copied byte for byte. Looping Meshcat playback follows exported fluid frames using the solver times and original vessel timestamps recorded in `fluid_frames.csv`. The `sph_start` entry in `spill_report.json` gives the onset, selected timestamp and method, measured quiet interval and peak speeds, safety results alongside their limits, settling duration, zero-velocity restart assumption, and any fallback reason. Legacy boxes use full replay and their saved spacing. Station exports use their saved fill fraction and spacing by default.

Use `--particle-spacing 0.003` to recover the old resolution. If a narrow geometry cannot hold the saved fill at the default 3.5 mm with the existing initialization clearance, the generator reports a fallback to 3 mm and records the reason in the geometry metadata. An explicit spacing is honored or rejected if it cannot fit. The radius controls native mass, kernel support and neighborhood search together. Volume-map cache filenames include the particle radius. The same goblet target of 148.127 mL initializes 148.122 mL with 5,486 particles at 3 mm and 148.133 mL with 3,455 particles at 3.5 mm, a 37% reduction in count.

`replay_samples.csv` samples timestep, active count, peak speed, fastest particle height and peak nonpressure acceleration every 0.5 solver seconds. `replay_timing.json` records the minimum and mean step size and timings for boundary/interpolation, native stepping plus export, and frame tracking. In the archived goblet replay, the fastest particles near 36 s were already below the workspace at about 29 m/s. Their subsequent speed increase was consistent with free fall. Native CFL considers all active particles, including misses, so those particles drove it to the existing 100 µs minimum. CFL, pressure, gravity and viscosity are unchanged.

### Goblet benchmark

All runs used the same recorded ML goblet trajectory, a 75% fill, and cached boundary maps. Simulated time includes the 0.25 s fixed-pose settling period.

| Replay | Particles | Simulated time | Steps | Min / mean timestep | Wall time |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3.0 mm, 12 FPS, start at zero | 5,486 | 0.301 s | 368 | 0.100 / 0.817 ms | 13.54 s |
| 3.5 mm, 6 FPS, start at zero | 3,455 | 0.301 s | 323 | 0.100 / 0.933 ms | 7.38 s |
| 3.5 mm, 6 FPS, auto start at 26.125 s | 3,455 | 0.302 s | 324 | 0.100 / 0.931 ms | 6.58 s |
| 3.0 mm, 12 FPS, matched pour interval | 5,486 | 9.151 s | 45,653 | 0.100 / 0.200 ms | 606.59 s |
| 3.5 mm, 6 FPS, matched pour interval | 3,455 | 9.151 s | 50,096 | 0.100 / 0.183 ms | 417.33 s |

The matched pour ran **1.45× faster**. Auto-start also skipped 26.125 s of recorded motion. The benchmark covered the onset and active pour, so whole-trajectory speedup remains unmeasured. The target was 148.127 mL with initialized volumes of 148.122 mL and 148.133 mL respectively. Both runs conserved all particles. At 3.5 mm, sampled rim outflow appeared 0.83 s earlier, and receiver collection at the endpoint increased from 40.6% to 48.3%. The fraction outside both vessels fell from 11.7% to 6.0% and may include liquid still in flight. These differences show that collection estimates are sensitive to particle spacing. Use the 3.0 mm override for comparisons with prior runs.

Native solver steps and exports together accounted for over 94% of matched-pour runtime. A short 6 vs. 12 FPS check at fixed spacing produced identical step counts, timestep statistics, and sampled peak speed, with two and four exported frames respectively. In the archived slow replay around 36 s, the fastest escaped particles were already well below the workspace and accelerating near gravity. Their speed was sufficient to drive CFL to the existing 0.100 ms floor. [Benchmark data](benchmarks/sph_performance.csv)

### Fast ML evaluation and visual evaluation

Rerun `bash scripts/setup_sph.sh` once to install the viscosity fix. Reuse the existing volume-map cache. For a new experiment, point `run_dir` at the directory exported by `visualize_station.py`.

```bash
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1
run_dir=sph_pouring_trial/test_tube_run_01
.venv/bin/python evaluate_sph_spill.py --station-export "$run_dir" \
  --scene-dir sph_pouring_trial/station_cache --headless \
  --output sph_pouring_trial/fast_01

.venv/bin/python evaluate_sph_spill.py --station-export "$run_dir" \
  --scene-dir sph_pouring_trial/station_cache --export-fps 12 \
  --output sph_pouring_trial/visual_01
.venv/bin/python visualize_pour.py --fluid-output sph_pouring_trial/visual_01 --no-robot
```

Headless evaluation still writes the final spill report and `replay_timing.json`. In Python use `evaluate(trajectory, config=config, export_fps=0)` for the same path. The default is 30 fps. Use `--diagnostics` to collect per-step data when investigating a run. It adds work and is off by default.

Use `--duration` to cap the requested interval after the selected start. A standalone evaluation retains the bundle’s saved spacing and starts at the first pose unless you pass an override. Generated runs use auto-start, 3.5 mm spacing and 6 fps by default. Use `--sph-start-time 0` when earlier liquid motion matters.

### Free-flight instability and runtime

With the previous Weiler implicit-viscosity warm start, a particle could carry its last viscous velocity increment even after it had left all neighbors and boundary support. The linear-system row for an isolated particle is the identity. The old initial guess was the current velocity plus that stale increment. Once the global relative residual passed the convergence test, the solver could accept the guess and apply the increment again. In the robot trace the largest repeated nonpressure impulse reached 0.196 m/s per 0.099 ms step, outside both wall maps, at nearly the self-density of 318.31 kg/m³. The source was moving slowly at the event. The pressure solver remained at two iterations. A diagnostic isolated particle gained 0.20 m/s horizontally on each step with the old build and conserved horizontal velocity with the patched build. The patch initializes the same implicit system at its current right-hand side. It leaves viscosity, gravity, geometry, and particle count intact.

See [the benchmark CSV](benchmarks/sph_replay.csv) and [the robot event CSV](benchmarks/sph_events.csv) for the recorded comparisons. The conservative default is CFL 0.125. Override it with `--cfl-factor` for experiments. On the 2.4 s controlled tilt with 448 particles, CFL 0.125, 0.25 and 0.5 required 12,453, 8,498 and 5,796 steps, respectively. Their largest single-step velocity changes were 0.54, 0.67 and 1.24 m/s. CFL 0.25 is a useful controlled-test setting. The larger 0.5 kicks make it a poor default. At CFL 0.25, headless and 12 fps produced identical final counts on the control. Replay measurement was 9.05 s headless and 9.70 s visual, with about 7 s in native stepping and 2 s in motion and boundary updates. The old full robot run spent 157 of its 288 native solver seconds on neighborhood search. More frame-rate reduction yields limited benefit while native stepping dominates.

The corrected 62 s robot replay conserved 448 particles and finished with 229 in the receiver, 215 in the source, and four escaped. The high global peak of 223 m/s occurred below the scene during the long fall. Above world Z zero the peak was 2.95 m/s. The largest single-step change was 1.20 m/s and reduced the affected particle's kinetic energy from 62.7 to 22.2 μJ. Receiver count was already 229 at 47 s for this particular take. Use a shorter scoring horizon only when it represents the comparison you intend. Full headless validation with per-step tracing took 440 s, including 139 s spent sampling diagnostics. The approximate 301 s without sampling is a timing subtraction, with no second full run to measure it directly.

The coarse check used `"particle_radius": 0.0018` in a copy of the control's cup config. It reduced the fill to 207 particles and shortened the solver portion, but changed collection from 434/448 to 145/207 at CFL 0.125. Keep 448 particles as the baseline for comparisons. The corrected receiver counts are still sensitive to solver settings and aim. Escaped particles remain in the simulation and accelerate under gravity because the scene has no ground. Counts are provisional.

### Diagnose receiver impact and particle speeds

Generate the 120° control above, then use the same bundle for stationary and moving runs. Reuse the cache and choose fresh output directories.

```bash
export PYTHONPATH="$PWD/.venv/splishsplash/build/lib"
export OMP_NUM_THREADS=1
control=sph_pouring_trial/control_120deg_01
.venv/bin/python evaluate_sph_spill.py --station-export "$control" \
  --scene-dir sph_pouring_trial/station_cache --diagnostics --hold-source \
  --duration 0.6 --output sph_pouring_trial/diagnostic_fixed
.venv/bin/python evaluate_sph_spill.py --station-export "$control" \
  --scene-dir sph_pouring_trial/station_cache --diagnostics \
  --duration 2.4 --output sph_pouring_trial/diagnostic_tilt
.venv/bin/python visualize_pour.py --fluid-output sph_pouring_trial/diagnostic_tilt --no-robot
```

Repeat with an earlier template by adding `--template old_template.json`. To isolate the CFL change, copy `DoubleDamBreak_export.json` and set `Configuration.cflFactor` to `1` in the copy. For a measured station export, use its directory in `--station-export`. The first 5.6 seconds of the deterministic test-tube run include approach and moving gripper closure.

`step_diagnostics.csv` reports speeds, density, active particle count and boundary velocities on every solver step. Velocity changes follow particle IDs across native sorting. The largest change includes the nonpressure impulse and the vector residual from the other stages. The two magnitudes can oppose each other. Signed wall distance and the boundary density term are sampled at that particle's pre-step center. A blank distance means the particle is outside the map. A position residual beyond roundoff can indicate the solver's penetration fallback. The native binding does not accumulate reaction forces for the prescribed vessels.

In the original 120° test, the first greater-than-1 m/s jump ended at 1.71684 s at the receiver floor. The particle center was 3.26 mm from the wall and the receiver map contributed about 304 kg/m³ to its density. There was no detectable penetration projection at that event. At the largest jump the receiver contributed 525 kg/m³, and the pressure-side residual was 3.28 m/s. The jump persisted with divergence disabled, implicating the density-pressure response to under-resolved receiver impact. The patched DFSPH step uses one timestep for both pressure solves and integration. The replay supplies world-frame rigid-body point velocities through `v + ω × (x − origin)`. The large jumps in the controlled pour occurred at the stationary receiver when the source was slowing.

The measured robot replay exposed another impulse at 5.02123 s in the source. At the 5.02 s CSV knot, the piecewise-interpolated wall velocity jumped from rest to about 0.60 m/s of translation and 6.89 rad/s of rotation. Replay and Meshcat now share continuous-velocity interpolation through the recorded poses using componentwise monotone cubics for position and a rotation spline. In the same 5.6 s prefix, peak particle speed fell from 2.61 to 0.85 m/s and the largest single-step velocity change fell from 2.55 to 0.22 m/s. All 448 particles stayed in the tube.

The default template now uses a CFL factor of 0.125. At 2 m/s and a 3 mm particle diameter, the native bound requests a 0.075 ms step before the 0.1 ms solver floor, compared with 0.6 ms at factor 1. This resolves the impact across more pressure updates while retaining the existing implicit viscosity. The result is still sensitive to the timestep, map resolution, fill and aim. The replay has no ground boundary, so missed particles continue to accelerate under gravity below the workspace. The reported particle counts are provisional.

### Evaluate a trajectory from your own Drake planner or ML script

Use `TrajectoryRecorder` from `drake_trajectory_export.py`. After each simulator advance, record the measured vessel pose with `recorder.record(t, "source", X_WC)`, then save it as `candidate.csv`. Supply meters in the Z-up world frame and strictly increasing timestamps. Quaternions use wxyz order and are normalized on read. Replay and the viewer subtract the first timestamp to obtain elapsed time. The scene manifest and spill report retain the original time origin. If logging gripper or center-of-mass poses, compose with the fixed transform from that frame to the vessel bottom-center frame first.

```bash
# Choose a matching geometry bundle, or supply your own cups.json.
.venv/bin/python station_sph_export.py --controlled sph_pouring_trial/flask_geometry_01 --vessel erlenmeyer
PYTHONPATH="$PWD/.venv/splishsplash/build/lib" OMP_NUM_THREADS=1 \
  .venv/bin/python evaluate_sph_spill.py --trajectory candidate.csv \
  --config sph_pouring_trial/flask_geometry_01/cups.json \
  --scene-dir sph_pouring_trial/station_cache --output sph_pouring_trial/candidate_01
.venv/bin/python visualize_pour.py --fluid-output sph_pouring_trial/candidate_01 --no-robot
```

Use the generated geometry only if the local vessel frame, source shape, receiver shape, and world receiver placement match your experiment. Edit `receiver_origin` for a different world placement. The stationary receiver uses the same bottom-center frame convention as the source. Set `receiver_quaternion_wxyz` for a rotated receiver. Its default is identity. For cylindrical vessels use `source_cup` and `receiver_cup`. For other axisymmetric shapes use `source_profile` and/or `receiver_profile`. Each profile takes precedence over its corresponding cylinder dimensions and describes both the fluid boundary and classifier cavity. Arbitrary non-axisymmetric meshes are outside this interface.

The following is an example profile configuration for a conical flask source and cylindrical default receiver. Each pair is `[z, radius]` in meters, ordered from base to rim. The outer wall starts at zero, the cavity floor is above it, and both profiles end at the same rim height. The inner wall must stay strictly within the outer wall and the complete initial fill box must fit in the cavity.

```json
{
  "source_profile": {
    "outer": [[0, 0.040], [0.006, 0.040], [0.080, 0.016], [0.160, 0.016]],
    "inner": [[0.004, 0.037], [0.006, 0.037], [0.080, 0.013], [0.160, 0.013]]
  },
  "receiver_origin": [0.72, 0.14, 0.20],
  "fill_start": [-0.014, -0.014, 0.010],
  "fill_end": [0.014, 0.014, 0.045],
  "particle_radius": 0.0015,
  "max_time_step": 0.001
}
```

In Python pass the same data as `config=PouringConfig(**geometry)` to `evaluate(recorder, config=...)` or use a CSV path. `VesselProfile` also accepts the two ring lists directly.

## Team and responsibilities

| Member | Owns |
| --- | --- |
| Shasha | Starter simulation, workspace and table setup, robot placement, glasses and cups, robot controls |
| Kevin | Repository organization and collaboration instructions; initial ML optimization experiments |
| Wyatt | Physics modeling |

Kevin uses macOS. Shasha and Wyatt use Windows. The branch includes a runnable Drake-to-SPlisHSPlasH pouring demo.

The physics scope and ML objective are still open. Agree on the information and commands exchanged between the scene, controls, physics model, and ML experiments before building dependencies between them.

## How we work

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the complete Git workflow and copyable commands.

- Keep `main` as the shared, working version.
- Each person works on a separate branch for each task, such as `shasha/scene-setup`, `wyatt/physics-model`, or `kevin/ml-experiments`.
- Push the task branch and open a pull request into `main`.
- Have another teammate review the change before merging.
- Start the next task from an updated `main` on a new branch.

## Evaluator reference

The reference environment used Python 3.14.4 and a Release build of SPlisHSPlasH 2.19.1 at commit `dba6b26b9387ab6bdd187d6df0a7c9d490d03500`, with single precision and CompactNSearch. If you build the bindings yourself, apply the patches from `scripts/setup_sph.sh` to the pinned source before compiling. Use the setup script for the tested build configuration.

The checked-in CSV runs for 3 seconds. Use `--duration 0.03` for a quick run or `--hold-source` for a fixed-vessel control. Each run keeps its own scene, VTK frames, log, and `spill_report.json`. Use a new or empty output directory for each run. Reuse `--scene-dir` to share volume maps across serial runs with the same SPlisHSPlasH build. Use separate cache directories for parallel processes. The first run for a cup shape builds its map. Without `--scene-dir`, the maps live with the run's scene.

**Call from Python.** The same API accepts a CSV path or a `TrajectoryRecorder` instance. The returned dictionary matches the JSON report. If `output` is omitted, the function creates a fresh temporary directory and returns its path as `output_dir`.

```python
from evaluate_sph_spill import evaluate

result = evaluate(
    "drake_vessel_trajectory.csv",
    output="sph_pouring_trial/python_run",
    scene_dir="sph_pouring_trial/eval_scene",
)
print(result["collected_fraction"], result["remaining_fraction"], result["outside_fraction"])
```

**Configure cups and fill.** Pass `config=PouringConfig(...)` in Python (import it from `evaluate_sph_spill`), or add `--config cups.json` to the command. Omitted fields keep the demo defaults. The JSON below shows all defaults.

```json
{
  "source_cup": [0.035, 0.003, 0.110, 0.004],
  "receiver_cup": [0.060, 0.004, 0.120, 0.004],
  "receiver_origin": [0.110, 0, 0],
  "fill_start": [-0.022, -0.022, 0.012],
  "fill_end": [0.022, 0.022, 0.080],
  "particle_radius": 0.003,
  "max_time_step": null
}
```

Cup dimensions are inner radius, wall thickness, total height, and base thickness, in meters. The stationary receiver origin is its bottom center in world coordinates. The fill bounds are the opposite corners of a box in the initial source frame, measured from the bottom center. Keep the entire box inside the source cavity. Compare candidate motions with the same build, geometry, fill, duration, and thread count.

**Interpret the report.** The three counts `in_source`, `in_receiver`, and `outside_both` partition the final particle centers using the generated cups' faceted interiors and the source pose actually applied on the last step. The fractions are normalized by the final particle count. `outside_fraction` is a provisional spill proxy and includes any fluid still in flight or inside a solid. `below_world_z_zero` is a separate diagnostic. The replay is one-way, with prescribed vessel motion and no table or ground contact. Use the counts for motion comparisons under identical settings. `spill_measurement_validated: false` indicates that quantitative spill accuracy remains unverified, so treat these outputs as provisional metrics for optimization and training.

### Optional stationary settling

Add `--settle-duration 0.2` to the evaluation command or pass `settle_duration=0.2` to `evaluate()`. The source stays at its first pose with zero linear and angular velocity during settling. The full requested replay begins at Drake t=0 on the next solver step. The solver clock and VTK output include the settling phase. Settling is off by default, and all particles carry through to the final report. With the original solver settings, a short fixed-source comparison ejected one particle with and without settling. Use the phase to separate startup from trajectory motion.

### Containment investigation

Both generated cups have 258 vertices and 512 nondegenerate triangles. Every edge is paired in opposite directions and both meshes have positive signed solid volume. The bases are 4 mm thick and the source and receiver walls are 3 mm and 4 mm thick, respectively. Probes of the Bender2019 distance field were positive in the cup cavities and outside the solids, and negative inside the base and wall at the sampled points. DFSPH samples each map in the body's local frame and uses positive nearby boundary volumes for pressure and divergence. The 1.3-second moving run showed up to 239 active center contributions from the source map and 42 from the receiver map. The penetration fallback moves a center slowly along the distance gradient and clears its velocity. It evaluates particle positions at discrete steps, so the collision model can still miss crossings.

In per-step particle-ID traces, every particle that ended below world Z zero first cleared the source opening, including all escaped particles in the fixed-vessel control. The initial particles were inside the source. The fixed trace had abrupt speed increases during early settling, followed by rim exits between about 0.14 and 0.18 s. One particle jumped from 2.39 to 8.89 m/s in a single step inside the cavity. The timing and speed jumps point to a startup fluid-solver transient. The force term responsible for the jumps remains unresolved. With no ground boundary, particles that miss the receiver continue to fall. The tracked paths explain the escapes through the open rim, and the fixed control shows they can occur independently of commanded motion. Cup geometry and simulation parameters were kept fixed.

### Startup ejection diagnostic

The fill starts at rest on a 6 mm lattice. The nearest centers are 12.36 mm from the source wall and 14.00 mm above its bottom. The first computed densities are 607–999.97 kg/m³ against a rest density of 1,000 kg/m³. The first three steps match gravity alone. The startup issue develops after the fill reaches the cup boundary.

In the original fixed-source run, particle 175 had a 12.01 m/s velocity change in the step ending at 0.18452 s. Nonpressure acceleration accounted for 12.00 m/s, with zero density-pressure impulse and about 0.008 m/s remaining. Explicit viscosity was the only other nonpressure force besides gravity. In the next step, the particle was 3.85 mm from the inner wall. The Bender2019 density term `rho0 * V * W` contributed about 1,511 of the 2,026 kg/m³, with the sampled normal pointing into the cavity and the vessel velocity zero. That step included an 8.63 m/s density-pressure contribution, 0.20 m/s from nonpressure acceleration, and a 3.30 m/s residual attributed to divergence correction. These are magnitudes of vectors and do not add algebraically. Particle 129 was the one that crossed the rim at 0.221 s. Its largest prior velocity change was also dominated by explicit viscosity while clear of the boundary. The volume-map density term follows the published method. [Bender et al., Volume Maps](https://animation.rwth-aachen.de/media/papers/65/2019-MIG-VolumeMaps.pdf)

**Implementation fix.** The pinned DFSPH code replaced the current timestep before the density-pressure solve but still advanced positions and time with the original step. At the 0.18452 s event, the step advanced by 1.275 ms while the pressure solve used 0.237 ms. The setup patch computes the next CFL timestep after the current integration so both pressure solves and the update use one timestep. This corrects the inconsistency. The fixed-source run still ejected particles with the original explicit viscosity.

**Stabilization.** The demo now uses Weiler’s implicit viscosity at the same kinematic coefficient of 0.01 m²/s. The short control showed no ejections and much smaller velocities. Lowering the explicit coefficient to a water-scale 10⁻⁶ m²/s also reduced velocities but still ejected particles, so that variant was not adopted. The implicit default retains a local density spike and has only been checked for this startup window. The settings do not establish quantitatively accurate water behavior.

| Fixed source, first 0.25 s, one thread | Peak density (kg/m³) | Peak speed (m/s) | Distinct uphill rim crossings | Final source / receiver / outside | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original build and explicit viscosity | 2,176 | 10.14 | 1 | 359 / 0 / 1 | 360 |
| Timestep fix and original explicit viscosity | 1,679 | 12.34 | 9 | 351 / 0 / 9 | 360 |
| Timestep fix and water-scale explicit viscosity | 2,013 | 1.69 | 7 | 353 / 0 / 7 | 360 |
| Timestep fix and implicit viscosity (current demo) | 1,607 | 0.57 | 0 | 360 / 0 / 0 | 360 |

Repeat the current control using the Quick Start evaluation command with `--hold-source --duration 0.25` and a fresh output directory.

Before the startup changes, the single-threaded 3-second reference runs produced these final geometric counts. Multithreaded diagnostic runs produced different exact tallies.

| Run | Source | Receiver | Outside both | Below world Z zero |
| --- | ---: | ---: | ---: | ---: |
| Drake trajectory | 18 | 341 | 1 | 1 |
| Fixed-vessel control | 354 | 0 | 6 | 6 |

For solver diagnostics, call `replay_scene(..., on_step=callback)` from Python. The callback receives the simulation and the start and end times of each step. Read `getParticleId(i)` with position and velocity so particles can be tracked across neighbor-search reordering.

## Open decisions

- Robot gripper and vessel details.
- Physics scope and modeling assumptions.
- ML inputs, outputs, objective, and evaluation metrics.
- Shared interfaces and units between components.
- Reproducible runtime setup and dependency versions.

Record decisions in the relevant pull request and update this README when they affect the whole team.
