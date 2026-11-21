# Pouring Bot

A collaborative project simulating a KUKA iiwa robotic arm for pouring, using Python and `pydrake`.

## Team and responsibilities

| Member | Owns |
| --- | --- |
| Shasha | Starter simulation, workspace and table setup, robot placement, glasses and cups, robot controls |
| Kevin | Repository organization and collaboration instructions; initial ML optimization experiments |
| Wyatt | Physics modeling |

Kevin uses macOS. Shasha and Wyatt use Windows. There is no starter simulation yet; Shasha will establish the first runnable example.

The physics scope and ML objective are still open. Agree on the information and commands exchanged between the scene, controls, physics model, and ML experiments before building dependencies between them.

## How we work

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the complete Git workflow and copyable commands.

- Keep `main` as the shared, working version.
- Each person works on a separate branch for each task, such as `shasha/scene-setup`, `wyatt/physics-model`, or `kevin/ml-experiments`.
- Push the task branch and open a pull request into `main`.
- Have another teammate review the change before merging.
- Start the next task from an updated `main` on a new branch.

## Runtime setup status

No Python version, Drake version, installation procedure, or run command has been selected yet. With the first runnable simulation, document the exact environment and commands here, including how the Windows teammates run it. Verify setup on the team's systems before treating it as shared infrastructure.

## Evaluate a Drake pour with SPlisHSPlasH

Run this from the repository root on `wyatt/drake-sph-integration`. The tested setup was Linux/WSL with Python 3.14.4, NumPy, SciPy, and a Release build of SPlisHSPlasH 2.19.1 (commit `dba6b26b9387ab6bdd187d6df0a7c9d490d03500`, single precision, CompactNSearch). The extension must be built for the Python interpreter that runs the evaluator. If you already have that build, start at the `export PYTHONPATH` command.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install numpy scipy
# For a fresh SPlisHSPlasH checkout, install its Linux build dependencies first.
git clone --recurse-submodules https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git "$HOME/tools/SPlisHSPlasH"
git -C "$HOME/tools/SPlisHSPlasH" checkout dba6b26b9387ab6bdd187d6df0a7c9d490d03500
git -C "$HOME/tools/SPlisHSPlasH" submodule update --init --recursive
cmake -S "$HOME/tools/SPlisHSPlasH" -B "$HOME/tools/SPlisHSPlasH/build" \
  -DCMAKE_BUILD_TYPE=Release -DUSE_PYTHON_BINDINGS=ON \
  -DPython_EXECUTABLE="$PWD/.venv/bin/python"
cmake --build "$HOME/tools/SPlisHSPlasH/build" --target pysplishsplash --parallel
export PYTHONPATH="$HOME/tools/SPlisHSPlasH/build/lib${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1
.venv/bin/python -c 'import pysplishsplash; print(pysplishsplash.__file__)'

.venv/bin/python evaluate_sph_spill.py \
  --trajectory drake_vessel_trajectory.csv \
  --scene-dir sph_pouring_trial/eval_scene \
  --output sph_pouring_trial/moving_3s
```

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
  "fill_end": [0.022, 0.022, 0.080]
}
```

Cup dimensions are inner radius, wall thickness, total height, and base thickness, in meters. The stationary receiver origin is its bottom center in world coordinates. The fill bounds are the opposite corners of a box in the initial source frame, measured from the bottom center. Keep the entire box inside the source cavity. Compare candidate motions with the same build, geometry, fill, duration, and thread count.

**Export robot motions.** Use `TrajectoryRecorder` in `drake_trajectory_export.py` to record `plant.EvalBodyPoseInWorld(context, source_body)` at each sample. Record the vessel's bottom-center origin with body name `source`, in meters, with Z up, at strictly increasing timestamps starting at zero. The CSV columns are `time_s,body,x_m,y_m,z_m,qw,qx,qy,qz`. Transform the gripper pose to the vessel frame before export if they differ. The evaluator uses the demo's upright initial fill, requires an identity initial source rotation, and defaults the receiver origin to `(0.110, 0, 0)` m. To regenerate the demo CSV, install Drake in an environment it supports and run `python drake_pouring_simulation.py --no-vis` from the repository root. Drake is only needed to produce the CSV.

**Interpret the report.** The three counts `in_source`, `in_receiver`, and `outside_both` partition the final particle centers using the generated cups' faceted interiors and the source pose actually applied on the last step. The fractions are normalized by the final particle count. `outside_fraction` is a provisional spill proxy and includes any fluid still in flight or inside a solid. `below_world_z_zero` is a separate diagnostic. The replay is one-way, with prescribed vessel motion and no table or ground contact. Use the counts for motion comparisons under identical settings. `spill_measurement_validated: false` indicates that quantitative spill accuracy remains unverified, so treat these outputs as provisional metrics for optimization and training.

### Containment investigation

Both generated cups have 258 vertices and 512 nondegenerate triangles. Every edge is paired in opposite directions and both meshes have positive signed solid volume. The bases are 4 mm thick and the source and receiver walls are 3 mm and 4 mm thick, respectively. Probes of the Bender2019 distance field were positive in the cup cavities and outside the solids, and negative inside the base and wall at the sampled points. DFSPH samples each map in the body's local frame and uses positive nearby boundary volumes for pressure and divergence. The 1.3-second moving run showed up to 239 active center contributions from the source map and 42 from the receiver map. The penetration fallback moves a center slowly along the distance gradient and clears its velocity. It evaluates particle positions at discrete steps, so the collision model can still miss crossings.

In per-step particle-ID traces, every particle that ended below world Z zero first cleared the source opening, including all escaped particles in the fixed-vessel control. The initial particles were inside the source. The fixed trace had abrupt speed increases during early settling, followed by rim exits between about 0.14 and 0.18 s. One particle jumped from 2.39 to 8.89 m/s in a single step inside the cavity. The timing and speed jumps point to a startup fluid-solver transient. The force term responsible for the jumps remains unresolved. With no ground boundary, particles that miss the receiver continue to fall. The tracked paths explain the escapes through the open rim, and the fixed control shows they can occur independently of commanded motion. Cup geometry and simulation parameters were kept fixed.

The single-threaded 3-second reference runs produced these final geometric counts. Multithreaded diagnostic runs produced different exact tallies.

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
