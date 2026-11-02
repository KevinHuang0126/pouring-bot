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

## Open decisions

- Robot gripper and vessel details.
- Physics scope and modeling assumptions.
- ML inputs, outputs, objective, and evaluation metrics.
- Shared interfaces and units between components.
- Reproducible runtime setup and dependency versions.

Record decisions in the relevant pull request and update this README when they affect the whole team.
