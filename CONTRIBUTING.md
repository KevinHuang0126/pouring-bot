# Working together

## Branches and ownership

`main` is the shared integration branch. Do development on task branches, then merge through pull requests. A branch separates unfinished work until the team is ready to combine it.

Use `<name>/<short-task>` for branch names:

| Member | Example branches |
| --- | --- |
| Shasha | `shasha/scene-setup`, `shasha/vessels`, `shasha/controls` |
| Kevin | `kevin/repo-docs`, `kevin/ml-experiments` |
| Wyatt | `wyatt/physics-model` |

Create one branch per focused task. Each person can own their branch independently; a new branch starts from the latest `main`, rather than continuing indefinitely on one personal branch. Coordinate shared-file edits with the person responsible for that component.

## First-time clone

After the initial repository commit has been pushed, clone the repository:

```sh
git clone https://github.com/KevinHuang0126/pouring-bot.git
cd pouring-bot
```

The Git commands in this guide work in macOS Terminal and Windows PowerShell with Git installed. Python and Drake setup will be documented in the README when the starter simulation is available.

## Start a task

First check that your current branch has no unfinished changes:

```sh
git status
```

If you have changes, commit them on their existing task branch before switching. Then update `main` and create a task branch. Replace the example branch name with your own:

```sh
git switch main
git pull --ff-only origin main
git switch -c shasha/scene-setup
```

`--ff-only` stops if local and remote history have diverged, so an ordinary update does not silently create a merge commit. If it stops, inspect the history with a teammate before proceeding.

## Save and share work

Make focused commits. Inspect the changes and stage the specific files you intend to include. Replace the example paths below with real paths from `git status`:

```sh
git status
git diff
git add path/to/changed_file.py path/to/another_file.py
git diff --cached
git commit -m "Add initial iiwa scene and tables"
git push -u origin shasha/scene-setup
```

Push regularly so teammates can see progress. Pushing a task branch does not change `main`.

## Open a pull request

On GitHub, open a pull request from your task branch into `main`. A draft pull request is useful while a task is still in progress.

Include:

- What changed and how to run or inspect it.
- What you actually checked, including the command and outcome when applicable.
- Any setup changes, dependencies, modeling assumptions, or interface changes.
- Any unfinished work or checks you could not run.

Ask at least one other teammate to review it. Before merging, make sure the change works with the current `main` and that required checks pass. If no automated checks exist yet, say what manual verification you performed. A documentation-only change needs a review of its instructions and links.

The reviewer checks that the change is understandable, runnable where applicable, and compatible with the other components. For a change to another person's component, include that person in the review.

Once approved, use GitHub's **Squash and merge** option to keep one commit per completed task on `main`. Delete the remote task branch after merging, then start the next task from an updated `main`.

Review and pull-request rules are team conventions until repository settings enforce them. The repository maintainer can configure a rule for `main` requiring pull requests and one approval, if available for this repository.

## Bring changes from main into an active task

Commit your current work first and confirm `git status` is clean. While on your task branch:

```sh
git fetch origin
git merge origin/main
```

If Git reports conflicts, open the listed files, reconcile both changes, and remove the conflict markers. Coordinate with the other author when the intended behavior is unclear. Then stage the resolved files, finish the merge, rerun the relevant checks, and push:

```sh
git add path/to/resolved_file.py
git commit
git push
```

If you need to stop resolving the merge, `git merge --abort` returns to the pre-merge state. Starting with a clean working tree makes this recovery safer. Do not use force pushes on `main` or on branches shared with teammates.

## Coordinate the components

Shasha establishes the starter simulation and documents its entry point. Kevin and Wyatt can begin independent experiments while it is being built; work that depends on the scene should integrate after its interface is agreed and merged into `main`.

Before connecting components, agree on:

- Which component creates and owns the Drake plant, scene, and simulator.
- Which commands the controls accept, and what observations the physics and ML code receive.
- Units, coordinate frames, timing, and the meaning of each shared quantity.
- How an experiment starts, resets, and reports results.

Write down these agreements in the relevant code documentation and pull request. Discuss changes to shared interfaces before implementing them, and document dependency changes with reproducible setup instructions.

Do not commit credentials, local virtual environments, or generated training outputs. Keep large datasets, model checkpoints, and simulation recordings outside Git unless the team explicitly agrees on a storage method. Share enough configuration, seed information, and commands to reproduce an experiment.
