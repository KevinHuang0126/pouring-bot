# WSL setup and run guide — simple pouring trajectory

This prototype is **not pushed to GitHub yet**. The GitHub connector returned a permission error when attempting to create `shasha/simple-pour-trajectory` in `KevinHuang0126/pouring-bot`. The instructions below create and push this branch using your own GitHub credentials, **without modifying `main`**.

## A. Get WSL ready (one-time setup)

From **Windows PowerShell** (not Ubuntu), if WSL / Ubuntu is not already installed:

```powershell
wsl --install -d Ubuntu-24.04
```

Restart Windows if prompted, then open **Ubuntu / WSL** and create your Linux username and password. The following commands run in the **Ubuntu WSL terminal**, not PowerShell.

```bash
sudo apt update
sudo apt install -y git unzip python3 python3-venv python3-pip
lsb_release -a
python3 --version
```

The easiest supported setup is Ubuntu 24.04 with Python 3.12. If your Ubuntu / Python versions differ, check https://drake.mit.edu/installation.html for an officially supported pairing. This demonstration does not need the `manipulation` package used by `visualize_station.py`.

## B. Clone cleanly and make a task branch

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/KevinHuang0126/pouring-bot.git
cd pouring-bot
git switch main
git pull --ff-only origin main
git switch -c shasha/simple-pour-trajectory
git branch --show-current
```

Expected branch name: `shasha/simple-pour-trajectory`. This branch starts at the current `main` but does not change `main` or other branches. If you already have a directory named `~/projects/pouring-bot`, either use a different empty clone directory or reuse your clone only after `git status` confirms it is clean.

## C. Copy the downloaded prototype into the new branch

Download **simple-pour-trajectory-package.zip** from the ChatGPT answer using your Windows browser; normally it will be saved to Windows Downloads.

In WSL, replace `YOUR_WINDOWS_USERNAME` with the name of your Windows profile folder (not your WSL username):

```bash
cd ~/projects/pouring-bot
cp '/mnt/c/Users/YOUR_WINDOWS_USERNAME/Downloads/simple-pour-trajectory-package.zip' .
unzip -n simple-pour-trajectory-package.zip
rm simple-pour-trajectory-package.zip
ls simple_pour_trajectory.py tests/test_simple_pour_trajectory.py models/package.xml
```

If the ZIP downloaded elsewhere, adjust the path in the `cp` command. `unzip -n` avoids overwriting existing files; on a clean clone those two Python paths should be new. The ZIP includes `SETUP.md` (technical notes), `WSL_SETUP.md` (this guide), the script, and tests.

## D. Install Drake for this standalone demo

```bash
cd ~/projects/pouring-bot
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install drake numpy
python -c 'import pydrake, numpy; print("Drake and NumPy import successfully")'
```

**Important:** `drake` is the official pip package, while Python imports its bindings as `pydrake`. Run `source .venv/bin/activate` in each new WSL session, or use `.venv/bin/python` explicitly. The local `.venv` should not be committed to Git.

## E. Run the numerical tests

```bash
cd ~/projects/pouring-bot
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_simple_pour_trajectory.py' -v
```

Expected: **7 tests, OK**. These cover fill and tilt activation, a ballistic path, vessel velocity influence, invalid fill input, and opening hit scoring.

## F. Run the Meshcat demonstration

```bash
python simple_pour_trajectory.py
```

Wait for the console to print a **Meshcat URL**, typically `http://localhost:7000` (the actual port can vary). Copy that exact URL into a **Windows** browser. In the 3D view you should see a table, receiving beaker, three different tilted source vessels suspended in the air, colored schematic fill cylinders, and predicted blue trajectories.

Keep the terminal process running while viewing the scene. Click the terminal and press **Enter** to exit.

The script is a **static display** without the iiwa arm. It shares the same `models/*.sdf` as the original station, but does not run robot motion or physical fluid dynamics.

## G. Commit and push to GitHub without touching other branches

```bash
cd ~/projects/pouring-bot
source .venv/bin/activate
git status --short
git branch --show-current
git add simple_pour_trajectory.py tests/test_simple_pour_trajectory.py SETUP.md WSL_SETUP.md
git diff --cached --stat
git commit -m 'Add standalone Drake simple pouring trajectory demo'
git push -u origin shasha/simple-pour-trajectory
```

The first push may ask you to authenticate. GitHub does not accept your account password for HTTPS Git pushes. If you already use Git Credential Manager or GitHub CLI in WSL, sign in with that mechanism. Alternatively install and configure GitHub CLI:

```bash
sudo apt install -y gh
gh auth login -h github.com -p https -w
gh auth setup-git
git push -u origin shasha/simple-pour-trajectory
```

You must have **write access** to `KevinHuang0126/pouring-bot` for a direct push. Otherwise ask Kevin to grant access or push to your own fork and open a pull request.

Once pushed, see the branch at:

https://github.com/KevinHuang0126/pouring-bot/tree/shasha/simple-pour-trajectory

This URL will only work **after the push succeeds**. You may open a pull request targeting `main`, but do not merge until teammates review it.

## H. Running the demo on a later day

```bash
cd ~/projects/pouring-bot
git switch shasha/simple-pour-trajectory
source .venv/bin/activate
python simple_pour_trajectory.py
```

## Troubleshooting

- **`ModuleNotFoundError: pydrake`** — make sure `.venv` is activated and `python -m pip install drake` succeeded.
- **`No matching distribution found for drake`** — check `lsb_release -a`, `python --version`, and supported configurations at https://drake.mit.edu/installation.html.
- **Meshcat page not reachable** — use the exact URL printed by the script, not a hard-coded port; modern WSL typically forwards localhost to Windows. Check firewall or VPN settings if forwarding is blocked.
- **Git push rejected (403)** — verify the GitHub account and repository write permission. The connector used by ChatGPT was denied branch-creation permission; your own credentials may differ.
- **`fatal: a branch named ... already exists`** — use `git switch shasha/simple-pour-trajectory` instead of `git switch -c ...`.
- **`git clone` says directory exists** — choose a fresh directory or use your existing clone carefully after checking `git status`.
