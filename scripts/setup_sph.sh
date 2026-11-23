#!/usr/bin/env bash
# Build the pinned SPlisHSPlasH bindings for an isolated demo environment.
set -euo pipefail

root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
venv="${POURING_VENV:-$root/.venv}"
python="${PYTHON:-python3}"
jobs="${BUILD_JOBS:-2}"
revision=dba6b26b9387ab6bdd187d6df0a7c9d490d03500
upstream=https://github.com/InteractiveComputerGraphics/SPlisHSPlasH.git

for command in "$python" git cmake c++ pkg-config; do
    command -v "$command" >/dev/null || {
        echo "Missing $command. See the Linux/WSL prerequisites in README.md." >&2
        exit 1
    }
done
"$python" -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] <= (3, 14) else "Use Python 3.12–3.14.")'
pkg-config --exists x11 xrandr xinerama xcursor xi glu || {
    echo "Missing X11 or OpenGL development libraries. See README.md." >&2
    exit 1
}

if [[ ! -x "$venv/bin/python" ]]; then
    "$python" -m venv "$venv"
fi
venv=$(cd "$venv" && pwd)
python="$venv/bin/python"
"$python" -m pip install 'drake==1.58.0' 'numpy==2.5.5' 'scipy==1.18.1'

source_dir="$venv/splishsplash"
if [[ ! -d "$source_dir/.git" ]]; then
    git init "$source_dir"
fi
if [[ "$(git -C "$source_dir" rev-parse --verify HEAD 2>/dev/null || true)" != "$revision" ]]; then
    git -C "$source_dir" fetch --depth=1 "$upstream" "$revision"
    git -C "$source_dir" checkout --detach FETCH_HEAD
fi
for patch in "$root"/patches/splishsplash-{dfsph-timestep,viscosity-initial-guess,particle-retirement}.patch; do
    if ! git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
        git -C "$source_dir" apply "$patch"
    fi
done
build_dir="$source_dir/build"
cmake -S "$source_dir" -B "$build_dir" \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.10 \
    -DUSE_PYTHON_BINDINGS=ON -DPython_EXECUTABLE="$python" \
    -DNeighborhoodSearch=CompactNSearch -DUSE_DOUBLE_PRECISION=OFF
cmake --build "$build_dir" --target pysplishsplash --parallel "$jobs"

"$python" - "$build_dir/lib" <<'PY'
from pathlib import Path
import sys
import sysconfig

path = Path(sysconfig.get_path("platlib")) / "pouring_sph.pth"
path.write_text(sys.argv[1] + "\n")
PY
"$python" -c 'import pydrake, pysplishsplash; print("SPH binding:", pysplishsplash.__file__)'
printf 'Demo Python: %s\n' "$python"
