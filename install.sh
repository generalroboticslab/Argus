#!/usr/bin/env bash
# One-click install for Argus.
#
# Creates the conda env, installs requirements + the Isaac Gym fork +
# IsaacGymEnvs, then patches IsaacGymEnvs so it imports without urdfpy
# (see envs/patch_isaacgymenvs.py). Idempotent: safe to re-run.
#
# Usage:
#   bash install.sh
#
# Env vars (override defaults):
#   ENV_NAME       conda env name              (default: argus)
#   PY_VERSION     python version              (default: 3.8 -- fork is 3.8-only)
#   ISAACGYM_URL   Isaac Gym fork git URL      (default: boxiXia/isaacgym)
#   IGE_URL        IsaacGymEnvs git URL        (default: isaac-sim/IsaacGymEnvs)
#   ISAACGYM_DIR   local Isaac Gym path        (default: ../isaacgym)
#   IGE_DIR        local IsaacGymEnvs path     (default: ../IsaacGymEnvs)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-argus}"
PY_VERSION="${PY_VERSION:-3.8}"
ISAACGYM_URL="${ISAACGYM_URL:-https://github.com/boxiXia/isaacgym.git}"
IGE_URL="${IGE_URL:-https://github.com/isaac-sim/IsaacGymEnvs.git}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$REPO_ROOT/../isaacgym}"
IGE_DIR="${IGE_DIR:-$REPO_ROOT/../IsaacGymEnvs}"

# Pick an available conda-like tool.
if command -v conda >/dev/null 2>&1; then CONDA=conda
elif command -v micromamba >/dev/null 2>&1; then CONDA=micromamba
elif command -v mamba >/dev/null 2>&1; then CONDA=mamba
else echo "ERROR: need conda, micromamba, or mamba on PATH" >&2; exit 2; fi

run() { "$CONDA" run -n "$ENV_NAME" "$@"; }

echo ">> [1/5] Creating env '$ENV_NAME' (python=$PY_VERSION)"
"$CONDA" create -y -n "$ENV_NAME" "python=$PY_VERSION"

echo ">> [2/5] Installing requirements.txt"
run python -m pip install -r "$REPO_ROOT/requirements.txt" --no-cache-dir

echo ">> [3/5] Installing Isaac Gym fork"
[[ -d "$ISAACGYM_DIR" ]] || git clone "$ISAACGYM_URL" "$ISAACGYM_DIR"
run python -m pip install -e "$ISAACGYM_DIR/python"

echo ">> [4/5] Installing IsaacGymEnvs"
[[ -d "$IGE_DIR" ]] || git clone "$IGE_URL" "$IGE_DIR"
run python -m pip install -e "$IGE_DIR" --no-deps

echo ">> [5/5] Patching IsaacGymEnvs (lazy urdfpy import)"
run python "$REPO_ROOT/envs/patch_isaacgymenvs.py"

echo ""
echo ">> Done. Activate with: $CONDA activate $ENV_NAME"
echo ">> Then: cd envs && export LD_LIBRARY_PATH=\${CONDA_PREFIX}/lib"
