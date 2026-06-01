#!/usr/bin/env bash
# Create a fresh conda/micromamba env, install requirements.txt, and run the
# dependency regression checks. This proves a clean install works on the
# modern pinned versions (no numpy-1.19 downgrade hacks).
#
# Usage:
#   bash test/test_fresh_env.sh [ENV_NAME]
#
# Env vars:
#   KEEP_ENV=1      keep the env after the run (default: remove it)
#   PY_VERSION      python version (default: 3.8 -- the Isaac Gym fork is 3.8-only)
#   FULL=1          also install the Isaac Gym fork + IsaacGymEnvs and run a
#                   bounded headless training smoke (needs an NVIDIA GPU)
#   ISAACGYM_DIR    path to the isaacgym fork    (default: ../isaacgym)
#   IGE_DIR         path to IsaacGymEnvs         (default: ../IsaacGymEnvs)
#
# Default tier is GPU-free: it installs requirements.txt and runs the
# dependency regression checks. FULL=1 adds the GPU end-to-end smoke.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-argus_freshtest}"
PY_VERSION="${PY_VERSION:-3.8}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$REPO_ROOT/../isaacgym}"
IGE_DIR="${IGE_DIR:-$REPO_ROOT/../IsaacGymEnvs}"

# Pick an available conda-like tool.
if command -v micromamba >/dev/null 2>&1; then
    CONDA=micromamba
elif command -v mamba >/dev/null 2>&1; then
    CONDA=mamba
elif command -v conda >/dev/null 2>&1; then
    CONDA=conda
else
    echo "ERROR: need micromamba, mamba, or conda on PATH" >&2
    exit 2
fi
echo ">> Using $CONDA, env=$ENV_NAME, python=$PY_VERSION"

cleanup() {
    if [[ "${KEEP_ENV:-0}" != "1" ]]; then
        echo ">> Removing env $ENV_NAME"
        "$CONDA" env remove -y -n "$ENV_NAME" >/dev/null 2>&1 || true
    else
        echo ">> Keeping env $ENV_NAME (KEEP_ENV=1)"
    fi
}
trap cleanup EXIT

echo ">> Creating fresh env"
"$CONDA" create -y -n "$ENV_NAME" "python=$PY_VERSION"

# Run a command inside the new env.
run() { "$CONDA" run -n "$ENV_NAME" "$@"; }

echo ">> Upgrading pip"
run python -m pip install --upgrade pip

echo ">> Installing requirements.txt"
run python -m pip install -r "$REPO_ROOT/requirements.txt" --no-cache-dir

echo ">> Running regression checks"
run python "$REPO_ROOT/test/regression_check.py"

if [[ "${FULL:-0}" == "1" ]]; then
    echo ">> FULL tier: installing Isaac Gym fork + IsaacGymEnvs"
    [[ -d "$ISAACGYM_DIR/python" ]] || { echo "ERROR: no isaacgym fork at $ISAACGYM_DIR (set ISAACGYM_DIR)"; exit 3; }
    [[ -d "$IGE_DIR" ]] || { echo "ERROR: no IsaacGymEnvs at $IGE_DIR (set IGE_DIR)"; exit 3; }

    run python -m pip install -e "$ISAACGYM_DIR/python"
    # --no-deps: keep the versions pinned in requirements.txt
    run python -m pip install -e "$IGE_DIR" --no-deps

    echo ">> FULL tier: patching IsaacGymEnvs (lazy urdfpy import)"
    run python "$REPO_ROOT/envs/patch_isaacgymenvs.py"

    echo ">> FULL tier: bounded headless training smoke (2 iterations)"
    # 256 timesteps / (16 envs * 8 steps) = 2 iterations, then exits cleanly.
    # Isaac Gym needs the env's libs on LD_LIBRARY_PATH.
    ENV_PREFIX="$("$CONDA" run -n "$ENV_NAME" python -c 'import sys;print(sys.prefix)')"
    ( cd "$REPO_ROOT/envs" && \
      LD_LIBRARY_PATH="$ENV_PREFIX/lib" "$CONDA" run -n "$ENV_NAME" \
        python ppo_isaacgym.py \
          --task_name=argus --agent_name=baseline --train_mode=train \
          --headless=True --track=False \
          --num_envs=16 --num_steps=8 --total_timesteps=256 \
          ++task.env.defaultJointPositions=0 \
          ++task.env.initialJointPositions=-0.105 \
          ++task.env.desiredJointPositions=-0.105 )
    echo ">> FULL tier: training smoke completed"
fi

echo ">> PASS: fresh environment installs and dependency checks succeed"
