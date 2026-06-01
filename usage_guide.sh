#!/usr/bin/env bash
# =============================================================================
# Argus Usage Guide
# =============================================================================
# This file contains all commands for installing, training, evaluating, and
# running the Argus robot policies described in:
#
#   Liu, J.*, Xia, B.*, & Chen, B. (2025). Extreme dynamic symmetry enables
#   omnidirectional and multifunctional robots. Science Robotics, 11.
#   https://doi.org/10.1126/scirobotics.aec1725
#
# HOW TO USE THIS FILE
# --------------------
# This is not meant to be executed all at once. Copy individual commands or
# sections as needed. Working directory is the repository root unless a
# section header specifies otherwise.
#
# SOFTWARE REQUIREMENTS
# ---------------------
#   - Linux (Ubuntu 20.04 or 22.04)
#   - Python 3.8 (required by Isaac Gym)
#   - CUDA 12.x with GPU >= 16 GB VRAM (for training; inference needs no GPU)
#   - Conda (Miniconda or Anaconda)
#   - Isaac Gym custom fork: https://github.com/boxiXia/isaacgym
#   - IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs
# =============================================================================


# =============================================================================
# SECTION 1: INSTALLATION
# =============================================================================

# --- Option A: Automated (recommended) ---
# Run from the repository root. Creates an 'argus' conda environment,
# installs all dependencies, clones Isaac Gym and IsaacGymEnvs, and applies
# compatibility patches. Safe to re-run (idempotent).

bash install.sh

# Override defaults with environment variables if needed:
#   ENV_NAME=myenv PY_VERSION=3.8 bash install.sh


# --- Option B: Manual step-by-step ---

# Step 1: Create and activate conda environment
conda create --name argus python=3.8
conda activate argus
pip install -r requirements.txt --no-cache-dir

# Step 2: Install Isaac Gym (custom fork required)
# Clone alongside the Argus repo (i.e., in the parent directory)
cd ..
git clone https://github.com/boxiXia/isaacgym.git
cd isaacgym/python
pip install -e .
cd ../../Argus

# Step 3: Install IsaacGymEnvs
cd ..
git clone https://github.com/isaac-sim/IsaacGymEnvs.git
cd IsaacGymEnvs
pip install -e . --no-deps
cd ../Argus

# Step 4: Apply compatibility patch (removes dependency on urdfpy)
cd envs
python patch_isaacgymenvs.py
cd ..


# --- Option C: Verify installation ---
# Run dependency-only checks (no GPU required)
bash test/test_fresh_env.sh

# Run full check including headless Isaac Gym training smoke test (GPU required)
FULL=1 bash test/test_fresh_env.sh


# =============================================================================
# SECTION 2: RUNNING PRETRAINED CHECKPOINTS (Quick Start)
# =============================================================================
# All play commands run from the envs/ directory.
# Append -k to any command to enable keyboard control (see Section 6).

cd envs
conda activate argus
export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib

# Flat-ground rolling (20-leg, primary demo)
bash run.sh argus_base -p

# Discrete terrain traversal (stepping stones)
bash run.sh argus_terrain -p

# Disabled-leg robustness (20-leg, random legs disabled at runtime)
bash run.sh argus_disable_leg_dof_20_const_vel -p

# Payload carrying (20-leg)
bash run.sh argus_carry_object_dof_20_const_vel -p

# Push rejection (20-leg, random impulse forces applied)
bash run.sh argus_push -p

# Object pushing with point-cloud perception (imitation learning policy)
bash run.sh argus_object_pushing_IL -p

# Object tracking with point-cloud perception (imitation learning policy)
bash run.sh argus_object_tracking_IL -p


# =============================================================================
# SECTION 3: TRAINING FROM SCRATCH
# =============================================================================
# All training commands run from the envs/ directory.
# Training logs and checkpoints are saved under envs/runs/.
# WandB logging is enabled by default; set wandb_entity in envs/cfg/config.yaml.

cd envs

# --- Flat-ground locomotion ---
bash run.sh argus_base

# --- Discrete terrain traversal ---
# 20-leg
bash run.sh argus_terrain_dof_20_const_vel
# 12-leg
bash run.sh argus_terrain_dof_12_const_vel
# 32-leg
bash run.sh argus_terrain_dof_32_const_vel

# --- Disabled-leg robustness ---
# 20-leg
bash run.sh argus_disable_leg_dof_20_const_vel
# 12-leg
bash run.sh argus_disable_leg_dof_12_const_vel
# 32-leg
bash run.sh argus_disable_leg_dof_32_const_vel

# --- Payload carrying ---
# 20-leg
bash run.sh argus_carry_object_dof_20_const_vel
# 12-leg
bash run.sh argus_carry_object_dof_12_const_vel
# 32-leg
bash run.sh argus_carry_object_dof_32_const_vel

# --- Push rejection ---
bash run.sh argus_push

# --- Object pushing (two-stage pipeline) ---
# Stage 1: Train base policy with privileged object state (no perception)
bash run.sh argus_object_pushing_base
# Stage 2: Collect offline point-cloud data and train encoder via imitation
bash run.sh argus_object_pushing_IL
python train_point_could_encoder.py --task object_pushing

# --- Object tracking (two-stage pipeline) ---
# Stage 1: Train base policy with privileged object state
bash run.sh argus_object_tracking_base
# Stage 2: Collect offline point-cloud data and train encoder via imitation
bash run.sh argus_object_tracking_IL
python train_point_could_encoder.py --task object_tracking

# --- Object tracking with 32-leg variant and variable perception units ---
# Stage 1
bash run.sh argus_object_tracking_base_32legs
# Stage 2: data collection (select cube size)
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5      # 0.5 m cube
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25     # 0.25 m cube
# Stage 3: encoder training (select number of perception units: 12, 20, or 32)
python train_point_could_encoder.py --task object_tracking --num_perception_units 32
python train_point_could_encoder.py --task object_tracking --num_perception_units 20
python train_point_could_encoder.py --task object_tracking --num_perception_units 12


# =============================================================================
# SECTION 4: EVALUATION
# =============================================================================
# Evaluation reuses the play (-p) infrastructure. JSON results are written to
# envs/eval/. Specify checkpoint paths in exp.sh under PLAY_ARGS before running.

cd envs

# Flat-ground locomotion
bash run.sh argus_base_eval -p

# Discrete terrain
bash run.sh argus_terrain_dof_20_const_vel_eval -p
bash run.sh argus_terrain_dof_32_const_vel_eval -p
bash run.sh argus_terrain_dof_12_const_vel_eval -p

# Disabled-leg robustness
bash run.sh argus_disable_leg_dof_20_const_vel_eval -p
bash run.sh argus_disable_leg_dof_32_const_vel_eval -p
bash run.sh argus_disable_leg_dof_12_const_vel_eval -p

# Payload carrying
bash run.sh argus_carry_object_dof_20_const_vel_eval -p
bash run.sh argus_carry_object_dof_32_const_vel_eval -p
bash run.sh argus_carry_object_dof_12_const_vel_eval -p

# Push rejection
bash run.sh argus_push_eval -p

# Object pushing
bash run.sh argus_object_pushing_base_eval -p         # base (no perception)
bash run.sh argus_object_pushing_eval -p              # with point-cloud encoder

# Object tracking (20-leg)
bash run.sh argus_object_tracking_base_eval -p        # base (no perception)
bash run.sh argus_object_tracking_eval -p             # with point-cloud encoder

# Object tracking (32-leg, various perception unit counts)
bash run.sh argus_object_tracking_base_32legs_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_12percetion_cube0_25_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_20percetion_cube0_25_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_5_eval -p
bash run.sh argus_object_tracking_IL_offline_32legs_32percetion_cube0_25_eval -p


# =============================================================================
# SECTION 5: ONNX EXPORT AND INFERENCE
# =============================================================================

# --- Exporting a .pt checkpoint to ONNX format ---
# Does NOT require Isaac Gym. Run from envs/.
cd envs

# Export the 20-leg flat-plane actor (default)
python export_onnx.py

# Export 12-leg or 32-leg actor
python export_onnx.py --dof 12
python export_onnx.py --dof 32

# Export from a custom checkpoint
python export_onnx.py --dof 20 --checkpoint ../assets/checkpoint/my_checkpoint.pt

# Output is written to assets/checkpoint/argus_actor_dof<N>.onnx by default.
# Override with --output path/to/output.onnx


# --- Running ONNX inference (no Isaac Gym required) ---
# The following is a minimal Python example. Run with:
#   python - <<'EOF'  (or save as a .py file and run normally)
#
# Python 3.8+, requires: pip install onnxruntime numpy
#
# import numpy as np
# import onnxruntime as ort
#
# # Load the 20-leg actor
# session = ort.InferenceSession("assets/checkpoint/argus_actor_dof20.onnx")
#
# # Build a dummy observation (batch_size=1, obs_dim=222)
# # In real use, fill this with actual sensor readings in the order described
# # in README_DRYAD.md Section 1 (Input Variable: obs).
# obs = np.zeros((1, 222), dtype=np.float32)
#
# # Run inference
# action = session.run(["action"], {"obs": obs})[0]
# # action shape: (1, 20) — target_rad = action * 0.25 + default_joint_pos
# joint_targets_rad = action * 0.25  # action_scale=0.25; add default_joint_pos for full target
# print("Joint position targets (rad):", joint_targets_rad)
#
# EOF


# =============================================================================
# SECTION 6: KEYBOARD CONTROL
# =============================================================================
# Add -k to any play command to enable keyboard velocity control.
# Example:
#   bash run.sh argus_base -p -k
#
# Key bindings (active when -k flag is used):
#   i  — increase forward speed   (+0.05 m/s in x)
#   k  — increase backward speed  (-0.05 m/s in x)
#   j  — strafe left               (+0.05 m/s in y)
#   l  — strafe right              (-0.05 m/s in y)


# =============================================================================
# SECTION 7: BLENDER VISUALIZATION
# =============================================================================
# Download the Blender scene file from:
#   https://drive.google.com/drive/folders/1OIzNFc4BLMO4p8IIS8wud3FVtw9b3A39
#
# Tested with Blender 3.x. After opening the .blend file:
#   1. Enable Viewport Shading (top-right of the 3D viewport)
#   2. Press the Play button in the timeline to run the animation
# See blender_rendering/README.md for screenshots of each step.
