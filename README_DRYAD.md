# Extreme Dynamic Symmetry Enables Omnidirectional and Multifunctional Robots

**Access this dataset on Dryad:** [https://doi.org/10.5061/dryad.3j9kd520k](https://doi.org/10.5061/dryad.3j9kd520k)

**Related publication:** Liu, J.\*, Xia, B.\*, & Chen, B. (2025). Extreme dynamic symmetry enables omnidirectional and multifunctional robots. *Science Robotics*, 11. https://doi.org/10.1126/scirobotics.aec1725

**Authors:** Jiaxun Liu\*, Boxi Xia\*, Boyuan Chen (\*equal contribution) — Duke University

---

## Description of the Data and File Structure

This dataset contains pretrained neural network policy checkpoints and robot description files for the Argus family of spherical robots, as described in the associated publication. Argus robots use radially oriented linear actuators (legs) and are designed to achieve high *dynamic isotropy* — a measure of how uniformly a robot can accelerate its center of mass in all directions. Three morphological variants are included: a 12-leg, 20-leg, and 32-leg design.

The policies were trained using Proximal Policy Optimization (PPO) in the Isaac Gym physics simulator (GPU-accelerated). Each policy takes a history of proprioceptive observations as input and outputs target positions for the robot's joints. Policies were trained and evaluated across six task categories: flat-ground locomotion, discrete terrain traversal, disabled-leg robustness, payload carrying, push rejection, and object interaction (pushing and tracking using point-cloud perception).

All neural networks are multilayer perceptrons (MLPs). Actor networks have two hidden layers (256 units each) with ELU activations. A RunningMeanStd normalizer is applied to observations before the MLP. Critic networks (used only during training, not included in the ONNX exports) use a separate normalizer and the same two-hidden-layer MLP (256, 256 units) with a scalar output.

---

## Repository Contents

```
Argus/
├── README.md                         GitHub-facing overview and quick-start guide
├── README_DRYAD.md                   This file — Dryad data dictionary
├── usage_guide.sh                    All installation and usage commands (see Code/Software section)
├── requirements.txt                  Python package dependencies (pip format)
├── install.sh                        Automated one-command installation script (bash)
├── LICENSE-CC-BY-NC-ND-4.0.md        License file
│
├── assets/
│   ├── checkpoint/                   Pretrained model weights (described in detail below)
│   │   ├── argus_actor_dof12.onnx    ONNX actor policy — 12-leg variant
│   │   ├── argus_actor_dof20.onnx    ONNX actor policy — 20-leg variant (physical robot)
│   │   ├── argus_actor_dof32.onnx    ONNX actor policy — 32-leg variant
│   │   ├── flat_plane.pt             PyTorch checkpoint — flat-ground locomotion (20-leg)
│   │   ├── argus_disable_leg/        PyTorch checkpoints — disabled-leg robustness (all variants)
│   │   ├── argus_carry_object/       PyTorch checkpoints — payload carrying (all variants)
│   │   ├── argus_push/               PyTorch checkpoint — push rejection (20-leg)
│   │   ├── discrete_terrain/         PyTorch checkpoints — discrete terrain traversal (all variants)
│   │   ├── argus_object_pushing/     PyTorch checkpoints — object pushing base policy + encoder
│   │   └── argus_object_tracking/    PyTorch checkpoints — object tracking base policies + encoders
│   └── urdf/argus/                   Robot description files (XML, URDF format)
│
├── envs/                             Training and simulation environment source code
│   ├── train.py                      Training entry point
│   ├── ppo_isaacgym.py               PPO algorithm implementation
│   ├── export_onnx.py                Script to export .pt checkpoints to .onnx format
│   ├── train_point_could_encoder.py  Point cloud encoder training (object interaction tasks)
│   ├── run.sh                        Experiment launcher (training and evaluation)
│   ├── exp.sh                        Per-experiment configuration (checkpoint paths, hyperparameters)
│   ├── patch_isaacgymenvs.py         Compatibility patch for IsaacGymEnvs
│   ├── torch_utils.py                Neural network definitions (Agent, RunningMeanStd, etc.)
│   ├── utils.py                      Miscellaneous utility functions
│   ├── cfg/                          Hydra configuration files (YAML)
│   ├── tasks/                        Isaac Gym task environment definitions
│   │   └── legged_terrain.py         Primary environment implementing all Argus tasks
│   ├── common/                       Shared simulation utilities
│   │   ├── terrain.py                Procedural terrain generation
│   │   ├── tof_sensor.py             Time-of-flight (point cloud) sensor simulation
│   │   ├── buffer.py                 Replay buffer for imitation learning
│   │   ├── torch_runner.py           Training loop and logging utilities
│   │   └── publisher.py              Data publishing utilities
│   └── utils/                        Offline helper scripts
│       ├── argus_generate_varient.py  Batch URDF variant generation
│       ├── gen_argus_12legs_urdf.py   12-leg URDF generator
│       ├── gen_argus_20legs_urdf.py   20-leg URDF generator (icosahedron layout)
│       ├── gen_argus_20legs_urdf_dodecahedron.py  20-leg URDF (dodecahedron layout)
│       ├── gen_argus_30legs_urdf.py   30-leg URDF generator
│       └── ray_casting_cone_on_foot.py  Foot ray-casting utility for perception
│
├── test/
│   ├── test_fresh_env.sh             End-to-end installation verification script
│   └── regression_check.py          Regression tests for policy inference
│
├── blender_rendering/
│   └── README.md                     Blender visualization instructions
│
└── visualization/                    Demo images and videos (GIF/MP4/PNG)
```

---

## File Descriptions and Data Dictionary

### 1. ONNX Actor Policy Files (`assets/checkpoint/argus_actor_dof*.onnx`)

These three files are the primary deliverable of this dataset. Each is a self-contained, deployment-ready neural network locomotion policy exported in the Open Neural Network Exchange (ONNX) format (opset version 17). The 20-leg file is the flat-ground policy; the 12-leg and 32-leg files are exported from disabled-leg robustness checkpoints (which also generalize to flat-ground locomotion). ONNX is an open standard readable by many inference engines (e.g., ONNX Runtime, TensorFlow, PyTorch) without requiring Isaac Gym or any robotics simulation software.

Each file encodes an **actor-only** network consisting of:
1. An observation normalizer (RunningMeanStd): applies learned per-dimension mean subtraction and variance scaling to raw observations, then clips to the range [−5, 5].
2. An MLP policy: two hidden layers (256 units each) with ELU activations, producing a deterministic action (the mean of the policy's action distribution).

| File | Variant | Source checkpoint | Input Dimension | Output Dimension | File Size |
|---|---|---|---|---|---|
| `argus_actor_dof12.onnx` | 12-leg | `argus_disable_leg/argus_disable_leg_dof_12_const_vel.pt` | 150 | 12 | ~423 KB |
| `argus_actor_dof20.onnx` | 20-leg | `flat_plane.pt` | 222 | 20 | ~504 KB |
| `argus_actor_dof32.onnx` | 32-leg | `argus_disable_leg/argus_disable_leg_dof_32_const_vel.pt` | 330 | 32 | ~624 KB |

#### Input Variable: `obs` (float32 tensor, shape: [batch\_size, obs\_dim])

The input is a stacked history of 3 consecutive observation frames (the most recent frame and 2 prior frames), concatenated in chronological order (oldest first). Each frame has dimension 14 + 3N, where N is the number of legs (degrees of freedom, DOF). The overall input dimension is therefore 3 × (14 + 3N).

**Per-frame observation layout** (N = number of legs/DOF):

All values entering the observation vector are pre-multiplied by a scale factor defined in `envs/cfg/task/argus.yaml` under `learn`. The ONNX model expects these **pre-scaled** values as input; raw sensor readings must be converted using the scale factors in the table below.

| Index range | Variable name | Size | Raw physical unit | Scale factor | Description |
|---|---|---|---|---|---|
| 0 – 2 | `worldSpaceAngularVelocity` | 3 | rad/s | 0.25 | Angular velocity of the robot body in the world (global) frame, multiplied by 0.25. Components are [ω\_x, ω\_y, ω\_z] (roll rate, pitch rate, yaw rate). To convert: obs\_value = raw\_rad\_per\_s × 0.25. |
| 3 – 4 | `commands_xy` | 2 | m/s | 2.0 | Target linear velocity command in the horizontal (x–y) plane, multiplied by 2.0. [v\_x, v\_y] where x is forward and y is lateral (left). To convert: obs\_value = raw\_m\_per\_s × 2.0. |
| 5 – (4+N) | `dofPosition` | N | rad | 1.0 | Current joint position of each leg actuator (scale factor = 1.0, so values are in radians). Index order follows the URDF joint ordering. |
| (5+N) – (4+2N) | `dofVelocity` | N | rad/s | 0.05 | Current joint velocity of each leg actuator, multiplied by 0.05. To convert: obs\_value = raw\_rad\_per\_s × 0.05. |
| (5+2N) – (4+3N) | `actions` | N | unitless | 1.0 | The action vector output by the policy at the previous time step. Values are in the normalized action space (approximately [−1, 1]). Used to give the network awareness of its own recent command history. |
| (5+3N) – (13+3N) | `base_rotation_matrix_filtered` | 9 | unitless | 1.0 | Orientation of the robot body as a flattened 3×3 rotation matrix (row-major), low-pass filtered to reduce IMU noise. Entries are dimensionless direction cosines. Column vectors are the body x-, y-, and z-axes in world frame. |

**Single frame size by variant:**
- 12-leg: 14 + 3(12) = 50 values per frame; 3 frames → input dimension 150
- 20-leg: 14 + 3(20) = 74 values per frame; 3 frames → input dimension 222
- 32-leg: 14 + 3(32) = 110 values per frame; 3 frames → input dimension 330

#### Output Variable: `action` (float32 tensor, shape: [batch\_size, N\_dof])

The output is a vector of N target joint positions (one per leg), in normalized units. To convert to a physical joint position target: target\_rad = 0.25 × action + default\_joint\_pos, where 0.25 rad is the action scale (`actionScale` in `argus.yaml`) and `default_joint_pos` is the resting joint angle. The network is called at a fixed control frequency of 50 Hz (sim dt = 0.005 s, decimation = 4).

| Index | Variable | Units | Description |
|---|---|---|---|
| 0 – (N−1) | target joint position | unitless (normalized) | Desired position for each joint. Multiply by action scale (0.25 rad) and add the default joint position to get the physical target in radians: target\_rad = 0.25 × action + default\_joint\_pos. Index order matches `dofPosition` in the input. |

---

### 2. PyTorch Checkpoint Files (`assets/checkpoint/**/*.pt`)

These files are full training checkpoints saved in PyTorch's serialized format (`torch.save`). They include the complete model state (actor + critic networks, observation normalizers, optimizer state) and are used to resume training or run evaluation in Isaac Gym. Each file is a Python dictionary with the following top-level key:

| Key | Type | Description |
|---|---|---|
| `agent` | `dict` | State dictionary for the `Agent` module (see `envs/torch_utils.py`). Contains all network weights and normalizer buffers. |

The `agent` state dictionary contains these named sub-modules:

| Sub-module key prefix | Description |
|---|---|
| `normalize_obs.*` | Observation normalizer for the actor: buffers `running_mean`, `running_var`, `count` (float64, shape [obs\_dim]) |
| `actor_mean.*` | Actor MLP weights: linear layers `0`, `2`, `4` with corresponding `.weight` (float32) and `.bias` (float32) tensors |
| `normalize_critic_obs.*` | Observation normalizer for the critic (float64 buffers, shape [critic\_obs\_dim]) |
| `critic.*` | Critic MLP weights: linear layers `0`, `2`, `4` (256→256→1); scalar value output |
| `actor_logstd` | Log standard deviation of the policy's action distribution (float32, shape [1, N\_dof]) |

**Critic observation layout** (used only during training; shape per frame: 20 + 4N):
The critic receives a privileged state vector not available on the physical robot, including linear velocity, contact forces, and unfiltered orientation — information available only from the simulator. All values are pre-scaled using the same factors as the actor observations (see scale factor table above).

| Variable | Size | Raw physical unit | Scale factor | Description |
|---|---|---|---|---|
| `linearVelocity` | 3 | m/s | 2.0 | Robot body linear velocity in world frame (obs\_value = raw\_m\_per\_s × 2.0) |
| `worldSpaceAngularVelocity` | 3 | rad/s | 0.25 | Same as actor |
| `projectedGravity` | 3 | unitless | 1.0 | Unit gravity vector projected into body frame; inherently unitless (components ≈ [−1, 1]) |
| `commands_xy` | 2 | m/s | 2.0 | Same as actor |
| `dofPosition` | N | rad | 1.0 | Same as actor |
| `dofVelocity` | N | rad/s | 0.05 | Same as actor |
| `actions` | N | unitless | 1.0 | Same as actor |
| `contact` | N | binary (0 or 1) | 1.0 | Foot contact state: 1 if foot is in contact with ground, 0 otherwise |
| `base_rotation_matrix` | 9 | unitless | 1.0 | Unfiltered rotation matrix (same layout as actor's filtered version) |

**Checkpoint files by task:**

| File path | Variant | Task | Notes |
|---|---|---|---|
| `flat_plane.pt` | 20-leg | Flat-ground locomotion | Primary flat-ground policy; also base for ONNX export |
| `argus_disable_leg/argus_disable_leg_dof_12_const_vel.pt` | 12-leg | Disabled-leg robustness | Trained to locomote with randomly disabled legs at constant velocity command |
| `argus_disable_leg/argus_disable_leg_dof_20_const_vel.pt` | 20-leg | Disabled-leg robustness | |
| `argus_disable_leg/argus_disable_leg_dof_32_const_vel.pt` | 32-leg | Disabled-leg robustness | |
| `argus_carry_object/argus_carry_object_dof_12_const_vel.pt` | 12-leg | Payload carrying | Robot locomotes while supporting a payload on its shell |
| `argus_carry_object/argus_carry_object_dof_20_const_vel.pt` | 20-leg | Payload carrying | |
| `argus_carry_object/argus_carry_object_dof_32_const_vel.pt` | 32-leg | Payload carrying | |
| `argus_push/push.pt` | 20-leg | Push rejection | Policy trained to resist random external impulse forces |
| `discrete_terrain/argus_terrain_dof_20_const_vel.pt` | 20-leg | Discrete terrain | Trained on stepping-stone terrain at constant velocity command |
| `discrete_terrain/argus_terrain_dof_20.pt` | 20-leg | Discrete terrain | Stepping-stone terrain; variable velocity command variant |
| `discrete_terrain/argus_terrain_dof_32_const_vel.pt` | 32-leg | Discrete terrain | |
| `discrete_terrain/argus_terrain_dof_12_const_vel.pt` | 12-leg | Discrete terrain | |
| `argus_object_pushing/argus_object_pushing_base.pt` | 20-leg | Object pushing (Stage 1) | Base policy trained with privileged object state (no perception) |
| `argus_object_pushing/object_pushing_encoder.pt` | 20-leg | Object pushing (Stage 2) | Point-cloud encoder trained via imitation of Stage 1 policy |
| `argus_object_tracking/argus_object_tracking_base.pt` | 20-leg | Object tracking (Stage 1) | Base policy trained with privileged object state |
| `argus_object_tracking/argus_object_tracking_base_32legs.pt` | 32-leg | Object tracking (Stage 1) | 32-leg variant base policy |
| `argus_object_tracking/argus_object_tracking_base_0.5-0.8_withdynamicsetup_nostackframe_32legs_newest.pt` | 32-leg | Object tracking (Stage 1) | 32-leg variant; dynamic object setup, no observation frame stacking, object distance range 0.5–0.8 m |
| `argus_object_tracking/object_tracking_encoder.pt` | 20-leg | Object tracking (Stage 2) | Point-cloud encoder (20-leg, 20 perception units) |
| `argus_object_tracking/immitation_12perception_05cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 12 perception units, 0.5 m cube object |
| `argus_object_tracking/immitation_12perception_025cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 12 perception units, 0.25 m cube object |
| `argus_object_tracking/immitation_20perception_05cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 20 perception units, 0.5 m cube object |
| `argus_object_tracking/immitation_20perception_025cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 20 perception units, 0.25 m cube object |
| `argus_object_tracking/immitation_32perception_05cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 32 perception units, 0.5 m cube object |
| `argus_object_tracking/immitation_32perception_025cube_encoder/imitation_model_best_val.pt` | 32-leg | Object tracking encoder | 32 perception units, 0.25 m cube object |

---

### 3. Robot Description Files (`assets/urdf/argus/`)

These files describe the physical properties of the Argus robot in the Unified Robot Description Format (URDF), an XML-based standard used by robotics simulators.

| File | Description |
|---|---|
| `argus_dof12.urdf` | 12-leg Argus; full joint limits, contact geometry, and inertial parameters |
| `argus_dof12_minimum.urdf` | 12-leg Argus; simplified version with minimal mesh references |
| `argus_dof20.urdf` | 20-leg Argus (physical robot configuration) |
| `argus_dof20_minimum.urdf` | 20-leg Argus; simplified |
| `argus_dof20_minimum_load.urdf` | 20-leg Argus; includes payload body for carry task |
| `argus_dof20_load.urdf` | 20-leg Argus with payload body (full geometry) |
| `argus_dof20_minimum_robstride03.urdf` | 20-leg Argus parameterized for RobStride-03 actuator model |
| `argus_dof20_vel1_5.urdf` | 20-leg Argus with increased velocity limit (1.5×) |
| `argus_dof32.urdf` | 32-leg Argus |
| `argus_dof32_minimum.urdf` | 32-leg Argus; simplified |
| `sim_rand_joint_00_argus_dof20_minimum.urdf` through `sim_rand_joint_NN_...` | Domain randomization variants of the 20-leg robot with perturbed joint origins, used for sim-to-real transfer training |

**Key URDF parameters relevant to policy operation:**

| URDF Parameter | Units | Description |
|---|---|---|
| `joint/limit/@lower` | rad | Minimum joint position allowed |
| `joint/limit/@upper` | rad | Maximum joint position allowed |
| `joint/limit/@effort` | N·m | Maximum joint torque |
| `joint/limit/@velocity` | rad/s | Maximum joint velocity |
| `link/inertial/mass` | kg | Link mass |
| `link/inertial/inertia` | kg·m² | Link inertia tensor (ixx, ixy, ixz, iyy, iyz, izz) |

**Mesh files:**

`assets/urdf/argus/meshes/` (referenced by full-geometry URDFs):

| File | Description |
|---|---|
| `base_link_inner.stl` | Inner core geometry |
| `main_simple.stl` | Leg link mesh |

`assets/urdf/argus/simple_meshes/` (referenced by `_minimum` URDFs):

| File | Description |
|---|---|
| `base_link.stl` | Outer shell geometry |
| `base_link_inner.stl` | Inner core geometry |
| `main_simple.stl` | Simplified leg link mesh |

---

### 4. Source Code Files (`envs/`)

All source code is written in Python (tested on Python 3.8). The files define the simulation environment, training loop, and evaluation logic. No standalone tabular data files (CSV, HDF5, etc.) are produced by training; outputs are PyTorch checkpoint files (`.pt`) and evaluation JSON files written to `envs/eval/`.

| File | Description |
|---|---|
| `envs/torch_utils.py` | Defines the `Agent` class (actor-critic network) and `RunningMeanStd` normalizer. This file is the canonical definition of the network architecture stored in all `.pt` checkpoint files. |
| `envs/train.py` | Main training entry point. Parses Hydra config and launches the PPO training loop. |
| `envs/ppo_isaacgym.py` | Proximal Policy Optimization (PPO) algorithm, interfacing with Isaac Gym. |
| `envs/export_onnx.py` | Converts a `.pt` actor checkpoint to `.onnx` format. Does not require Isaac Gym. Accepts `--dof` (12, 20, or 32) and `--checkpoint` arguments. |
| `envs/train_point_could_encoder.py` | Trains the point-cloud encoder via imitation learning (Stage 2 of perception tasks). |
| `envs/tasks/legged_terrain.py` | Defines all Argus task environments (reward functions, observation computation, terrain setup, etc.). |
| `envs/common/tof_sensor.py` | Simulates time-of-flight (ToF) distance sensors mounted on the robot's legs for point-cloud perception. |
| `envs/common/terrain.py` | Generates procedural terrain meshes (flat, stepping-stone, etc.) for simulation. |
| `envs/run.sh` | Bash launcher: selects the correct config and runs training or evaluation for a named experiment. |
| `envs/exp.sh` | Defines all experiment configurations: hyperparameters, checkpoint paths, task names. |
| `envs/cfg/config.yaml` | Root Hydra configuration: simulation device, random seed, WandB logging settings. |
| `envs/cfg/task/argus.yaml` | Primary task configuration: observation names, state names, scale factors, action scale, decimation, terrain settings, and all per-task hyperparameters. |
| `envs/utils.py` | Top-level utility functions shared across training scripts. |
| `envs/utils/gen_argus_*_urdf.py` | Scripts that procedurally generate URDF files for each morphological variant. |

---

### 5. Evaluation Output Format (`envs/eval/*.json`)

Running an evaluation command (e.g., `bash run.sh argus_base_eval -p`) writes a JSON file to `envs/eval/`. Each JSON file contains a dictionary with the following keys (all values are averages over the evaluation episode):

| Key | Units | Description |
|---|---|---|
| `episode_length` | timesteps (1 step = 0.02 s) | Mean episode length before timeout or reset |
| `tracking_lin_vel` | m/s | Mean linear velocity tracking error (lower is better; 0 = perfect) |
| `tracking_ang_vel` | rad/s | Mean angular velocity tracking error |
| `success_rate` | fraction [0, 1] | Fraction of trials where the robot reached the goal (task-dependent) |
| `energy` | J/step | Mean mechanical energy consumed per timestep |

*Note: exact keys vary by task. See `envs/tasks/legged_terrain.py` for the full list of logged metrics per task.*

---

## Sharing/Access Information

- **Code repository:** https://github.com/generalroboticslab/Argus
- **Project website:** https://generalroboticslab.com/Argus
- **Paper:** https://doi.org/10.1126/scirobotics.aec1725
- **License:** CC BY-NC-ND 4.0 (data and model weights); see `LICENSE-CC-BY-NC-ND-4.0.md`
- **Blender visualization files:** https://drive.google.com/drive/folders/1OIzNFc4BLMO4p8IIS8wud3FVtw9b3A39

---

## Code/Software

All executable commands (installation, training, evaluation, and ONNX export) are provided in the separate script file `usage_guide.sh` (plain text, readable in any text editor). The file is organized into numbered sections and is not meant to be executed as a whole; individual commands should be copied and run as needed.

**Software requirements:**
- Operating system: Linux (tested on Ubuntu 20.04 and 22.04)
- Python: 3.8 (required by Isaac Gym)
- CUDA: 12.x (GPU with ≥ 16 GB VRAM recommended for training; inference with ONNX Runtime requires no GPU)
- Conda: Miniconda or Anaconda (for environment management)
- Isaac Gym: custom fork at https://github.com/boxiXia/isaacgym (the official NVIDIA release is not compatible due to dependency version conflicts)
- IsaacGymEnvs: https://github.com/isaac-sim/IsaacGymEnvs (with compatibility patch applied by `envs/patch_isaacgymenvs.py`)
- Key Python packages (see `requirements.txt` for pinned versions): torch, onnx, onnxruntime, hydra-core, wandb, numpy

**Working directory:** All commands in `usage_guide.sh` that reference relative paths assume the working directory is `envs/` unless otherwise noted.

**ONNX inference (no Isaac Gym required):** The three `.onnx` files can be run with ONNX Runtime alone. A minimal Python usage example is included in `usage_guide.sh` (Section 5).
