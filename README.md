# GF-FM: 2nd-Order Flow Matching with Geometric Fabrics Guidance

**Standalone Implementation** - No Isaac Lab dependencies.

## Overview

GF-FM combines 2nd-order Flow Matching for trajectory generation with Geometric Fabrics (FABRICS) for safety guidance during inference. This implementation uses:

- **cuRobo** (standalone) for motion planning and data collection
- **FABRICS** (standalone) for joint limit repulsion guidance
- **PyTorch** for neural network training and inference

### Key Features

| Feature | Description |
|---------|-------------|
| **2nd-Order State** | State x = [q, q_dot] ∈ R^14 for Franka Panda |
| **Velocity Field** | v = [q_dot, q_ddot] ∈ R^14 |
| **GF Guidance** | v_final = v_FM + λ * v_GF |
| **Standalone** | No Isaac Lab or Isaac Sim required |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         GF-FM Pipeline                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   cuRobo     │───▶│   HDF5       │───▶│   Training   │      │
│  │  MotionGen   │    │  Dataset     │    │   (FM)       │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                                        │              │
│         │ (q, q_dot, q_ddot)                     │ Policy       │
│         ▼                                        ▼              │
│  ┌──────────────┐                       ┌──────────────┐       │
│  │  Trajectory  │                       │  Inference   │       │
│  │  Validation  │                       │  + GF Guide  │       │
│  └──────────────┘                       └──────────────┘       │
│                                                 │               │
│                                    FABRICS ─────┘               │
│                                  (Joint Limit                   │
│                                   Repulsion)                    │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
scripts/imitation_learning/gf_fm/
├── README.md                          # This file
├── __init__.py                        # Package exports
│
├── generate_2nd_order_standalone.py   # Data collection (cuRobo)
├── train_2nd_order.py                 # Training script
├── play_standalone.py                 # Inference with GF guidance
├── visualize_trajectory.py            # Visualization (matplotlib/PyBullet)
│
├── data/
│   ├── __init__.py
│   └── second_order_dataset.py        # HDF5 dataset loader
│
├── policy/
│   ├── __init__.py
│   └── second_order_flow_policy.py    # 2nd-order FM policy
│
├── guidance/
│   ├── __init__.py
│   └── gf_guidance_field.py           # FABRICS standalone guidance
│
├── config/
│   ├── __init__.py
│   ├── franka_gf_fm.yaml              # Training config
│   └── franka_data_collection.yaml    # Data collection config
│
└── *.bak                              # Backup of Isaac Lab versions
```

---

## Installation

### Dependencies

```bash
# Core dependencies
pip install torch>=2.0 numpy h5py pyyaml tqdm termcolor

# cuRobo (for data collection)
# Follow cuRobo installation: https://curobo.org/get_started/1_install_instructions.html

# FABRICS (for GF guidance)
pip install warp-lang>=1.5.0 urdfpy

# Training monitoring (optional)
pip install wandb tensorboard

# Visualization (optional)
pip install matplotlib pybullet
```

### Verify Installation

```bash
# Check cuRobo
python -c "from curobo.wrap.reacher.motion_gen import MotionGen; print('cuRobo OK')"

# Check FABRICS
python -c "from fabrics_sim.fabrics.franka_panda_cspace_fabric import FrankaPandaCspaceFabric; print('FABRICS OK')"
```

---

## Usage

### 1. Data Collection

Generate 2nd-order trajectory dataset using cuRobo standalone:

```bash
./isaaclab.sh -p scripts/gf_fm/train_2nd_order.py \
    --num_demos 1000 \
    --output scripts/gf_fm/datasets/franka_2nd_order.hdf5 \
    --device cuda:0 \
    --interpolation_dt 0.02 \
    --seed 42
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num_demos` | 100 | Number of demonstrations |
| `--output` | `./datasets/franka_2nd_order.hdf5` | Output path |
| `--device` | `cuda:0` | CUDA device |
| `--interpolation_dt` | 0.02 | Trajectory timestep (50Hz) |
| `--seed` | 42 | Random seed |

**Output Format (HDF5):**
```
/data/demo_0/
    obs/
        joint_pos: (T, 7)    # Joint positions
        joint_vel: (T, 7)    # Joint velocities
    actions: (T, 14)         # [q_dot, q_ddot] velocity field
/data/demo_1/
    ...
```

---

### 2. Training

Train 2nd-order Flow Matching policy:

# Train with wandb
```bash
./isaaclab.sh -p scripts/gf_fm/train_2nd_order.py \
    --config scripts/gf_fm/config/franka_gf_fm.yaml \
    --data_path scripts/gf_fm/datasets/franka_2nd_order.hdf5 \
    --wandb \
    --wandb_project gf-fm
```

**Training Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | (required) | Path to YAML config |
| `--data_path` | None | Override dataset path |
| `--log_dir` | None | Override log directory |
| `--device` | None | Override device (cuda:0/cpu) |
| `--wandb` | False | Enable W&B logging |
| `--wandb_project` | `gf-fm` | W&B project name |
| `--wandb_entity` | None | W&B entity (username/team) |
| `--wandb_name` | auto | Custom run name |

**Key Config Options (`franka_gf_fm.yaml`):**
```yaml
# Model
joint_dim: 7
n_obs_steps: 2
horizon: 16

# Training
batch_size: 256
num_epochs: 500
learning_rate: 1.0e-4

# Flow Matching
num_inference_steps: 10
sigma_min: 0.001
```

**W&B Logged Metrics:**
- `train/loss` - Training loss (per step)
- `train/consistency_loss` - Consistency loss component
- `train/velocity_loss` - Velocity loss component
- `epoch/loss` - Average epoch loss
- `epoch/time` - Epoch duration
- `epoch/lr` - Learning rate
- `best_loss` - Best validation loss (summary)
- `best_epoch` - Epoch with best loss (summary)

---

### 3. Inference with GF Guidance

Run policy inference with Geometric Fabrics guidance:

```bash
# With GF guidance
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/franka/checkpoints/best_model.pth \
    --num_rollouts 100 \
    --guidance_scale 1.0

# Without guidance (baseline)
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/franka/checkpoints/best_model.pth \
    --num_rollouts 100 \
    --no_guidance

# Save results
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/franka/checkpoints/best_model.pth \
    --num_rollouts 10 \
    --no_guidance \
    --save_results scripts/gf_fm/results/franka/results.json \
    --save_trajectories scripts/gf_fm/results/franka/trajectories.npz

./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/franka/checkpoints/best_model.pth \
    --receding_horizon \
    --replan_steps 4 \
    --goal_threshold 1.0 \
    --max_steps 1000 \
    --num_rollouts 10 \
    --no_guidance \
    --save_trajectories scripts/gf_fm/results/franka/trajectories_rh.npz
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint` | (required) | Model checkpoint path |
| `--num_rollouts` | 10 | Number of evaluation rollouts |
| `--guidance_scale` | 1.0 | GF guidance strength (λ) |
| `--no_guidance` | False | Disable GF guidance |
| `--save_results` | None | Save summary to JSON |
| `--save_trajectories` | None | Save trajectories to NPZ |

---

### 4. Visualization

#### Matplotlib (2D Plots)

```bash
# Visualize dataset demo
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --dataset ./datasets/franka_2nd_order.hdf5 \
    --demo_idx 0

# Save plot
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --dataset ./datasets/franka_2nd_order.hdf5 \
    --save_plot trajectory_plot.png
```

#### PyBullet (3D Simulation)

```bash
# Interactive 3D visualization
# ./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
#     --dataset ./datasets/franka_2nd_order.hdf5 \
#     --demo_idx 0 \
#     --pybullet

# # Save video
# ./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
#     --dataset ./datasets/franka_2nd_order.hdf5 \
#     --pybullet \
#     --save_video trajectory.mp4

# # Slow playback
# ./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
#     --dataset ./datasets/franka_2nd_order.hdf5 \
#     --pybullet \
#     --playback_speed 0.5

```

#### Inference Results

```bash
# Visualize inference trajectories
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz scripts/gf_fm/results/franka/trajectories_rh.npz \
    --traj_idx 5 \
    --pybullet
```

---

## Technical Details

### 2nd-Order Flow Matching

Standard Flow Matching learns a velocity field v(x, t) that transports samples from noise to data distribution. For 2nd-order dynamics:

```
State:  x = [q, q_dot] ∈ R^14
Action: v = [q_dot, q_ddot] ∈ R^14

Flow ODE:
    dx/dt = v(x, t)

    Expanded:
    dq/dt = q_dot
    d(q_dot)/dt = q_ddot
```

### GF Guidance

At inference time, FABRICS computes repulsion accelerations from joint limits:

```
v_final = v_FM + λ * v_GF

where:
    v_FM = policy output (learned velocity field)
    v_GF = FABRICS guidance (joint limit repulsion)
    λ = guidance_scale (hyperparameter)
```

### Franka Panda Joint Limits

| Joint | Lower (rad) | Upper (rad) |
|-------|-------------|-------------|
| 1 | -2.8973 | 2.8973 |
| 2 | -1.7628 | 1.7628 |
| 3 | -2.8973 | 2.8973 |
| 4 | -3.0718 | -0.0698 |
| 5 | -2.8973 | 2.8973 |
| 6 | -0.0175 | 3.7525 |
| 7 | -2.8973 | 2.8973 |

---

## API Reference

### SecondOrderFlowPolicy

```python
from gf_fm import SecondOrderFlowPolicy

policy = SecondOrderFlowPolicy(
    joint_dim=7,
    n_obs_steps=2,
    horizon=16,
    num_inference_steps=10,
)

# Set guidance
from gf_fm import GFGuidanceField
guidance = GFGuidanceField(guidance_scale=1.0)
policy.set_guidance_field(guidance)

# Inference
obs_dict = {
    'joint_pos': torch.randn(1, 2, 7),  # (B, n_obs_steps, 7)
    'joint_vel': torch.randn(1, 2, 7),
}
result = policy.predict_action({'obs': obs_dict})
action = result['action']  # (B, horizon, 14)
```

### GFGuidanceField

```python
from gf_fm import GFGuidanceField

guidance = GFGuidanceField(
    batch_size=1,
    device='cuda:0',
    joint_dim=7,
    guidance_scale=1.0,
    max_guidance_strength=5.0,
)

# Compute guidance
q = torch.randn(1, 7)      # Joint positions
q_dot = torch.randn(1, 7)  # Joint velocities
t = 0.5                    # Flow time

guidance_vel = guidance.compute_guidance_with_state(q, q_dot, t)
# Returns (B, 14) guidance velocity
```

### SimpleJointLimitGuidance

Lightweight fallback without full FABRICS:

```python
from gf_fm import SimpleJointLimitGuidance

guidance = SimpleJointLimitGuidance(
    guidance_scale=1.0,
    repulsion_threshold=0.2,  # Start repulsion 0.2 rad from limit
    repulsion_gain=5.0,
)
```

---

## Troubleshooting

### cuRobo Errors

```
ImportError: No module named 'curobo'
```
→ Install cuRobo following official instructions

### FABRICS Errors

```
ImportError: No module named 'fabrics_sim'
```
→ Add FABRICS to PYTHONPATH:
```bash
export PYTHONPATH=$PYTHONPATH:/path/to/FABRICS/src
```

### CUDA Out of Memory

Reduce batch size in config:
```yaml
batch_size: 128  # or smaller
```

### PyBullet URDF Not Found

The visualizer tries multiple URDF paths. If Franka URDF is not found:
```bash
pip install pybullet
# PyBullet includes a Franka URDF in pybullet_data
```

---

## References

- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Geometric Fabrics for Motion Generation](https://arxiv.org/abs/2010.14750)
- [cuRobo: Accelerated Robot Planning](https://curobo.org/)
- [NVIDIA FABRICS](https://github.com/NVlabs/FABRICS)

---

## License

Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

SPDX-License-Identifier: Apache-2.0


Step 1: 데이터 생성

./isaaclab.sh -p scripts/gf_fm/generate_2nd_order_standalone.py \
      --num_demos 1000 \
      --fixed_start \
      --goal_conditioned \
      --output scripts/gf_fm/datasets/franka_goal_cond.hdf5



Step 2: 학습

./isaaclab.sh -p scripts/gf_fm/train_2nd_order.py \
    --config scripts/gf_fm/config/franka_gf_fm_goal.yaml \
    --data_path scripts/gf_fm/datasets/franka_goal_cond.hdf5



Step 3: 검증

./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --no_guidance \
    --num_rollouts 10 \
    --save_trajectories scripts/gf_fm/results/trajectories_goal.npz


Step 4: 시각화

./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz scripts/gf_fm/results/trajectories_goal.npz \
    --traj_idx 5 \
    --pybullet
    --playback_speed 0.2