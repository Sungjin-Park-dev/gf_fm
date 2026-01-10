# GF-FM: Usage Guide

This guide details how to use the standalone GF-FM implementation for data collection, training, and inference with obstacle avoidance.

---

## 1. Installation

Ensure you have the required dependencies:
```bash
# Core
pip install torch numpy h5py pyyaml tqdm termcolor

# FABRICS & Visualization
pip install warp-lang>=1.5.0 urdfpy pybullet matplotlib
```
*Note: `cuRobo` is required for data collection.*

---

## 2. Data Collection (cuRobo)

Generate a dataset of optimal trajectories using cuRobo.

```bash
# Generate 1000 demonstrations with goal conditioning
./isaaclab.sh -p scripts/gf_fm/run/generate_dataset.py \
    --num_demos 1000 \
    --fixed_start \
    --goal_conditioned \
    --output scripts/gf_fm/datasets/franka_goal_cond.hdf5
```

---

## 3. Training

Train the Flow Matching policy.

```bash
# Train with default config
./isaaclab.sh -p scripts/gf_fm/run/train.py \
    --config scripts/gf_fm/config/franka_gf_fm_goal.yaml \
    --data_path scripts/gf_fm/datasets/franka_goal_cond.hdf5 \
    --wandb
```

---

## 4. Inference & Guidance

Run inference with Geometric Fabrics guidance to avoid obstacles.

### No Guidance
```bash
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --fixed_start \
    --num_rollouts 10 \
    --no_guidance \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --save_trajectories scripts/gf_fm/results/traj_baseline.npz
```

**Guidance (Post):**
```bash
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --fixed_start \
    --num_rollouts 10 \
    --guidance_scal 1.0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --save_trajectories scripts/gf_fm/results/traj_guidance.npz
```

### Advanced Guidance Modes
Use **ODE-Coupled** mode for better stability and overshoot prevention.
```bash
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --fixed_start \
    --num_rollouts 10 \
    --guidance_mode ode_coupled \
    --n_substeps 4 \
    --lambda_schedule constant \
    --guidance_scale 1.0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --save_trajectories scripts/gf_fm/results/traj_ode_coupled.npz
```

---

## 5. Visualization

Visualize generated trajectories using PyBullet.

```bash
# Visualize specific trajectory index
./isaaclab.sh -p scripts/gf_fm/run/visualize.py \
    --npz scripts/gf_fm/results/trajectories.npz \
    --traj_idx 0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --pybullet \
    --playback_speed 0.5
```

---

## 6. Comparison Experiment

To verify obstacle avoidance, run a comparison between baseline (no guidance) and GF guidance.

### Scenario
- **Start:** Fixed Home
- **Goal:** `[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]`
- **Obstacle:** Sphere at `[0.4, 0.0, 0.5]`, radius `0.15`

### Step 1: Baseline (Expected Collision)
```bash
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/best_model.pth \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --no_guidance \
    --save_trajectories scripts/gf_fm/results/traj_baseline.npz
```

### Step 2: With Guidance (Expected Avoidance)
```bash
./isaaclab.sh -p scripts/gf_fm/run/run_inference.py \
    --checkpoint scripts/gf_fm/logs/best_model.pth \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --guidance_mode ode_coupled \
    --guidance_scale 1.5 \
    --save_trajectories scripts/gf_fm/results/traj_guidance.npz
```

### Step 3: Compare
Visualize both `traj_baseline.npz` and `traj_guidance.npz` to see the difference.

```