# GF Guidance Comparison Commands

This document outlines the commands to compare trajectory generation with and without Geometric Fabrics (GF) guidance in a deterministic scenario.

## 1. Scenario Definition

We use a fixed scenario where the direct path to the goal is blocked by an obstacle.

- **Start Config**: Fixed (Home position)
- **Goal Config**: `[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]` (Requires moving forward and left)
- **Obstacle**: Sphere at `[0.4, 0.0, 0.5]`, radius `0.15`m (Blocks the forward path)

---

## 2. Execution Commands

Run these commands from the `isaaclab` root directory.

### A. Baseline: No Guidance

Generates a trajectory without any obstacle avoidance.

```bash
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --no_guidance \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_baseline.npz
```

### B. Experiment: With GF Guidance

Generates a trajectory with active GF repulsion to avoid the obstacle.

```bash
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --guidance_scale 1.0 \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_guidance.npz
```

---

## 3. Visualization

Visualize the generated trajectories using PyBullet to verify obstacle avoidance.

### Visualize Baseline (Collision Expected)

```bash
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz scripts/gf_fm/results/traj_baseline.npz \
    --traj_idx 0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --pybullet \
    --playback_speed 0.5
```

### Visualize Guidance (Avoidance Expected)

```bash
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz scripts/gf_fm/results/traj_guidance.npz \
    --traj_idx 0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --pybullet \
    --playback_speed 0.5
```

---

## 4. Expected Results

- **Baseline**: The robot should move directly towards the goal, likely passing through the red obstacle sphere. The `Average final error` should be low (e.g., ~0.3 rad), indicating it reached the vicinity of the goal but ignored the collision.
- **Guidance**: The robot should deviate from the direct path to go around the obstacle. The `Average final error` may be higher (e.g., >1.0 rad) if the avoidance maneuver prevents it from reaching the goal within the fixed time horizon, or it might reach the goal via a longer path. Visually, the blue robot should stay clear of the red sphere.
