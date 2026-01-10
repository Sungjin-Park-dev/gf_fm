#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""
Debug script to verify GF Guidance isolated from Flow Matching policy.
It simulates the robot dynamics driven ONLY by GF repulsion forces.

Goal: Confirm if the robot moves AWAY from the obstacle (Repulsion) or TOWARDS it (Attraction).
"""

import argparse
import os
import sys
import numpy as np
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import GF
from guidance.gf_guidance_field import GFGuidanceField

# Configuration
DEVICE = "cuda:0"
# Obstacle placed further in front (~0.45)
# Home EE (~0.306). Radius 0.1. Surface at 0.35.
# Dist ~0.044 (Inside 0.05 engage depth)
OBSTACLE_POS = [0.45, 0.0, 0.48]
OBSTACLE_RADIUS = 0.10
DT = 0.02
NUM_STEPS = 50
GUIDANCE_SCALE = 1.0  # Strong guidance to see effect clearly
# Franka Home
FIXED_START_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)

def main():
    print(f"[{sys.argv[0]}] Starting GF Isolation Test...")
    print(f"  Obstacle: {OBSTACLE_POS}, r={OBSTACLE_RADIUS}")
    print(f"  Start Q: {FIXED_START_Q}")
    print(f"  Guidance Scale: {GUIDANCE_SCALE}")
    
    # 1. Initialize GF Field
    # Enable debug to print dot products (accel_dir vs away_vec)
    gf_field = GFGuidanceField(
        batch_size=1,
        device=DEVICE,
        guidance_scale=GUIDANCE_SCALE,
        max_guidance_strength=10.0,
        debug=True,       # Enable internal debug prints
        debug_every=10    # Print every 10 steps
    )
    
    # 2. Set Obstacle
    gf_field.set_sphere_obstacles([(tuple(OBSTACLE_POS), OBSTACLE_RADIUS)])
    
    # 3. Initialize State
    q = torch.tensor(FIXED_START_Q, device=DEVICE).unsqueeze(0) # (1, 7)
    q_dot = torch.zeros_like(q)
    
    # Dummy latent z (not used for guidance calculation if we pass q, q_dot explicitly)
    # But compute_guidance needs z shape.
    z_dummy = torch.zeros((1, 14), device=DEVICE)
    
    trajectory_q = [q.cpu().numpy().squeeze()]
    
    print("\nStarting Simulation Loop (Pure GF Dynamics)...")
    print("-" * 60)
    
    for i in range(NUM_STEPS):
        # Reset velocity to zero to avoid damping feedback loop (pure geometric repulsion)
        q_dot = torch.zeros_like(q)

        # 4. Compute Guidance
        # We assume flow time t=0.5 (middle of generation) just to have a valid t
        t_sim = 0.5 
        
        # This returns v_final correction. 
        # v_GF is usually acceleration-based. GFGuidanceField converts it to velocity correction.
        # guidance = [v_q, v_qdot]
        guidance = gf_field.compute_guidance_with_state(q, q_dot, t=t_sim)
        
        # Extract purely the velocity component to update position
        # guidance is (B, 14). First 7 are v_q correction.
        v_guidance = guidance[:, :7]
        a_guidance = guidance[:, 7:] # This is q_ddot
        
        # 5. Integrate Dynamics
        # Pure Euler integration driven by guidance velocity
        # q_{t+1} = q_t + v_guidance * dt
        
        # Note: In real inference, v_total = v_FM + v_guidance.
        # Here v_FM = 0. So v_total = v_guidance.
        
        q_next = q + v_guidance * DT
        
        # Simple velocity update (optional, if we want 2nd order dynamics)
        # But GFGuidanceField returns a velocity correction meant to be added to v_FM.
        # So treating it as velocity is correct for position update.
        
        q = q_next
        # q_dot = a_guidance # Removed to avoid damping feedback loop
        
        trajectory_q.append(q.cpu().numpy().squeeze())
        
        # Log magnitude and distance
        vel_norm = torch.norm(v_guidance).item()
        
        # Calculate minimum distance to obstacle
        body_points, _ = gf_field._fabric.get_taskmap("body_points")(q, q_dot)
        body_points = body_points.view(1, -1, 3) # (B, N, 3)
        obs_pos = torch.tensor(OBSTACLE_POS, device=DEVICE).view(1, 1, 3)
        dists = torch.norm(body_points - obs_pos, dim=-1) - OBSTACLE_RADIUS
        min_dist = dists.min().item()
        
        if i % 10 == 0:
            print(f"Step {i:03d}: |v_guide| = {vel_norm:.4f}, Min Dist (signed) = {min_dist:.4f}")

    print("-" * 60)
    print("Simulation Complete.")
    
    # 6. Save Trajectory
    save_path = "scripts/gf_fm/results/debug_gf.npz"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Dummy limits for visualization
    joint_limits_lower = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    joint_limits_upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
    
    np.savez(
        save_path,
        trajectories=np.array([trajectory_q]), # Shape (1, T, 7)
        joint_limits_lower=joint_limits_lower,
        joint_limits_upper=joint_limits_upper
    )
    print(f"Trajectory saved to: {save_path}")
    print("Visualize with:")
    print(f"  ./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py --npz {save_path} --traj_idx 0 --pybullet --obstacles '[{{\"pos\":{OBSTACLE_POS},\"radius\":{OBSTACLE_RADIUS}}}]'")

if __name__ == "__main__":
    main()
