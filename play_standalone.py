#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone inference for SecondOrderFlowPolicy with GF guidance.

NO Isaac Lab dependencies - uses policy directly for trajectory generation.

Usage:
    python play_standalone.py \
        --checkpoint ./logs/gf_fm/checkpoints/best_model.pth \
        --num_rollouts 10 \
        --guidance_scale 1.0
"""

import argparse
import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'flowpolicy_curobo'))


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone inference with GF guidance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_rollouts", type=int, default=10, help="Number of rollouts")
    parser.add_argument("--horizon", type=int, default=16, help="Trajectory horizon")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="GF guidance scale")
    parser.add_argument("--no_guidance", action="store_true", help="Disable GF guidance")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--save_results", type=str, default=None, help="Save results to JSON")
    parser.add_argument("--save_trajectories", type=str, default=None, help="Save trajectories to NPZ")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


# Franka joint limits
FRANKA_JOINT_LIMITS = {
    "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    "upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
}


def load_checkpoint(checkpoint_path: str, device: str):
    """Load trained model from checkpoint.

    Returns:
        policy, config
    """
    from policy.second_order_flow_policy import SecondOrderFlowPolicy
    from model.normalizer import LinearNormalizer

    print(f"[Load] Checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    shape_meta = checkpoint['shape_meta']

    print(f"  State dim: {shape_meta.get('state', {}).get('shape', 'N/A')}")
    print(f"  Action dim: {shape_meta['action']['shape']}")

    # Create policy
    policy = SecondOrderFlowPolicy(**config)

    # Load weights
    policy.load_state_dict(checkpoint['policy_state_dict'])

    # Load normalizer
    normalizer = LinearNormalizer()
    normalizer.load_state_dict(checkpoint['normalizer_state_dict'])
    policy.set_normalizer(normalizer)

    policy.to(device)
    policy.eval()

    return policy, config


def sample_random_state(rng: np.random.Generator, device: str, margin: float = 0.1):
    """Sample random initial state within joint limits.

    Returns:
        q: (1, 7) joint positions
        q_dot: (1, 7) joint velocities (zeros)
    """
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]

    range_size = upper - lower
    lower_safe = lower + margin * range_size
    upper_safe = upper - margin * range_size

    q = rng.uniform(lower_safe, upper_safe)
    q = torch.tensor(q, device=device, dtype=torch.float32).unsqueeze(0)

    q_dot = torch.zeros(1, 7, device=device, dtype=torch.float32)

    return q, q_dot


def check_joint_limits(q: np.ndarray) -> tuple:
    """Check if trajectory stays within joint limits.

    Returns:
        (valid, min_margin, violation_count)
    """
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]

    lower_margin = q - lower  # Should be positive
    upper_margin = upper - q  # Should be positive

    min_margin = min(lower_margin.min(), upper_margin.min())
    violation_count = np.sum((q < lower) | (q > upper))

    return min_margin > 0, min_margin, violation_count


def run_inference(
    policy,
    guidance_field,
    q_init: torch.Tensor,
    q_dot_init: torch.Tensor,
    config: dict,
    device: str,
) -> dict:
    """Run policy inference from initial state.

    Args:
        policy: SecondOrderFlowPolicy
        guidance_field: GFGuidanceField or None
        q_init: (1, 7) initial joint positions
        q_dot_init: (1, 7) initial joint velocities
        config: Model config
        device: Device

    Returns:
        result dict with trajectory
    """
    n_obs_steps = config.get('n_obs_steps', 2)
    joint_dim = config.get('joint_dim', 7)

    # Build observation dict
    # Repeat initial state for n_obs_steps
    obs_dict = {
        'joint_pos': q_init.unsqueeze(1).repeat(1, n_obs_steps, 1),  # (1, n_obs_steps, 7)
        'joint_vel': q_dot_init.unsqueeze(1).repeat(1, n_obs_steps, 1),  # (1, n_obs_steps, 7)
    }

    # Run policy
    with torch.no_grad():
        result = policy.predict_action({'obs': obs_dict})

    # Extract action (velocity field)
    action_pred = result['action_pred']  # (1, horizon, 14)

    # Split into q_dot and q_ddot
    q_dot_pred = action_pred[..., :joint_dim].cpu().numpy()  # (1, H, 7)
    q_ddot_pred = action_pred[..., joint_dim:].cpu().numpy()  # (1, H, 7)

    # Integrate trajectory
    dt = config.get('interpolation_dt', 0.02)
    q_init_np = q_init.cpu().numpy()  # (1, 7)

    H = q_dot_pred.shape[1]
    q_traj = np.zeros((1, H + 1, joint_dim))
    q_traj[:, 0, :] = q_init_np

    # Simple Euler integration
    for t in range(H):
        q_traj[:, t + 1, :] = q_traj[:, t, :] + dt * q_dot_pred[:, t, :]

    return {
        'q_init': q_init_np.squeeze(0),
        'q_trajectory': q_traj.squeeze(0),  # (H+1, 7)
        'q_dot_pred': q_dot_pred.squeeze(0),  # (H, 7)
        'q_ddot_pred': q_ddot_pred.squeeze(0),  # (H, 7)
        'dt': dt,
    }


def main():
    args = parse_args()

    # Set seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("GF-FM Standalone Inference")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rollouts: {args.num_rollouts}")
    print(f"  Guidance: {'DISABLED' if args.no_guidance else f'scale={args.guidance_scale}'}")
    print(f"  Device: {args.device}")
    print(f"{'='*60}\n")

    # Load policy
    policy, config = load_checkpoint(args.checkpoint, args.device)

    # Create guidance field
    guidance_field = None
    if not args.no_guidance:
        from guidance.gf_guidance_field import GFGuidanceField

        guidance_field = GFGuidanceField(
            batch_size=1,
            device=args.device,
            joint_dim=config.get('joint_dim', 7),
            guidance_scale=args.guidance_scale,
        )
        policy.set_guidance_field(guidance_field)
        print("[Guidance] GF guidance enabled")
    else:
        print("[Guidance] DISABLED")

    # Run rollouts
    results = []
    valid_count = 0
    total_violations = 0

    all_trajectories = []

    for i in tqdm(range(args.num_rollouts), desc="Rollouts"):
        # Sample random initial state
        q_init, q_dot_init = sample_random_state(rng, args.device)

        # Run inference
        result = run_inference(
            policy, guidance_field, q_init, q_dot_init, config, args.device
        )

        # Check joint limits
        valid, min_margin, violations = check_joint_limits(result['q_trajectory'])

        result['valid'] = valid
        result['min_margin'] = float(min_margin)
        result['violations'] = int(violations)

        if valid:
            valid_count += 1
        total_violations += violations

        results.append(result)
        all_trajectories.append(result['q_trajectory'])

    # Summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"  Valid trajectories: {valid_count}/{args.num_rollouts} ({100*valid_count/args.num_rollouts:.1f}%)")
    print(f"  Total limit violations: {total_violations}")
    print(f"  GF guidance: {'ENABLED' if not args.no_guidance else 'DISABLED'}")
    print(f"{'='*60}\n")

    # Save results
    if args.save_results:
        summary = {
            'num_rollouts': args.num_rollouts,
            'valid_count': valid_count,
            'valid_rate': valid_count / args.num_rollouts,
            'total_violations': total_violations,
            'guidance_enabled': not args.no_guidance,
            'guidance_scale': args.guidance_scale,
            'checkpoint': args.checkpoint,
        }
        with open(args.save_results, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[Save] Results: {args.save_results}")

    # Save trajectories
    if args.save_trajectories:
        np.savez(
            args.save_trajectories,
            trajectories=np.stack(all_trajectories),
            joint_limits_lower=FRANKA_JOINT_LIMITS["lower"],
            joint_limits_upper=FRANKA_JOINT_LIMITS["upper"],
        )
        print(f"[Save] Trajectories: {args.save_trajectories}")


if __name__ == "__main__":
    main()
