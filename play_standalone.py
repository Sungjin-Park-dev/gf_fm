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
    # Receding horizon arguments
    parser.add_argument("--receding_horizon", action="store_true", help="Enable receding horizon mode")
    parser.add_argument("--replan_steps", type=int, default=4, help="Steps to execute before replanning")
    parser.add_argument("--goal_threshold", type=float, default=0.05, help="Goal position threshold (rad)")
    parser.add_argument("--max_steps", type=int, default=1000, help="Maximum steps before timeout")
    parser.add_argument("--fixed_start", action="store_true", help="Use fixed start configuration")
    return parser.parse_args()


# Fixed start configuration (Franka home position)
FIXED_START_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)


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


def sample_goal_state(rng: np.random.Generator, device: str, margin: float = 0.1):
    """Sample random goal joint position within joint limits.

    Args:
        rng: Random number generator
        device: Device
        margin: Margin from joint limits (fraction of range)

    Returns:
        q_goal: (1, 7) goal joint positions
    """
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]

    range_size = upper - lower
    lower_safe = lower + margin * range_size
    upper_safe = upper - margin * range_size

    q_goal = rng.uniform(lower_safe, upper_safe)
    q_goal = torch.tensor(q_goal, device=device, dtype=torch.float32).unsqueeze(0)

    return q_goal


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


def run_receding_horizon_inference(
    policy,
    guidance_field,
    q_init: torch.Tensor,
    q_dot_init: torch.Tensor,
    q_goal: torch.Tensor,
    config: dict,
    device: str,
    replan_steps: int = 4,
    goal_threshold: float = 0.05,
    max_steps: int = 500,
) -> dict:
    """Run receding horizon inference until goal is reached.

    Args:
        policy: SecondOrderFlowPolicy
        guidance_field: GFGuidanceField or None
        q_init: (1, 7) initial joint positions
        q_dot_init: (1, 7) initial joint velocities
        q_goal: (1, 7) goal joint positions
        config: Model config
        device: Device
        replan_steps: Number of steps to execute before replanning
        goal_threshold: Goal position threshold (rad)
        max_steps: Maximum steps before timeout

    Returns:
        result dict with full trajectory and goal status
    """
    n_obs_steps = config.get('n_obs_steps', 2)
    joint_dim = config.get('joint_dim', 7)
    horizon = config.get('horizon', 16)
    dt = config.get('interpolation_dt', 0.02)

    # Initialize state
    q_current = q_init.clone()
    q_dot_current = q_dot_init.clone()

    # Trajectory storage
    full_trajectory = [q_init.cpu().numpy().squeeze()]
    full_velocity = []

    step_count = 0
    replan_count = 0
    reached_goal = False

    while step_count < max_steps:
        # Build observation dict (with goal conditioning)
        obs_dict = {
            'joint_pos': q_current.unsqueeze(1).repeat(1, n_obs_steps, 1),
            'joint_vel': q_dot_current.unsqueeze(1).repeat(1, n_obs_steps, 1),
            'goal_q': q_goal.unsqueeze(1).repeat(1, n_obs_steps, 1),  # Goal conditioning
        }

        # Run policy
        with torch.no_grad():
            result = policy.predict_action({'obs': obs_dict})

        # Extract velocity field
        action_pred = result['action_pred']  # (1, horizon, 14)
        q_dot_pred = action_pred[..., :joint_dim]  # (1, H, 7)

        replan_count += 1

        # Execute replan_steps (or remaining horizon)
        steps_to_execute = min(replan_steps, horizon)

        for t in range(steps_to_execute):
            # Euler integration
            q_next = q_current + dt * q_dot_pred[:, t, :]
            q_dot_next = q_dot_pred[:, t, :]

            # Store trajectory
            full_trajectory.append(q_next.cpu().numpy().squeeze())
            full_velocity.append(q_dot_next.cpu().numpy().squeeze())

            # Update state
            q_current = q_next
            q_dot_current = q_dot_next
            step_count += 1

            # Check goal reached
            error = torch.norm(q_current - q_goal).item()
            if error < goal_threshold:
                reached_goal = True
                break

            # Check max steps
            if step_count >= max_steps:
                break

        if reached_goal or step_count >= max_steps:
            break

    # Compute final error
    final_error = torch.norm(q_current - q_goal).item()

    # Check joint limits on trajectory
    q_traj_np = np.array(full_trajectory)
    valid, min_margin, violations = check_joint_limits(q_traj_np)

    return {
        'q_init': q_init.cpu().numpy().squeeze(),
        'q_goal': q_goal.cpu().numpy().squeeze(),
        'q_trajectory': q_traj_np,
        'q_dot_trajectory': np.array(full_velocity) if full_velocity else np.zeros((0, 7)),
        'dt': dt,
        'reached_goal': reached_goal,
        'final_error': final_error,
        'total_steps': step_count,
        'replan_count': replan_count,
        'valid': valid,
        'min_margin': float(min_margin),
        'violations': int(violations),
    }


def main():
    args = parse_args()

    # Set seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    if args.receding_horizon:
        print("GF-FM Receding Horizon Inference")
    else:
        print("GF-FM Standalone Inference")
    print(f"{'='*60}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Rollouts: {args.num_rollouts}")
    print(f"  Guidance: {'DISABLED' if args.no_guidance else f'scale={args.guidance_scale}'}")
    print(f"  Device: {args.device}")
    if args.receding_horizon:
        print(f"  Mode: RECEDING HORIZON")
        print(f"  Replan steps: {args.replan_steps}")
        print(f"  Goal threshold: {args.goal_threshold} rad")
        print(f"  Max steps: {args.max_steps}")
        print(f"  Fixed start: {args.fixed_start}")
        if args.fixed_start:
            print(f"  Start config: {FIXED_START_Q.tolist()}")
    else:
        print(f"  Mode: SINGLE-SHOT")
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

    # Receding horizon specific stats
    goal_reached_count = 0
    total_steps_list = []
    final_errors = []

    for i in tqdm(range(args.num_rollouts), desc="Rollouts"):
        # Sample initial state (fixed or random)
        if args.fixed_start:
            q_init = torch.tensor(FIXED_START_Q, device=args.device, dtype=torch.float32).unsqueeze(0)
            q_dot_init = torch.zeros(1, 7, device=args.device, dtype=torch.float32)
        else:
            q_init, q_dot_init = sample_random_state(rng, args.device)

        if args.receding_horizon:
            # Sample random goal state
            q_goal = sample_goal_state(rng, args.device)

            # Run receding horizon inference
            result = run_receding_horizon_inference(
                policy, guidance_field,
                q_init, q_dot_init, q_goal,
                config, args.device,
                replan_steps=args.replan_steps,
                goal_threshold=args.goal_threshold,
                max_steps=args.max_steps,
            )

            # Track receding horizon stats
            if result['reached_goal']:
                goal_reached_count += 1
            total_steps_list.append(result['total_steps'])
            final_errors.append(result['final_error'])

        else:
            # Single-shot inference
            result = run_inference(
                policy, guidance_field, q_init, q_dot_init, config, args.device
            )

            # Check joint limits
            valid, min_margin, violations = check_joint_limits(result['q_trajectory'])
            result['valid'] = valid
            result['min_margin'] = float(min_margin)
            result['violations'] = int(violations)

        if result['valid']:
            valid_count += 1
        total_violations += result['violations']

        results.append(result)
        all_trajectories.append(result['q_trajectory'])

    # Summary
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")

    if args.receding_horizon:
        avg_steps = np.mean(total_steps_list)
        avg_error = np.mean(final_errors)
        dt = config.get('interpolation_dt', 0.02)
        avg_time = avg_steps * dt

        print(f"  Goal reached: {goal_reached_count}/{args.num_rollouts} ({100*goal_reached_count/args.num_rollouts:.1f}%)")
        print(f"  Avg steps to goal: {avg_steps:.1f}")
        print(f"  Avg trajectory time: {avg_time:.2f}s")
        print(f"  Avg final error: {avg_error:.4f} rad")
        print(f"  Valid trajectories: {valid_count}/{args.num_rollouts} ({100*valid_count/args.num_rollouts:.1f}%)")
        print(f"  Total limit violations: {total_violations}")
    else:
        print(f"  Valid trajectories: {valid_count}/{args.num_rollouts} ({100*valid_count/args.num_rollouts:.1f}%)")
        print(f"  Total limit violations: {total_violations}")

    print(f"  GF guidance: {'ENABLED' if not args.no_guidance else 'DISABLED'}")
    print(f"{'='*60}\n")

    # Save results
    if args.save_results:
        summary = {
            'num_rollouts': args.num_rollouts,
            'valid_count': int(valid_count),
            'valid_rate': float(valid_count / args.num_rollouts),
            'total_violations': int(total_violations),
            'guidance_enabled': not args.no_guidance,
            'guidance_scale': args.guidance_scale,
            'checkpoint': args.checkpoint,
            'receding_horizon': args.receding_horizon,
        }

        if args.receding_horizon:
            summary.update({
                'replan_steps': args.replan_steps,
                'goal_threshold': args.goal_threshold,
                'max_steps': args.max_steps,
                'goal_reached_count': int(goal_reached_count),
                'goal_reached_rate': float(goal_reached_count / args.num_rollouts),
                'avg_steps': float(avg_steps),
                'avg_final_error': float(avg_error),
            })

        with open(args.save_results, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"[Save] Results: {args.save_results}")

    # Save trajectories
    if args.save_trajectories:
        # Handle variable length trajectories for receding horizon
        if args.receding_horizon:
            # Save as object array for variable lengths
            save_dict = {
                'joint_limits_lower': FRANKA_JOINT_LIMITS["lower"],
                'joint_limits_upper': FRANKA_JOINT_LIMITS["upper"],
                'num_trajectories': len(all_trajectories),
            }
            # Save each trajectory separately
            for idx, traj in enumerate(all_trajectories):
                save_dict[f'traj_{idx}'] = traj
                save_dict[f'goal_{idx}'] = results[idx]['q_goal']
                save_dict[f'reached_{idx}'] = results[idx]['reached_goal']
                save_dict[f'steps_{idx}'] = results[idx]['total_steps']

            np.savez(args.save_trajectories, **save_dict)
        else:
            np.savez(
                args.save_trajectories,
                trajectories=np.stack(all_trajectories),
                joint_limits_lower=FRANKA_JOINT_LIMITS["lower"],
                joint_limits_upper=FRANKA_JOINT_LIMITS["upper"],
            )
        print(f"[Save] Trajectories: {args.save_trajectories}")


if __name__ == "__main__":
    main()
