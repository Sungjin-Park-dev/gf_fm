#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generate 2nd-order (q, q_dot, q_ddot) trajectory dataset using cuRobo standalone.

NO Isaac Lab dependencies - uses cuRobo directly for motion planning.

Usage:
    python generate_2nd_order_standalone.py \
        --num_demos 100 \
        --output ./datasets/franka_2nd_order.hdf5 \
        --device cuda:0
"""

import argparse
import os
import h5py
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Generate 2nd-order trajectory dataset (standalone)")
    parser.add_argument("--num_demos", type=int, default=100, help="Number of demonstrations")
    parser.add_argument("--output", type=str, default="./datasets/franka_2nd_order.hdf5", help="Output HDF5 path")
    parser.add_argument("--device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--interpolation_dt", type=float, default=0.02, help="Trajectory interpolation dt")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    # Goal-conditioned options
    parser.add_argument("--fixed_start", action="store_true", help="Use fixed start joint configuration")
    parser.add_argument("--goal_conditioned", action="store_true", help="Store goal_q for goal-conditioned training")
    return parser.parse_args()


# Franka Panda joint limits (from URDF)
FRANKA_JOINT_LIMITS = {
    "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    "upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
}

FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
]

# Fixed start configuration (Franka home position)
FIXED_START_Q = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float32)


def setup_curobo(device: str, interpolation_dt: float):
    """Initialize cuRobo MotionGen standalone (no Isaac Lab).

    Returns:
        motion_gen: cuRobo MotionGen instance
    """
    # cuRobo imports (standalone - no Isaac Lab)
    from curobo.types.base import TensorDeviceType
    from curobo.geom.types import WorldConfig, Cuboid
    from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

    print("[cuRobo] Initializing MotionGen standalone...")

    tensor_args = TensorDeviceType(device=torch.device(device))

    # Load Franka config - no world (free-space trajectories)
    # Use a far-away dummy obstacle to keep primitive collision checks enabled
    # while maintaining an effectively free-space environment.
    world_cfg = WorldConfig(
        cuboid=[
            Cuboid(
                name="dummy_far",
                dims=[0.01, 0.01, 0.01],
                pose=[1000.0, 1000.0, 1000.0, 1.0, 0.0, 0.0, 0.0],
            )
        ]
    )

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "franka.yml",           # Robot config
        world_cfg,              # Far-away dummy obstacle (free-space)
        tensor_args,
        interpolation_dt=interpolation_dt,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        trajopt_tsteps=32,
    )

    motion_gen = MotionGen(motion_gen_config)

    print("[cuRobo] Warming up...")
    motion_gen.warmup(warmup_js_trajopt=True)

    print("[cuRobo] Ready!")
    return motion_gen


def sample_random_joint_config(rng: np.random.Generator, margin: float = 0.1) -> np.ndarray:
    """Sample random joint configuration within limits.

    Args:
        rng: NumPy random generator
        margin: Margin from joint limits (fraction)

    Returns:
        q: (7,) joint configuration
    """
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]

    # Add margin to avoid edge cases
    range_size = upper - lower
    lower_safe = lower + margin * range_size
    upper_safe = upper - margin * range_size

    q = rng.uniform(lower_safe, upper_safe)
    return q.astype(np.float32)


def plan_trajectory(motion_gen, q_start: np.ndarray, q_goal: np.ndarray, device: str):
    """Plan trajectory from start to goal joint configuration.

    Args:
        motion_gen: cuRobo MotionGen instance
        q_start: (7,) start joint config
        q_goal: (7,) goal joint config
        device: CUDA device

    Returns:
        trajectory dict with positions, velocities, accelerations or None if failed
    """
    from curobo.types.robot import JointState
    from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

    # Create JointState objects
    start_state = JointState.from_position(
        torch.tensor(q_start, device=device).unsqueeze(0),
        joint_names=FRANKA_JOINT_NAMES
    )

    goal_state = JointState.from_position(
        torch.tensor(q_goal, device=device).unsqueeze(0),
        joint_names=FRANKA_JOINT_NAMES
    )

    # Plan config
    plan_config = MotionGenPlanConfig(
        max_attempts=5,
        enable_finetune_trajopt=True,
    )

    # Plan joint-to-joint motion
    result = motion_gen.plan_single_js(
        start_state,
        goal_state,
        plan_config
    )

    if not result.success.item():
        return None

    # Get interpolated trajectory
    traj = result.get_interpolated_plan()

    # Extract trajectory data
    positions = traj.position.cpu().numpy()           # (T, 7)
    velocities = traj.velocity.cpu().numpy()          # (T, 7)
    accelerations = traj.acceleration.cpu().numpy()   # (T, 7)

    # Squeeze batch dimension if present
    if positions.ndim == 3:
        positions = positions.squeeze(0)
        velocities = velocities.squeeze(0)
        accelerations = accelerations.squeeze(0)

    return {
        "positions": positions,
        "velocities": velocities,
        "accelerations": accelerations,
        "dt": result.interpolation_dt,
    }


def compute_actions(positions: np.ndarray, velocities: np.ndarray, accelerations: np.ndarray) -> np.ndarray:
    """Compute action array [q_dot, q_ddot] for FM training.

    Args:
        positions: (T, 7) joint positions
        velocities: (T, 7) joint velocities
        accelerations: (T, 7) joint accelerations

    Returns:
        actions: (T, 14) velocity field [q_dot, q_ddot]
    """
    # Actions for 2nd-order FM: [q_dot, q_ddot]
    actions = np.concatenate([velocities, accelerations], axis=-1)
    return actions.astype(np.float32)


def save_dataset(demos: list, output_path: str, goal_conditioned: bool = False):
    """Save demonstrations to HDF5 file.

    Args:
        demos: List of demo dicts
        output_path: Output HDF5 path
        goal_conditioned: Whether to save goal_q for goal-conditioned training
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # Metadata
        f.attrs["num_demos"] = len(demos)
        f.attrs["format"] = "gf_fm_2nd_order_standalone"
        f.attrs["joint_names"] = FRANKA_JOINT_NAMES
        f.attrs["goal_conditioned"] = goal_conditioned

        data_grp = f.create_group("data")

        for i, demo in enumerate(demos):
            demo_grp = data_grp.create_group(f"demo_{i}")

            # Observations
            obs_grp = demo_grp.create_group("obs")
            obs_grp.create_dataset("joint_pos", data=demo["joint_pos"], compression="gzip")
            obs_grp.create_dataset("joint_vel", data=demo["joint_vel"], compression="gzip")

            # Goal observation (for goal-conditioned training)
            if goal_conditioned and "goal_q" in demo:
                obs_grp.create_dataset("goal_q", data=demo["goal_q"], compression="gzip")

            # Actions: [q_dot, q_ddot]
            demo_grp.create_dataset("actions", data=demo["actions"], compression="gzip")

            # Metadata
            demo_grp.attrs["num_samples"] = len(demo["actions"])
            demo_grp.attrs["dt"] = demo["dt"]

            # Store start/goal as metadata
            if "q_start" in demo:
                demo_grp.attrs["q_start"] = demo["q_start"]
            if "q_goal" in demo:
                demo_grp.attrs["q_goal"] = demo["q_goal"]

    print(f"[Dataset] Saved {len(demos)} demos to {output_path}")
    if goal_conditioned:
        print(f"[Dataset] Goal-conditioned: YES (goal_q stored)")


def main():
    args = parse_args()

    # Set seed
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    print(f"\n{'='*60}")
    print("GF-FM Standalone Data Collection (cuRobo)")
    print(f"{'='*60}")
    print(f"  Demos: {args.num_demos}")
    print(f"  Output: {args.output}")
    print(f"  Device: {args.device}")
    print(f"  Interpolation dt: {args.interpolation_dt}")
    print(f"  Fixed start: {args.fixed_start}")
    print(f"  Goal-conditioned: {args.goal_conditioned}")
    if args.fixed_start:
        print(f"  Start config: {FIXED_START_Q.tolist()}")
    print(f"{'='*60}\n")

    # Initialize cuRobo
    motion_gen = setup_curobo(args.device, args.interpolation_dt)

    # Collect demonstrations
    demos = []
    failed_count = 0

    pbar = tqdm(total=args.num_demos, desc="Collecting demos")

    while len(demos) < args.num_demos:
        # Sample start (fixed or random)
        if args.fixed_start:
            q_start = FIXED_START_Q.copy()
        else:
            q_start = sample_random_joint_config(rng)

        # Sample random goal
        q_goal = sample_random_joint_config(rng)

        # Plan trajectory
        result = plan_trajectory(motion_gen, q_start, q_goal, args.device)

        if result is None:
            failed_count += 1
            continue

        # Add stationary padding at the goal to teach the model to stop
        # This is critical for Flow Matching to learn v=0 at the goal
        n_pad = 20  # Cover at least one horizon (16)
        
        # Get last position (should be close to q_goal, but use actual plan end)
        last_pos = result["positions"][-1]
        
        # Create padding arrays
        pad_pos = np.tile(last_pos, (n_pad, 1))
        pad_vel = np.zeros((n_pad, 7), dtype=np.float32)
        pad_acc = np.zeros((n_pad, 7), dtype=np.float32)
        
        # Append padding
        result["positions"] = np.concatenate([result["positions"], pad_pos], axis=0)
        result["velocities"] = np.concatenate([result["velocities"], pad_vel], axis=0)
        result["accelerations"] = np.concatenate([result["accelerations"], pad_acc], axis=0)

        # Compute actions
        actions = compute_actions(
            result["positions"],
            result["velocities"],
            result["accelerations"]
        )

        T = len(result["positions"])

        # Store demo
        demo = {
            "joint_pos": result["positions"].astype(np.float32),
            "joint_vel": result["velocities"].astype(np.float32),
            "actions": actions,
            "dt": result["dt"],
            "q_start": q_start,
            "q_goal": q_goal,
        }

        # Add goal_q observation (replicated for each timestep)
        if args.goal_conditioned:
            demo["goal_q"] = np.tile(q_goal, (T, 1)).astype(np.float32)

        demos.append(demo)

        pbar.update(1)
        pbar.set_postfix({"failed": failed_count, "T": T})

    pbar.close()

    # Save dataset
    save_dataset(demos, args.output, goal_conditioned=args.goal_conditioned)

    # Summary
    total_samples = sum(len(d["actions"]) for d in demos)
    avg_len = total_samples / len(demos)

    print(f"\n{'='*60}")
    print("Collection Complete!")
    print(f"{'='*60}")
    print(f"  Demos collected: {len(demos)}")
    print(f"  Failed attempts: {failed_count}")
    print(f"  Total samples: {total_samples}")
    print(f"  Average trajectory length: {avg_len:.1f}")
    print(f"  Fixed start: {args.fixed_start}")
    print(f"  Goal-conditioned: {args.goal_conditioned}")
    print(f"  Output: {args.output}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
