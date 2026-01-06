#!/usr/bin/env python3
# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone trajectory visualization for GF-FM.

NO Isaac Lab dependencies - uses matplotlib and optionally PyBullet.

Usage:
    # Visualize generated dataset (matplotlib)
    python visualize_trajectory.py --dataset ./datasets/franka_2nd_order.hdf5 --demo_idx 0

    # Visualize with PyBullet 3D
    python visualize_trajectory.py --dataset ./datasets/franka_2nd_order.hdf5 --demo_idx 0 --pybullet

    # Visualize NPZ trajectories from play_standalone.py
    python visualize_trajectory.py --npz ./results/trajectories.npz --traj_idx 0
"""

import argparse
import os
import sys
import numpy as np

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize GF-FM trajectories (standalone)")
    parser.add_argument("--dataset", type=str, default=None, help="HDF5 dataset path")
    parser.add_argument("--npz", type=str, default=None, help="NPZ trajectories path (from play_standalone.py)")
    parser.add_argument("--demo_idx", type=int, default=0, help="Demo index to visualize (for HDF5)")
    parser.add_argument("--traj_idx", type=int, default=0, help="Trajectory index (for NPZ)")
    parser.add_argument("--pybullet", action="store_true", help="Use PyBullet for 3D visualization")
    parser.add_argument("--save_plot", type=str, default=None, help="Save plot to file instead of showing")
    parser.add_argument("--save_video", type=str, default=None, help="Save PyBullet video to file")
    parser.add_argument("--playback_speed", type=float, default=0.5, help="Playback speed for PyBullet")
    return parser.parse_args()


# Franka joint limits
FRANKA_JOINT_LIMITS = {
    "lower": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
    "upper": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]),
}

FRANKA_JOINT_NAMES = [
    "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
    "panda_joint5", "panda_joint6", "panda_joint7"
]


def load_from_hdf5(hdf5_path: str, demo_idx: int) -> dict:
    """Load trajectory from HDF5 dataset."""
    import h5py

    with h5py.File(hdf5_path, "r") as f:
        num_demos = f.attrs.get("num_demos", len(f["data"]))
        print(f"[Dataset] {hdf5_path}")
        print(f"  Total demos: {num_demos}")

        if demo_idx >= num_demos:
            raise ValueError(f"demo_idx {demo_idx} >= num_demos {num_demos}")

        demo_grp = f[f"data/demo_{demo_idx}"]
        dt = demo_grp.attrs.get("dt", 0.02)

        joint_pos = demo_grp["obs/joint_pos"][:]
        joint_vel = demo_grp["obs/joint_vel"][:]
        actions = demo_grp["actions"][:]

        # Actions are [q_dot, q_ddot]
        joint_acc = actions[:, 7:] if actions.shape[1] == 14 else np.zeros_like(joint_vel)

    return {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_acc": joint_acc,
        "dt": dt,
    }


def load_from_npz(npz_path: str, traj_idx: int) -> dict:
    """Load trajectory from NPZ file (from play_standalone.py).

    Supports two formats:
    1. Single-shot mode: trajectories array (num_traj, T, 7)
    2. Receding horizon mode: traj_0, traj_1, ... with variable lengths
    """
    data = np.load(npz_path, allow_pickle=True)

    print(f"[NPZ] {npz_path}")

    # Check format: receding horizon vs single-shot
    if "num_trajectories" in data:
        # Receding horizon format
        num_traj = int(data["num_trajectories"])
        print(f"  Format: RECEDING HORIZON")
        print(f"  Total trajectories: {num_traj}")

        if traj_idx >= num_traj:
            raise ValueError(f"traj_idx {traj_idx} >= num_trajectories {num_traj}")

        joint_pos = data[f"traj_{traj_idx}"]  # (T, 7) - variable length
        q_goal = data[f"goal_{traj_idx}"]
        reached = bool(data[f"reached_{traj_idx}"])
        steps = int(data[f"steps_{traj_idx}"])

        print(f"  Trajectory {traj_idx}: {joint_pos.shape[0]} steps")
        print(f"  Goal reached: {reached}")
        print(f"  Goal position: [{', '.join(f'{g:.2f}' for g in q_goal)}]")

    elif "trajectories" in data:
        # Single-shot format
        trajectories = data["trajectories"]  # (num_traj, T, 7)
        print(f"  Format: SINGLE-SHOT")
        print(f"  Total trajectories: {trajectories.shape[0]}")
        print(f"  Trajectory length: {trajectories.shape[1]}")

        if traj_idx >= trajectories.shape[0]:
            raise ValueError(f"traj_idx {traj_idx} >= num_trajectories {trajectories.shape[0]}")

        joint_pos = trajectories[traj_idx]  # (T, 7)
        q_goal = None
        reached = None
        steps = None

    else:
        raise ValueError(f"Unknown NPZ format. Keys: {list(data.keys())}")

    # Compute velocity and acceleration from positions
    dt = 0.02  # Default dt
    joint_vel = np.gradient(joint_pos, dt, axis=0)
    joint_acc = np.gradient(joint_vel, dt, axis=0)

    result = {
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "joint_acc": joint_acc,
        "dt": dt,
    }

    # Add goal info if available (receding horizon)
    if q_goal is not None:
        result["q_goal"] = q_goal
        result["reached_goal"] = reached
        result["total_steps"] = steps

    return result


def plot_trajectory(data: dict, save_path: str = None):
    """Plot trajectory with matplotlib."""
    import matplotlib.pyplot as plt

    joint_pos = data["joint_pos"]
    joint_vel = data["joint_vel"]
    joint_acc = data["joint_acc"]
    dt = data["dt"]

    T = joint_pos.shape[0]
    timesteps = np.arange(T) * dt

    fig, axs = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    colors = plt.cm.tab10(np.linspace(0, 1, 7))

    # Joint positions
    for i in range(7):
        axs[0].plot(timesteps, joint_pos[:, i], label=f"Joint {i+1}", color=colors[i])
        # Plot joint limits
        axs[0].axhline(y=FRANKA_JOINT_LIMITS["lower"][i], color=colors[i], linestyle="--", alpha=0.3)
        axs[0].axhline(y=FRANKA_JOINT_LIMITS["upper"][i], color=colors[i], linestyle="--", alpha=0.3)
    axs[0].set_ylabel("Position (rad)")
    axs[0].set_title("Joint Positions with Limits")
    axs[0].legend(loc="upper right", fontsize=8, ncol=4)
    axs[0].grid(True, alpha=0.3)

    # Joint velocities
    for i in range(7):
        axs[1].plot(timesteps, joint_vel[:, i], label=f"Joint {i+1}", color=colors[i])
    axs[1].set_ylabel("Velocity (rad/s)")
    axs[1].set_title("Joint Velocities")
    axs[1].grid(True, alpha=0.3)

    # Joint accelerations
    for i in range(7):
        axs[2].plot(timesteps, joint_acc[:, i], label=f"Joint {i+1}", color=colors[i])
    axs[2].set_ylabel("Acceleration (rad/s²)")
    axs[2].set_title("Joint Accelerations")
    axs[2].grid(True, alpha=0.3)

    # Joint limit violations
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]
    violations = np.sum((joint_pos < lower) | (joint_pos > upper), axis=1)
    axs[3].bar(timesteps, violations, width=dt * 0.8, color="red", alpha=0.7)
    axs[3].set_ylabel("Violations")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_title(f"Joint Limit Violations (Total: {int(np.sum(violations))})")
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[Plot] Saved to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_pybullet(data: dict, playback_speed: float = 1.0, save_video: str = None):
    """Visualize trajectory in PyBullet with optional goal visualization."""
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        print("[Error] PyBullet not installed. Install with: pip install pybullet")
        return

    joint_pos = data["joint_pos"]
    dt = data["dt"]
    q_goal = data.get("q_goal", None)  # Goal position (if available)

    # Connect to PyBullet
    if save_video:
        p.connect(p.GUI, options=f"--mp4={save_video}")
    else:
        p.connect(p.GUI)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load ground plane
    p.loadURDF("plane.urdf")

    # Try to find Franka URDF
    urdf_paths = [
        "/workspace/isaaclab/src/nvidia-curobo/src/curobo/content/assets/robot/franka_description/franka_panda.urdf",
        os.path.expanduser("~/.local/share/ov/pkg/isaac-sim-4.2.0/extscache/omni.importer.urdf-106.1.2+cp310/data/urdf/robots/franka_description/robots/panda_arm.urdf"),
    ]

    robot_urdf = None
    for path in urdf_paths:
        if os.path.exists(path):
            robot_urdf = path
            break

    if robot_urdf is None:
        # Try pybullet_data Franka
        robot_urdf = os.path.join(pybullet_data.getDataPath(), "franka_panda/panda.urdf")
        if not os.path.exists(robot_urdf):
            print("[Error] Could not find Franka URDF")
            p.disconnect()
            return

    print(f"[PyBullet] Loading URDF: {robot_urdf}")

    # Load main robot (blue - current trajectory)
    robot_id = p.loadURDF(
        robot_urdf,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )

    # Color main robot blue
    for i in range(p.getNumJoints(robot_id)):
        p.changeVisualShape(
            robot_id, i,
            rgbaColor=[0.0, 0.0, 1.0, 1.0]  # Blue
        )
    p.changeVisualShape(robot_id, -1, rgbaColor=[0.0, 0.0, 1.0, 1.0])

    # Get joint info
    num_joints = p.getNumJoints(robot_id)
    joint_indices = []
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_type = joint_info[2]
        if joint_type == p.JOINT_REVOLUTE:
            joint_indices.append(i)
            if len(joint_indices) >= 7:
                break

    print(f"[PyBullet] Found {len(joint_indices)} revolute joints")

    # Load goal robot (green transparent ghost) if goal is available
    goal_robot_id = None
    if q_goal is not None:
        print(f"[PyBullet] Loading goal robot (green ghost)")

        goal_robot_id = p.loadURDF(
            robot_urdf,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Disable collisions for goal robot (visual only)
        for i in range(p.getNumJoints(goal_robot_id)):
            p.setCollisionFilterGroupMask(goal_robot_id, i, 0, 0)

        # Set goal robot to goal position
        for i, joint_idx in enumerate(joint_indices[:7]):
            p.resetJointState(goal_robot_id, joint_idx, q_goal[i])

        # Make goal robot transparent green
        for i in range(p.getNumJoints(goal_robot_id)):
            p.changeVisualShape(
                goal_robot_id, i,
                rgbaColor=[0.2, 0.8, 0.2, 0.4]  # Green, 40% opacity
            )
        # Base link
        p.changeVisualShape(
            goal_robot_id, -1,
            rgbaColor=[0.2, 0.8, 0.2, 0.4]
        )

    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.3, 0, 0.4]
    )

    # Disable rendering during setup
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

    # Add trajectory info text
    goal_info = ""
    if q_goal is not None:
        reached = data.get("reached_goal", False)
        goal_info = f" | Goal: {'REACHED' if reached else 'NOT REACHED'}"

    text_id = p.addUserDebugText(
        f"Trajectory: {joint_pos.shape[0]} steps, dt={dt:.3f}s{goal_info}",
        [0, 0, 1.0],
        textColorRGB=[0, 0, 0],
        textSize=1.5
    )

    # Add legend
    if q_goal is not None:
        p.addUserDebugText(
            "Green = Goal | Blue = Current",
            [0, 0, 1.1],
            textColorRGB=[0.3, 0.3, 0.3],
            textSize=1.0
        )

    # Playback loop
    print(f"[PyBullet] Playing trajectory ({joint_pos.shape[0]} steps)...")
    print("  Press 'q' to quit")

    import time
    step_time = dt / playback_speed

    for t in range(joint_pos.shape[0]):
        q = joint_pos[t]

        # Set joint positions
        for i, joint_idx in enumerate(joint_indices[:7]):
            p.resetJointState(robot_id, joint_idx, q[i])

        # Reset goal robot position every frame to prevent gravity sagging
        if goal_robot_id is not None:
             for i, joint_idx in enumerate(joint_indices[:7]):
                p.resetJointState(goal_robot_id, joint_idx, q_goal[i])

        # Update step text
        p.addUserDebugText(
            f"Step: {t+1}/{joint_pos.shape[0]}",
            [0, 0, 0.9],
            textColorRGB=[0, 0, 1],
            textSize=1.2,
            replaceItemUniqueId=text_id if t > 0 else -1
        )

        p.stepSimulation()
        time.sleep(step_time)

        # Check for quit
        keys = p.getKeyboardEvents()
        if ord('q') in keys:
            break

    print("[PyBullet] Done. Press any key to close...")
    input()
    p.disconnect()


def main():
    args = parse_args()

    if args.dataset is None and args.npz is None:
        print("[Error] Must specify --dataset (HDF5) or --npz")
        return

    # Load trajectory
    if args.dataset:
        data = load_from_hdf5(args.dataset, args.demo_idx)
    else:
        data = load_from_npz(args.npz, args.traj_idx)

    print(f"\n[Trajectory Info]")
    print(f"  Length: {data['joint_pos'].shape[0]} steps")
    print(f"  Duration: {data['joint_pos'].shape[0] * data['dt']:.2f}s")
    print(f"  dt: {data['dt']:.4f}s")

    # Check joint limits
    lower = FRANKA_JOINT_LIMITS["lower"]
    upper = FRANKA_JOINT_LIMITS["upper"]
    violations = np.sum((data["joint_pos"] < lower) | (data["joint_pos"] > upper))
    print(f"  Joint limit violations: {violations}")

    # Plot trajectory
    print("\n[Plotting trajectory...]")
    plot_trajectory(data, args.save_plot)

    # PyBullet visualization
    if args.pybullet:
        print("\n[Starting PyBullet visualization...]")
        visualize_pybullet(data, args.playback_speed, args.save_video)


if __name__ == "__main__":
    main()
