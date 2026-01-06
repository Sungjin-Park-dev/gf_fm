# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Dataset loader for 2nd-order Flow Matching on (q, q_dot) states."""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


class SecondOrderFlowDataset(Dataset):
    """Dataset for 2nd-order Flow Matching on (q, q_dot) states.

    State representation: x = [q, q_dot] in R^14 (for 7-DOF robot)
    Velocity field: v = [q_dot, q_ddot] in R^14

    Flow Matching learns v_theta(x, t) that transports noise to data.

    HDF5 Structure (from generate_2nd_order_dataset.py):
        /data/demo_0/obs/joint_pos: (T, 7)
        /data/demo_0/obs/joint_vel: (T, 7)
        /data/demo_0/obs/eef_pos: (T, 3)
        /data/demo_0/obs/eef_quat: (T, 4)
        /data/demo_0/obs/obstacle_pos: (T, K*3)
        /data/demo_0/actions: (T, 14)  # [q_dot, q_ddot]

    Returns:
        dict with keys:
            - 'obs': dict of observation tensors for conditioning
            - 'state': 2nd-order state [q, q_dot] for flow matching
            - 'action': velocity field [q_dot, q_ddot] as target
    """

    def __init__(
        self,
        dataset_path: str,
        horizon: int = 16,
        n_obs_steps: int = 2,
        n_action_steps: int = 8,
        obs_keys: List[str] = None,
        joint_dim: int = 7,
        pad_before: int = 0,
        pad_after: int = 0,
    ):
        """Initialize SecondOrderFlowDataset.

        Args:
            dataset_path: Path to HDF5 file
            horizon: Length of trajectory sequence (default: 16)
            n_obs_steps: Number of observation timesteps for conditioning (default: 2)
            n_action_steps: Number of action steps to execute (default: 8)
            obs_keys: List of observation keys for conditioning
            joint_dim: Dimension of joint space (default: 7 for Franka)
            pad_before: Padding before sequence
            pad_after: Padding after sequence
        """
        super().__init__()

        self.dataset_path = dataset_path
        self.horizon = horizon
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.joint_dim = joint_dim
        self.pad_before = pad_before
        self.pad_after = pad_after

        # Default observation keys for conditioning
        if obs_keys is None:
            obs_keys = ['eef_pos', 'eef_quat', 'obstacle_pos']
        self.obs_keys = obs_keys

        # State and action dimensions
        self.state_dim = 2 * joint_dim  # [q, q_dot]
        self.action_dim = 2 * joint_dim  # [q_dot, q_ddot]

        # Open HDF5 file
        self.hdf5_file = h5py.File(dataset_path, 'r', swmr=True)

        # Build sequence index
        self.demo_names = sorted(list(self.hdf5_file['data'].keys()))
        self.sequence_indices = []

        for demo_idx, demo_name in enumerate(self.demo_names):
            demo_group = self.hdf5_file[f'data/{demo_name}']
            demo_len = demo_group['actions'].shape[0]

            # Valid starting indices
            required_length = max(n_obs_steps, horizon)
            for start_idx in range(demo_len - required_length + 1):
                self.sequence_indices.append((demo_idx, start_idx))

        print(f"[SecondOrderFlowDataset] Loaded {len(self.demo_names)} demos")
        print(f"[SecondOrderFlowDataset] Total sequences: {len(self.sequence_indices)}")
        print(f"[SecondOrderFlowDataset] state_dim={self.state_dim}, action_dim={self.action_dim}")

    def __len__(self) -> int:
        return len(self.sequence_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sequence sample.

        Returns:
            dict with keys:
                - 'obs': dict of observation tensors for conditioning
                - 'state': 2nd-order state [q, q_dot] (horizon, 14)
                - 'action': velocity field [q_dot, q_ddot] (horizon, 14)
        """
        demo_idx, start_idx = self.sequence_indices[idx]
        demo_name = self.demo_names[demo_idx]
        demo_group = self.hdf5_file[f'data/{demo_name}']

        # Load observations for conditioning (n_obs_steps timesteps)
        obs_dict = {}
        for key in self.obs_keys:
            if f'obs/{key}' in demo_group:
                obs_data = demo_group[f'obs/{key}'][start_idx:start_idx + self.n_obs_steps]
                obs_dict[key] = torch.from_numpy(np.array(obs_data)).float()

        # Always include joint_pos and joint_vel in obs for state encoder
        joint_pos = demo_group['obs/joint_pos'][start_idx:start_idx + self.n_obs_steps]
        joint_vel = demo_group['obs/joint_vel'][start_idx:start_idx + self.n_obs_steps]
        obs_dict['joint_pos'] = torch.from_numpy(np.array(joint_pos)).float()
        obs_dict['joint_vel'] = torch.from_numpy(np.array(joint_vel)).float()

        # Load state: [q, q_dot] for horizon timesteps
        joint_pos_horizon = demo_group['obs/joint_pos'][start_idx:start_idx + self.horizon]
        joint_vel_horizon = demo_group['obs/joint_vel'][start_idx:start_idx + self.horizon]

        joint_pos_horizon = np.array(joint_pos_horizon)
        joint_vel_horizon = np.array(joint_vel_horizon)

        # Construct 2nd-order state: [q, q_dot]
        state = np.concatenate([joint_pos_horizon, joint_vel_horizon], axis=-1)
        state = torch.from_numpy(state).float()  # (horizon, 14)

        # Load actions: velocity field [q_dot, q_ddot]
        actions = demo_group['actions'][start_idx:start_idx + self.horizon]
        actions = torch.from_numpy(np.array(actions)).float()  # (horizon, 14)

        return {
            'obs': obs_dict,
            'state': state,
            'action': actions,
        }

    def get_all_states(self) -> np.ndarray:
        """Get all states from all demos for normalization.

        Returns:
            all_states: (N, 14) array of [q, q_dot] states
        """
        all_states = []
        for demo_name in self.demo_names:
            demo_group = self.hdf5_file[f'data/{demo_name}']
            joint_pos = demo_group['obs/joint_pos'][:]
            joint_vel = demo_group['obs/joint_vel'][:]
            state = np.concatenate([joint_pos, joint_vel], axis=-1)
            all_states.append(state)
        return np.concatenate(all_states, axis=0)

    def get_all_actions(self) -> np.ndarray:
        """Get all actions (velocity fields) for normalization.

        Returns:
            all_actions: (N, 14) array of [q_dot, q_ddot]
        """
        all_actions = []
        for demo_name in self.demo_names:
            actions = self.hdf5_file[f'data/{demo_name}/actions'][:]
            all_actions.append(actions)
        return np.concatenate(all_actions, axis=0)

    def get_all_observations(self) -> Dict[str, np.ndarray]:
        """Get all observations for normalization.

        Returns:
            dict with observation arrays
        """
        # Include joint_pos and joint_vel
        all_keys = list(set(self.obs_keys) | {'joint_pos', 'joint_vel'})
        all_obs = {key: [] for key in all_keys}

        for demo_name in self.demo_names:
            demo_group = self.hdf5_file[f'data/{demo_name}']
            for key in all_keys:
                if f'obs/{key}' in demo_group:
                    obs_data = demo_group[f'obs/{key}'][:]
                    all_obs[key].append(obs_data)

        for key in list(all_obs.keys()):
            if len(all_obs[key]) > 0:
                all_obs[key] = np.concatenate(all_obs[key], axis=0)
            else:
                del all_obs[key]

        return all_obs

    def get_normalizer_stats(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Compute normalization statistics.

        Returns:
            stats dict with min/max for state, action, and observations
        """
        stats = {}

        # State stats [q, q_dot]
        all_states = self.get_all_states()
        stats['state'] = {
            'min': torch.from_numpy(all_states.min(axis=0)).float(),
            'max': torch.from_numpy(all_states.max(axis=0)).float(),
        }

        # Action stats [q_dot, q_ddot]
        all_actions = self.get_all_actions()
        stats['action'] = {
            'min': torch.from_numpy(all_actions.min(axis=0)).float(),
            'max': torch.from_numpy(all_actions.max(axis=0)).float(),
        }

        # Observation stats
        all_obs = self.get_all_observations()
        for key, data in all_obs.items():
            stats[key] = {
                'min': torch.from_numpy(data.min(axis=0)).float(),
                'max': torch.from_numpy(data.max(axis=0)).float(),
            }

        return stats

    def __del__(self):
        """Close HDF5 file on deletion."""
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()


def get_shape_meta_from_dataset(
    dataset_path: str,
    obs_keys: List[str] = None,
    joint_dim: int = 7,
) -> Dict:
    """Extract shape metadata from HDF5 dataset.

    Args:
        dataset_path: Path to HDF5 file
        obs_keys: List of observation keys
        joint_dim: Joint space dimension

    Returns:
        shape_meta dict
    """
    with h5py.File(dataset_path, 'r') as f:
        first_demo = list(f['data'].keys())[0]
        demo_group = f[f'data/{first_demo}']

        shape_meta = {
            'obs': {},
            'state': {'shape': (2 * joint_dim,)},  # [q, q_dot]
            'action': {'shape': (2 * joint_dim,)},  # [q_dot, q_ddot]
        }

        # Observation shapes
        if obs_keys is None:
            obs_keys = ['joint_pos', 'joint_vel', 'eef_pos', 'eef_quat', 'obstacle_pos']

        for obs_key in obs_keys:
            if f'obs/{obs_key}' in demo_group:
                obs_shape = demo_group[f'obs/{obs_key}'].shape[1:]
                shape_meta['obs'][obs_key] = {'shape': obs_shape}

    return shape_meta
