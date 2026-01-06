import torch
import torch.nn as nn
from typing import List, Type, Dict


def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim: Dimension of the output vector
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return: List of nn.Module layers
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class StateOnlyEncoder(nn.Module):
    """
    State-only encoder for manipulation tasks with low-dimensional observations.

    Replaces the PointNet encoder from FlowPolicyReference with a simple MLP
    that processes low-dimensional state observations.

    Input: Dictionary with keys (flexible based on dataset):
        Default for PickPlace task:
        - 'joint_pos': (B, T, 9) or (B, 9) - Joint positions
        - 'joint_vel': (B, T, 9) or (B, 9) - Joint velocities
        - 'eef_pos': (B, T, 3) or (B, 3) - End-effector position
        - 'eef_quat': (B, T, 4) or (B, 4) - End-effector quaternion
        - 'cube_positions': (B, T, 9) or (B, 9) - Cube positions

    Output: (B, T, out_dim) or (B, out_dim) - Global conditioning vector
    """

    def __init__(
        self,
        obs_dim: int = 34,  # Default: 9 (joint_pos) + 9 (joint_vel) + 3 (eef_pos) + 4 (eef_quat) + 9 (cube_pos)
        hidden_dims: List[int] = [128, 256],
        out_dim: int = 128,
        activation_fn: Type[nn.Module] = nn.ReLU,
    ):
        """
        Initialize StateOnlyEncoder.

        Args:
            obs_dim: Total dimension of concatenated state observations (default: 19)
            hidden_dims: Hidden layer dimensions (default: [128, 256])
            out_dim: Output dimension (default: 128)
            activation_fn: Activation function class (default: nn.ReLU)
        """
        super().__init__()

        self.obs_dim = obs_dim
        self.out_dim = out_dim

        # Create MLP: obs_dim → hidden[0] → hidden[1] → ... → out_dim
        self.mlp = nn.Sequential(
            *create_mlp(
                input_dim=obs_dim,
                output_dim=out_dim,
                net_arch=hidden_dims,
                activation_fn=activation_fn,
                squash_output=False,
            )
        )

    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass: concatenate state observations and pass through MLP.

        Args:
            obs_dict: Dictionary with observation keys (flexible)
                Each value can be (B, obs_dim) or (B, T, obs_dim)
                Supported keys: 'eef_pos', 'eef_quat', 'gripper_pos', 'object',
                               'joint_pos', 'joint_vel', 'cube_positions', 'cube_orientations', 'pose_command'

        Returns:
            encoded: (B, out_dim) or (B, T, out_dim) - Encoded state representation
        """
        # Concatenate all observations in a consistent order
        # Order matters for consistent encoding
        obs_list = []

        # Priority: task-critical observations for manipulation
        if 'eef_pos' in obs_dict:
            obs_list.append(obs_dict['eef_pos'])
        if 'eef_quat' in obs_dict:
            obs_list.append(obs_dict['eef_quat'])
        if 'gripper_pos' in obs_dict:
            obs_list.append(obs_dict['gripper_pos'])
        if 'object' in obs_dict:
            obs_list.append(obs_dict['object'])

        # Legacy observations for backward compatibility
        if 'joint_pos' in obs_dict:
            obs_list.append(obs_dict['joint_pos'])
        if 'joint_vel' in obs_dict:
            obs_list.append(obs_dict['joint_vel'])
        if 'cube_positions' in obs_dict:
            obs_list.append(obs_dict['cube_positions'])
        if 'cube_orientations' in obs_dict:
            obs_list.append(obs_dict['cube_orientations'])

        # Fallback for Reach task (backward compatibility)
        if 'pose_command' in obs_dict:
            obs_list.append(obs_dict['pose_command'])

        # Concatenate all state observations along last dimension
        x = torch.cat(obs_list, dim=-1)

        # Pass through MLP
        encoded = self.mlp(x)

        return encoded

    def output_shape(self) -> int:
        """
        Return the output dimension of the encoder.

        This method matches the interface expected by FlowPolicy.

        Returns:
            out_dim: Output dimension (default: 128)
        """
        return self.out_dim
