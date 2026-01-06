# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometric Fabrics guidance field for FM inference (STANDALONE - no Isaac Lab).

Uses FABRICS with Warp/PyTorch directly for collision avoidance guidance.
No Isaac Lab or Isaac Sim dependencies.
"""

from typing import Dict, Optional
import torch
import sys
import os

# Add FABRICS path
_fabrics_src = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'FABRICS', 'src')
if _fabrics_src not in sys.path:
    sys.path.insert(0, _fabrics_src)


class GFGuidanceField:
    """Geometric Fabrics guidance field for 2nd-order FM inference (standalone).

    Uses FABRICS standalone (no Isaac Lab) to compute repulsion forces.
    At each ODE step, computes joint-space accelerations that steer
    trajectories away from joint limits.

    Guidance equation:
        v_final = v_FM + lambda * v_GF

    Args:
        batch_size: Number of parallel trajectories
        device: Torch device ('cuda:0', etc.)
        joint_dim: Joint space dimension (7 for Franka)
        timestep: Integration timestep
        guidance_scale: Scale factor for guidance (lambda)
        max_guidance_strength: Maximum clamp for guidance
    """

    def __init__(
        self,
        batch_size: int = 1,
        device: str = 'cuda:0',
        joint_dim: int = 7,
        timestep: float = 0.02,
        guidance_scale: float = 1.0,
        max_guidance_strength: float = 5.0,
    ):
        self.batch_size = batch_size
        self.device = device
        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim  # [q, q_dot]
        self.timestep = timestep
        self.guidance_scale = guidance_scale
        self.max_guidance_strength = max_guidance_strength
        self.enabled = True

        # Lazy initialization
        self._fabric = None
        self._integrator = None
        self._initialized = False

    def _lazy_init(self):
        """Lazy initialization of FABRICS components."""
        if self._initialized:
            return

        try:
            from fabrics_sim.fabrics.franka_panda_cspace_fabric import FrankaPandaCspaceFabric
            from fabrics_sim.integrator.integrators import DisplacementIntegrator
            from fabrics_sim.utils.utils import initialize_warp

            # Extract device index
            device_idx = self.device.split(':')[-1] if ':' in self.device else '0'

            # Initialize Warp
            initialize_warp(device_idx)

            # Create Franka fabric (joint-space with joint limit repulsion)
            self._fabric = FrankaPandaCspaceFabric(
                batch_size=self.batch_size,
                device=self.device,
                timestep=self.timestep,
                graph_capturable=False,  # Flexibility for varying batch sizes
            )

            # Create integrator
            self._integrator = DisplacementIntegrator(self._fabric)

            # Get joint limits
            self._joint_mins = []
            self._joint_maxs = []
            for joint in self._fabric.urdfpy_robot.joints:
                if joint.joint_type == "revolute":
                    if joint.limit is None:
                        self._joint_mins.append(-3.1416)
                        self._joint_maxs.append(3.1416)
                    else:
                        self._joint_mins.append(joint.limit.lower)
                        self._joint_maxs.append(joint.limit.upper)

            self._joint_mins = torch.tensor(self._joint_mins, device=self.device)
            self._joint_maxs = torch.tensor(self._joint_maxs, device=self.device)

            self._initialized = True
            print(f"[GFGuidanceField] Initialized FABRICS standalone (batch={self.batch_size})")

        except ImportError as e:
            print(f"[GFGuidanceField] Warning: Could not import FABRICS: {e}")
            print("[GFGuidanceField] Guidance will return zeros.")
            self._initialized = True
            self._fabric = None

    def compute_guidance(
        self,
        z: torch.Tensor,
        t: float,
        normalizer=None,
        q: Optional[torch.Tensor] = None,
        q_dot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute GF guidance velocity.

        Args:
            z: Current ODE state (B, T, 14) or (B, 14) - velocity field
            t: Flow time [0, 1]
            normalizer: Optional normalizer for denormalization
            q: Optional explicit joint positions (B, 7)
            q_dot: Optional explicit joint velocities (B, 7)

        Returns:
            guidance: Same shape as z, guidance to add to FM prediction
        """
        if not self.enabled:
            return torch.zeros_like(z)

        self._lazy_init()

        if self._fabric is None:
            return torch.zeros_like(z)

        original_shape = z.shape
        B = z.shape[0]
        has_time_dim = z.dim() == 3

        # Handle batch size mismatch
        if B != self.batch_size:
            # Re-initialize with correct batch size
            self.batch_size = B
            self._initialized = False
            self._lazy_init()
            if self._fabric is None:
                return torch.zeros_like(z)

        # Get q, q_dot from explicit values or use defaults
        if q is None:
            # Use middle of joint range as default
            q = self._fabric.default_config[:B].clone()
        if q_dot is None:
            q_dot = torch.zeros(B, self.joint_dim, device=self.device)

        # Ensure shapes
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q_dot.dim() == 1:
            q_dot = q_dot.unsqueeze(0)

        # Use current position as target (this creates joint limit repulsion only)
        # For obstacle repulsion, would need to add world configuration
        target = q.clone()

        try:
            # Set FABRICS features
            self._fabric.set_features(target, q.detach(), q_dot.detach())

            # Get fabric outputs (metric, force)
            # The fabric internally computes repulsion from joint limits
            qdd_zeros = torch.zeros_like(q_dot)
            q_new, qd_new, qdd_new = self._integrator.step(
                q.detach(), q_dot.detach(), qdd_zeros, self.timestep
            )

            # Compute guidance acceleration
            # qdd_new contains the fabric-induced acceleration (repulsion from limits)
            a_guidance = qdd_new - qdd_zeros  # Pure guidance component

            # Scale by flow time (less guidance as t -> 1)
            time_scale = max(0.1, 1.0 - t)

            # Convert to velocity field guidance
            # For [q_dot, q_ddot] velocity field:
            # - guidance for q_dot: small position correction via velocity
            # - guidance for q_ddot: direct acceleration
            dt = self.timestep
            v_q_guidance = dt * a_guidance * time_scale  # (B, 7)
            v_qdot_guidance = a_guidance * time_scale    # (B, 7)

            # Combine into 14D guidance
            guidance_14d = torch.cat([v_q_guidance, v_qdot_guidance], dim=-1)  # (B, 14)

            # Apply scale and clamp
            guidance_14d = self.guidance_scale * guidance_14d
            guidance_14d = torch.clamp(
                guidance_14d,
                -self.max_guidance_strength,
                self.max_guidance_strength
            )

            # Expand to match input shape
            if has_time_dim:
                T = original_shape[1]
                guidance = guidance_14d.unsqueeze(1).expand(-1, T, -1)
            else:
                guidance = guidance_14d

            return guidance

        except Exception as e:
            print(f"[GFGuidanceField] Warning: Guidance failed: {e}")
            return torch.zeros_like(z)

    def compute_guidance_with_state(
        self,
        q: torch.Tensor,
        q_dot: torch.Tensor,
        t: float,
    ) -> torch.Tensor:
        """Compute GF guidance given explicit state.

        Args:
            q: Joint positions (B, 7)
            q_dot: Joint velocities (B, 7)
            t: Flow time

        Returns:
            guidance: (B, 14) guidance [v_q_guidance, v_qdot_guidance]
        """
        B = q.shape[0]
        z_dummy = torch.zeros(B, self.state_dim, device=self.device)
        return self.compute_guidance(z_dummy, t, None, q, q_dot)

    def get_joint_limits(self):
        """Get joint limits from FABRICS.

        Returns:
            (joint_mins, joint_maxs) tensors or None if not initialized
        """
        self._lazy_init()
        if self._fabric is None:
            return None, None
        return self._joint_mins, self._joint_maxs

    def set_enabled(self, enabled: bool):
        """Enable/disable guidance."""
        self.enabled = enabled

    def set_guidance_scale(self, scale: float):
        """Set guidance scale."""
        self.guidance_scale = scale

    def reset(self):
        """Reset guidance state."""
        pass  # Stateless


class SimpleJointLimitGuidance:
    """Simple joint limit guidance without full FABRICS.

    Lightweight fallback using basic repulsion from joint limits.
    """

    def __init__(
        self,
        batch_size: int = 1,
        device: str = 'cuda:0',
        joint_dim: int = 7,
        guidance_scale: float = 1.0,
        repulsion_threshold: float = 0.2,  # Radians from limit to start repulsion
        repulsion_gain: float = 5.0,
    ):
        self.batch_size = batch_size
        self.device = device
        self.joint_dim = joint_dim
        self.guidance_scale = guidance_scale
        self.repulsion_threshold = repulsion_threshold
        self.repulsion_gain = repulsion_gain
        self.enabled = True

        # Franka joint limits
        self.joint_mins = torch.tensor(
            [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
            device=device
        )
        self.joint_maxs = torch.tensor(
            [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
            device=device
        )

    def compute_guidance(
        self,
        z: torch.Tensor,
        t: float,
        normalizer=None,
        q: Optional[torch.Tensor] = None,
        q_dot: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute simple joint limit repulsion guidance."""
        if not self.enabled:
            return torch.zeros_like(z)

        original_shape = z.shape
        B = z.shape[0]
        has_time_dim = z.dim() == 3

        if q is None:
            # Cannot compute guidance without position
            return torch.zeros_like(z)

        if q.dim() == 1:
            q = q.unsqueeze(0)

        # Distance to limits
        dist_to_lower = q - self.joint_mins
        dist_to_upper = self.joint_maxs - q

        # Repulsion from lower limits (push positive)
        lower_repulsion = torch.zeros_like(q)
        lower_mask = dist_to_lower < self.repulsion_threshold
        lower_repulsion[lower_mask] = self.repulsion_gain * (
            self.repulsion_threshold - dist_to_lower[lower_mask]
        )

        # Repulsion from upper limits (push negative)
        upper_repulsion = torch.zeros_like(q)
        upper_mask = dist_to_upper < self.repulsion_threshold
        upper_repulsion[upper_mask] = -self.repulsion_gain * (
            self.repulsion_threshold - dist_to_upper[upper_mask]
        )

        # Combined acceleration guidance
        a_guidance = lower_repulsion + upper_repulsion

        # Scale by time
        time_scale = max(0.1, 1.0 - t)
        a_guidance = a_guidance * time_scale

        # Build 14D guidance
        dt = 0.02
        v_q = dt * a_guidance
        v_qdot = a_guidance

        guidance_14d = torch.cat([v_q, v_qdot], dim=-1)
        guidance_14d = self.guidance_scale * guidance_14d

        if has_time_dim:
            T = original_shape[1]
            guidance = guidance_14d.unsqueeze(1).expand(-1, T, -1)
        else:
            guidance = guidance_14d

        return guidance

    def set_enabled(self, enabled: bool):
        self.enabled = enabled
