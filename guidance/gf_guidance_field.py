# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Geometric Fabrics guidance field for FM inference (STANDALONE - no Isaac Lab).

Uses FABRICS with Warp/PyTorch directly for collision avoidance guidance.
Supports joint limit repulsion and sphere obstacle avoidance.
No Isaac Lab or Isaac Sim dependencies.
"""

from typing import Dict, List, Optional, Tuple
import torch
import sys
import os

# Add FABRICS path
_fabrics_src = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'FABRICS', 'src')
if _fabrics_src not in sys.path:
    sys.path.insert(0, _fabrics_src)

try:
    import warp as wp
except ImportError:
    wp = None


class GFGuidanceField:
    """Geometric Fabrics guidance field for 2nd-order FM inference (standalone).

    Uses FABRICS standalone (no Isaac Lab) to compute repulsion forces.
    At each ODE step, computes joint-space accelerations that steer
    trajectories away from joint limits and obstacles.

    Supports:
        - Joint limit repulsion
        - Sphere obstacle avoidance (runtime configurable)

    Guidance equation:
        v_final = v_FM + lambda * v_GF

    Args:
        batch_size: Number of parallel trajectories
        device: Torch device ('cuda:0', etc.)
        joint_dim: Joint space dimension (7 for Franka)
        timestep: Integration timestep
        guidance_scale: Scale factor for guidance (lambda)
        max_guidance_strength: Maximum clamp for guidance
        max_spheres: Maximum number of sphere obstacles
    """

    def __init__(
        self,
        batch_size: int = 1,
        device: str = 'cuda:0',
        joint_dim: int = 7,
        timestep: float = 0.02,
        guidance_scale: float = 1.0,
        max_guidance_strength: float = 5.0,
        max_spheres: int = 20,
        debug: bool = False,
        debug_every: int = 50,
    ):
        self.batch_size = batch_size
        self.device = device
        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim  # [q, q_dot]
        self.timestep = timestep
        self.guidance_scale = guidance_scale
        self.max_guidance_strength = max_guidance_strength
        self.max_spheres = max_spheres
        self.enabled = True
        self.debug = debug
        self.debug_every = max(1, int(debug_every))
        self._debug_counter = 0

        # Lazy initialization
        self._fabric = None
        self._integrator = None
        self._initialized = False

        # Sphere obstacle world model
        self._world_model = None
        self._sphere_centers = None
        self._sphere_radii = None

    def _lazy_init(self):
        """Lazy initialization of FABRICS components."""
        if self._initialized:
            return

        try:
            from guidance.franka_fabric import FrankaPandaRepulsionFabric
            from fabrics_sim.integrator.integrators import DisplacementIntegrator
            from fabrics_sim.utils.utils import initialize_warp

            # Extract device index
            device_idx = self.device.split(':')[-1] if ':' in self.device else '0'

            # Initialize Warp
            initialize_warp(device_idx)

            # Create Franka fabric (joint limit + body sphere repulsion)
            self._fabric = FrankaPandaRepulsionFabric(
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

            # Build default config (middle of joint range)
            default_config = 0.5 * (self._joint_mins + self._joint_maxs)
            self._default_config = default_config.unsqueeze(0).repeat(self.batch_size, 1)

            self._initialized = True
            if self.debug:
                import guidance.body_sphere_repulsion as repulsion_mod
                import guidance.franka_fabric as fabric_mod
                print(f"[GFGuidanceField] Debug: fabric module={fabric_mod.__file__}")
                print(f"[GFGuidanceField] Debug: repulsion module={repulsion_mod.__file__}")
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
            
            # Update world model batch size if it exists
            if self._world_model is not None:
                self._world_model.update_batch_size(B)

            if self._fabric is None:
                return torch.zeros_like(z)

        # Get q, q_dot from explicit values or use defaults
        if q is None:
            # Use middle of joint range as default
            q = self._default_config[:B].clone()
        if q_dot is None:
            q_dot = torch.zeros(B, self.joint_dim, device=self.device)

        # Ensure shapes
        if q.dim() == 1:
            q = q.unsqueeze(0)
        if q_dot.dim() == 1:
            q_dot = q_dot.unsqueeze(0)

        # Use current position as target for c-space attractor
        target = q.clone()

        try:
            # Get obstacle data from world model
            if self._world_model is not None and self._world_model.sphere_count > 0:
                object_ids, object_indicator = self._world_model.get_object_ids()
            else:
                # Empty obstacles - create minimal indicator arrays
                object_ids = wp.zeros((B, 1), dtype=wp.uint64, device=self.device)
                object_indicator = wp.zeros((B, 1), dtype=wp.uint64, device=self.device)

            # Set FABRICS features (joint limit + obstacle repulsion)
            self._fabric.set_features(
                q.detach(),
                q_dot.detach(),
                object_ids,
                object_indicator,
                cspace_target=target
            )
            self._debug_repulsion(q)

            # Get fabric outputs (metric, force)
            # The fabric internally computes repulsion from joint limits
            qdd_zeros = torch.zeros_like(q_dot)
            q_new, qd_new, qdd_new = self._integrator.step(
                q.detach(), q_dot.detach(), qdd_zeros, self.timestep
            )

            # Compute guidance acceleration
            # qdd_new contains the fabric-induced acceleration (repulsion from limits)
            a_guidance = qdd_new - qdd_zeros  # Pure guidance component

            # Scale by flow time - KEEP CONSTANT for obstacle avoidance
            time_scale = 1.0

            # Convert to velocity field guidance
            # For [q_dot, q_ddot] velocity field:
            # - guidance for q_dot: velocity correction (apply stronger impulse)
            # - guidance for q_ddot: direct acceleration
            dt = self.timestep
            # Use larger factor for velocity correction to ensure responsiveness
            v_q_guidance = 0.5 * a_guidance * time_scale  # (B, 7)
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

    def _debug_repulsion(self, q: torch.Tensor) -> None:
        if not self.debug:
            return
        if self._sphere_centers is None or self._sphere_centers.numel() == 0:
            return
        self._debug_counter += 1
        if self._debug_counter % self.debug_every != 0:
            return
        base_repulsion = getattr(self._fabric, "base_fabric_repulsion", None)
        if base_repulsion is None or base_repulsion.accel_dir is None:
            print("[GFGuidanceField] Debug: base repulsion not ready.")
            return
        accel_dir = base_repulsion.accel_dir.detach()
        signed_dist = base_repulsion.signed_distance.detach()
        if accel_dir.ndim != 3:
            print(f"[GFGuidanceField] Debug: unexpected accel_dir shape {accel_dir.shape}.")
            return
        batch_size, num_points, _ = accel_dir.shape
        body_pos, _ = self._fabric.get_taskmap("body_points")(q, None)
        body_pos = body_pos.view(batch_size, num_points, 3)
        centers = self._sphere_centers.to(body_pos.device, dtype=body_pos.dtype)
        diff = body_pos.unsqueeze(2) - centers.unsqueeze(0).unsqueeze(0)
        dist = torch.linalg.norm(diff, dim=-1)
        min_idx = dist.argmin(dim=-1)
        nearest_centers = centers[min_idx]
        away_vec = body_pos - nearest_centers
        dot = torch.sum(accel_dir * away_vec, dim=-1)
        engage_depth = self._fabric.fabric_params["body_repulsion"]["engage_depth"]
        active = signed_dist <= engage_depth
        if not active.any():
            print("[GFGuidanceField] Debug: no active repulsion points.")
            return
        dot_active = dot[active]
        neg_frac = (dot_active < 0).float().mean().item()
        min_dot = dot_active.min().item()
        max_dot = dot_active.max().item()
        print(
            "[GFGuidanceField] Debug: accel_dir·away_vec "
            f"min={min_dot:.4f} max={max_dot:.4f} neg_frac={neg_frac:.3f} "
            f"active_pts={dot_active.numel()}"
        )

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

    def set_sphere_obstacles(
        self,
        spheres: List[Tuple[Tuple[float, float, float], float]]
    ):
        """Set sphere obstacles for collision avoidance.

        Args:
            spheres: List of ((x, y, z), radius) tuples defining sphere obstacles
                     in world coordinates.

        Example:
            guidance_field.set_sphere_obstacles([
                ((0.5, 0.0, 0.4), 0.1),   # sphere at (0.5, 0, 0.4) with radius 0.1
                ((0.3, 0.2, 0.5), 0.08),  # another sphere
            ])
        """
        self._lazy_init()

        # Import here to avoid circular import
        from .sphere_obstacle import SphereWorldModel

        # Create world model if needed
        if self._world_model is None:
            self._world_model = SphereWorldModel(
                batch_size=self.batch_size,
                max_spheres=self.max_spheres,
                device=self.device
            )

        # Clear existing and add new spheres
        self._world_model.clear()
        for pos, radius in spheres:
            self._world_model.add_sphere(pos, radius)

        if spheres:
            centers = [pos for pos, _ in spheres]
            radii = [radius for _, radius in spheres]
            self._sphere_centers = torch.tensor(centers, device=self.device, dtype=torch.float32)
            self._sphere_radii = torch.tensor(radii, device=self.device, dtype=torch.float32)
        else:
            self._sphere_centers = None
            self._sphere_radii = None

        print(f"[GFGuidanceField] Set {len(spheres)} sphere obstacle(s)")

    def clear_obstacles(self):
        """Remove all sphere obstacles."""
        if self._world_model is not None:
            self._world_model.clear()
            self._sphere_centers = None
            self._sphere_radii = None
            print("[GFGuidanceField] Cleared all obstacles")

    @property
    def obstacle_count(self) -> int:
        """Number of active sphere obstacles."""
        if self._world_model is None:
            return 0
        return self._world_model.sphere_count

    @property
    def collision_status(self):
        """Check if any collision is detected.

        Returns:
            Tensor of collision status per batch, or None if not available
        """
        if self._fabric is None:
            return None
        if hasattr(self._fabric, 'collision_status'):
            return self._fabric.collision_status
        return None


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
