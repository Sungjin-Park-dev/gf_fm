# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Guidance integration strategies for Flow Matching ODE.

Provides different methods for incorporating Geometric Fabrics guidance
into the FM ODE integration process:

- AdditiveGuidanceIntegrator: Original method - adds guidance to FM prediction then integrates
- ODECoupledGuidanceIntegrator: New method - sub-step integration with continuous guidance coupling

Usage:
    from guidance import GuidanceMode, AdditiveGuidanceIntegrator, ODECoupledGuidanceIntegrator

    # Select integrator based on mode
    if mode == GuidanceMode.ADDITIVE:
        integrator = AdditiveGuidanceIntegrator()
    else:
        integrator = ODECoupledGuidanceIntegrator(n_substeps=4)

    # Use in ODE loop
    z_next = integrator.integrate_step(z, tau, dt, fm_velocity, guidance_field, ...)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Dict, Any
import numpy as np
import torch


class GuidanceMode(Enum):
    """Guidance integration modes.

    ADDITIVE: Original method - pred = v_FM + guidance, then integrate once
    ODE_COUPLED: New method - sub-step integration with v_FM + lambda * Phi_GF
    """
    ADDITIVE = "additive"
    ODE_COUPLED = "ode_coupled"


class GuidanceIntegrator(ABC):
    """Abstract base class for guidance integration strategies.

    Subclasses implement different methods for combining FM velocity field
    with GF guidance during ODE integration.
    """

    @abstractmethod
    def integrate_step(
        self,
        z: torch.Tensor,
        tau: float,
        dt: float,
        fm_velocity: torch.Tensor,
        guidance_field: Any,  # GFGuidanceField
        obs_data: Dict[str, torch.Tensor],
        sde: Any,  # ConsistencyFM
        normalizer: Any,  # LinearNormalizer
        joint_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Perform one integration step with guidance.

        Args:
            z: Current ODE state (B, T, 14) in normalized space
            tau: Current flow time in [eps, 1-eps]
            dt: Step size (1/num_inference_steps)
            fm_velocity: FM network prediction (B, T, 14) in normalized space
            guidance_field: GFGuidanceField instance for computing guidance
            obs_data: Observation dict with 'joint_pos', 'joint_vel'
            sde: ConsistencyFM instance for sigma_t calculation
            normalizer: LinearNormalizer for space conversion
            joint_dim: Joint dimension (7 for Franka)
            device: Torch device

        Returns:
            z_next: Updated state after integration step (B, T, 14)
        """
        pass


class AdditiveGuidanceIntegrator(GuidanceIntegrator):
    """Original additive guidance method.

    Guidance is added to FM prediction before integration:
        pred_final = v_FM + guidance
        z_next = z + pred_final * dt + noise

    This is the existing implementation, extracted into a separate class
    for clarity and comparison with the ODE-coupled method.
    """

    def integrate_step(
        self,
        z: torch.Tensor,
        tau: float,
        dt: float,
        fm_velocity: torch.Tensor,
        guidance_field: Any,
        obs_data: Dict[str, torch.Tensor],
        sde: Any,
        normalizer: Any,
        joint_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        B, T, action_dim = z.shape

        # 1. Get initial q from observation
        if 'joint_pos' in obs_data:
            q_init = obs_data['joint_pos'][:, -1, :]  # (B, 7)
        else:
            q_init = torch.zeros(B, joint_dim, device=device)

        # 2. Unnormalize current state to get physical velocities
        z_unnorm = normalizer['action'].unnormalize(z)
        q_dot_est = z_unnorm[..., :joint_dim]  # (B, T, 7)

        # 3. Integrate to get q trajectory
        dt_traj = 0.02  # Standard trajectory timestep
        q_delta = torch.cumsum(q_dot_est * dt_traj, dim=1)
        q_traj = q_init.unsqueeze(1) + q_delta  # (B, T, 7)

        # 4. Compute guidance in physical space
        q_flat = q_traj.reshape(-1, joint_dim)
        q_dot_flat = q_dot_est.reshape(-1, joint_dim)

        guidance_flat = guidance_field.compute_guidance(
            z.view(-1, action_dim),
            t=tau,
            q=q_flat,
            q_dot=q_dot_flat
        )

        # 5. Reshape and normalize guidance
        guidance_phys = guidance_flat.view(B, T, -1)
        scale = normalizer['action'].params_dict['scale']
        guidance_norm = guidance_phys * scale

        # 6. Add guidance to FM prediction
        pred = fm_velocity + guidance_norm

        # 7. SDE correction and Euler step
        sigma_t = sde.sigma_t(tau)
        noise_scale = sde.noise_scale

        # Consistency FM correction term
        if noise_scale > 0 and (1.0 - tau) > 1e-6:
            correction = (sigma_t**2) / (2 * (noise_scale**2) * ((1.0 - tau)**2)) * \
                        (0.5 * tau * (1.0 - tau) * pred - 0.5 * (2.0 - tau) * z.detach())
            pred_sigma = pred + correction
        else:
            pred_sigma = pred

        # Euler step with noise
        noise = sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device) if sigma_t > 0 else 0
        z_next = z.detach().clone() + pred_sigma * dt + noise

        return z_next


class ODECoupledGuidanceIntegrator(GuidanceIntegrator):
    """ODE-coupled guidance with sub-step integration.

    The ODE is defined as:
        dx/dtau = v_FM(x, tau) + lambda(tau) * Phi_GF(x)

    Sub-step integration allows more accurate coupling between
    FM velocity and GF guidance field. At each sub-step, the guidance
    is recomputed based on the current state, allowing the two fields
    to interact continuously.

    Args:
        n_substeps: Number of sub-steps per main ODE step (default: 4)
        lambda_schedule: How lambda varies with tau ("constant", "linear_decay",
                        "linear_increase", "cosine")
        lambda_base: Base value for lambda (guidance scale)
    """

    def __init__(
        self,
        n_substeps: int = 4,
        lambda_schedule: str = "constant",
        lambda_base: float = 1.0,
    ):
        self.n_substeps = n_substeps
        self.lambda_schedule = lambda_schedule
        self.lambda_base = lambda_base

    def _compute_lambda(self, tau: float) -> float:
        """Compute time-varying guidance scale.

        Args:
            tau: Flow time in [0, 1]

        Returns:
            lambda_t: Guidance scale at time tau
        """
        if self.lambda_schedule == "constant":
            return self.lambda_base
        elif self.lambda_schedule == "linear_decay":
            # More guidance early, less later (let FM take over)
            return self.lambda_base * (1.0 - tau)
        elif self.lambda_schedule == "linear_increase":
            # Less guidance early, more later (safety priority)
            return self.lambda_base * tau
        elif self.lambda_schedule == "cosine":
            # Smooth cosine decay
            return self.lambda_base * 0.5 * (1 + np.cos(np.pi * tau))
        else:
            return self.lambda_base

    def integrate_step(
        self,
        z: torch.Tensor,
        tau: float,
        dt: float,
        fm_velocity: torch.Tensor,
        guidance_field: Any,
        obs_data: Dict[str, torch.Tensor],
        sde: Any,
        normalizer: Any,
        joint_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        B, T, action_dim = z.shape
        sub_dt = dt / self.n_substeps

        z_current = z.clone()

        # Get initial q from observation
        if 'joint_pos' in obs_data:
            q_init = obs_data['joint_pos'][:, -1, :]  # (B, 7)
        else:
            q_init = torch.zeros(B, joint_dim, device=device)

        dt_traj = 0.02  # Standard trajectory timestep
        scale = normalizer['action'].params_dict['scale']

        for s in range(self.n_substeps):
            sub_tau = tau + s * sub_dt
            lambda_t = self._compute_lambda(sub_tau)

            # 1. Unnormalize current state to physical space
            z_unnorm = normalizer['action'].unnormalize(z_current)
            q_dot_est = z_unnorm[..., :joint_dim]  # (B, T, 7)

            # 2. Integrate to get q trajectory from initial position
            q_delta = torch.cumsum(q_dot_est * dt_traj, dim=1)
            q_traj = q_init.unsqueeze(1) + q_delta  # (B, T, 7)

            # 3. Compute GF guidance at current state
            q_flat = q_traj.reshape(-1, joint_dim)
            q_dot_flat = q_dot_est.reshape(-1, joint_dim)

            guidance_flat = guidance_field.compute_guidance(
                z_current.view(-1, action_dim),
                t=sub_tau,
                q=q_flat,
                q_dot=q_dot_flat
            )

            guidance_phys = guidance_flat.view(B, T, -1)
            guidance_norm = guidance_phys * scale

            # 4. Combined velocity: v_FM + lambda * Phi_GF
            v_combined = fm_velocity + lambda_t * guidance_norm

            # 5. SDE correction term
            sigma_t = sde.sigma_t(sub_tau)
            noise_scale = sde.noise_scale

            if noise_scale > 0 and (1.0 - sub_tau) > 1e-6:
                correction = (sigma_t**2) / (2 * (noise_scale**2) * ((1.0 - sub_tau)**2)) * \
                            (0.5 * sub_tau * (1.0 - sub_tau) * v_combined -
                             0.5 * (2.0 - sub_tau) * z_current.detach())
                pred_sigma = v_combined + correction
            else:
                pred_sigma = v_combined

            # 6. Euler sub-step
            noise = sigma_t * np.sqrt(sub_dt) * torch.randn_like(pred_sigma).to(device) if sigma_t > 0 else 0
            z_current = z_current.detach().clone() + pred_sigma * sub_dt + noise

        return z_current


def create_guidance_integrator(
    mode: str = "additive",
    n_substeps: int = 4,
    lambda_schedule: str = "constant",
    lambda_base: float = 1.0,
) -> GuidanceIntegrator:
    """Factory function to create guidance integrator.

    Args:
        mode: "additive" or "ode_coupled"
        n_substeps: Number of sub-steps for ODE-coupled mode
        lambda_schedule: Lambda schedule type
        lambda_base: Base lambda value

    Returns:
        GuidanceIntegrator instance
    """
    guidance_mode = GuidanceMode(mode)

    if guidance_mode == GuidanceMode.ADDITIVE:
        return AdditiveGuidanceIntegrator()
    else:
        return ODECoupledGuidanceIntegrator(
            n_substeps=n_substeps,
            lambda_schedule=lambda_schedule,
            lambda_base=lambda_base,
        )
