# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""2nd-order Flow Matching Policy for (q, q_dot) state space.

State representation: x = [q, q_dot] in R^14 (for 7-DOF robot)
Velocity field: v = [q_dot, q_ddot] in R^14

The network learns v_theta(x, t, c) that transports noise to trajectory data.
At inference, GF guidance can be added to steer trajectories away from obstacles.
"""

from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
from termcolor import cprint

from common.sde_lib import ConsistencyFM
from model.normalizer import LinearNormalizer
from policy.base_policy import BasePolicy
from model.conditional_unet1d import ConditionalUnet1D
from model.mask_generator import LowdimMaskGenerator
from common.pytorch_util import dict_apply
from common.model_util import print_params
from model.state_encoder import StateOnlyEncoder


class SecondOrderFlowPolicy(BasePolicy):
    """2nd-order Flow Matching Policy for (q, q_dot) states.

    This policy learns a velocity field in the 2nd-order state space [q, q_dot].
    The velocity field v_theta = [v_q, v_qdot] represents:
        - v_q: rate of change of joint positions (should match q_dot)
        - v_qdot: rate of change of joint velocities (acceleration)

    During inference, GF guidance can be injected to ensure collision avoidance.

    Args:
        shape_meta: Dict with 'state' (14,) and 'action' (14,) shapes
        horizon: Length of trajectory to predict
        n_action_steps: Number of steps to execute
        n_obs_steps: Number of observation steps for conditioning
        joint_dim: Dimension of joint space (default: 7 for Franka)
        obs_as_global_cond: Use observations as global conditioning
        encoder_output_dim: Output dimension of observation encoder
        Conditional_ConsistencyFM: CFM hyperparameters
    """

    def __init__(
        self,
        shape_meta: dict,
        horizon: int,
        n_action_steps: int,
        n_obs_steps: int,
        joint_dim: int = 7,
        obs_as_global_cond: bool = True,
        diffusion_step_embed_dim: int = 256,
        down_dims: tuple = (256, 512, 1024),
        kernel_size: int = 5,
        n_groups: int = 8,
        condition_type: str = "film",
        use_down_condition: bool = True,
        use_mid_condition: bool = True,
        use_up_condition: bool = True,
        encoder_output_dim: int = 256,
        Conditional_ConsistencyFM: Optional[dict] = None,
        eta: float = 0.01,
        **kwargs
    ):
        super().__init__()

        self.joint_dim = joint_dim
        self.state_dim = 2 * joint_dim  # [q, q_dot]
        self.action_dim = 2 * joint_dim  # [q_dot, q_ddot] velocity field
        self.condition_type = condition_type

        # Parse shape_meta
        if 'state' in shape_meta:
            state_shape = shape_meta['state']['shape']
            self.state_dim = state_shape[0] if isinstance(state_shape, tuple) else state_shape

        if 'action' in shape_meta:
            action_shape = shape_meta['action']['shape']
            self.action_dim = action_shape[0] if isinstance(action_shape, tuple) else action_shape

        cprint(f"[SecondOrderFlowPolicy] state_dim={self.state_dim}, action_dim={self.action_dim}", "yellow")

        # Compute observation dimension for encoder
        obs_shape_meta = shape_meta.get('obs', {})
        obs_dim = 0
        for key, shape_info in obs_shape_meta.items():
            obs_shape = shape_info['shape']
            if isinstance(obs_shape, (tuple, list)):
                obs_dim += int(np.prod(obs_shape))
            else:
                obs_dim += int(obs_shape)

        # If no obs in shape_meta, use default (state itself as condition)
        if obs_dim == 0:
            # Use first n_obs_steps of state as conditioning
            obs_dim = self.state_dim
            cprint(f"[SecondOrderFlowPolicy] Using state as observation, obs_dim={obs_dim}", "cyan")
        else:
            cprint(f"[SecondOrderFlowPolicy] obs_dim={obs_dim}", "cyan")
            for key, shape_info in obs_shape_meta.items():
                cprint(f"  {key}: {shape_info['shape']}", "cyan")

        # State encoder for conditioning
        obs_encoder = StateOnlyEncoder(
            obs_dim=obs_dim,
            hidden_dims=[128, 256],
            out_dim=encoder_output_dim
        )

        obs_feature_dim = obs_encoder.output_shape()

        # Setup conditioning
        global_cond_dim = None
        input_dim = self.action_dim  # Network input is velocity field

        if obs_as_global_cond:
            if "cross_attention" in condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        cprint(f"[SecondOrderFlowPolicy] encoder_output_dim={encoder_output_dim}", "yellow")
        cprint(f"[SecondOrderFlowPolicy] global_cond_dim={global_cond_dim}", "yellow")

        # ConditionalUnet1D for velocity field prediction
        model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        # CFM hyperparameters
        if Conditional_ConsistencyFM is None:
            Conditional_ConsistencyFM = {
                'eps': 1e-2,
                'num_segments': 2,
                'boundary': 1,
                'delta': 1e-2,
                'alpha': 1e-5,
                'num_inference_step': 10  # More steps for 2nd-order
            }

        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        # GF Guidance field (optional)
        self.guidance_field = None

        print_params(self)

    def set_guidance_field(self, guidance_field) -> None:
        """Set GF guidance field for inference-time safety."""
        self.guidance_field = guidance_field

    def disable_guidance(self) -> None:
        """Disable GF guidance."""
        if self.guidance_field is not None:
            self.guidance_field.set_enabled(False)

    def enable_guidance(self) -> None:
        """Enable GF guidance."""
        if self.guidance_field is not None:
            self.guidance_field.set_enabled(True)

    # ========= inference ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Predict velocity field trajectory with optional GF guidance.

        The ODE solver integrates:
            dx/dt = v_FM(x, t, c) + lambda * v_GF(x)

        where:
            x = [q, q_dot] in R^14
            v_FM = network prediction
            v_GF = geometric fabrics repulsion

        Args:
            obs_dict: Observation dict for conditioning

        Returns:
            result: Dict with:
                - 'action': Velocity field [q_dot, q_ddot] (B, n_action_steps, 14)
                - 'state_pred': Predicted state trajectory (B, horizon, 14)
                - 'q_pred': Predicted positions (B, horizon, 7)
                - 'qdot_pred': Predicted velocities (B, horizon, 7)
        """
        # Handle nested obs dict
        if 'obs' in obs_dict:
            obs_data = obs_dict['obs']
        else:
            obs_data = obs_dict

        # Normalize observations
        nobs = self.normalizer.normalize(obs_data)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        To = self.n_obs_steps

        device = self.device
        dtype = self.dtype

        # Build conditioning
        local_cond = None
        global_cond = None

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(B, -1)

            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            this_nobs = dict_apply(nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(B, To, -1)
            Do = self.obs_feature_dim
            cond_data = torch.zeros(size=(B, T, Da + Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # Initialize from noise
        noise = torch.randn(size=cond_data.shape, dtype=cond_data.dtype, device=device)
        z = noise.detach().clone()

        # ODE integration
        sde = ConsistencyFM(
            'gaussian',
            noise_scale=1.0,
            use_ode_sampler='rk45',
            sigma_var=0.0,
            ode_tol=1e-5,
            sample_N=self.num_inference_step
        )

        dt = 1.0 / self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i / sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=device) * num_t

            # FM velocity prediction
            pred = self.model(z, t * 99, local_cond=local_cond, global_cond=global_cond)

            # === GF GUIDANCE for 2nd-order states ===
            if self.guidance_field is not None and self.guidance_field.enabled:
                # Extract q and q_dot from current state estimate
                # z has shape (B, T, 14) where first 7 are q_dot, next 7 are q_ddot
                # But we need actual q, q_dot for GF computation

                # For guidance, we need to maintain a running state estimate
                # Approximate: use first observation's state + integrated velocity
                # This is a simplification - in practice, might track state explicitly

                guidance_velocity = self.guidance_field.compute_guidance(
                    z[..., :Da],  # Current velocity field estimate
                    t=num_t,
                    normalizer=self.normalizer,
                )
                pred[..., :Da] = pred[..., :Da] + guidance_velocity
            # === END GF GUIDANCE ===

            # SDE integration step
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2) / (2 * (sde.noise_scale**2) * ((1.0 - num_t)**2)) * \
                        (0.5 * num_t * (1.0 - num_t) * pred - 0.5 * (2.0 - num_t) * z.detach().clone())
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)

        z[cond_mask] = cond_data[cond_mask]

        # Unnormalize prediction
        naction_pred = z[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # Extract action window
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        # Split into q_dot and q_ddot
        q_dot_pred = action_pred[..., :self.joint_dim]
        q_ddot_pred = action_pred[..., self.joint_dim:]

        result = {
            'action': action,
            'action_pred': action_pred,
            'q_dot_pred': q_dot_pred,
            'q_ddot_pred': q_ddot_pred,
        }
        return result

    # ========= training ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        """Set normalizer from training data statistics."""
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> tuple:
        """Compute Consistency Flow Matching loss.

        Args:
            batch: Dict with:
                - 'obs': Observation dict for conditioning
                - 'action': Target velocity field (B, T, 14)
                - 'state': 2nd-order state [q, q_dot] (B, T, 14) [optional]

        Returns:
            loss: Scalar loss value
            loss_dict: Dict with loss components
        """
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta = self.delta
        alpha = self.alpha
        reduce_op = torch.mean

        # Normalize
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # Build conditioning
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            this_nobs = dict_apply(nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # Generate mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Noise
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        # Time sampling
        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])

        # Interpolate
        xt = t_expand * target + (1.0 - t_expand) * a0
        xr = r_expand * target + (1.0 - r_expand) * a0

        # Apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        # Segment computation
        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1)
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1.0 - segment_ends_expand) * a0

        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt

        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if isinstance(threshold, int) and threshold == 0:
                return x_at_segment_ends
            less_than_threshold = t_expand < threshold
            res = less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt) + \
                  (~less_than_threshold) * x_at_segment_ends
            return res

        # Forward passes
        vt = self.model(xt, t * 99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r * 99, local_cond=local_cond, global_cond=global_cond)

        # Apply mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]
        vr = torch.nan_to_num(vr)

        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)

        # Loss computation
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)

        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if isinstance(threshold, int) and threshold == 0:
                return 0
            less_than_threshold = t_expand < threshold
            far_from_segment_ends = (segment_ends - t) > 1.01 * delta
            far_from_segment_ends = far_from_segment_ends.view(-1, 1, 1).repeat(1, trajectory.shape[1], trajectory.shape[2])
            losses_v = torch.square(vt - vr)
            losses_v = less_than_threshold * far_from_segment_ends * losses_v
            losses_v = reduce_op(losses_v.reshape(losses_v.shape[0], -1), dim=-1)
            return losses_v

        losses_v = masked_losses_v(vt, vr, boundary, segment_ends, t)

        loss = torch.mean(losses_f + alpha * losses_v)
        loss_dict = {
            'bc_loss': loss.item(),
            'consistency_loss': torch.mean(losses_f).item(),
            'velocity_loss': torch.mean(losses_v).item() if isinstance(losses_v, torch.Tensor) else 0.0,
        }

        return loss, loss_dict
