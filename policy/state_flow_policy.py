from typing import Dict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
import copy
import time
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.sde_lib import ConsistencyFM
from model.normalizer import LinearNormalizer
from policy.base_policy import BasePolicy
from model.conditional_unet1d import ConditionalUnet1D
from model.mask_generator import LowdimMaskGenerator
from common.pytorch_util import dict_apply
from common.model_util import print_params
from model.state_encoder import StateOnlyEncoder
import warnings
warnings.filterwarnings("ignore")


class StateFlowPolicy(BasePolicy):
    """
    FlowPolicy adapted for state-only observations (no point cloud).

    Uses Consistency Flow Matching for 1-step inference on low-dimensional
    state observations from Isaac-Reach-UR10-v0.

    Observations:
        - joint_pos: (B, T, 6) - Joint positions
        - joint_vel: (B, T, 6) - Joint velocities
        - pose_command: (B, T, 7) - Target end-effector pose

    Actions:
        - (B, T, 6) - UR10 joint position commands
    """

    def __init__(self,
            shape_meta: dict,
            horizon,
            n_action_steps,
            n_obs_steps,
            obs_as_global_cond=True,
            diffusion_step_embed_dim=256,
            down_dims=(256,512,1024),
            kernel_size=5,
            n_groups=8,
            condition_type="film",
            use_down_condition=True,
            use_mid_condition=True,
            use_up_condition=True,
            encoder_output_dim=128,
            Conditional_ConsistencyFM=None,
            eta=0.01,
            **kwargs):
        super().__init__()

        self.condition_type = condition_type

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:
            # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']

        # Extract shapes from shape_meta (don't use dict_apply to avoid recursion issues)
        obs_dict = {}
        for key, shape_info in obs_shape_meta.items():
            obs_dict[key] = shape_info['shape']

        # State-only encoder (replaces FlowPolicyEncoder)
        # Automatically compute obs_dim from shape_meta
        obs_dim = 0
        for key, shape_info in obs_shape_meta.items():
            obs_shape = shape_info['shape']
            # Handle both tuple and single int
            if isinstance(obs_shape, (tuple, list)):
                obs_dim += int(np.prod(obs_shape))
            else:
                obs_dim += int(obs_shape)

        cprint(f"[StateFlowPolicy] Total observation dimension: {obs_dim}", "yellow")
        for key, shape_info in obs_shape_meta.items():
            cprint(f"  {key}: {shape_info['shape']}", "cyan")

        obs_encoder = StateOnlyEncoder(
            obs_dim=obs_dim,
            hidden_dims=[128, 256],
            out_dim=encoder_output_dim
        )

        obs_feature_dim = obs_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None

        # obs_as_global_cond=true
        if obs_as_global_cond:
            input_dim = action_dim
            if "cross_attention" in self.condition_type:
                global_cond_dim = obs_feature_dim
            else:
                global_cond_dim = obs_feature_dim * n_obs_steps

        cprint(f"[StateFlowPolicy] encoder_output_dim: {encoder_output_dim}", "yellow")
        cprint(f"[StateFlowPolicy] global_cond_dim: {global_cond_dim}", "yellow")

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
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs

        if Conditional_ConsistencyFM is None:
            Conditional_ConsistencyFM = {
                'eps': 1e-2,
                'num_segments': 2,
                'boundary': 1,
                'delta': 1e-2,
                'alpha': 1e-5,
                'num_inference_step': 1
            }
        self.eta = eta
        self.eps = Conditional_ConsistencyFM['eps']
        self.num_segments = Conditional_ConsistencyFM['num_segments']
        self.boundary = Conditional_ConsistencyFM['boundary']
        self.delta = Conditional_ConsistencyFM['delta']
        self.alpha = Conditional_ConsistencyFM['alpha']
        self.num_inference_step = Conditional_ConsistencyFM['num_inference_step']

        # Guidance field for GF repulsion (optional, set via set_guidance_field)
        self.guidance_field = None

        print_params(self)

    def set_guidance_field(self, guidance_field) -> None:
        """Set guidance field for inference-time trajectory repulsion.

        Args:
            guidance_field: GuidanceField instance or None to disable.
        """
        self.guidance_field = guidance_field

    def disable_guidance(self) -> None:
        """Disable guidance field repulsion."""
        if self.guidance_field is not None:
            self.guidance_field.set_enabled(False)

    # ========= inference  ============
    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Predict action sequence from state observations.

        Args:
            obs_dict: Must include "obs" key with nested dict containing:
                - joint_pos: (B, T, obs_dim)
                - joint_vel: (B, T, obs_dim)
                - eef_pos: (B, T, 3)
                - eef_quat: (B, T, 4)
                - cube_positions: (B, T, 9)

        Returns:
            result: Dict with "action" key containing (B, n_action_steps, action_dim)
        """
        # Extract observations from nested dict
        # Input is {'obs': {observation_keys...}}
        # Normalizer expects {observation_keys...}
        if 'obs' in obs_dict:
            obs_data = obs_dict['obs']
        else:
            obs_data = obs_dict

        # normalize input
        nobs = self.normalizer.normalize(obs_data)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(B, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        noise = torch.randn(
            size=cond_data.shape,
            dtype=cond_data.dtype,
            device=cond_data.device,
            generator=None)
        z = noise.detach().clone() # a0

        sde = ConsistencyFM('gaussian',
                            noise_scale=1.0,
                            use_ode_sampler='rk45', # unused
                            sigma_var=0.0,
                            ode_tol=1e-5,
                            sample_N=self.num_inference_step)

        # Uniform
        dt = 1./self.num_inference_step
        eps = self.eps

        for i in range(sde.sample_N):
            num_t = i /sde.sample_N * (1 - eps) + eps
            t = torch.ones(z.shape[0], device=noise.device) * num_t
            pred = self.model(z, t*99, local_cond=local_cond, global_cond=global_cond)

            # === GF REPULSION GUIDANCE ===
            # Add guidance velocity to steer trajectories away from obstacles
            if self.guidance_field is not None and self.guidance_field.enabled:
                guidance_velocity = self.guidance_field.compute_guidance(
                    z[..., :Da],  # Only action dimensions
                    t=num_t,
                    normalizer=self.normalizer,
                )
                # Add guidance to predicted velocity
                pred[..., :Da] = pred[..., :Da] + guidance_velocity
            # === END GF REPULSION GUIDANCE ===

            # convert to diffusion models if sampling.sigma_variance > 0.0 while perserving the marginal probability
            sigma_t = sde.sigma_t(num_t)
            pred_sigma = pred + (sigma_t**2)/(2*(sde.noise_scale**2)*((1.-num_t)**2)) * (0.5 * num_t * (1.-num_t) * pred - 0.5 * (2.-num_t)*z.detach().clone())
            z = z.detach().clone() + pred_sigma * dt + sigma_t * np.sqrt(dt) * torch.randn_like(pred_sigma).to(device)
        z[cond_mask] = cond_data[cond_mask] # a1

        # unnormalize prediction
        naction_pred = z[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]

        result = {
            'action': action,
            'action_pred': action_pred,
        }
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        """
        Compute Consistency Flow Matching loss.

        Args:
            batch: Dict with keys:
                - 'obs': Dict with joint_pos, joint_vel, pose_command
                - 'action': (B, T, action_dim)

        Returns:
            loss: Scalar loss value
            loss_dict: Dict with loss components
        """
        eps = self.eps
        num_segments = self.num_segments
        boundary = self.boundary
        delta  = self.delta
        alpha =  self.alpha
        reduce_op = torch.mean

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])
        target = nactions

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory

        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)

            if "cross_attention" in self.condition_type:
                # treat as a sequence
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # gt & noise
        target = target
        a0 = torch.randn(trajectory.shape, device=trajectory.device)

        t = torch.rand(target.shape[0], device=target.device) * (1 - eps) + eps # 1=sde.T
        r = torch.clamp(t + delta, max=1.0)
        t_expand = t.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        r_expand = r.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        xt = t_expand * target + (1.-t_expand) * a0
        xr = r_expand * target + (1.-r_expand) * a0

        # apply mask
        xt[condition_mask] = cond_data[condition_mask]
        xr[condition_mask] = cond_data[condition_mask]

        segments = torch.linspace(0, 1, num_segments + 1, device=target.device)
        seg_indices = torch.searchsorted(segments, t, side="left").clamp(min=1) # .clamp(min=1) prevents the inclusion of 0 in indices.
        segment_ends = segments[seg_indices]
        segment_ends_expand = segment_ends.view(-1, 1, 1).repeat(1, target.shape[1], target.shape[2])
        x_at_segment_ends = segment_ends_expand * target + (1.-segment_ends_expand) * a0

        def f_euler(t_expand, segment_ends_expand, xt, vt):
            return xt + (segment_ends_expand - t_expand) * vt

        def threshold_based_f_euler(t_expand, segment_ends_expand, xt, vt, threshold, x_at_segment_ends):
            if (threshold, int) and threshold == 0:
                return x_at_segment_ends

            less_than_threshold = t_expand < threshold

            res = (
                less_than_threshold * f_euler(t_expand, segment_ends_expand, xt, vt)
                + (~less_than_threshold) * x_at_segment_ends
            )
            return res

        vt = self.model(xt, t*99, cond=local_cond, global_cond=global_cond)
        vr = self.model(xr, r*99, local_cond=local_cond, global_cond=global_cond)

        # mask
        vt[condition_mask] = cond_data[condition_mask]
        vr[condition_mask] = cond_data[condition_mask]

        vr = torch.nan_to_num(vr)

        ft = f_euler(t_expand, segment_ends_expand, xt, vt)
        fr = threshold_based_f_euler(r_expand, segment_ends_expand, xr, vr, boundary, x_at_segment_ends)

        ##### loss #####
        losses_f = torch.square(ft - fr)
        losses_f = reduce_op(losses_f.reshape(losses_f.shape[0], -1), dim=-1)

        def masked_losses_v(vt, vr, threshold, segment_ends, t):
            if (threshold, int) and threshold == 0:
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
        }

        return loss, loss_dict
