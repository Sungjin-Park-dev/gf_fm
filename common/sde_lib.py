"""Abstract SDE classes, Reverse SDE, and VE/VP SDEs."""
import abc
import torch
import numpy as np


class ConsistencyFM():
    def __init__(self, init_type='gaussian', noise_scale=1.0,  use_ode_sampler='rk45', sigma_var=0.0, ode_tol=1e-5, sample_N=None):
      if sample_N is not None:
        self.sample_N = sample_N
        #print('Number of sampling steps:', self.sample_N)
      self.init_type = init_type
      
      self.noise_scale = noise_scale
      self.use_ode_sampler = use_ode_sampler
      self.ode_tol = ode_tol
      self.sigma_t = lambda t: (1. - t) * sigma_var
      #print('Init. Distribution Variance:', self.noise_scale)
      #print('SDE Sampler Variance:', sigma_var)
      #print('ODE Tolerence:', self.ode_tol)
      
      self.consistencyfm_hyperparameters = {
        "delta": 1e-3,
        "num_segments": 2,
        "boundary": 1, # NOTE If wanting zero, use 0 but not 0. or 0.0, since the former is integar.
        "alpha": 1e-5,
      }

    def T(self):
      return 1.

    def ode(self, init_input, model, reverse=False):
      """
      ODE solver for flow matching (not used in current implementation).
      We use Euler integration in predict_action instead.
      """
      raise NotImplementedError("ODE solver not implemented. Use Euler integration in predict_action().")

    @torch.no_grad()
    def euler_ode(self, init_input, model, reverse=False, N=100):
      """
      Euler ODE solver (not used in current implementation).
      We use Euler integration in predict_action instead.
      """
      raise NotImplementedError("Euler ODE solver not implemented. Use Euler integration in predict_action().")

    def get_z0(self, batch):
      B, N, D = batch.shape 

      if self.init_type == 'gaussian':
          ### standard gaussian #+ 0.5
          cur_shape = (B, N, D)
          return torch.randn(cur_shape)*self.noise_scale
      else:
          raise NotImplementedError("INITIALIZATION TYPE NOT IMPLEMENTED") 
      
