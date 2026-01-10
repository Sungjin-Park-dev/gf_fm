# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Guidance fields for GF-FM inference.

Provides:
    - GFGuidanceField: Full FABRICS-based guidance (joint limits + obstacles)
    - SimpleJointLimitGuidance: Lightweight fallback (joint limits only)
    - SphereWorldModel: Sphere obstacle management for collision avoidance
    - GuidanceMode: Enum for guidance integration modes
    - GuidanceIntegrator: Base class for guidance integration strategies
    - AdditiveGuidanceIntegrator: Original additive guidance method
    - ODECoupledGuidanceIntegrator: ODE-coupled sub-step integration
"""

from .gf_guidance_field import GFGuidanceField, SimpleJointLimitGuidance
from .sphere_obstacle import SphereWorldModel, generate_icosphere
from .guidance_integrator import (
    GuidanceMode,
    GuidanceIntegrator,
    AdditiveGuidanceIntegrator,
    ODECoupledGuidanceIntegrator,
    create_guidance_integrator,
)

__all__ = [
    # Guidance fields
    "GFGuidanceField",
    "SimpleJointLimitGuidance",
    # Obstacle model
    "SphereWorldModel",
    "generate_icosphere",
    # Guidance integrators
    "GuidanceMode",
    "GuidanceIntegrator",
    "AdditiveGuidanceIntegrator",
    "ODECoupledGuidanceIntegrator",
    "create_guidance_integrator",
]
