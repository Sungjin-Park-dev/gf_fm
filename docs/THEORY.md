# GF-FM: Theoretical Background & Implementation Details

This document covers the theoretical foundations of the GF-FM (Geometric Fabrics + Flow Matching) framework, including the integration of physical guidance into generative flow and the resolution of common pitfalls in vector field interpretation.

---

## 1. Geometric Fabrics (GF) Guidance

### Overview
GF-FM enables obstacle avoidance and joint limit compliance during inference by injecting **Geometric Fabrics (GF)** repulsion terms into the Flow Matching (FM) generation process.

The core equation for guidance injection is:
```math
v_{final} = v_{FM}(x, \tau) + \lambda(\tau) \cdot \Phi_{GF}(x)
```
Where:
- $v_{FM}$: Velocity field predicted by the Flow Matching network (learned from data).
- $\Phi_{GF}$: Repulsion field calculated by Geometric Fabrics (physics-based).
- $\lambda$: Guidance scale (tunable parameter).
- $\tau$: Flow time (0 to 1).

### Architecture
The inference pipeline combines the generative model with a physics-based safety layer:

```
┌──────────────┐     ┌──────────────┐     ┌────────────┐
│ Flow Policy  │────▶│  GF Guidance │────▶│  v_final   │
│   (v_FM)     │     │   (v_GF)     │     │            │
└──────────────┘     └──────────────┘     └────────────┘
                              │
                     ┌────────┴────────┐
              ┌──────▼──────┐  ┌───────▼───────┐
              │ Joint Limit │  │ Sphere Obstacle│
              │  Repulsion  │  │   Repulsion    │
              └─────────────┘  └────────────────┘
```

### Sphere Obstacle Avoidance
- **World Model:** Obstacles are represented as spheres in the world frame.
- **Collision Detection:** The robot body is approximated by a set of collision spheres. Distances are computed between body spheres and obstacle meshes.
- **Repulsion Logic:**
  - **Inside:** Pushes the robot body point towards the nearest surface point (Outwards).
  - **Outside:** Pushes the robot body point away from the obstacle (Outwards).
  - *Note:* Early versions had a bug where "outside" repulsion was attractive; this has been fixed in `body_sphere_repulsion.py`.

---

## 2. Vector Field vs. Physical Dynamics

A critical distinction in GF-FM is the difference between the **Generative Vector Field** and **Physical Dynamics**.

### Two Time Scales
1.  **Diffusion Time ($\tau \in [0, 1]$):** The abstract time used for the generative process (noise $\to$ data).
2.  **Physical Time ($t \in [0, T]$):** The actual execution time of the robot trajectory.

### The Misconception
It is a common mistake to interpret the FM vector field $v_{FM} = \frac{dx}{d\tau}$ as physical velocity or acceleration.
- **Scenario:** Generating a "stationary" robot at the goal.
- **FM View:** The model must transport noise values to the fixed goal value. $v_{FM}$ is **large** (high transport speed).
- **Physical View:** The robot should have zero velocity. Physical velocity is **zero**.

**Conclusion:** Never use $v_{FM}$ directly as a motor command. The output of the FM process is the **state** $x = [q, \dot{q}]$, and this generated state contains the correct physical velocities.

### Guidance Implementation
Since $v_{FM}$ is not physical acceleration, simply adding physical forces to it requires careful scaling.
- We interpret GF guidance as modifying the **target state** of the generation.
- By adding repulsion to the vector field, we steer the "flow" of generation towards regions of the state space that are collision-free.

---

## 3. ODE-Coupled Guidance Integration

To effectively combine the generative process with physical constraints, we use an **ODE-Coupled** integration strategy.

### Additive vs. Coupled
| Feature | Additive (Naive) | ODE-Coupled (Recommended) |
|---------|------------------|---------------------------|
| **Method** | `pred = v_FM + guidance` then integrate | Sub-step integration with continuous guidance |
| **Interaction** | Loose coupling | Tight feedback loop |
| **Precision** | Low (Linear approximation) | High (Continuous path correction) |

### Coupled Integration Algorithm
In the `ode_coupled` mode (`OCECoupledGuidanceIntegrator`), each main ODE step is divided into sub-steps:

```python
for step in ode_steps:
    v_fm = policy.forward(z, t)           # 1. Predict flow direction
    
    for sub_step in sub_steps:
        # 2. Re-evaluate physical guidance at current intermediate state
        guidance = gf_field.compute_guidance(z_current)
        
        # 3. Combine and advance
        v_total = v_fm + lambda * guidance
        z_current = z_current + v_total * dt_sub
```

This ensures that if the guidance pushes the state into a new configuration, the next sub-step's guidance calculation reflects that change immediately, preventing oscillation and overshoot.

### Lambda Scheduling
The guidance strength $\lambda$ can be scheduled over flow time $\tau$:
- **Constant:** Uniform pushing throughout generation.
- **Linear Decay:** Strong initial shaping, fine-tuning later.
- **Linear Increase:** Safety priority at the end of generation.

---

## References
- [Geometric Fabrics for Motion Generation](https://arxiv.org/abs/2010.14750)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
