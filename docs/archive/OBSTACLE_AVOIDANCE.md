# Sphere Obstacle Avoidance with GF Guidance

## Overview

학습 시 장애물이 없는 환경에서 훈련된 Flow Policy가 inference 시 **Geometric Fabrics (GF) guidance**를 통해 sphere 장애물을 회피할 수 있도록 하는 기능입니다.

```
v_final = v_FM + λ * v_GF

v_FM  : Flow Matching policy 출력 (학습된 velocity field)
v_GF  : FABRICS guidance (obstacle + joint limit repulsion)
λ     : guidance_scale (조절 가능)
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Inference Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐     ┌──────────────┐     ┌────────────┐  │
│  │ Flow Policy  │────▶│  GF Guidance │────▶│  v_final   │  │
│  │   (v_FM)     │     │   (v_GF)     │     │            │  │
│  └──────────────┘     └──────────────┘     └────────────┘  │
│                              │                              │
│                              │                              │
│                     ┌────────┴────────┐                    │
│                     │                 │                    │
│              ┌──────▼──────┐  ┌───────▼───────┐           │
│              │ Joint Limit │  │ Sphere Obstacle│           │
│              │  Repulsion  │  │   Repulsion    │           │
│              └─────────────┘  └────────────────┘           │
│                                      │                     │
│                              ┌───────▼───────┐             │
│                              │SphereWorldModel│             │
│                              │ (Warp Meshes) │             │
│                              └───────────────┘             │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `guidance/sphere_obstacle.py` | SphereWorldModel - Warp mesh 생성 및 관리 |
| `guidance/gf_guidance_field.py` | GFGuidanceField - FABRICS 기반 guidance 계산 |
| `guidance/__init__.py` | Module exports |

---

## Usage

### CLI

```bash
# Single sphere obstacle
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint ./logs/best_model.pth \
    --obstacles '[{"pos":[0.5,0.0,0.4],"radius":0.1}]' \
    --guidance_scale 1.0

# Multiple obstacles
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint ./logs/best_model.pth \
    --obstacles '[{"pos":[0.5,0.0,0.4],"radius":0.1},{"pos":[0.3,0.2,0.5],"radius":0.08}]' \
    --guidance_scale 1.5

# With receding horizon
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint ./logs/best_model.pth \
    --receding_horizon \
    --obstacles '[{"pos":[0.5,0.0,0.4],"radius":0.1}]' \
    --guidance_scale 1.0
```

### Python API

```python
from guidance import GFGuidanceField

# Create guidance field
guidance = GFGuidanceField(
    batch_size=1,
    device='cuda:0',
    guidance_scale=1.0,      # λ scaling factor
    max_spheres=20,          # Maximum obstacles
)

# Set sphere obstacles: List of ((x, y, z), radius)
guidance.set_sphere_obstacles([
    ((0.5, 0.0, 0.4), 0.1),   # sphere 1
    ((0.3, 0.2, 0.5), 0.08),  # sphere 2
])

# Attach to policy
policy.set_guidance_field(guidance)

# Run inference (guidance applied automatically)
result = policy.predict_action({'obs': obs_dict})
```

### Dynamic Obstacle Update

```python
# Clear and set new obstacles
guidance.clear_obstacles()
guidance.set_sphere_obstacles([
    ((0.6, 0.1, 0.3), 0.12),  # new obstacle position
])

# Check obstacle count
print(f"Active obstacles: {guidance.obstacle_count}")
```

---

## Parameters

### GFGuidanceField

| Parameter | Default | Description |
|-----------|---------|-------------|
| `guidance_scale` | 1.0 | GF guidance 강도 (λ) |
| `max_guidance_strength` | 5.0 | Guidance clamp 최대값 |
| `max_spheres` | 20 | 최대 obstacle 개수 |
| `timestep` | 0.02 | Integration timestep |

### FABRICS Internal (from `franka_panda_pose_params.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `engage_depth` | 0.8 | Repulsion 시작 거리 (m) |
| `constant_accel` | 5.0 | Repulsion 가속도 |
| `damping_gain` | 1.0 | Velocity damping |
| `velocity_gate` | True | Velocity gating 활성화 |

---

## How It Works

### 1. Sphere Mesh Generation

Icosphere (정이십면체 세분화)를 사용하여 sphere mesh 생성:

```python
# subdivisions=2 → 162 vertices, 320 faces
vertices, faces = generate_icosphere(subdivisions=2)
```

### 2. Collision Detection

FABRICS는 로봇을 **61개의 collision sphere**로 표현하고, 각 sphere와 obstacle mesh 간의 거리를 GPU에서 계산합니다.

```
Robot Body → 61 Collision Spheres → Distance Query → Repulsion Force
```

### 3. Barrier Function

거리 기반 barrier function으로 smooth repulsion 생성:

```
acceleration = metric_scalar * (1/distance) * normal
```

- 거리가 가까울수록 강한 repulsion
- Velocity gating으로 oscillation 방지

### 4. Guidance Integration

ODE solver의 각 step에서 guidance 적용:

```python
for step in ode_steps:
    v_fm = policy.forward(z, t)           # FM prediction
    v_gf = guidance.compute_guidance(...)  # GF guidance
    v_final = v_fm + λ * v_gf              # Combined
    z = z + v_final * dt                   # Integration
```

---

## Coordinate System

Obstacle 위치는 **world frame** (로봇 base 기준) 좌표계입니다:

```
    Z (up)
    │
    │
    │_____ Y (left)
   /
  /
 X (forward)

Robot base at origin (0, 0, 0)
```

**Franka Panda workspace 참고:**
- X: ~0.3 ~ 0.8 m (forward reach)
- Y: ~-0.5 ~ 0.5 m (lateral)
- Z: ~0.0 ~ 0.8 m (height)

---

## Troubleshooting

### Import Error: No module named 'warp'

```bash
pip install warp-lang>=1.5.0
```

### FABRICS Import Error

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/scripts/FABRICS/src
```

### Guidance Too Strong/Weak

```python
# Adjust guidance scale
guidance.set_guidance_scale(0.5)   # Weaker
guidance.set_guidance_scale(2.0)   # Stronger
```

### Collision Still Occurs

1. `engage_depth` 증가 (더 먼 거리에서 repulsion 시작)
2. `guidance_scale` 증가
3. Obstacle radius 약간 증가 (safety margin)

---

## Visualization

`visualize_trajectory.py`에서 장애물을 PyBullet으로 시각화할 수 있습니다:

```bash
# 장애물과 함께 trajectory 시각화
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz ./results/trajectories.npz \
    --traj_idx 0 \
    --obstacles '[{"pos":[0.5,0.0,0.4],"radius":0.1}]' \
    --pybullet \
    --playback_speed 0.3
```

**색상 범례:**
- **Blue**: 로봇 (현재 trajectory)
- **Green**: 목표 위치 (goal-conditioned인 경우)
- **Red**: 장애물 (sphere)

---

## Limitations

1. **Static obstacles only**: Rollout 중 obstacle 위치 변경 미지원
2. **Sphere only**: Box, mesh 등 다른 형태 미지원
3. **No self-collision**: 로봇 자체 충돌 회피는 FABRICS 내부에서 처리

---

## References

- [Geometric Fabrics for Motion Generation](https://arxiv.org/abs/2010.14750)
- [NVIDIA FABRICS Repository](https://github.com/NVlabs/FABRICS)
- FABRICS params: `scripts/FABRICS/src/fabrics_sim/fabric_params/franka_panda_pose_params.yaml`
