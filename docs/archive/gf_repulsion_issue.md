# GF Guidance Repulsion Issue Analysis

## Problem Description

GF (Geometric Fabrics) guidance가 장애물에 대해 repulsion이 아닌 attraction처럼 동작합니다.

### 현상
- Flow policy는 goal 방향으로 정상 동작
- GF guidance를 켜면 end-effector가 장애물 쪽으로 이동
- 장애물과 부딪히지는 않지만 "품는" 느낌으로 가까워짐
- 장애물이 goal 반대 방향에 있어도 장애물 쪽으로 끌려감

---

## Code Flow Analysis

### 1. GF Guidance 적용 경로

```
play_standalone.py
  → GFGuidanceField 생성 (guidance_scale=5.0)
  → set_sphere_obstacles() 호출
  → policy.set_guidance_field()

SecondOrderFlowPolicy.predict_action()
  → self.guidance_field.compute_guidance()
  → guidance를 pred에 더함: pred = pred + guidance_norm
```

### 2. FABRICS Repulsion 계산 경로

```
GFGuidanceField.compute_guidance()
  → _lazy_init()에서 FrankaPandaRepulsionFabric 생성
  → DisplacementIntegrator.step() 호출
  → 내부적으로 collision_response kernel 실행
  → a_guidance = qdd_new - qdd_zeros (가속도 추출)
```

### 3. 방향 벡터 계산 (핵심 의심 지점)

**파일:** `FABRICS/src/fabrics_sim/fabric_terms/body_sphere_3d_repulsion.py`

**Line 79-80:**
```python
# Direction from sphere to closest point on the mesh
n = wp.normalize(closest_point - sphere_center_point)
```

**변수 의미:**
- `closest_point`: 장애물 메시에서 로봇 충돌구에 가장 가까운 점
- `sphere_center_point`: 로봇의 충돌 구체 중심
- `n`: 방향 벡터

**Line 108:** 가속도 적용
```python
wp.atomic_add(base_acceleration, batch_index, point_index, metric_scalar * (1./d) * n)
```

---

## Attempted Solutions

### 시도 1: gf_guidance_field.py에서 부호 반전

**파일:** `scripts/gf_fm/guidance/gf_guidance_field.py:242`

```python
# Before
guidance_14d = self.guidance_scale * guidance_14d

# After (시도)
guidance_14d = -self.guidance_scale * guidance_14d
```

**결과:** 실패 - 여전히 장애물 쪽으로 이동

### 시도 2: 부호 반전 제거

```python
# 원래대로 복원
guidance_14d = self.guidance_scale * guidance_14d
```

**결과:** 실패 - 동일한 문제 지속

### 시도 3: body_sphere_3d_repulsion.py 복사 후 방향 벡터 수정

**파일:** `scripts/gf_fm/guidance/body_sphere_3d_repulsion_fixed.py:80`

```python
# Before
n = wp.normalize(closest_point - sphere_center_point)

# After
n = wp.normalize(sphere_center_point - closest_point)
```

**결과:** 실패 - 여전히 장애물 쪽으로 이동

---

## Current File Structure

```
scripts/gf_fm/guidance/
├── gf_guidance_field.py                    # GF guidance 계산
├── sphere_obstacle.py                      # 장애물 메시 생성
├── body_sphere_3d_repulsion_fixed.py       # 수정된 repulsion (Line 80 방향 반전)
└── franka_panda_repulsion_fabric_fixed.py  # 수정된 fabric import

scripts/FABRICS/src/fabrics_sim/
├── fabric_terms/
│   └── body_sphere_3d_repulsion.py         # 원본 repulsion
├── fabrics/
│   └── franka_panda_repulsion_fabric.py    # 원본 fabric
└── integrator/
    └── integrators.py                      # DisplacementIntegrator
```

---

## Key Code Snippets

### 1. GFGuidanceField.compute_guidance() (gf_guidance_field.py:180-256)

```python
def compute_guidance(self, state_14d, time_scale=1.0):
    # state_14d: (B, T, 14) or (B, 14) - [q, q_dot]

    # FABRICS 초기화
    self._lazy_init()

    # 장애물 메시 업데이트
    self._update_obstacle_meshes()

    # 현재 상태 추출
    q = state_flat[:, :7]      # joint positions
    q_dot = state_flat[:, 7:]  # joint velocities

    # FABRICS integrator step
    qdd_zeros = torch.zeros_like(q_dot)
    q_new, qd_new, qdd_new = self._integrator.step(
        q.detach(), q_dot.detach(), qdd_zeros, self.timestep
    )

    # Guidance acceleration 추출
    a_guidance = qdd_new - qdd_zeros  # (B, 7)

    # Velocity field로 변환
    v_q_guidance = 0.5 * a_guidance * time_scale
    v_qdot_guidance = a_guidance * time_scale
    guidance_14d = torch.cat([v_q_guidance, v_qdot_guidance], dim=-1)

    # Scale 적용
    guidance_14d = self.guidance_scale * guidance_14d

    return guidance
```

### 2. collision_response kernel (body_sphere_3d_repulsion.py:18-188)

```python
@wp.kernel
def collision_response(...):
    # 메시와 로봇 충돌구 사이 거리 계산
    got_dist = wp.mesh_query_point(object_mesh, sphere_center_point, max_depth, ...)

    if got_dist:
        closest_point = wp.mesh_eval_position(object_mesh, f, bary_u, bary_v)

        # 방향 계산 (이 부분이 의심됨)
        n = wp.normalize(closest_point - sphere_center_point)

        # 거리 계산
        d = wp.length(closest_point - sphere_center_point)

        if d_signed <= engage_depth:
            # 가속도 추가
            wp.atomic_add(base_acceleration, batch_index, point_index,
                          metric_scalar * (1./d) * n)
```

### 3. SecondOrderFlowPolicy guidance 적용 (second_order_flow_policy.py:290-350)

```python
def predict_action(self, obs, ...):
    # ... flow matching prediction ...

    if self.guidance_field is not None:
        guidance = self.guidance_field.compute_guidance(pred, time_scale=1.0)

        # Reshape and normalize
        guidance_flat = guidance.reshape(B_traj * T_traj, -1)
        guidance_phys = self.normalizer['action'].unnormalize(...)

        # Add to prediction
        pred = pred + guidance_norm
```

---

## Hypotheses (Not Yet Verified)

### 가설 1: 방향 벡터 해석 오류
- `closest_point - sphere_center_point`가 실제로 어느 방향을 가리키는지 불명확
- Warp의 mesh_query_point 반환값의 의미 확인 필요

### 가설 2: Taskmap 변환 문제
- 3D 공간의 가속도가 joint space로 변환될 때 방향이 바뀔 수 있음
- RobotFrameOriginsTaskMap의 Jacobian 처리 확인 필요

### 가설 3: Metric 부호 문제
- base_acceleration 외에도 metric 계산에서 방향이 반영됨
- `force = -torch.bmm(metric, acceleration)` 에서 음수 처리

### 가설 4: Normalizer 문제
- action normalizer의 scale/offset이 방향에 영향을 줄 수 있음

### 가설 5: Feature 설정 문제
- `set_features()`에서 cspace_target 설정이 잘못될 수 있음
- C-space attractor가 의도치 않은 attraction을 생성

---

## Test Configuration

```yaml
# obstacles_example.yaml
obstacles:
  - pos: [0.4, -0.3, 0.5]
    radius: 0.15
```

```bash
# 테스트 명령어
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --guidance_scale 5.0 \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_guidance.npz
```

---

## Questions for Further Investigation

1. Warp의 `mesh_query_point`에서 반환되는 `closest_point`의 정확한 좌표계는?

2. `BodySphereRepulsion.force_eval()`에서 `accel_dir`이 어떻게 처리되는지?
   - Line 534-546: `accel_dir`에 음수를 곱하고 있음

3. `DisplacementIntegrator.step()`에서 forces에 음수를 곱하는 이유?
   - `joint_accel = -torch.bmm(self._masses_inv, self._forces.unsqueeze(2)).squeeze(2)`

4. FABRICS의 원래 의도된 repulsion 동작은 무엇인가?
   - 원본 FABRICS 예제에서 장애물 회피가 정상 동작하는지 확인 필요

---

## Related Files (Full Paths)

- `/workspace/isaaclab/scripts/gf_fm/guidance/gf_guidance_field.py`
- `/workspace/isaaclab/scripts/gf_fm/guidance/sphere_obstacle.py`
- `/workspace/isaaclab/scripts/gf_fm/guidance/body_sphere_3d_repulsion_fixed.py`
- `/workspace/isaaclab/scripts/gf_fm/guidance/franka_panda_repulsion_fabric_fixed.py`
- `/workspace/isaaclab/scripts/gf_fm/policy/second_order_flow_policy.py`
- `/workspace/isaaclab/scripts/FABRICS/src/fabrics_sim/fabric_terms/body_sphere_3d_repulsion.py`
- `/workspace/isaaclab/scripts/FABRICS/src/fabrics_sim/fabrics/franka_panda_repulsion_fabric.py`
- `/workspace/isaaclab/scripts/FABRICS/src/fabrics_sim/integrator/integrators.py`
