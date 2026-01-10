# ODE-Coupled GF Guidance

## 개요

FM의 ODE 적분 과정 중에 GF의 힘을 가이드(Guidance)로 주입하는 방식입니다.

**핵심 수식:**
$$\frac{dx}{d\tau} = v_{FM}(x, \tau) + \lambda(\tau) \cdot \Phi_{GF}(x)$$

여기서:
- $x = [q, \dot{q}] \in \mathbb{R}^{14}$ : 2차 상태 (관절 위치 + 속도)
- $\tau \in [\epsilon, 1]$ : Flow time (생성 시간)
- $v_{FM}$ : FM 네트워크가 예측한 속도장
- $\Phi_{GF}$ : Geometric Fabrics 반발력 필드
- $\lambda(\tau)$ : 시간에 따라 변하는 guidance 강도

---

## 두 가지 Guidance 모드 비교

| 항목 | Additive (기존) | ODE-Coupled (새로운) |
|------|----------------|---------------------|
| **적용 방식** | `pred = v_FM + guidance` 후 1회 적분 | Sub-step마다 guidance 재계산 |
| **상호작용** | FM과 GF가 독립적 | ODE 내에서 지속적 상호작용 |
| **Guidance 계산** | Step당 1회 | Step당 n_substeps회 (기본 4회) |
| **정확도** | 낮음 (근사) | 높음 (연속적 결합) |
| **계산 비용** | 낮음 | 약간 높음 (FM forward는 1회만) |

### Additive 방식 (기존)

```
for each ODE step:
    v_FM = FM_network(z, tau)
    guidance = GF_field(state)
    pred = v_FM + guidance        # 한 번만 더함
    z = z + pred * dt + noise     # Euler 적분
```

### ODE-Coupled 방식 (새로운)

```
for each ODE step:
    v_FM = FM_network(z, tau)     # FM은 한 번만 계산

    for each sub-step:            # n_substeps 반복
        guidance = GF_field(z)    # 현재 상태에서 재계산
        lambda_t = compute_lambda(tau)
        v_combined = v_FM + lambda_t * guidance
        z = z + v_combined * sub_dt + noise
```

---

## Lambda Schedule 옵션

Guidance 강도 $\lambda$를 flow time $\tau$에 따라 조절할 수 있습니다.

| Schedule | 수식 | 그래프 | 용도 |
|----------|------|--------|------|
| `constant` | $\lambda$ | `────────` | 기본값, 일정한 강도 |
| `linear_decay` | $\lambda(1-\tau)$ | `╲________` | 초반 강한 회피, 후반 FM 우선 |
| `linear_increase` | $\lambda \tau$ | `________╱` | 후반 강한 회피 (안전 우선) |
| `cosine` | $\frac{\lambda}{2}(1+\cos(\pi\tau))$ | `╲___╱` | 부드러운 감소 |

### 예시 (lambda_base=1.0)

| tau | constant | linear_decay | linear_increase | cosine |
|-----|----------|--------------|-----------------|--------|
| 0.0 | 1.0 | 1.0 | 0.0 | 1.0 |
| 0.5 | 1.0 | 0.5 | 0.5 | 0.5 |
| 1.0 | 1.0 | 0.0 | 1.0 | 0.0 |

---

## 사용법

### CLI

```bash
# Guidance 없이
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --no_guidance \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_baseline.npz

# 기존 방식 (Additive) - 기본값
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --guidance_scale 1.0 \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_additive.npz


# 새로운 방식 (ODE-Coupled)
./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
    --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
    --receding_horizon \
    --fixed_start \
    --force_goal '[0.5, 0.5, 0.0, -1.5, 0.0, 1.5, 0.0]' \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --guidance_mode ode_coupled \
    --n_substeps 4 \
    --lambda_schedule constant \
    --guidance_scale 1.0 \
    --num_rollouts 1 \
    --save_trajectories scripts/gf_fm/results/traj_guidance.npz

# ODE-Coupled + 장애물 회피
# ./isaaclab.sh -p scripts/gf_fm/play_standalone.py \
#     --checkpoint scripts/gf_fm/logs/goal_cond_stop/checkpoints/best_model.pth \
#     --guidance_mode ode_coupled \
#     --n_substeps 4 \
#     --lambda_schedule linear_decay \
#     --obstacles '[{"pos":[0.4,0.0,0.5],"radius":0.1}]' \
#     --receding_horizon \
#     --num_rollouts 10

```
## 시각화
```bash
./isaaclab.sh -p scripts/gf_fm/visualize_trajectory.py \
    --npz scripts/gf_fm/results/traj_guidance.npz \
    --traj_idx 0 \
    --obstacles_file scripts/gf_fm/config/obstacles_example.yaml \
    --pybullet \
    --playback_speed 0.5 \
    --save_video traj_guidance.mp4
```


### CLI 인자

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--guidance_mode` | `additive` | `additive` 또는 `ode_coupled` |
| `--n_substeps` | `4` | ODE-coupled 모드의 sub-step 수 |
| `--lambda_schedule` | `constant` | Lambda 스케줄 유형 |
| `--guidance_scale` | `1.0` | Lambda base 값 (guidance 강도) |

### Python API

```python
from policy.second_order_flow_policy import SecondOrderFlowPolicy
from guidance import GFGuidanceField

# 방법 1: 생성 시 설정
policy = SecondOrderFlowPolicy(
    **config,
    guidance_mode="ode_coupled",
    n_guidance_substeps=4,
    lambda_schedule="cosine",
    lambda_base=1.0,
)

# 방법 2: 런타임에 변경
policy.set_guidance_mode(
    mode="ode_coupled",
    n_substeps=8,
    lambda_schedule="linear_decay",
    lambda_base=2.0,
)

# Guidance field 설정
guidance_field = GFGuidanceField(guidance_scale=1.0, ...)
policy.set_guidance_field(guidance_field)

# Inference
result = policy.predict_action(obs_dict)
```

---

## 파일 구조

```
scripts/gf_fm/
├── guidance/
│   ├── __init__.py                  # Export 정의
│   ├── gf_guidance_field.py         # GFGuidanceField (FABRICS 기반)
│   ├── guidance_integrator.py       # [NEW] Guidance 적분 전략
│   │   ├── GuidanceMode              # Enum: additive, ode_coupled
│   │   ├── GuidanceIntegrator        # ABC
│   │   ├── AdditiveGuidanceIntegrator    # 기존 방식
│   │   └── ODECoupledGuidanceIntegrator  # 새로운 방식
│   └── sphere_obstacle.py           # 장애물 메시 생성
│
├── policy/
│   └── second_order_flow_policy.py  # [MODIFIED] 새 파라미터 추가
│
└── play_standalone.py               # [MODIFIED] CLI 인자 추가
```

---

## 구현 세부사항

### GuidanceIntegrator 클래스 계층

```python
class GuidanceMode(Enum):
    ADDITIVE = "additive"
    ODE_COUPLED = "ode_coupled"

class GuidanceIntegrator(ABC):
    @abstractmethod
    def integrate_step(self, z, tau, dt, fm_velocity, guidance_field, ...) -> torch.Tensor:
        """한 ODE step을 guidance와 함께 적분"""
        pass

class AdditiveGuidanceIntegrator(GuidanceIntegrator):
    """기존 방식: pred = v_FM + guidance 후 적분"""

class ODECoupledGuidanceIntegrator(GuidanceIntegrator):
    """새로운 방식: Sub-step 적분으로 ODE 결합"""
    def __init__(self, n_substeps=4, lambda_schedule="constant", lambda_base=1.0):
        ...
```

### SecondOrderFlowPolicy 변경사항

```python
class SecondOrderFlowPolicy(BasePolicy):
    def __init__(self, ...,
        guidance_mode: str = "additive",
        n_guidance_substeps: int = 4,
        lambda_schedule: str = "constant",
        lambda_base: float = 1.0,
    ):
        ...
        self._guidance_integrator = None  # Lazy init

    def _get_guidance_integrator(self) -> Optional[GuidanceIntegrator]:
        """Lazy initialization of guidance integrator"""
        ...

    def set_guidance_mode(self, mode: str, **kwargs) -> None:
        """Runtime에서 guidance 모드 변경"""
        ...
```

### ODE Loop (predict_action)

```python
# 기존: guidance를 직접 적용
pred = pred + guidance_norm
z = z + pred_sigma * dt + noise

# 변경: integrator에 위임
if guidance_integrator is not None:
    z = guidance_integrator.integrate_step(z, tau, dt, pred, ...)
else:
    z = z + pred_sigma * dt + noise
```

---

## 하이퍼파라미터 튜닝 가이드

### n_substeps 선택

| 값 | 특징 | 권장 상황 |
|----|------|----------|
| 2 | 빠름, 정확도 낮음 | 빠른 테스트 |
| 4 | **권장** (기본값) | 일반적 사용 |
| 8 | 정밀, 느림 | 복잡한 장애물 환경 |

### lambda_schedule 선택

| Schedule | 권장 상황 |
|----------|----------|
| `constant` | 일반적 사용, 일정한 회피 필요 시 |
| `linear_decay` | 초반에 강한 회피 후 목표 도달 우선 |
| `linear_increase` | 마지막 순간 안전 우선 |
| `cosine` | 부드러운 전환이 필요할 때 |

### guidance_scale (lambda_base) 선택

| 값 | 효과 |
|----|------|
| 0.5 | 약한 회피, FM 우선 |
| 1.0 | 균형 (기본값) |
| 2.0+ | 강한 회피, 안전 우선 |

---

## 호환성

### 기존 Checkpoint 호환

- 모든 새 파라미터에 기본값 제공
- `guidance_mode="additive"`가 기본값 → 기존 동작 완전 유지
- 기존 CLI 명령어 그대로 사용 가능

### API 호환

```python
# 기존 코드 (변경 없이 동작)
policy = SecondOrderFlowPolicy(**config)
policy.set_guidance_field(guidance_field)
result = policy.predict_action(obs_dict)
```

---

## 관련 문서

- [OBSTACLE_AVOIDANCE.md](./OBSTACLE_AVOIDANCE.md) - 장애물 회피 전반
- [flow_matching_vector_field_analysis.md](./flow_matching_vector_field_analysis.md) - FM Vector Field 분석
- [gf_repulsion_issue.md](./gf_repulsion_issue.md) - GF Repulsion 문제 분석
- [README.md](./README.md) - 프로젝트 전체 개요

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2025-01-09 | ODE-Coupled Guidance 초기 구현 |
