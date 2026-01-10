# Flow Matching Vector Field와 Physical Dynamics의 관계 재정의

## 1. 배경 및 문제 의식

우리는 로봇의 궤적 생성 모델로 **Flow Matching (FM)**을 사용하고 있으며, 장애물 회피를 위해 **Geometric Fabrics (GF)** 기반의 Repulsion을 Guidance로 적용하고 있습니다. 이 과정에서 FM이 학습하는 **Vector Field ($v_{FM}$)**의 물리적 의미에 대한 혼동이 있었습니다.

### 사용자의 가설
> "FM이 Joint Position ($q$)을 학습한다면, 이때의 Vector Field는 Joint Velocity ($\dot{q}$)가 될 것이고, Joint Velocity ($\dot{q}$)를 학습한다면 Vector Field는 Joint Acceleration ($\ddot{q}$)으로 해석할 수 있지 않을까?"

### 결론 요약
**아니요, 그렇게 해석할 수 없습니다.** 가장 큰 이유는 **시간(Time)의 정의가 다르기 때문**입니다.

---

## 2. 두 가지 시간(Time)의 차이

이 문제를 이해하기 위해서는 두 가지 다른 시간 축을 명확히 구분해야 합니다.

### A. Diffusion Time ($\tau$)
- **정의:** 노이즈($x_0$)에서 데이터($x_1$)로 변환되는 생성 과정상의 시간.
- **범위:** $0$ (Pure Noise) $\to$ $1$ (Clean Data)
- **단위:** 무차원 (Abstract Unit)
- **FM Vector Field ($v_\tau$):** $\frac{dx}{d\tau}$. 데이터 분포로 이동하기 위해 노이즈를 어떻게 변화시켜야 하는지를 나타내는 **"생성 방향"**입니다.

### B. Physical Time ($t$)
- **정의:** 로봇이 실제로 움직이는 물리적 시간.
- **범위:** $0$초 $\to$ $T$초 (Trajectory Duration)
- **단위:** 초 (Seconds)
- **Physical Dynamics:** $\dot{q} = \frac{dq}{dt}$ (속도), $\ddot{q} = \frac{d\dot{q}}{dt}$ (가속도).

---

## 3. 왜 $v_{FM} \neq \text{Physical Velocity}$ 인가?

사용자의 가설인 "$q$를 학습하면 $v_{FM}$은 $\dot{q}$이다"를 수식으로 풀면 다음과 같습니다.

$$ v_{FM} = \frac{dq}{d\tau} $$

반면, 물리적 속도는:

$$ \dot{q} = \frac{dq}{dt} $$

여기서 $d\tau$ (생성 스텝)와 $dt$ (물리적 시간 스텝)는 아무런 관계가 없습니다.
- FM의 Vector Field $v_{FM}$은 **"이 노이즈를 걷어내면 $q$가 된다"**는 방향을 가리킬 뿐, 로봇이 $t$ 시간에 얼마나 빠르게 움직이는지를 나타내지 않습니다.

### 예시
- **정지해 있는 로봇($\dot{q}=0$)**을 생성한다고 가정해 봅시다.
- FM은 노이즈 상태에서 정지 상태($q_{target}$)로 가기 위해 열심히 값을 수정해야 합니다. 즉, $v_{FM}$은 큽니다.
- 하지만 물리적 속도 $\dot{q}$는 0이어야 합니다.
- 따라서 $v_{FM} \neq \dot{q}$ 입니다.

---

## 4. 현재 구현 (2nd Order FM)의 올바른 해석

우리의 모델(`SecondOrderFlowPolicy`)은 **Joint Position ($q$)과 Velocity ($\dot{q}$)**를 포함하는 State $x = [q, \dot{q}]$를 다룹니다. 그리고 모델은 **Physical Velocity Field**를 출력하도록 설계되었습니다.

### 모델의 출력 ($x_{pred}$)
엄밀히 말해 우리 모델은 **Trajectory 그 자체**를 생성하고 있습니다.
Consistency Model 관점에서, 현재 스텝의 예측값 $\text{pred}$는 **"최종적으로 생성될 물리적 궤적의 속도와 가속도"**를 추정하는 것입니다.

$$ \text{Output} = [\dot{q}_{phys}, \ddot{q}_{phys}] $$

### GF Guidance의 적용
GF Guidance는 물리적 상태에서 장애물을 피하기 위한 **가속도($a_{GF}$)**를 계산합니다.

$$ a_{GF} = \text{FABRICS}(q, \dot{q}) $$

우리는 이 $a_{GF}$를 FM의 예측값(Output)에 더해주고 있습니다. 이것은 **Vector Field를 수정하는 것이 아니라, 예측된 최종 목적지(Target Output)를 수정하는 것**으로 해석해야 합니다.

$$ [\dot{q}_{new}, \ddot{q}_{new}] = [\dot{q}_{old}, \dot{q}_{old}] + \text{Scale} \times [a_{GF} \cdot \Delta t, a_{GF}] $$

이것은 "생성 과정의 흐름"을 바꾸는 것이지만, 물리적으로는 **"생성될 궤적이 장애물 반대 방향으로 가속하도록 강제"**하는 것입니다.

---

## 5. 해결 방안 및 구현 전략

아까 발생했던 Guidance가 약했던 문제는, 우리가 **Generative Flow ($d\tau$)와 Physical Update ($dt$)를 혼동하여 스케일링을 잘못했기 때문**입니다.

### 수정된 전략 (현재 적용됨)

1.  **Time Scale 제거:** Diffusion Time $\tau$가 1에 가까워진다고 해서(생성이 거의 끝났다고 해서) 장애물 회피를 멈추면 안 됩니다. 물리적으로 충돌 위험이 있다면 끝까지 밀어내야 합니다. $\to$ `time_scale = 1.0` 고정.
2.  **직관적 물리 적용:**
    *   FABRICS는 "지금 당장 피하려면 이만큼 가속해($a_{GF}$)"라고 말합니다.
    *   FM 모델은 "내 생각엔 속도가 $\dot{q}$이고 가속도가 $\ddot{q}$일 것 같아"라고 예측합니다.
    *   우리는 이 예측을 수정합니다:
        *   **가속도 수정:** $\ddot{q} \leftarrow \ddot{q} + a_{GF}$ (직접적 반영)
        *   **속도 수정:** $\dot{q} \leftarrow \dot{q} + \alpha \cdot a_{GF}$ (가속도를 적분한 효과를 즉시 반영하여 반응성 향상)

### 요약
Flow Matching의 Vector Field를 물리적 속도/가속도로 해석하려 하지 말고, **"모델이 예측한 물리적 궤적(Output)에 외력(Guidance)을 더해 수정한다"**는 관점으로 접근해야 올바른 제어가 가능합니다.
