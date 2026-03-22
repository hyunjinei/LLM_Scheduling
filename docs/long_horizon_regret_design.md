# Long-Horizon Regret Design for Reason / Reflection

## 목표

현재 reason / reflection 신호는 projected Cmax, estimated start time, post-route 같은 one-step transition feature에 크게 의존한다. 이는 빠르고 해석 가능하지만, 장기 horizon의 makespan regret와 항상 일치하지는 않는다. 본 설계의 목표는 action quality를 downstream rollout 기준으로 더 직접 측정하는 보강 경로를 정의하는 것이다.

## 현재 한계

- one-step projected Cmax는 local surrogate일 뿐이다.
- post-route가 좋아 보여도 실제 suffix rollout에서는 병목이 달라질 수 있다.
- reflection이 immediate gap 위주로 critical step을 잡으면, 진짜 regret가 큰 step을 놓칠 수 있다.

## 보강 방향

### 1. Short suffix rollout regret

각 candidate action에 대해 action 이후 짧은 horizon H만큼 rollout하여 surrogate return을 계산한다.

예:
- current state s_t
- candidate a_t
- short rollout depth H=3 or 5
- heuristic / policy rollout로 suffix cost 추정

regret 정의 예시:
- `regret_H(a) = rollout_cost_H(a) - min_b rollout_cost_H(b)`

장점:
- 구현 난이도 낮음
- 현재 환경/마스킹 구조 재사용 가능

단점:
- 계산량 증가
- rollout policy 품질에 의존

### 2. Value model / critic estimate

state 또는 state-action value를 근사하는 별도 value head / critic을 두고,
reflection과 rationale에 long-horizon quality 신호로 사용한다.

가능 신호:
- `V(s_t)`
- `Q(s_t, a_t)`
- `advantage(s_t, a_t) = Q - V`

장점:
- 추론 시 빠름
- reflection score를 더 안정적으로 만들 수 있음

단점:
- 추가 학습 필요
- critic bias 가능성

### 3. Counterfactual suffix regeneration

critical step 후보에서 chosen action과 alternative action 각각에 대해 동일 prefix 이후 suffix를 재생성하여 최종 makespan 차이를 측정한다.

정의:
- same prefix up to t-1
- choose action a vs b at step t
- regenerate suffix under same decode/mask policy
- compare final makespan

장점:
- 가장 직접적인 long-horizon regret 측정
- reflection supervision 품질이 높음

단점:
- 계산량 큼
- variance 큼

## 권장 단계적 적용

### Phase 1. Short rollout regret 추가
- reflection score에 `short_rollout_regret` 추가
- top alternatives 정렬에도 long-horizon tie-break로 사용
- reason target에 `downstream regret` 한 문장 추가 가능

### Phase 2. Counterfactual suffix evaluation for critical steps
- top-k critical step에만 적용
- chosen vs best_alt의 final makespan 차이 측정
- reflection memory에 즉시 반영

### Phase 3. Critic / value model 도입
- RL과 결합
- reason/reflection과 shared value signal 사용

## 현재 코드와의 결합 지점

### A. Critical step selection
현재:
- immediate_gap
- makespan_jump
- start_delay

확장:
- `critical_score = w1*immediate_gap + w2*makespan_jump + w3*start_delay + w4*short_rollout_regret`

### B. Reason target
현재:
- immediate timing/Cmax impact
- bottleneck relation
- post-route exposure

확장:
- `downstream regret estimate`
- `short suffix projected makespan`

### C. Reflection memory
현재:
- episode postmortem
- bottleneck / idle / post-route 규칙

확장:
- “alternative X reduced final suffix makespan by Y”
- “chosen action caused downstream regret on bottleneck machine M_k”

## 최소 구현안

1. step record에 `short_rollout_regret` 필드 추가
2. top alternatives 계산 시 short rollout tie-break 사용
3. critical step score에 regret 포함
4. reflection memory에 regret 문장 추가

## 계산 비용 제어

- full candidate 전부가 아니라 top-K alternatives만 rollout
- depth H를 2~5로 제한
- critical-step mining 이후에만 적용
- serial/dispatch 모두 같은 interface 유지

## 권장 실험

1. one-step reflection baseline
2. + short rollout regret
3. + counterfactual suffix regeneration on critical steps

비교 지표:
- reflection correction rate
- final makespan improvement
- inference overhead
