# Multiple Teachers / Preference Supervision Design

## 목표

현재 step dataset은 한 문제당 하나의 teacher schedule만 step row로 분해한다. 하지만 JSSP에는 동일 makespan의 여러 선형화가 존재할 수 있고, near-optimal schedule도 충분히 학습 가치가 있다. 본 설계의 목적은 single-teacher imitation에서 벗어나, multiple teacher view와 pairwise preference supervision을 함께 지원하는 확장 경로를 정의하는 것이다.

## 왜 필요한가

- 같은 instance에도 여러 좋은 step choice가 존재한다.
- 현재 single-target imitation은 정답을 하나의 linearization으로 과도하게 고정한다.
- action masking으로 feasibility는 통제할 수 있지만, quality supervision은 여전히 teacher 1개에 종속된다.
- reason / reflection / RL을 더 강하게 만들려면 pairwise winner-loser 데이터가 필요하다.

## 확장 방향

### 1. Multiple exact / optimal teachers

가장 이상적인 경로는 동일 instance에 대해 makespan-optimal schedule을 여러 개 확보하는 것이다.

가능 경로:
- solver tie-breaking 변경
- solver warm start / randomized branching
- optimality gap 0 상태에서 다른 linearization 추출
- dispatch-valid teacher projection을 teacher별로 각각 수행

저장 단위:
- instance 1개
- teacher 1..K개
- teacher별 step rows 생성

### 2. Multiple near-optimal teachers

exact-optimal 다양성이 부족하면 near-optimal teachers도 같이 쓴다.

가능 경로:
- heuristic schedules
- beam / rollout candidates
- RL rollout 중 상위 trajectory
- bounded-regret variants

저장 원칙:
- teacher quality metric을 함께 저장
- exact optimal과 near-optimal을 구분하는 provenance 필드 추가

### 3. Pairwise preference supervision

teacher를 여러 개 확보하면 단순 imitation 대신 preference 학습이 가능하다.

기본 단위:
- 같은 state에서
- winner action / loser action
- 또는 winner suffix / loser suffix

활용처:
- DPO / IPO / BOPO형 학습
- critic / value regression target 보조신호
- reason target에서 contrastive explanation 강화

## 권장 dataset schema

### A. Multi-teacher step row schema

기존 필드 유지 + 아래 추가:
- `teacher_id`: 같은 instance 내 teacher 식별자
- `teacher_kind`: `optimal`, `near_optimal`, `heuristic`, `rl_rollout` 등
- `teacher_rank`: instance 내 quality rank
- `teacher_makespan`: teacher schedule의 최종 makespan
- `teacher_gap`: best known makespan 대비 gap
- `teacher_source`: solver / heuristic / rollout provenance
- `teacher_weight`: sampling 또는 loss weighting에 사용할 값

### B. Pairwise preference row schema

같은 state 기준 pair row:
- `instance_id`
- `state_id`
- `env_mode`
- `state_text`
- `winner_action_code`
- `loser_action_code`
- `winner_job_id`
- `loser_job_id`
- `winner_effect`
- `loser_effect`
- `winner_teacher_id`
- `loser_teacher_id`
- `preference_margin`
- `preference_reason`

reason/reflection 확장용:
- `winner_suffix_makespan`
- `loser_suffix_makespan`
- `shared_prefix_len`
- `long_horizon_regret`

## state alignment 원칙

pairwise preference는 반드시 같은 state에서 비교해야 한다.

안전한 기준:
- 같은 `instance_id`
- 같은 `env_mode`
- 같은 `step_idx`
- 같은 `state_signature`

`state_signature` 후보:
- job ready time vector
- machine ready time vector
- unfinished next-op index vector
- current time / current cmax

주의:
- 같은 `step_idx`라도 trajectory가 diverge하면 state가 달라질 수 있다.
- 따라서 장기적으로는 `step_idx`만으로 pair를 만들면 안 되고, `shared-prefix` 또는 `state_signature` 기준이 필요하다.

## 최소 구현 순서

### Phase 1. Multi-code / multi-teacher metadata 준비
- current schema에 `teacher_id=0`, `teacher_kind='single_teacher'` 추가
- 이후 multiple teacher가 들어와도 backward compatibility 유지

### Phase 2. Near-optimal extra teachers 생성
- 동일 instance에 대해 heuristic / rollout teacher 1~2개 추가
- best known makespan과 gap 저장
- 같은 state에서 action disagreement가 발생하는 step 식별

### Phase 3. Preference pair 생성기
- shared-prefix states만 pair 생성
- 같은 state에서 lower final makespan trajectory의 action을 winner로 지정
- margin = loser_suffix_makespan - winner_suffix_makespan

### Phase 4. Trainer integration
- supervised imitation + preference loss 혼합
- action_reason에서는 winner action token 가중 강화
- reason target에 contrastive preference explanation 반영

## 본 연구와의 연결점

본 프레임워크는 이미 다음 조건을 만족하므로 preference 확장에 유리하다.
- step-wise state representation 존재
- feasible action set 명시
- action code / inverse mapping 구조 존재
- reflection / RL에서 alternative 비교 정보 존재

즉 multiple-teacher / preference supervision은 새로운 틀을 만들기보다, 현재 step formulation 위에 quality supervision을 추가하는 형태로 확장 가능하다.

## 구현시 주의사항

- exact-optimal multiple teachers가 없더라도 near-optimal teacher를 섞을 때 provenance를 명확히 저장할 것
- dispatch teacher는 raw sequence replay가 아니라 dispatch-valid teacher projection을 teacher별로 다시 수행할 것
- pair construction은 같은 index가 아니라 같은 state를 기준으로 할 것
- evaluation은 single-teacher imitation과 preference-augmented 학습을 분리하여 비교할 것

## 권장 실험 비교

1. single teacher imitation
2. single teacher + action-centric loss
3. multiple teacher imitation
4. single teacher + preference pairs
5. multiple teacher + preference pairs

핵심 비교 지표:
- feasibility
- makespan / PRD
- larger-size generalization
- reflection correction rate
