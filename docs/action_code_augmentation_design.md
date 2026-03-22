# Action-Code Augmentation Design

## 목표

현재 dataset generation은 같은 state에 대해 사실상 하나의 고정 action-code permutation만 제공한다. action code randomization은 이미 bias를 줄이지만, 동일 state를 여러 code view로 학습시키면 code identity에 대한 잔여 편향을 더 줄일 수 있다.

## 현재 상태

- step-local randomized action code 사용
- 그러나 dataset generation 시 `source_index + step_idx` 기반 seed로 사실상 state당 1개 permutation 고정
- 즉 같은 state는 학습 내내 같은 `<Axxxx> ↔ job` 배치를 본다

## 확장 방향

### 1. Offline multi-code augmentation

같은 step row를 여러 code-view로 복제한다.

예:
- original state row
- code view #1
- code view #2
- code view #3

장점:
- trainer 수정 거의 없음
- 재현성 좋음

단점:
- dataset 크기 증가
- HF 저장비용 증가

### 2. Epoch-level online remapping

학습 epoch마다 같은 state의 action-code permutation을 다시 샘플링한다.

예:
- epoch 1: `<A0410> -> Job 2`
- epoch 2: `<A4507> -> Job 2`
- epoch 3: `<A1198> -> Job 2`

장점:
- dataset 크기 증가 없음
- code identity memorization 더 강하게 차단

단점:
- trainer/data collator 쪽 추가 구현 필요
- 재현성/logging 설계 필요

## 권장 순서

### Phase 1. Offline 2-view augmentation
- dataset generation에 `num_code_views` 추가
- 같은 row를 K개 생성하되 mapping만 다르게 생성
- first benchmark에 가장 쉬움

### Phase 2. Online remapping prototype
- raw state row + teacher action만 저장
- collator/map 단계에서 epoch seed 기반 code remapping 적용
- action token label도 동적으로 재생성

## schema 확장

추가 필드:
- `code_view_id`
- `action_code_seed`
- `mapping_version`

필요 시:
- `teacher_job_id`
- `teacher_action_code`
- `feasible_action_codes`

## 본 연구와의 연결점

이 프레임워크는 이미
- feasible action set 명시
- action code special token
- inverse mapping
을 갖고 있으므로, code augmentation은 문제 formulation을 바꾸지 않고도 일반화 강화를 시도할 수 있는 자연스러운 확장이다.

## 권장 실험

1. fixed single permutation per state
2. offline 2-view augmentation
3. offline 4-view augmentation
4. epoch-level online remapping

비교 지표:
- larger-size generalization
- action hallucination / invalid decode rate
- output bias reduction
