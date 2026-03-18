# Step-by-Step LLM Scheduling Methodology

이 문서는 PPT/논문 방법론 설명용 초안이다. 현재 코드 구현 기준으로, 본 연구의 핵심 구조를 다음 세 가지 축으로 정리한다.

1. 왜 `Job i` 대신 `<Axxxx>` action code를 사용하는가
2. decode-time masking이 어떻게 동작하는가
3. 본 연구 방법론의 핵심이 무엇인가

---

## 1. 전체 프레임워크 개요

본 연구는 JSSP를 "전체 해(sequence)를 한 번에 생성하는 문제"가 아니라, **현재 상태에서 다음 action 하나를 선택하는 step-by-step decision problem**으로 재정식화한다.

전체 흐름은 다음과 같다.

1. 원본 JSSP instance와 reference schedule로부터 step dataset 생성
2. environment가 현재 상태(state)와 feasible action set 계산
3. 각 feasible candidate를 자연어로 표현
4. LLM은 feasible action code 중 하나를 선택
5. 선택된 action code를 다시 실제 job으로 역매핑하여 environment에 입력
6. environment가 다음 상태로 전이
7. 이를 episode 종료까지 반복

즉, 본 연구의 LLM은 "전체 schedule 생성기"가 아니라 **state-conditioned next-action policy**로 동작한다.

---

## 2. 왜 `Job i` 대신 `<Axxxx>`를 사용하는가

### 2.1 기본 아이디어

기존 방식처럼 모델이 직접 `Job 0`, `Job 1`, `Job 2`를 출력하게 만들면, 다음과 같은 문제가 생길 수 있다.

- 절대 job index에 대한 출력 편향이 생길 수 있음
- 학습 시 본 적 없는 더 큰 문제 크기에서 출력 일반화가 약해질 수 있음
- `Job 3` 같은 정적 label이 문제 크기와 직접 결합되어 있음

이를 줄이기 위해, 본 연구는 **각 step마다 feasible job들을 랜덤한 action code `<Axxxx>`로 재부호화**한다.

예를 들어 현재 feasible job이 `[0, 1, 5]`라면, 현재 step에서만 다음과 같이 매핑할 수 있다.

```text
<A4507> -> Job 0
<A0410> -> Job 1
<A8880> -> Job 5
```

LLM은 이제 `Job 1`을 직접 출력하는 대신, `<A0410>`을 출력한다.

이후 시스템은 inverse map을 사용해 이를 다시 실제 job으로 바꾼다.

```text
LLM output: <A0410>
inverse map: <A0410> -> Job 1
environment step input: Job 1
```

즉, **LLM은 action code를 선택하고, environment는 실제 job을 실행**한다.

### 2.2 왜 `<Axxxx>`가 매 step마다 바뀌는가

`<Axxxx>`는 global job identity가 아니라 **step-local action identity**다.

예를 들어:

```text
Step t:
<A4507> -> Job 0
<A0410> -> Job 1

Step t+1:
<A4507> -> Job 3
<A0410> -> Job 0
```

즉 같은 code라도 다음 step에서는 전혀 다른 job을 가리킬 수 있다.

이렇게 매 step마다 랜덤 재매핑을 하는 이유는:

- 절대 job index memorization을 줄이기 위해
- action selection을 "상태 기반 상대적 선택 문제"로 만들기 위해
- 학습 크기와 추론 크기가 다를 때 output bias를 줄이기 위해

따라서 본 연구에서 일반화의 대상은 `Job 7` 같은 정적 label이 아니라, **현재 상태와 feasible candidate 특징을 보고 어떤 action을 고를지에 대한 policy 자체**다.

### 2.3 PPT용 한 줄 요약

- 각 step의 feasible job을 랜덤 action code `<Axxxx>`로 재부호화하고, LLM은 code를 선택한 뒤 이를 실제 job으로 역매핑하여 environment에 입력한다.

---

## 3. Policy Prompt는 어떻게 구성되는가

본 연구의 policy prompt는 크게 세 부분으로 구성된다.

1. Static problem context
2. Dynamic state
3. Candidate transition features

### 3.1 Static problem context

현재는 최소 정보만 남긴다.

```text
Problem: 2 jobs x 2 machines (total_ops=4)
```

이 정보는 문제 크기 자체를 알려주지만, `Job 0 route=...` 같은 절대 job route 정보는 넣지 않는다.

### 3.2 Dynamic state

현재 step에서 실제로 바뀌는 환경 요약이다.

예:

```text
decision_step=1/4 scheduled_ratio=0
current_time=0
current_cmax=0
unfinished_jobs_count=2 unfinished_jobs_ratio=1
idle_machines=['M0', 'M1'] num_running_ops=0
bottleneck_machine=M1 bottleneck_load=54 bottleneck_ops_left=2
```

이 정보는 매 step마다 바뀐다.

### 3.3 Candidate transition features

각 candidate action code가 선택되었을 때의 projected effect를 자연어로 표현한다.

예:

```text
<A4507> | operation machine=M0 | processing time=21 | decision_t=0 | est_start=0 | est_end=21 | cmax:0->21 | delta_cmax=21 | job_ready:0->21 | machine_ready:0->21 | rem_ops:2->1 | rem_work:49->28 | machine_load=48 | machine_ops_left=2 | rem_work_after_ratio=0.5714 | post_route=[M1:28]
```

여기서:

- `operation machine=M0`
  - 현재 이 action을 선택하면 즉시 수행할 operation의 machine
- `processing time=21`
  - 현재 operation의 processing time
- `est_start=0`, `est_end=21`
  - 예상 시작/종료 시각
- `cmax:0->21`, `delta_cmax=21`
  - makespan에 미치는 immediate effect
- `rem_ops:2->1`, `rem_work:49->28`
  - 해당 job의 남은 operation/remaining work 변화
- `post_route=[M1:28]`
  - 현재 operation을 수행한 뒤 남는 후속 route

즉 LLM은 단순히 "어느 job을 고를까"를 보는 것이 아니라, **각 action이 현재 상태를 어떻게 바꾸는지**를 보고 선택한다.

### 3.4 Policy output

Policy 모델은 아래처럼 **action code 하나만 출력**한다.

```text
<A4507>
```

---

## 4. Action Code를 다시 Job으로 바꾸는 과정

LLM이 `<A4507>`를 출력하면 시스템은 같은 step에서 사용한 `action_code_to_job` 사전을 이용해 역변환한다.

예:

```text
action_code_to_job = {
    "<A0410>": 1,
    "<A4507>": 0,
}

LLM output = <A4507>
chosen_job = action_code_to_job["<A4507>"] = 0
env.step(chosen_job)
```

즉:

1. environment가 feasible jobs 계산
2. feasible jobs를 `<Axxxx>`로 랜덤 매핑
3. LLM이 code 출력
4. code를 실제 job으로 역매핑
5. environment가 그 job을 실행

이 구조 덕분에 LLM은 항상 **고정 vocabulary 위에서 현재 후보 집합 중 하나를 선택하는 문제**를 푼다.

---

## 5. Masking 로직은 어떻게 동작하는가

본 연구의 feasibility handling 핵심은 **decode-time hard masking**이다.

### 5.1 핵심 아이디어

매 step마다 environment는 현재 feasible action code 집합을 계산한다.

예:

```text
Feasible action codes now: [<A0410>, <A4507>]
```

이때 디코딩 시 허용되는 다음 토큰은 오직:

- `<A0410>`로 이어질 수 있는 prefix
- `<A4507>`로 이어질 수 있는 prefix

뿐이다.

즉 `<A9999>`처럼 현재 feasible set에 없는 action code는 **생성 단계에서 바로 차단**된다.

### 5.2 Prefix-level masking 예시

현재 feasible action set이 다음과 같다고 하자.

```text
[<A0410>, <A4507>]
```

그럼 생성 중 가능한 prefix는 예를 들어:

```text
<
<A
<A0
<A04
<A041
<A0410

<A4
<A45
<A450
<A4507
```

처럼 feasible code의 prefix에 해당하는 경우만 허용된다.

반대로:

```text
<A8
<A99
<A1234
```

같이 어떤 feasible code의 prefix도 아닌 문자열은 즉시 마스킹된다.

### 5.3 형식 제약 + feasibility 제약

이 masking은 동시에 두 가지를 만족시킨다.

1. 형식 제약
   - 출력이 반드시 `<Axxxx>` 형식이 되도록 제한

2. feasibility 제약
   - `<Axxxx>` 중에서도 현재 feasible set에 속한 code만 허용

즉 본 연구는 **문자열 형식 오류와 infeasible action 생성을 동시에 차단**한다.

### 5.4 결과

이 구조에서는 모델이 아무리 이상한 방향으로 확률을 두더라도,

- 현재 feasible set 밖의 action
- 잘못된 형식의 action

은 최종적으로 출력될 수 없다.

따라서 feasibility는 사후 repair가 아니라, **디코딩 시점에서 직접 통제**된다.

### 5.5 PPT용 한 줄 요약

- Environment가 계산한 feasible action set만 디코딩 가능하도록 prefix-level hard masking을 적용하여, infeasible action과 잘못된 출력 형식을 생성 단계에서 즉시 차단한다.

---

## 6. Reason Prompt는 왜 필요한가

Reason model은 action을 새로 고르는 모델이 아니라, **이미 고른 action이 왜 좋은지를 설명하는 모델**이다.

### 6.1 입력

Reason input은 다음 요소를 포함한다.

1. 현재 step의 policy 자연어 문제
2. `Selected action: <Axxxx>`
3. `Critical context`
4. `Top alternatives`

예:

```text
Selected action: <A4507>

Critical context:
- bottleneck_machine: M1 (remaining_load=54, ops_left=2)

Chosen transition:
- operation machine: M0
- bottleneck_relation: releases_future_bottleneck
- cmax: 0->21
- delta_cmax: 21
- ...
- post_route: [M1:28]

Top alternatives:
- <A0410>: operation machine=M1, post_route=[M0:27], cmax 0->26, ...
```

### 6.2 의미

- `Critical context`
  - 현재 state에서 가장 중요한 global signal, 즉 bottleneck machine 요약
- `Top alternatives`
  - 선택하지 않은 후보들 중 projected makespan 등이 좋은 강한 대안들

이를 통해 모델은 단순히 "좋다"가 아니라,

- 왜 chosen action이 좋았는지
- 왜 alternative는 덜 좋았는지

를 비교 기반으로 학습한다.

### 6.3 출력

```text
Reason: ...
Not chosen:
- <A0410>: ...
```

즉 reason supervision은 policy 선택의 설명 가능성과 보조 학습 신호 역할을 동시에 가진다.

---

## 7. Self-Reflection은 어떻게 동작하는가

Self-reflection은 기본 정책이 한 번 rollout을 끝낸 뒤, **episode hindsight를 이용해 일부 critical step을 다시 고쳐보는 2차 개선 모듈**이다.

### 7.1 1차 rollout

먼저 policy가 episode를 끝까지 생성한다.

각 step마다 아래 정보를 저장한다.

- chosen action code
- chosen job
- chosen start/end time
- makespan before / after
- all candidate options

### 7.2 critical step 선정

episode 종료 후, 각 step에 대해 chosen action과 가장 강한 alternative를 비교한다.

현재 구현은 대략 다음 세 신호를 사용한다.

- `immediate_gap`
  - chosen의 projected makespan이 best alternative보다 얼마나 나쁜지
- `makespan_jump`
  - 해당 step에서 실제 makespan이 얼마나 증가했는지
- `start_delay`
  - chosen start가 best alternative보다 얼마나 늦은지

이 값들이 모두 충분히 나쁘지 않으면 critical step으로 보지 않는다.

즉 critical step은 한마디로,

> "더 나은 대안이 있었고, 현재 선택이 immediate schedule quality 측면에서 손해였던 step"

이다.

### 7.3 episode reflection memory 생성

critical steps를 뽑은 뒤, 전체 episode hindsight를 요약한 memory를 만든다.

예:

```text
Current episode makespan=87.
Episode postmortem: the schedule likely lost quality through one or more of the following patterns:
delaying bottleneck activation, accepting larger idle/wait gaps without structural payoff,
or choosing weaker post-routes when immediate Cmax signals were close.

Reflection rules:
1. Prefer actions that activate or release the bottleneck route earlier.
2. If immediate projected Cmax is tied or close, prefer lower regret in post-route progression.
3. Avoid larger machine idle gaps or waits unless they clearly unlock stronger bottleneck progress.
```

### 7.4 step re-selection prompt

그 다음 critical step에 대해, 원래 step prompt 뒤에 reflection memory와 step-specific guidance를 붙여 다시 action을 고르게 한다.

즉 최종 2차 prompt는 다음 세 요소의 결합이다.

1. 원래 step의 policy 자연어 문제
2. episode reflection memory
3. post-episode guidance

예:

```text
[원래 step policy prompt]

Episode reflection memory:
...

Apply these reflection rules while selecting this step action.

Post-episode guidance:
This step was identified as a Cmax bottleneck.
If feasible, prefer Job 1.
Avoid Job 0 if a stronger alternative exists.
Why: chosen=<A4507>/Job0 showed high bottleneck risk; prefer <A0410>/Job1
```

이후 모델은 다시 `<Axxxx>` 하나를 출력한다.

즉 self-reflection은 **전체 schedule refinement**가 아니라, **critical step local revision with episode hindsight** 구조다.

---

## 8. Serial formulation과 Dispatch formulation

본 연구는 두 가지 formulation을 지원한다.

### 8.1 Serial formulation

- unfinished job 전체에 가까운 후보 집합 위에서
- 다음 schedule construction order를 고르는 방식

특징:

- 미래에 실제 시작할 작업도 미리 sequence에 넣을 수 있음
- 정적 benchmark scheduling과 잘 맞음

### 8.2 Dispatch formulation

- 현재 시각 `t`에서 실제로 dispatch 가능한 job만 후보로 두는 방식
- 같은 시각에 더 넣을 수 있는 작업이 있으면 계속 배치
- 더 이상 배치할 게 없을 때 다음 event time으로 이동

특징:

- real-time dispatching과 더 가까움
- machine ready / current time / idle machine 개념이 직접적으로 의미를 가짐

즉 본 연구는 **정적 serial scheduling과 동적 dispatching을 모두 포괄하는 unified LLM scheduling framework**를 지향한다.

---

## 9. 본 연구 방법론의 핵심

본 연구의 방법론을 한 줄씩 압축하면 다음과 같다.

### 9.1 Step-by-step reformulation

- JSSP를 전체 해 생성 문제가 아니라, 현재 상태에서 다음 action 하나를 선택하는 문제로 재정식화

### 9.2 Randomized action-code policy

- feasible job을 매 step 랜덤한 `<Axxxx>` action code로 재부호화
- 절대 job index 출력 편향을 줄이고, 상태 기반 상대적 선택 문제로 변환

### 9.3 Decode-time hard masking

- 현재 feasible action set만 생성 가능하도록 prefix-level masking 적용
- infeasible action과 형식 오류를 생성 단계에서 즉시 차단

### 9.4 Environment-validated transition

- 선택된 action code를 실제 job으로 역매핑하여 environment에 입력
- 검증된 상태전이만 다음 step으로 전달

### 9.5 Policy / Reason / Reflection 구조

- Policy: 다음 action 선택
- Reason: 선택된 action의 설명 및 대안 비교
- Reflection: episode hindsight를 이용한 critical step 재선택

### 9.6 Generalization perspective

- 본 연구는 특정 job 번호나 전체 sequence pattern을 외우게 하지 않고
- **현재 feasible action set 위에서 어떤 action을 선택할지에 대한 state-conditioned policy**를 학습하게 한다

---

## 10. PPT용 핵심 문장

### 10.1 한 줄 요약

> 본 연구는 JSSP를 step-by-step action selection 문제로 재정식화하고, randomized action code와 decode-time hard masking을 결합하여 feasible action만 선택하도록 제약을 직접 통제하는 LLM scheduling framework를 제안한다.

### 10.2 차별점 요약

- 전체 해 생성이 아닌 **현재 상태 기반 next-action policy**
- `Job i` 대신 **step-local randomized action code**
- **decode-time hard masking**으로 infeasible action 차단
- **environment-validated transition**으로 실행 가능한 상태만 누적
- **serial + dispatch dual formulation**
- **policy + reason + self-reflection** 확장 구조

