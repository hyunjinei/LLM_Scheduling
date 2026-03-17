# LLM_JSSP_masking

## 연구 개요

이 저장소는 **Job Shop Scheduling Problem (JSSP)** 를 **step-by-step LLM policy** 로 푸는 연구 코드다. 핵심 아이디어는 매 step마다 LLM이 자연어 상태를 보고 다음 행동을 선택하되, **디코딩 시점에 feasible action만 남기고 나머지 행동은 마스킹**하는 것이다. 즉, 제약을 어긴 뒤 사후 repair 하거나 sampling으로 운 좋게 feasible 해를 찾는 방식이 아니라, **행동 생성 단계 자체를 제약 친화적으로 바꾸는 것**이 목적이다.

이 저장소는 다음 연구 질문에 답한다.

> LLM이 JSSP를 end-to-end하게 풀되, feasibility를 추론 후 보정이 아니라 **추론 중(decoding-time)** 에 직접 통제할 수 있는가?

본 연구의 답은 다음과 같다.
- JSSP를 **step-level decision process** 로 재정의한다.
- 현재 상태에서 가능한 행동 집합을 환경이 계산한다.
- LLM은 그 feasible action 집합 안에서만 다음 행동을 생성한다.
- 각 행동은 환경 전이로 즉시 검증된다.
- supervised fine-tuning과 RL을 통해 **makespan** 을 직접 줄이는 policy를 학습한다.

## 연구 포지셔닝

### 이 연구는 무엇인가
- **문제**: JSSP / scheduling
- **목표**: makespan minimization
- **출력 형태**: 전체 해를 한 번에 생성하는 것이 아니라, **한 step씩 행동을 선택**
- **제약 처리 방식**: decode-time feasible-action masking + environment transition validation
- **학습 방식**: self-labeling 기반 SFT, 이후 RL 연계 가능
- **환경 모드**:
  - `serial`: benchmark-oriented schedule-construction formulation
  - `dispatch`: event-driven dispatch formulation

### 이 연구는 무엇이 아닌가
- OR solver 코드를 자동 생성하는 **formulation/modeling 연구**가 아니다.
- LLM이 휴리스틱 코드를 발명하는 **automatic heuristic design 연구**가 아니다.
- 단순히 infeasible 해를 만든 뒤 사후 repair 하는 연구가 아니다.
- unconstrained whole-schedule free-form generation만을 강조하는 연구가 아니다.

## 핵심 기여

1. **Step-by-step JSSP policy learning**
   - 전체 스케줄 문자열을 한 번에 내는 대신, 현재 상태에서 다음 scheduling action을 선택하는 정책 학습 구조를 사용한다.
2. **Decode-time hard feasibility control**
   - 현재 상태에서 infeasible한 action은 아예 디코딩 후보에서 제거한다.
3. **Environment-grounded transition validation**
   - 선택된 action은 스케줄링 환경에서 즉시 전이되며, 상태와 makespan이 명시적으로 갱신된다.
4. **Serial/Dispatch dual formulation**
   - benchmark-oriented serial formulation과 real-time dispatch formulation을 모두 지원한다.
5. **Self-labeled supervision + RL bridge**
   - teacher schedule로부터 step-level `policy / reason / mixed` 데이터를 자동 생성하고, 이후 RL로 연결 가능하다.

## Step-by-step이 최초인가?

**넓은 의미의 step-by-step / autoregressive CO generation 자체를 최초라고 주장하는 것은 안전하지 않다.**
이유는 다음과 같다.
- [ACCORD](https://openreview.net/pdf?id=f0TBAdcJ8m)는 제목부터 *Autoregressive Constraint-satisfying Generation* 이다.
- [LLMs can Schedule](https://arxiv.org/abs/2408.06993), [STARJOB](https://openreview.net/forum?id=t0fU6t3Skw), [Self-Guiding Exploration](https://arxiv.org/abs/2405.17950) 역시 broad sense에서 sequential / multi-step solution generation에 속한다.
- [Large Language Models as End-to-end Combinatorial Optimization Solvers](https://arxiv.org/abs/2509.16865) 역시 direct solution generation 계열이다.

따라서 더 안전한 표현은 아래다.

> **본 연구의 차별점은 “step-by-step 그 자체”가 아니라, JSSP에 대해 step-level policy learning, decode-time feasible-action masking, explicit environment transition validation, serial/dispatch dual formulation을 하나의 프레임워크로 결합한 점”이다.**

논문 문장으로 쓰려면 아래 표현이 안전하다.

> To the best of our knowledge, prior LLM-based scheduling studies have explored sequential or autoregressive generation, but explicit **step-level feasible-action masking with environment-validated transitions** under both **serial and dispatch** formulations remains underexplored in JSSP.

## Related Work Review Protocol

문헌 정리는 다음 기준을 사용했다.
- **source**: [awesome-fm4co: LLMs for Combinatorial Optimization](https://github.com/ai4co/awesome-fm4co/blob/main/README.md)
- **coverage**: 2026-03-17 기준 해당 섹션 전체 **157편**
- **분류 축**:
  1. feasibility / constraint satisfaction
  2. end-to-end direct solving
  3. scheduling / makespan relevance

전수 검토를 위해 별도 카탈로그도 생성했다.
- `literature_review/llm_for_co_catalog.json`
- `literature_review/llm_for_co_catalog.csv`

현재 자동 수집 기준 커버리지는 다음과 같다.
- abstract 수집 성공: **131 / 157**
- method snippet 수집 성공: **129 / 157**
- 둘 다 수집 성공: **128 / 157**

실패의 대부분은 OpenReview 원문 접근 제한(`403 Forbidden`) 때문이다. 따라서 아래 비교표는 **전체 목록 전수 수집 + 접근 가능한 논문의 abstract/method 확인 + 핵심 scheduling/feasibility 논문 수동 검토**를 결합해 작성했다.

awesome-fm4co `LLMs for Combinatorial Optimization` 섹션의 전체 157편을 remark 기준으로 세면 다음과 같다.

| Category | Count |
|---|---:|
| Algorithm | 91 |
| Formulation | 24 |
| Solution | 18 |
| Benchmark | 12 |
| Review | 2 |
| Interpretability | 2 |
| Others / mixed labels | 8 |

즉, 현재 문헌의 다수는 **직접 해를 생성하는 end-to-end solver** 보다는 **algorithm design / hyper-heuristic / formulation / benchmark** 에 치우쳐 있다. 이 점이 본 연구의 포지셔닝에 중요하다.

## 표 1. Feasibility 보장 또는 제약 준수 향상에 초점을 둔 LLM-for-CO 연구

| Author (Year) | Problem | Approach | Objective | Feasibility / Constraint handling | End-to-end direct solving | Inference-time hard guarantee | 본 연구와의 차이 |
|---|---|---|---|---|---:|---:|---|
| [Self-Guiding Exploration for Combinatorial Problems (2024)](https://arxiv.org/abs/2405.17950) | TSP, VRP, BPP, AP, KP, JSSP | multi-trajectory prompting + decomposition + refinement | solution quality improvement across CPs | reasoning/refinement로 품질과 validity 개선 | O | X | broad CP prompting 연구이며, explicit feasible-action mask는 없음 |
| [LLMs can Schedule (2024)](https://arxiv.org/abs/2408.06993) | JSSP | JSSP 전용 dataset 기반 supervised LLM scheduling + sampling | min makespan | 데이터와 sampling으로 feasibility/quality 개선 | O | X | JSSP direct solving이지만 decode-time hard masking은 아님 |
| [STARJOB: Dataset for LLM-Driven Job Shop Scheduling (2024)](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | JSSP dataset + fine-tuned LLM scheduler | min makespan | domain data로 scheduling validity 향상 | O | X | JSSP 특화 데이터셋 연구이나 hard feasible decoder는 아님 |
| [Large Language Models as End-to-end Combinatorial Optimization Solvers (2025)](https://arxiv.org/abs/2509.16865) | TSP, OP, CVRP, MIS, MVC, PFSP, JSSP | SFT → FOARL → Best-of-N | feasibility + optimality gap reduction | RL에서 feasibility/optimality-aware reward 사용 | O | X | feasibility를 높이지만 BoN/sampling 의존이 남음 |
| [ACCORD (2025)](https://openreview.net/pdf?id=f0TBAdcJ8m) | FSSP, JSSP, BPP, KP, TSP, VRP | autoregressive constraint-satisfying generation | feasibility + quality | constraint-aware autoregressive generation | O | X | 가장 가까운 prior 중 하나지만, env-grounded hard action masking과는 다름 |
| [Hard Constraints Meet Soft Generation (2026)](https://arxiv.org/abs/2602.01090) | TSP, OP, CVRP, MIS, MVC, PFSP, JSSP | FALCON: grammar-constrained decoding + feasibility repair + adaptive BoN + BOPO | 100% feasibility + quality | grammar + repair + adaptive sampling | O | O | 100% feasibility를 달성하지만 repair/adaptive BoN에 의존 |
| **This Research** | **JSSP** | **step-level policy + decode-time feasible-action masking + environment validation** | **min makespan** | **invalid action 제거 후 decode, step transition 즉시 검증** | **O** | **O*** | **repair/BoN 없이 action-level feasibility를 직접 강제** |

`*` 본 연구의 보장은 **모델링된 action space와 environment constraints에 대해** 성립한다. 즉 환경이 encode한 제약 범위 안에서 infeasible action은 생성되지 않는다.

## 표 2. End-to-end direct solving 관점의 주요 LLM-for-CO 연구

여기서 end-to-end는 “테스트 시 외부 OR solver가 해를 만들어 주는 것이 아니라, LLM이 직접 solution/action을 생성한다”는 뜻으로 사용한다.

| Author (Year) | Problem | End-to-end form | Objective | Constraint handling | 한계 |
|---|---|---|---|---|---|
| [Can Language Models Solve Graph Problems in Natural Language? (2023)](https://arxiv.org/abs/2305.10037) | graph problems | natural-language answer generation | task correctness | explicit hard feasibility control 없음 | 초기 direct-solver 계열이지만 scheduling과는 거리 있음 |
| [Self-Guiding Exploration for Combinatorial Problems (2024)](https://arxiv.org/abs/2405.17950) | multi-CP incl. JSSP | multiple thought trajectories + refinement | optimization performance | refinement 중심 | hard action-space control 없음 |
| [LLMs can Schedule (2024)](https://arxiv.org/abs/2408.06993) | JSSP | direct schedule generation from prompt | min makespan | training + sampling | step-level feasibility mask 부재 |
| [STARJOB (2024)](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | dataset-driven direct scheduler | min makespan | data-driven validity improvement | inference-time hard guarantee 부재 |
| [ReflecSched (2025)](https://arxiv.org/abs/2508.01724) | DFJSP | LLM-powered hierarchical reflection | dynamic scheduling quality | heuristic simulation summaries 사용 | reflection 기반이며 hard action masking과는 다름 |
| [Large Language Models as End-to-end Combinatorial Optimization Solvers (2025)](https://arxiv.org/abs/2509.16865) | 7 CO tasks incl. PFSP/JSSP | direct solution generation + SFT/RL/BoN | feasibility + objective gap | FOARL + BoN | high feasibility but not pure mask-based guarantee |
| [ACCORD (2025)](https://openreview.net/pdf?id=f0TBAdcJ8m) | TSP, VRP, KP, BPP, FSSP, JSSP | autoregressive direct generation | feasibility + quality | constraint-satisfying generation | still not explicit env-feasible action interface |
| [Hard Constraints Meet Soft Generation (2026)](https://arxiv.org/abs/2602.01090) | 7 CO tasks incl. PFSP/JSSP | direct generation + grammar/repair/adaptive BoN | perfect feasibility + quality | grammar + repair | guarantee는 있으나 repair가 핵심 구성요소 |
| **This Research** | **JSSP** | **text-state → next action direct generation** | **min makespan** | **decode-time feasible-action mask + env validation** | **whole-solution free-form이 아니라 step-policy라는 점이 차별점** |

## 표 3. Scheduling / makespan 관련 LLM-for-CO 연구 정리

아래 표는 awesome-fm4co의 LLM-for-CO 섹션 중 scheduling과 직접 관련된 항목만 따로 추렸다.

| Author (Year) | Problem | Approach | Objective / Task | Type | Constraint emphasis | 본 연구와의 관계 |
|---|---|---|---|---|---|---|
| [AI-Copilot for Business Optimisation (2023)](https://arxiv.org/pdf/2309.13218) | JSSP | optimization copilot / formulation | scheduling formulation | Formulation | X | direct scheduler가 아니라 modeling 지원 |
| [Self-Guiding Exploration for Combinatorial Problems (2024)](https://arxiv.org/abs/2405.17950) | JSSP 포함 | multi-trajectory prompting | solution quality | Solution | 제한적 | broad CP direct solving prior |
| [LLMs can Schedule (2024)](https://arxiv.org/abs/2408.06993) | JSSP | dataset-based LLM scheduling | makespan | Solution | X | 가장 직접적인 JSSP prior 중 하나 |
| [STARJOB (2024)](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | dataset + LLM-driven scheduling | makespan | Solution | X | JSSP dataset/fine-tuning prior |
| [Automatic programming via LLMs ... (2024)](https://arxiv.org/pdf/2410.22657) | DyJSSP | automatic rule/program generation | dynamic scheduling | Algorithm | X | direct policy가 아니라 heuristic design |
| [REMoH (2025)](https://arxiv.org/pdf/2506.07759) | FJSSP | reflective heuristic evolution | multi-objective heuristic quality | Algorithm | X | heuristic evolution 계열 |
| [ReflecSched (2025)](https://arxiv.org/abs/2508.01724) | DFJSP | hierarchical reflection | dynamic scheduling quality | Solution | 부분적 | reflection-based dynamic scheduling prior |
| [LLM as End-to-end CO Solvers (2025)](https://arxiv.org/abs/2509.16865) | PFSP, JSSP | direct solver + FOARL + BoN | feasibility + objective gap | Solution | 부분적 | strong end-to-end baseline |
| [LLM4EO (2025)](https://arxiv.org/pdf/2511.16485) | FJSP | evolutionary optimization with LLM | scheduling quality | Algorithm | X | heuristic/evolutionary optimization |
| [ACCORD (2025)](https://openreview.net/pdf?id=f0TBAdcJ8m) | FSSP, JSSP | autoregressive constraint-satisfying generation | feasibility + quality | Solution & Benchmark | O (improved) | 가장 가까운 autoregressive scheduling prior 중 하나 |
| [Leveraging LLMs for efficient scheduling ... (2025)](https://doi.org/10.1038/s44334-025-00061-w) | DFJSP | LLM-assisted scheduling in HRC-FMS | schedule efficiency | Algorithm | X | industrial scheduling relevance 큼 |
| [LLM-Assisted Automatic Dispatching Rule Design ... (2026)](https://arxiv.org/pdf/2601.15738) | Dynamic FAFSP | dispatching rule design | flow-shop scheduling quality | Algorithm | X | dispatch relevance는 높지만 direct masked policy는 아님 |
| [Hard Constraints Meet Soft Generation (2026)](https://arxiv.org/abs/2602.01090) | PFSP, JSSP | grammar + repair + adaptive BoN | perfect feasibility + quality | Solution | O | feasibility guarantee를 정면으로 다룬 강력한 baseline |
| **This Research** | **JSSP** | **step-level masked policy (serial + dispatch)** | **min makespan** | **Solution / policy learning** | **O** | **scheduling action-level hard feasibility control** |

## 기존 연구의 한계점

1. **feasibility가 “높아진다”와 “보장된다”는 다르다**
   - 많은 end-to-end LLM-for-CO 연구는 feasibility를 RL, prompting, Best-of-N, reflection, repair로 개선한다.
   - 그러나 추론 시점에 **infeasible action을 아예 생성하지 못하게 하는 hard action-space control** 은 상대적으로 드물다.

2. **sampling/post-selection 의존성이 크다**
   - 여러 solution paper는 multi-sample generation이나 Best-of-N으로 품질과 validity를 보완한다.
   - 이는 inference cost를 키우고, real-time scheduling에는 부담이 된다.

3. **scheduling direct policy보다 heuristic design / formulation 비중이 크다**
   - 전체 157편 중 다수는 algorithm generation, hyper-heuristic, solver modeling, benchmark에 속한다.
   - scheduling에서조차 direct next-action policy보다 heuristic/rule design이 더 흔하다.

4. **serial benchmark와 real-time dispatch를 함께 다루는 LLM scheduling 프레임워크가 드물다**
   - 많은 연구는 offline schedule generation 또는 heuristic discovery에 머문다.
   - real-time dispatch formulation을 동시에 고려하는 구조는 적다.

5. **step-level environment grounding이 약하다**
   - 자연어로 solution sequence를 생성하더라도, 각 step이 환경에서 실제 feasible transition인지 즉시 검증하지 않는 경우가 많다.

## 본 연구의 차별점

1. **JSSP를 “문장 생성”이 아니라 “step-level action policy”로 재정의**
   - 전체 해를 한 번에 서술하는 대신, 각 상태에서 다음 행동을 선택하는 정책 학습 문제로 바꾼다.

2. **decode-time hard feasibility mask**
   - invalid action은 아예 디코딩 후보에서 제거한다.
   - 즉 제약 처리를 사후 repair가 아니라 **행동 생성 시점**으로 끌어온다.

3. **environment-grounded transition**
   - 생성된 행동은 스케줄링 환경에서 즉시 적용/검증된다.
   - 따라서 text generation과 scheduling dynamics가 분리되지 않는다.

4. **serial + dispatch dual formulation**
   - benchmark-oriented serial formulation과 operational dispatch formulation을 모두 지원한다.
   - 이는 static benchmark 성능과 현장형 dispatching relevance를 함께 다룰 수 있게 한다.

5. **self-labeling 기반 dense supervision**
   - teacher schedule 하나를 whole-solution label로만 쓰지 않고, step-level row로 분해한다.
   - 따라서 supervision density가 높고 SFT/RL 연결이 쉽다.

6. **policy / reason / mixed supervision 분리 가능**
   - action-only, reason-only, action+reason을 분리해 ablation 할 수 있다.

## 본 연구의 장점

- **제약 준수 강함**: 모델링된 action space 안에서는 infeasible action이 decode되지 않는다.
- **설명 가능성**: step trace와 reason supervision을 통해 의사결정 과정을 추적할 수 있다.
- **RL 연결성**: step policy이므로 REINFORCE/GRPO/BOPO류 fine-tuning과 결합하기 쉽다.
- **운영 적합성**: dispatch formulation으로 real-time scheduling과의 연결 가능성이 높다.
- **문제 일반화 가능성**: “feasible set + masked decoding + env validation” 구조는 다른 scheduling/CO로 이식 가능하다.

## 본 연구의 한계

- **전역 최적성 보장은 없다**: feasibility를 강제해도 makespan optimality는 여전히 학습 성능에 좌우된다.
- **환경 모델 의존성**: guarantee는 encode된 제약에 대해서만 성립한다. 환경에 없는 산업 제약은 보장되지 않는다.
- **문제 특화 설계 필요**: state schema, mask logic, transition validation은 도메인별 구현이 필요하다.
- **step-by-step 추론 비용**: whole-solution 1회 생성보다 step 수만큼 반복 디코딩이 필요하다.
- **large-instance context issue**: 매우 큰 문제에서는 prompt/state 압축이 필요하다.

## 논문용 한 줄 주장 예시

### 안전한 주장
- 본 연구는 JSSP를 step-level action policy로 재정의하고, decode-time feasible-action masking을 통해 제약 위반 행동을 생성 단계에서 차단한다.
- 본 연구는 serial 및 dispatch scheduling formulation을 모두 지원하는 environment-grounded LLM scheduling framework를 제안한다.
- 본 연구는 self-labeled step supervision과 constrained decoding을 결합하여, feasibility와 makespan optimization을 동시에 겨냥한다.

### 과장 없이 강하게 쓰는 주장
- Prior LLM-based scheduling studies have explored direct generation, reflection, and heuristic design, but **hard feasible-action masking at decoding time with explicit environment-validated transitions** remains insufficiently explored for JSSP.
- Unlike prior methods that rely on sampling, post-selection, or repair to improve validity, our method constrains the action space **before** decoding, thereby enforcing feasibility at the decision level.

## 저장소 구성

- `generate_jssp_step_dataset.py`: self-labeled step dataset generation
- `unified_trainer.py`: local training entrypoint
- `inference_jssp_fssp.py`: local inference entrypoint
- `RL_jssp_fssp.py`: local RL entrypoint
- `notebooks/colab_00~06`: self-contained Colab workflows
- `llm_jssp/utils/*`: environment, prompting, masking, preprocessing utilities

## 부록 A. awesome-fm4co의 LLM-for-CO 전체 목록 (157편)

아래 표는 [awesome-fm4co](https://github.com/ai4co/awesome-fm4co/blob/main/README.md)의 `LLMs for Combinatorial Optimization` 섹션을 2026-03-17 기준으로 정리한 것이다.

<details>
<summary>전체 목록 펼치기</summary>

| Date | Paper | Problem | Type |
|---|---|---|---|
| 2023.07 | [Large Language Models for Supply Chain Optimization](https://arxiv.org/pdf/2307.03875) | Supply_Chain | Algorithm w. Interpretability |
| 2023.09 | [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/pdf/2305.10037) | Graph | Solution |
| 2023.09 | [Large Language Models as Optimizers](https://arxiv.org/pdf/2309.03409) | TSP | Solution |
| 2023.10 | [Chain-of-Experts: When LLMs Meet Complex Operations Research Problems](https://openreview.net/pdf?id=HobyL1B9CZ) | MILP | Formulation |
| 2023.10 | [OptiMUS: Scalable Optimization Modeling with (MI)LP Solvers and Large Language Models](https://arxiv.org/pdf/2402.10172) | MILP | Formulation |
| 2023.10 | [AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling](https://arxiv.org/pdf/2309.13218) | JSSP | Formulation |
| 2023.11 | [Large Language Models as Evolutionary Optimizers](https://arxiv.org/pdf/2310.19046) | TSP | Solution |
| 2023.11 | [Algorithm Evolution Using Large Language Model](https://arxiv.org/pdf/2311.15249) | TSP | Algorithm |
| 2023.12 | [Mathematical discoveries from program search with large language models](https://www.nature.com/articles/s41586-023-06924-6) | BPP | Algorithm |
| 2023.12 | [NPHardEval: Dynamic Benchmark on Reasoning Ability of Large Language Models via Complexity Classes](https://arxiv.org/pdf/2312.14890) | TSP,KP, GCP,MSP | Benchmark |
| 2024.02 | [ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution](https://arxiv.org/pdf/2402.01145) | TSP,VRP,OP, MKP,BPP,EDA | Algorithm |
| 2024.02 | [AutoSAT: Automatically Optimize SAT Solvers via Large Language Models](https://arxiv.org/pdf/2402.10705) | SAT | Algorithm |
| 2024.02 | [From Large Language Models and Optimization to Decision Optimization CoPilot: A Research Manifesto](https://arxiv.org/pdf/2402.16269) | MILP | Formulation |
| 2024.03 | [How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems](https://arxiv.org/pdf/2403.01757) | VRP | Solution |
| 2024.03 | [RouteExplainer: An Explanation Framework for Vehicle Routing Problem](https://arxiv.org/pdf/2403.03585.pdf) | VRP | Interpretability |
| 2024.03 | [Can Large Language Models Solve Robot Routing?](https://arxiv.org/pdf/2403.10795) | TSP,VRP | Algorithm |
| 2024.05 | [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://arxiv.org/pdf/2401.02051) | TSP,BPP,FSSP | Algorithm |
| 2024.05 | [ORLM: Training Large Language Models for Optimization Modeling](https://arxiv.org/pdf/2405.17743) | General OPT | Formulation |
| 2024.05 | [Self-Guiding Exploration for Combinatorial Problems](https://arxiv.org/pdf/2405.17950) | TSP,VRP,BPP, AP,KP,JSSP | Solution |
| 2024.06 | [Eyeballing Combinatorial Problems: A Case Study of Using Multimodal Large Language Models to Solve Traveling Salesman Problems](https://arxiv.org/pdf/2406.06865) | TSP | Solution |
| 2024.07 | [Visual Reasoning and Multi-Agent Approach in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP Combinatorial Challenges](https://arxiv.org/pdf/2407.00092) | TSP,mTSP | Solution |
| 2024.07 | [Solving General Natural-Language-Description Optimization Problems with Large Language Models](https://arxiv.org/pdf/2407.07924) | MILP | Formulation |
| 2024.08 | [Diagnosing Infeasible Optimization Problems Using Large Language Models](https://arxiv.org/pdf/2308.12923) | MILP | Formulation |
| 2024.08 | [LLMs can Schedule](https://arxiv.org/pdf/2408.06993) | JSSP | Solution |
| 2024.09 | [Multi-objective Evolution of Heuristic Using Large Language Model](https://arxiv.org/pdf/2409.16867) | TSP,BPP | Algorithm |
| 2024.10 | [Towards Foundation Models for Mixed Integer Linear Programming](https://arxiv.org/pdf/2410.08288) | MILP | Formulation |
| 2024.10 | [LLMOPT: Learning to Define and Solve General Optimization Problems from Scratch](https://arxiv.org/pdf/2410.13213) | General OPT | Formulation |
| 2024.10 | [OptiBench: Benchmarking Large Language Models in Optimization Modeling with Equivalence-Detection Evaluation](https://openreview.net/forum?id=KD9F5Ap878) | MILP | Benchmark |
| 2024.10 | [OptiBench Meets ReSocratic: Measure and Improve LLMs for Optimization Modeling](https://openreview.net/forum?id=fsDZwS49uY) | MILP | Benchmark |
| 2024.10 | [DRoC: Elevating Large Language Models for Complex Vehicle Routing via Decomposed Retrieval of Constraints](https://openreview.net/forum?id=s9zoyICZ4k) | 48VRPs | Formulation |
| 2024.10 | [STARJOB: Dataset for LLM-Driven Job Shop Scheduling](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | Solution |
| 2024.10 | [LLM4Solver: Large Language Model for Efficient Algorithm Design of Combinatorial Optimization Solver](https://openreview.net/forum?id=XTxdDEFR6D) | MILP | Algorithm |
| 2024.10 | [Unifying All Species: LLM-based Hyper-Heuristics for Multi-objective Optimization](https://openreview.net/forum?id=sUywd7UhFT) | TSP | Algorithm |
| 2024.10 | [Evo-Step: Evolutionary Generation and Stepwise Validation for Optimizing LLMs in OR](https://openreview.net/forum?id=aapUBU9U0D) | MILP | Formulation |
| 2024.10 | [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/pdf/2410.22657) | DyJSSP | Algorithm |
| 2024.11 | [Large Language Models for Combinatorial Optimization of Design Structure Matrix](https://arxiv.org/pdf/2411.12571) | DSM | Solution |
| 2024.12 | [HSEvo: Elevating Automatic Heuristic Design with Diversity-Driven Harmony Search and Genetic Algorithm Using LLMs](https://arxiv.org/pdf/2412.14995) | TSP,BPP,OP | Algorithm |
| 2024.12 | [Evaluating LLM Reasoning in the Operations Research Domain with ORQA](https://arxiv.org/pdf/2412.17874) | General OR | Benchmark |
| 2024.12 | [QUBE: Enhancing Automatic Heuristic Design via Quality-Uncertainty Balanced Evolution](https://arxiv.org/pdf/2412.20694) | OBP,TSP,CSP | Algorithm |
| 2025.01 | [Monte Carlo Tree Search for Comprehensive Exploration in LLM-Based Automatic Heuristic Design](https://arxiv.org/pdf/2501.08603) | TSP,CVRP,KP, BPP,MKP,ASP | Algorithm |
| 2025.01 | [Bridging Visualization and Optimization: Multimodal Large Language Models on Graph-Structured Combinatorial Optimization](https://arxiv.org/pdf/2501.11968) | Influence Maximization, Network Dismantling | Algorithm |
| 2025.01 | [Can Large Language Models Be Trusted as Black-Box Evolutionary Optimizers for Combinatorial Problems?](https://arxiv.org/pdf/2501.15081) | Influence Maximization | Algorithm |
| 2025.02 | [Improving Existing Optimization Algorithms with LLMs](https://arxiv.org/pdf/2502.08298) | MIS | Algorithm |
| 2025.02 | [Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search for Automating Heuristic Optimization](https://arxiv.org/pdf/2502.11422) | TSP,FSSP | Algorithm |
| 2025.02 | [GraphThought: Graph Combinatorial Optimization with Thought Generation](https://arxiv.org/pdf/2502.11607) | MIS,MVC,TSP | Algorithm |
| 2025.02 | [EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations](https://arxiv.org/pdf/2502.14760) | MILP | Algorithm |
| 2025.02 | [ARS: Automatic Routing Solver with Large Language Models](https://arxiv.org/pdf/2502.15359) | VRP | Benchmark & Algorithm |
| 2025.02 | [Text2Zinc: A Cross-Domain Dataset for Modeling Optimization and Satisfaction Problems in MiniZinc](https://arxiv.org/pdf/2503.10642) | LP,MIP,CP | Formulation (Dataset) |
| 2025.02 | [GraphArena: Evaluating and Exploring Large Language Models on Graph Computation](https://openreview.net/pdf?id=Y1r9yCMzeA) | MVC,MIS,MCP, TSP,MCS,GED | Benchmark & Dataset & Model |
| 2025.03 | [Leveraging Large Language Models to Develop Heuristics for Emerging Optimization Problems](https://arxiv.org/pdf/2503.03350) | UPMP | Algorithm |
| 2025.03 | [OR-LLM-Agent: Automating Modeling and Solving of Operations Research Optimization Problem with Reasoning Large Language Model](https://arxiv.org/pdf/2503.10009) | OP | Formulation |
| 2025.03 | [Combinatorial Optimization for All: Using LLMs to Aid Non-Experts in Improving Optimization Algorithms](https://arxiv.org/pdf/2503.10968) | TSP | Algorithm |
| 2025.03 | [Automatic MILP Model Construction for Multi-Robot Task Allocation and Scheduling Based on Large Language Models](https://arxiv.org/pdf/2503.13813) | MILP | Formulation |
| 2025.03 | [Code Evolution Graphs: Understanding Large Language Model Driven Design of Algorithms](https://arxiv.org/pdf/2503.16668) | BBO,TSP,BPP | Interpretability |
| 2025.04 | [CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization](https://arxiv.org/pdf/2504.04310) | General COP | Benchmark |
| 2025.04 | [Algorithm Discovery With LLMs: Evolutionary Search Meets Reinforcement Learning](https://arxiv.org/pdf/2504.05108) | BPP,TSP,FP | Algorithm |
| 2025.04 | [OptimAI: Optimization from Natural Language Using LLM-Powered AI Agents](https://arxiv.org/pdf/2504.16918) | MILP | Formulation |
| 2025.04 | [Fitness Landscape of Large Language Model-Assisted Automated Algorithm Search](https://arxiv.org/pdf/2504.19636) | OBP,TSP, CVRP,VRPTW | Benchmark & Interpretability |
| 2025.04 | [Large Language Models powered Neural Solvers for Generalized Vehicle Routing Problems](https://openreview.net/pdf?id=EVqlVjvlt8) | VRP | Algorithm |
| 2025.05 | [Efficient Heuristics Generation for Solving Combinatorial Optimization Problems Using Large Language Models](https://arxiv.org/pdf/2505.12627v1) | TSP,CVRP,BPP, MKP,OP | Algorithm |
| 2025.05 | [CALM: Co-evolution of Algorithms and Language Model for Automatic Heuristic Design](https://arxiv.org/pdf/2505.12285v1) | TSP,KP,OBP,OP | Algorithm |
| 2025.05 | [Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design](https://arxiv.org/pdf/2505.16979) | KP | Solution |
| 2025.05 | [A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization](https://arxiv.org/pdf/2505.16952) | MIS,MDS, TSP,CVRP,CFLP, CPMP,FJSP,STP | Benchmark |
| 2025.05 | [LLM-Meta-SR: In-Context Learning for Evolving Selection Operators in Symbolic Regression](https://arxiv.org/abs/2505.18602) | SR | Algorithm |
| 2025.05 | [RedAHD: Reduction-Based End-to-End Automatic Heuristic Design with Large Language Models](https://arxiv.org/pdf/2505.20242) | TSP,CVRP, KP,BPP,MKP | Algorithm |
| 2025.05 | [Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization](https://arxiv.org/pdf/2505.20881) | TSP,CVRP,BPP | Algorithm |
| 2025.05 | [Large Language Model-driven Large Neighborhood Search for Large-Scale MILP Problems](https://openreview.net/pdf?id=teUg2pMrF0) | MILP | Algorithm |
| 2025.05 | [Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling](https://arxiv.org/pdf/2505.11792) | General OPT | Formulation |
| 2025.06 | [LLM-Driven Instance-Specific Heuristic Generation and Selection](https://arxiv.org/pdf/2506.00490) | OBPP,CVRP | Algorithm |
| 2025.06 | [ORMind: A Cognitive-Inspired End-to-End Reasoning Framework for Operations Research](https://arxiv.org/pdf/2506.01326) | OR | Formulation |
| 2025.06 | [EALG: Evolutionary Adversarial Generation of Language Model–Guided Generators for Combinatorial Optimization](https://arxiv.org/pdf/2506.02594) | TSP,OP | Algorithm |
| 2025.06 | [CP-Bench: Evaluating Large Language Models for Constraint Modelling](https://arxiv.org/pdf/2506.06052) | CP | Benchmark |
| 2025.06 | [REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](https://arxiv.org/pdf/2506.07759) | FJSSP | Algorithm |
| 2025.06 | [HeuriGym: An Agentic Benchmark for LLM-Crafted Heuristics in Combinatorial Optimization](https://arxiv.org/pdf/2506.07972) | TSP,SAT | Benchmark |
| 2025.06 | [ALE-Bench: A Benchmark for Long-Horizon Objective-Driven Algorithm Engineering](https://arxiv.org/pdf/2506.09050) | General OPT | Benchmark |
| 2025.06 | [OPT-BENCH: Evaluating LLM Agent on Large-Scale Search Spaces Optimization Problems](https://arxiv.org/pdf/2506.10764) | GCP,KP,MCP, MIS,SCP,TSP | Benchmark |
| 2025.06 | [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/pdf/2506.11057) | SAT | Solution |
| 2025.06 | [AlphaEvolve: A coding agent for scientific and algorithmic discovery](https://arxiv.org/pdf/2506.13131) | OPT | Algorithm |
| 2025.06 | [OpenEvolve: an open-source evolutionary coding agent](https://huggingface.co/blog/codelion/openevolve) | OPT | Algorithm |
| 2025.06 | [HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges](https://arxiv.org/pdf/2506.15196) | TSP,CVRP,JSSP, MaxCut,MKP | Algorithm |
| 2025.07 | [Large Language Models for Combinatorial Optimization: A Systematic Review](https://arxiv.org/pdf/2507.03637) | CO | Review |
| 2025.07 | [Fine-tuning Large Language Model for Automated Algorithm Design](https://arxiv.org/pdf/2507.10614) | ASP,TSP,CVRP | Algorithm |
| 2025.07 | [DHEvo: Data-Algorithm Based Heuristic Evolution for Generalizable MILP Solving](https://arxiv.org/pdf/2507.15615) | MILP | Algorithm |
| 2025.07 | [MeLA: A Metacognitive LLM-Driven Architecture for Automatic Heuristic Design](https://arxiv.org/pdf/2507.20541) | TSP,BPP,ACS | Algorithm |
| 2025.07 | [Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](https://arxiv.org/pdf/2507.20923) | TSP,CVRP,KP | Algorithm |
| 2025.07 | [Automatically discovering heuristics in a complex SAT solver with large language models](https://arxiv.org/pdf/2507.22876) | SAT | Algorithm |
| 2025.07 | [Nested-Refinement Metamorphosis: Reflective Evolution for Efficient Optimization of Networking Problems](https://aclanthology.org/2025.findings-acl.895.pdf) | TSP,MKP,CVRP | Algorithm |
| 2025.08 | [ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection](https://arxiv.org/pdf/2508.01724) | DFJSP | Solution |
| 2025.08 | [OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling](https://arxiv.org/pdf/2508.02503) | MDVRP,WSCP | Formulation |
| 2025.08 | [EoH-S: Evolution of Heuristic Set using LLMs for Automated Heuristic Design](https://arxiv.org/pdf/2508.03082) | OBP,TSP,CVRP | Algorithm |
| 2025.08 | [X-evolve: Solution space evolution powered by large language models](https://arxiv.org/pdf/2508.07932) | CSP,BPP, Shannon capacity | Algorithm |
| 2025.08 | [EvoCut: Strengthening Integer Programs via Evolution-Guided Language Models](https://arxiv.org/pdf/2508.11850) | MILP | Formulation |
| 2025.08 | [HIFO-PROMPT: Prompting with Hindsight and Foresight For LLM-Based Automatic Heuristic Design](https://arxiv.org/pdf/2508.13333) | TSP,OBP,FSSP | Algorithm |
| 2025.08 | [MOTIF: Multi-strategy Optimization via Turn-based Interactive Framework](https://arxiv.org/abs/2508.03929) | TSP,CVRP,OP,BPP,MKP | Algorithm |
| 2025.09 | [LLM-QUBO: An End-to-End Framework for Automated QUBO Transformation from Natural Language Problem Descriptions](https://arxiv.org/pdf/2509.00099) | MILP | Formulation |
| 2025.09 | [AutoPBO: LLM-powered Optimization for Local Search PBO Solvers](https://arxiv.org/pdf/2509.04007) | PBO | Solution |
| 2025.09 | [Autonomous Code Evolution MeetsNP-Completeness](https://arxiv.org/pdf/2509.07367) | SAT | Solution |
| 2025.09 | [LLM-based Instance-driven Heuristic Bias in the Context of a Biased Random Key Genetic Algorithm](https://arxiv.org/pdf/2509.09707) | LRS | Algorithm |
| 2025.09 | [Learn to Relax with Large Language Models: Solving Nonlinear Combinatorial Optimization Problems via Bidirectional Coevolution](https://arxiv.org/pdf/2509.12643) | MDD,SFL,TSPTW | Algorithm |
| 2025.09 | [DaSAThco: Data-Aware SAT Heuristics Combinations Optimization via Large Language Models](https://arxiv.org/pdf/2509.12602) | SAT | Algorithm |
| 2025.09 | [Large Language Models as End-to-end Combinatorial Optimization Solvers](https://arxiv.org/pdf/2509.16865) | TSP,OP,CVRP, MIS,MVC,PFSP,JSSP | Solution |
| 2025.09 | [Large Language Models and Operations Research: A Structured Survey](https://arxiv.org/pdf/2509.18180) | OR | Review |
| 2025.09 | [ShinkaEvolve: Towards Open-Ended And Sample-Efficient Program Evolution](https://arxiv.org/pdf/2509.19349) | OPT | Algorithm |
| 2025.09 | [StepORLM: A Self-Evolving Framework With Generative Process Supervision For Operations Research Language Models](https://arxiv.org/pdf/2509.22558) | OR | Formulation |
| 2025.09 | [OptiMind: Teaching LLMs to Think Like Optimization Experts](https://arxiv.org/pdf/2509.22979) | MILP | Formulation |
| 2025.09 | [AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms](https://arxiv.org/pdf/2509.23189) | TSP,CVRP,FSSP | Algorithm |
| 2025.09 | [ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems](https://arxiv.org/pdf/2509.23465) | TSP | Algorithm |
| 2025.09 | [Experience-guided reflective co-evolution of prompts and heuristics for automatic algorithm design](https://arxiv.org/pdf/2509.24509) | TSP,BPP | Algorithm |
| 2025.10 | [EvoSpeak: Large Language Models for Interpretable Genetic Programming-Evolved Heuristics](https://arxiv.org/pdf/2510.02686) | DFJSS | Algorithm |
| 2025.10 | [VRPAgent: LLM-Driven Discovery of Heuristic Operators for Vehicle Routing Problems](https://arxiv.org/pdf/2510.07073) | VRP | Algorithm |
| 2025.10 | [Refining Hybrid Genetic Search for CVRP via Reinforcement Learning-Finetuned LLM](https://arxiv.org/pdf/2510.11121) | CVRP | Algorithm |
| 2025.10 | [CodeEvolve: an open source evolutionary coding agent for algorithm discovery and optimization](https://arxiv.org/pdf/2510.14150) | OPT | Algorithm |
| 2025.10 | [An Agentic Framework with LLMs for Solving Complex Vehicle Routing Problems](https://arxiv.org/pdf/2510.16701) | VRP | Formulation |
| 2025.10 | [OptiTree: Hierarchical Thoughts Generation with Tree Search for LLM Optimization Modeling](https://arxiv.org/pdf/2510.22192) | OR | Formulation |
| 2025.10 | [Glia: A Human-Inspired AI for Automated Systems Design and Optimization](https://arxiv.org/abs/2510.27176) | Systems routing + scheduling | Algorithm |
| 2025.10 | [Discovering Heuristics with Large Language Models (LLMs) for Mixed-Integer Programs: Single-Machine Scheduling](https://arxiv.org/pdf/2510.24013) | SMTT | Algorithm |
| 2025.11 | [Bridging Synthetic and Real Routing Problems via LLM-Guided Instance Generation and Progressive Adaptation](https://arxiv.org/pdf/2511.10233) | TSP,CVRP | Algorithm |
| 2025.11 | [irace-evo: Automatic Algorithm Configuration Extended With LLM-Based Code Evolution](https://arxiv.org/pdf/2511.14794) | VSBPP | Algorithm |
| 2025.11 | [LLM4EO: Large Language Model for Evolutionary Optimization in Flexible Job Shop Scheduling](https://arxiv.org/pdf/2511.16485) | FJSP | Algorithm |
| 2025.11 | [ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention](https://openreview.net/pdf?id=f0TBAdcJ8m) | FSSP,JSSP,BPP, KP,TSP,VRP | Solution & Benchmark |
| 2025.11 | [AutoFloorplan: Evolving Heuristics for Chip Floorplanning with Large Language Models and Textual Gradient-Guided Repair](https://openreview.net/pdf?id=DS2iool3nv) | Floorplanning | Algorithm |
| 2025.11 | [TPD-AHD: Textual Preference Differentiation for LLM-Based Automatic Heuristic Design](https://openreview.net/pdf?id=VEMknlIPtM) | TSP,CVRP,JSSP, MKP, VRP,MASP,CFLP | Algorithm |
| 2025.11 | [ALIGNING LLMS WITH GRAPH NEURAL SOLVERS FOR COMBINATORIAL OPTIMIZATION](https://openreview.net/pdf?id=KSfLDk3jxI) | TSP,CVRP,KP, MVCP,MISP | Algorithm |
| 2025.11 | [Large Language Model Guided Dynamic Branching Rule Scheduling in Branch-and-Bound](https://openreview.net/pdf?id=8LCdjf7uIk) | MILP | Algorithm |
| 2025.11 | [Online Algorithm Configuration for MILP Re-Optimization with LLM Guidance](https://openreview.net/pdf?id=xbyebbS1ZF) | MILP | Algorithm |
| 2025.11 | [Experience-Guided Reflective Co-Evolution of Prompts and Heuristics for Automatic Algorithm Design](https://openreview.net/pdf?id=oD9RwlFqEE) | TSP, BPP | Algorithm |
| 2025.11 | [Adversarial examples for heuristics in combinatorial optimization: An LLM based approach](https://openreview.net/pdf?id=fasU6t3hL4) | KP,BPP | Algorithm |
| 2025.11 | [Rethinking Code Similarity for Automated Algorithm Design with LLMs](https://openreview.net/pdf?id=HIUqeO9OOr) | ASP,TSP,CPP | Algorithm |
| 2025.11 | [AutoMOAE: Multi-Objective Auto-Algorithm Evolution](https://openreview.net/pdf?id=G8tP1Z9dLy) | GCP,TSP | Algorithm |
| 2025.11 | [Fusing LLMs with Scientific Literature for Heuristic Discovery](https://openreview.net/pdf?id=lwqeXDYKWJ) | TSP | Algorithm |
| 2025.11 | [Cognitively Inspired Reflective Evolution: Interactive Multi-Turn LLM–EA Synthesis of Heuristics for Combinatorial Optimization](https://openreview.net/pdf?id=31VTD5pS2v) | TSP,BPP | Algorithm |
| 2025.11 | [Hierarchical Representations for Cross-task Automated Heuristic Design using LLMs](https://openreview.net/pdf?id=dgvx86qybJ) | TSP,CVRP,FSSP, BPP,ASP | Algorithm |
| 2025.11 | [ThetaEvolve: Test-time Learning on Open Problems](https://arxiv.org/abs/2511.23473) | HadamardMatrix, CirclePacking | Algorithm |
| 2025.11 | [Leveraging large language models for efficient scheduling in Human–Robot collaborative flexible manufacturing systems](https://doi.org/10.1038/s44334-025-00061-w) | DFJSP | Algorithm |
| 2025.12 | [RoCo: Role-Based LLMs Collaboration for Automatic Heuristic Design](https://arxiv.org/pdf/2512.03762) | TSP,OP,CVRP, MKP,offline BPP | Algorithm |
| 2025.12 | [CogMCTS: A Novel Cognitive-Guided Monte Carlo Tree Search Framework for Iterative Heuristic Evolution with Large Language Models](https://arxiv.org/pdf/2512.08609) | TSP,OP,CVRP, MKP,KP | Algorithm |
| 2025.12 | [Behavior and Representation in Large Language Models for Combinatorial Optimization: From Feature Extraction to Algorithm Selection](https://arxiv.org/pdf/2512.13374) | BP,GCP,JSP,KP | Analysis |
| 2025.12 | [LAPPI: Interactive Optimization with LLM-Assisted Preference-Based Problem Instantiation](https://arxiv.org/pdf/2512.14138) | TSP,OP | Formulation |
| 2026.01 | [DRAGON: LLM-Driven Decomposition and Reconstruction Agents for Large-Scale Combinatorial Optimization](https://arxiv.org/pdf/2601.06502) | TSP,CVRP, BPP,MKP | Algorithm |
| 2026.01 | [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/pdf/2601.15738) | Dynamic FAFSP | Algorithm |
| 2026.01 | [Evolving Interdependent Operators with Large Language Models for Multi-Objective Combinatorial Optimization](https://arxiv.org/pdf/2601.17899) | MOCP | Algorithm |
| 2026.01 | [Algorithmic Prompt-Augmentation for Efficient LLM-Based Heuristic Design for A* Search](https://arxiv.org/pdf/2601.19622) | SP,ULPMP | Algorithm |
| 2026.01 | [Rethinking LLM-Driven Heuristic Design: Generating Efficient and Specialized Solvers via Dynamics-Aware Optimization](https://arxiv.org/pdf/2601.20868) | TSP,CVRP, BPP,MKP | Algorithm |
| 2026.01 | [PathWise: Planning through World Model for Automated Heuristic Design via Self-Evolving LLMs](https://arxiv.org/pdf/2601.20539) | TSP,KP,CVRP, MKP,OP,BPP | Algorithm |
| 2026.01 | [LLaMEA-SAGE: Guiding Automated Algorithm Design with Structural Feedback from Explainable AI](https://arxiv.org/pdf/2601.21511) | BBO | Algorithm |
| 2026.01 | [Game-Theoretic Co-Evolution for LLM-Based Heuristic Discovery](https://arxiv.org/pdf/2601.22896) | OBP,TSP,CVRP | Algorithm |
| 2026.01 | [Beyond the Node: Clade-level Selection for Efficient MCTS in Automatic Heuristic Design](https://arxiv.org/pdf/2602.00549) | TSP,KP,CVRP, MKP,BPP | Algorithm |
| 2026.02 | [Hard Constraints Meet Soft Generation: Guaranteed Feasibility for LLM-based Combinatorial Optimization](https://arxiv.org/pdf/2602.01090) | TSP,OP,CVRP,MIS, MVC,PFSP,JSSP | Solution |
| 2026.02 | [Reasoning in a Combinatorial and Constrained World: Benchmarking LLMs on Natural-Language Combinatorial Optimization](https://arxiv.org/pdf/2602.02188) | COP | Benchmark |
| 2026.02 | [G-LNS: Generative Large Neighborhood Search for LLM-Based Automatic Heuristic Design](https://arxiv.org/pdf/2602.08253) | TSP,CVRP,OVRP | Algorithm |
| 2026.02 | [OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery](https://arxiv.org/pdf/2602.13869) | OR | Solution |
| 2026.02 | [Heuristic Search as Language-Guided Program Optimization](https://arxiv.org/pdf/2602.16038) | PDPTW,crew pairing, TMP,Intra-op scheduling | Algorithm |
| 2026.02 | [AdaEvolve: Adaptive LLM Driven Zeroth-Order Optimization](https://arxiv.org/pdf/2602.20133) | CirclePacking | Algorithm |
| 2026.02 | [ConstraintBench: Benchmarking LLM Constraint Reasoning on Direct Optimization](https://arxiv.org/pdf/2602.22465) | Optimization | Benchmark |
| 2026.02 | [Enhancing CVRP Solver through LLM-driven Automatic Heuristic Design](https://arxiv.org/pdf/2602.23092) | CVRP | Algorithm |
| 2026.02 | [EvoX: Meta-Evolution for Automated Discovery](https://arxiv.org/pdf/2602.23413) | CirclePacking | Algorithm |
| 2026.03 | [From Heuristic Selection to Automated Algorithm Design: LLMs Benefit from Strong Priors](https://arxiv.org/pdf/2603.02792) | OneMax | Method |

</details>

## 부록 B. End-to-end / Solution 계열 논문 전체 목록

<details>
<summary>Solution-type 논문 목록 펼치기</summary>

| Date | Paper | Problem | Type |
|---|---|---|---|
| 2023.09 | [Can Language Models Solve Graph Problems in Natural Language?](https://arxiv.org/pdf/2305.10037) | Graph | Solution |
| 2023.09 | [Large Language Models as Optimizers](https://arxiv.org/pdf/2309.03409) | TSP | Solution |
| 2023.11 | [Large Language Models as Evolutionary Optimizers](https://arxiv.org/pdf/2310.19046) | TSP | Solution |
| 2024.03 | [How Multimodal Integration Boost the Performance of LLM for Optimization: Case Study on Capacitated Vehicle Routing Problems](https://arxiv.org/pdf/2403.01757) | VRP | Solution |
| 2024.05 | [Self-Guiding Exploration for Combinatorial Problems](https://arxiv.org/pdf/2405.17950) | TSP,VRP,BPP, AP,KP,JSSP | Solution |
| 2024.06 | [Eyeballing Combinatorial Problems: A Case Study of Using Multimodal Large Language Models to Solve Traveling Salesman Problems](https://arxiv.org/pdf/2406.06865) | TSP | Solution |
| 2024.07 | [Visual Reasoning and Multi-Agent Approach in Multimodal Large Language Models (MLLMs): Solving TSP and mTSP Combinatorial Challenges](https://arxiv.org/pdf/2407.00092) | TSP,mTSP | Solution |
| 2024.08 | [LLMs can Schedule](https://arxiv.org/pdf/2408.06993) | JSSP | Solution |
| 2024.10 | [STARJOB: Dataset for LLM-Driven Job Shop Scheduling](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | Solution |
| 2024.11 | [Large Language Models for Combinatorial Optimization of Design Structure Matrix](https://arxiv.org/pdf/2411.12571) | DSM | Solution |
| 2025.05 | [Know the Ropes: A Heuristic Strategy for LLM-based Multi-Agent System Design](https://arxiv.org/pdf/2505.16979) | KP | Solution |
| 2025.06 | [STRCMP: Integrating Graph Structural Priors with Language Models for Combinatorial Optimization](https://arxiv.org/pdf/2506.11057) | SAT | Solution |
| 2025.08 | [ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection](https://arxiv.org/pdf/2508.01724) | DFJSP | Solution |
| 2025.09 | [AutoPBO: LLM-powered Optimization for Local Search PBO Solvers](https://arxiv.org/pdf/2509.04007) | PBO | Solution |
| 2025.09 | [Autonomous Code Evolution MeetsNP-Completeness](https://arxiv.org/pdf/2509.07367) | SAT | Solution |
| 2025.09 | [Large Language Models as End-to-end Combinatorial Optimization Solvers](https://arxiv.org/pdf/2509.16865) | TSP,OP,CVRP, MIS,MVC,PFSP,JSSP | Solution |
| 2025.11 | [ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention](https://openreview.net/pdf?id=f0TBAdcJ8m) | FSSP,JSSP,BPP, KP,TSP,VRP | Solution & Benchmark |
| 2026.02 | [Hard Constraints Meet Soft Generation: Guaranteed Feasibility for LLM-based Combinatorial Optimization](https://arxiv.org/pdf/2602.01090) | TSP,OP,CVRP,MIS, MVC,PFSP,JSSP | Solution |
| 2026.02 | [OR-Agent: Bridging Evolutionary Search and Structured Research for Automated Algorithm Discovery](https://arxiv.org/pdf/2602.13869) | OR | Solution |

</details>

## 부록 C. Scheduling 관련 논문 전체 목록

<details>
<summary>Scheduling/makespan 관련 논문 목록 펼치기</summary>

| Date | Paper | Problem | Type |
|---|---|---|---|
| 2023.10 | [AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling](https://arxiv.org/pdf/2309.13218) | JSSP | Formulation |
| 2024.05 | [Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model](https://arxiv.org/pdf/2401.02051) | TSP,BPP,FSSP | Algorithm |
| 2024.05 | [Self-Guiding Exploration for Combinatorial Problems](https://arxiv.org/pdf/2405.17950) | TSP,VRP,BPP, AP,KP,JSSP | Solution |
| 2024.08 | [LLMs can Schedule](https://arxiv.org/pdf/2408.06993) | JSSP | Solution |
| 2024.10 | [STARJOB: Dataset for LLM-Driven Job Shop Scheduling](https://openreview.net/forum?id=t0fU6t3Skw) | JSSP | Solution |
| 2024.10 | [Automatic programming via large language models with population self-evolution for dynamic job shop scheduling problem](https://arxiv.org/pdf/2410.22657) | DyJSSP | Algorithm |
| 2025.02 | [Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search for Automating Heuristic Optimization](https://arxiv.org/pdf/2502.11422) | TSP,FSSP | Algorithm |
| 2025.05 | [A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization](https://arxiv.org/pdf/2505.16952) | MIS,MDS, TSP,CVRP,CFLP, CPMP,FJSP,STP | Benchmark |
| 2025.06 | [REMoH: A Reflective Evolution of Multi-objective Heuristics approach via Large Language Models](https://arxiv.org/pdf/2506.07759) | FJSSP | Algorithm |
| 2025.06 | [HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges](https://arxiv.org/pdf/2506.15196) | TSP,CVRP,JSSP, MaxCut,MKP | Algorithm |
| 2025.08 | [ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection](https://arxiv.org/pdf/2508.01724) | DFJSP | Solution |
| 2025.08 | [HIFO-PROMPT: Prompting with Hindsight and Foresight For LLM-Based Automatic Heuristic Design](https://arxiv.org/pdf/2508.13333) | TSP,OBP,FSSP | Algorithm |
| 2025.09 | [Large Language Models as End-to-end Combinatorial Optimization Solvers](https://arxiv.org/pdf/2509.16865) | TSP,OP,CVRP, MIS,MVC,PFSP,JSSP | Solution |
| 2025.09 | [AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms](https://arxiv.org/pdf/2509.23189) | TSP,CVRP,FSSP | Algorithm |
| 2025.10 | [Discovering Heuristics with Large Language Models (LLMs) for Mixed-Integer Programs: Single-Machine Scheduling](https://arxiv.org/pdf/2510.24013) | SMTT | Algorithm |
| 2025.11 | [LLM4EO: Large Language Model for Evolutionary Optimization in Flexible Job Shop Scheduling](https://arxiv.org/pdf/2511.16485) | FJSP | Algorithm |
| 2025.11 | [ACCORD: Autoregressive Constraint-satisfying Generation for COmbinatorial Optimization with Routing and Dynamic attention](https://openreview.net/pdf?id=f0TBAdcJ8m) | FSSP,JSSP,BPP, KP,TSP,VRP | Solution & Benchmark |
| 2025.11 | [TPD-AHD: Textual Preference Differentiation for LLM-Based Automatic Heuristic Design](https://openreview.net/pdf?id=VEMknlIPtM) | TSP,CVRP,JSSP, MKP, VRP,MASP,CFLP | Algorithm |
| 2025.11 | [Hierarchical Representations for Cross-task Automated Heuristic Design using LLMs](https://openreview.net/pdf?id=dgvx86qybJ) | TSP,CVRP,FSSP, BPP,ASP | Algorithm |
| 2025.11 | [Leveraging large language models for efficient scheduling in Human–Robot collaborative flexible manufacturing systems](https://doi.org/10.1038/s44334-025-00061-w) | DFJSP | Algorithm |
| 2026.01 | [LLM-Assisted Automatic Dispatching Rule Design for Dynamic Flexible Assembly Flow Shop Scheduling](https://arxiv.org/pdf/2601.15738) | Dynamic FAFSP | Algorithm |
| 2026.02 | [Hard Constraints Meet Soft Generation: Guaranteed Feasibility for LLM-based Combinatorial Optimization](https://arxiv.org/pdf/2602.01090) | TSP,OP,CVRP,MIS, MVC,PFSP,JSSP | Solution |

</details>
