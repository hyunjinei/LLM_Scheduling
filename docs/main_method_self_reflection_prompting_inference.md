# 4.8.1 Self-reflection prompting

본 절은 `[action_code_add][fix]colab_03_inference_step_full.ipynb`의 추론 절차에 맞춘 설명이다. 여기서 self-reflection은 모델 파라미터를 업데이트하는 학습 절차가 아니라, test-time에서 이미 완성된 rollout을 사후 분석한 뒤 특정 의사결정 지점을 다시 유도하여 schedule suffix를 재생성하는 추론-time 개선 절차이다. 따라서 본 절의 방법은 RL, GRPO, BOPO와 구분되며, supervised fine-tuning 이후 고정된 정책 모델과 candidate score head를 그대로 사용한다.

## 논문 붙여넣기용 최종 정리본

본 연구의 self-reflection prompting은 1차 step-by-step rollout 이후 완성된 스케줄을 사후적으로 분석하여, 최종 makespan에 불리하게 작용했을 가능성이 높은 의사결정 단계를 다시 검토하는 test-time refinement 절차이다. 본 연구의 기본 추론 정책은 자유 형식 텍스트 생성을 통해 action을 생성하지 않고, 현재 step의 feasible action set에 포함된 후보 action code들을 각각 candidate-scoring query로 평가한 뒤 가장 높은 점수 또는 sampling된 후보를 선택하는 reranking 구조를 사용한다. Self-reflection 역시 동일한 candidate-scoring 정책 위에서 작동하며, 추가적인 파라미터 업데이트 없이 prompt context와 step-level guidance를 통해 후속 rollout의 선택 분포를 국소적으로 조정한다.

먼저 초기 rollout을 다음과 같이 정의한다.

$$
\tau^{(0)} = \left(c_0^{(0)}, c_1^{(0)}, \ldots, c_{T-1}^{(0)}\right)
\tag{16}
$$

각 step \(t\)에서 환경은 현재 schedule state \(x_t\)와 feasible action set \(\mathcal{C}_t\)를 제공한다. Candidate scoring 모델은 각 후보 \(c \in \mathcal{C}_t\)에 대해 별도의 scoring query를 구성하고, \(\langle \mathrm{CAND\_SCORE} \rangle\) 위치의 hidden representation을 candidate score head에 통과시켜 다음과 같이 scalar score를 계산한다.

$$
s_{\theta,\phi}(x_t,c) = g_\phi\left(h_\theta(x_t,c)\right)
\tag{17}
$$

여기서 \(\theta\)는 LoRA가 적용된 base language model의 파라미터, \(\phi\)는 candidate score head의 파라미터, \(h_\theta(x_t,c)\)는 해당 후보를 평가하는 query에서 \(\langle \mathrm{CAND\_SCORE} \rangle\) 위치의 마지막 hidden vector이다. 추론 시 후보 선택 확률은 feasible candidate set 내부에서만 softmax로 정규화되며, \(T_s\)를 candidate-scoring temperature라고 할 때 다음과 같이 정의한다.

$$
p_{\theta,\phi}(c \mid x_t,\mathcal{C}_t)
=
\frac{\exp\left(s_{\theta,\phi}(x_t,c)/T_s\right)}
{\sum_{c' \in \mathcal{C}_t}\exp\left(s_{\theta,\phi}(x_t,c')/T_s\right)}
\tag{18}
$$

초기 rollout이 종료되면, 각 step의 선택 결과와 모든 후보의 one-step transition estimate를 이용해 critical step을 식별한다. 본 구현에서 critical step은 현재 선택된 action \(c_t^{(0)}\)와 가장 강한 대안 후보 \(b_t\)를 비교하여 선택한다. 대안 후보 \(b_t\)는 projected makespan, estimated start time, processing time 순으로 가장 유리한 후보로 정의된다. 각 step에 대해 다음과 같은 regret-like signal을 계산한다.

$$
\Delta_t^{\mathrm{proj}}
= \widehat{C}_t(c_t^{(0)}) - \widehat{C}_t(b_t),
\qquad
J_t = C_t^{\mathrm{after}} - C_t^{\mathrm{before}},
\qquad
D_t = \widehat{S}_t(c_t^{(0)}) - \widehat{S}_t(b_t)
\tag{19}
$$

여기서 \(\widehat{C}_t(c)\)는 step \(t\)에서 action \(c\)를 선택했을 때 예상되는 projected makespan, \(J_t\)는 실제 rollout에서 해당 step 이후 발생한 makespan 증가분, \(\widehat{S}_t(c)\)는 action \(c\)의 estimated start time이다. 구현 상으로는 \((\Delta_t^{\mathrm{proj}}, J_t, D_t, t)\)를 lexicographic score로 사용하여 내림차순 정렬하고, 상위 \(K\)개 step을 reflection 대상으로 선정한다.

선택된 critical step들에 대해서는 reflection memory가 생성된다. Reflection memory는 episode 전체 결과와 step-level diagnostics를 함께 포함한다. 구체적으로 현재 final makespan, critical step 수, makespan 증가 step 수, 반복적으로 등장하는 bottleneck machine 후보, 기존 선택 action의 start/end/idle gap/job wait, 그리고 대안 후보의 projected makespan, estimated start, delta Cmax, post-route 정보를 요약한다. 이 memory는 다음 guided rollout에서 step prompt에 추가되어, 모델이 이전 rollout의 실패 패턴을 조건부 문맥으로 인식하도록 한다.

Reflection prompt는 각 critical step에 대해 별도로 구성된다. Prompt에는 원래 step state, 이전 rollout에서 선택된 action code, feasible action code 목록, 선택 후보와 대안 후보의 transition diagnostics, 그리고 reflection memory가 포함된다. 그러나 이 prompt의 목적은 모델이 임의의 자연어 답변을 생성하도록 하는 것이 아니다. 현재 구현에서는 reflection prompt 역시 candidate-scoring reranker의 입력으로 사용된다. 즉, reflection prompt가 주어진 상태에서 feasible action code들을 다시 점수화하고, 더 나은 후보 action code를 선택한다.

만약 reflection prompt를 조건으로 한 candidate-scoring 재평가가 기존 선택과 다른 유효한 대체 action \(\tilde{c}_t\)를 선택하면, deterministic controller는 이 결과를 다음과 같은 step-level guidance로 변환한다.

$$
G_t = \left\{j_t^{+}, j_t^{-}, r_t\right\}
\tag{20}
$$

$$
j_t^{+} = \mathrm{job}(\tilde{c}_t),
\qquad
j_t^{-} = \mathrm{job}(c_t^{(0)})
\tag{21}
$$

여기서 \(\tilde{c}_t\)는 reflection prompt를 조건으로 한 candidate-scoring 재평가 또는 deterministic fallback이 제안한 대체 action code이며, \(c_t^{(0)}\)는 초기 rollout에서 실제 선택되었던 action code이다. \(j_t^{+}\)는 다음 rollout에서 선호하도록 유도되는 job이고, \(j_t^{-}\)는 hard constraint로 금지되는 job이 아니라 prompt상 비선호 신호를 받는 이전 선택 job이다. \(r_t\)는 이러한 변경을 유도하는 hindsight reason을 의미한다. 따라서 guidance는 LLM이 직접 작성한 독립적인 출력물이 아니라, reflection 재평가 결과를 deterministic controller가 prompt-level hint로 구조화한 것이다.

이 guidance는 최종 스케줄을 즉시 수정하는 hard replacement가 아니다. 실제 개선 스케줄은 prefix replay와 해당 step 이후의 남은 스케줄 구간에 대한 guided re-rollout을 통해 다시 생성된다. 먼저 guidance가 존재하는 가장 이른 step을 \(t_0\)라고 하면, \(0 \le t < t_0\) 구간은 초기 rollout의 선택 job을 그대로 replay한다. 이는 초반 prefix가 달라져 비교 자체가 불안정해지는 것을 막고, reflection이 겨냥한 의사결정 지점 이후의 효과를 평가하기 위함이다. 이후 \(t \ge t_0\) 구간에서는 reflection memory와 step guidance를 prompt에 추가한 상태로 candidate-scoring 정책을 다시 실행한다. 따라서 self-reflection은 단순히 한 step만 바꾸고 나머지를 고정하는 방식이 아니라, 문제의 prefix를 보존한 뒤 bottleneck 후보 step부터 해당 step 이후의 남은 스케줄 구간을 다시 생성하는 guided re-rollout의 성격을 가진다.

각 reflection pass에서 생성된 candidate schedule \(\tau'\)는 현재의 최적 스케줄 \(\tau^{\mathrm{best}}\)와 비교되어 다음과 같은 acceptance rule에 의해 업데이트된다.

$$
\tau^{\mathrm{best}} \leftarrow
\begin{cases}
\tau', & \text{if } M(\tau') < M(\tau^{\mathrm{best}}),\\
\tau^{\mathrm{best}}, & \text{otherwise.}
\end{cases}
\tag{22}
$$

이러한 inference-time refinement 절차는 별도의 파라미터 업데이트 없이도 1차 rollout에서 발생한 불리한 의사결정을 사후적으로 식별하고, bottleneck activation 지연이나 과도한 대기 시간을 완화하여 스케줄의 품질을 제고한다. 또한 acceptance rule에 의해 guided re-rollout이 기존 best보다 나쁜 makespan을 만들 경우 해당 결과는 폐기된다. 이는 초기 단계의 dispatching 선택이 makespan에 누적되어 증폭되는 JSSP와 같은 long-horizon 문제의 구조적 한계를 완화하는 데 기여한다.

## 논문용 수정 본문

Self-reflection prompting은 1차 step-by-step rollout 이후 완성된 schedule을 사후적으로 분석하여, final makespan에 불리하게 작용했을 가능성이 높은 의사결정 step을 다시 검토하는 test-time refinement 절차이다. 본 연구의 기본 추론 정책은 자유 형식 텍스트 생성을 통해 action을 생성하지 않고, 현재 step의 feasible action set에 포함된 후보 action code들을 각각 candidate-scoring query로 평가한 뒤, 가장 높은 점수 또는 sampling된 후보를 선택하는 reranking 구조를 사용한다. Self-reflection 역시 동일한 candidate-scoring 정책 위에서 작동하며, 추가적인 파라미터 업데이트 없이 prompt context와 guidance만을 통해 후속 rollout의 선택 분포를 국소적으로 조정한다.

먼저 초기 rollout을

```text
tau^(0) = (a_0, a_1, ..., a_{T-1})
```

라고 하자. 각 step `t`에서 환경은 현재 schedule state `x_t`와 feasible action set `C_t`를 제공한다. Candidate scoring 모델은 각 후보 `c in C_t`에 대해 별도의 scoring prompt를 구성하고, `<CAND_SCORE>` 위치의 hidden representation을 candidate score head에 통과시켜 scalar score를 계산한다.

```text
s_theta,phi(x_t, c) = g_phi(h_theta(x_t, c))
```

여기서 `theta`는 LoRA가 적용된 base language model의 파라미터, `phi`는 candidate score head의 파라미터, `h_theta(x_t, c)`는 해당 후보를 평가하는 query에서 `<CAND_SCORE>` 위치의 마지막 hidden vector이다. 추론 시 후보 선택 확률은 feasible candidate set 내부에서만 softmax로 정규화된다.

```text
p(a_t = c | x_t, C_t) =
    exp(s_theta,phi(x_t, c) / T)
    / sum_{c' in C_t} exp(s_theta,phi(x_t, c') / T)
```

`T`는 candidate-scoring temperature이며, greedy 추론에서는 가장 높은 score를 갖는 후보가 선택된다. Sampling 추론에서는 위 분포로부터 후보 action code를 샘플링한다.

초기 rollout이 종료되면, 각 step의 선택 결과와 모든 후보의 one-step transition estimate를 이용해 critical step을 찾는다. 본 구현에서 critical step은 단순히 makespan이 증가한 step만을 의미하지 않는다. 현재 선택 action `a_t`와 가장 강한 대안 후보 `b_t`를 비교하여, 다음과 같은 regret-like signal을 계산한다.

```text
Delta_t^proj = C_hat_t(a_t) - C_hat_t(b_t)
J_t          = C_t^after - C_t^before
D_t          = S_hat_t(a_t) - S_hat_t(b_t)
```

여기서 `C_hat_t(a)`는 step `t`에서 action `a`를 선택했을 때의 projected makespan, `J_t`는 실제 rollout에서 해당 step 이후 makespan이 증가한 정도, `S_hat_t(a)`는 action `a`의 estimated start time이다. 구현에서는 `(Delta_t^proj, J_t, D_t, t)`를 lexicographic score로 사용하여 내림차순 정렬하고, 상위 `K`개 step을 reflection 대상 step으로 선택한다. 즉, projected makespan 관점에서 더 나은 대안이 존재하거나, 해당 step에서 makespan jump가 크거나, 선택 action이 좋은 대안보다 늦게 시작되는 경우가 우선적으로 검토된다.

선택된 critical step들에 대해서는 reflection memory가 생성된다. Reflection memory는 episode 전체 결과와 step-level diagnostics를 함께 포함한다. 구체적으로 현재 final makespan, critical step 수, makespan 증가 step 수, 반복적으로 등장하는 bottleneck machine 후보, 기존 선택 action의 start/end/idle gap/job wait, 그리고 대안 후보의 projected makespan, estimated start, delta Cmax, post-route 정보를 요약한다. 이 memory는 다음 guided rollout에서 모든 step prompt에 추가되어, 모델이 이전 rollout의 실패 패턴을 조건부 문맥으로 인식하도록 한다.

Reflection prompt는 각 critical step에 대해 별도로 구성된다. Prompt에는 원래 step state, 이전 rollout에서 선택된 action code, feasible action code 목록, 선택 후보와 대안 후보의 transition diagnostics, 그리고 reflection memory가 포함된다. 그러나 이 prompt의 목적은 모델이 임의의 자연어 답변을 생성하도록 하는 것이 아니다. 현재 구현에서는 reflection prompt 역시 candidate-scoring reranker의 입력으로 사용된다. 즉, reflection prompt가 주어진 상태에서 feasible action code들을 다시 점수화하고, 더 나은 후보 action code를 제안한다.

만약 reflection prompt를 조건으로 한 candidate-scoring 재평가가 기존 선택과 다른 유효한 대체 action을 선택하면, 해당 action이 가리키는 job을 preferred job으로 둔다. 반대로 초기 rollout에서 해당 step에 선택되었던 job은 다음 guided rollout에서 비선호하도록 표시되는 discouraged job이 된다. 이때 guidance 문장을 LLM이 직접 생성하는 것이 아니라, 재평가 결과를 deterministic controller가 step-level guidance로 변환한다. 만약 reflection 재평가가 기존 선택과 동일한 action을 제안하거나 유효한 대안을 만들지 못하면, 구현은 deterministic critic fallback을 사용한다. 이 fallback은 해당 step의 대안 후보 중 projected makespan, estimated start, processing time 순으로 가장 좋은 후보를 선택한다.

```text
G_t = { j_t^+, j_t^-, r_t }

j_t^+ = job(tilde c_t)
j_t^- = job(c_t^(0))
```

여기서 `tilde c_t`는 reflection prompt를 조건으로 한 candidate-scoring 재평가 또는 deterministic fallback이 제안한 대체 action code이고, `c_t^(0)`는 초기 rollout에서 실제 선택되었던 action code이다. `j_t^+`는 다음 rollout에서 선호하도록 유도되는 job이며, `j_t^-`는 hard constraint로 금지되는 job이 아니라 prompt상 비선호 신호를 받는 이전 선택 job이다. `r_t`는 이러한 변경을 유도하는 hindsight reason이다.

이 guidance는 최종 schedule을 즉시 수정하는 hard replacement가 아니다. 실제 개선 schedule은 prefix replay와 guided suffix re-rollout을 통해 다시 생성된다. 먼저 guidance가 존재하는 가장 이른 step을 `t0`라고 하면, `0 <= t < t0` 구간은 초기 rollout의 선택 job을 그대로 replay한다. 이는 초반 prefix가 달라져 비교 자체가 불안정해지는 것을 막고, reflection이 겨냥한 의사결정 지점 이후의 효과를 평가하기 위함이다. 이후 `t >= t0` 구간에서는 reflection memory와 step guidance를 prompt에 추가한 상태로 candidate-scoring 정책을 다시 실행한다. 따라서 self-reflection은 "한 step만 바꾸고 나머지를 고정"하는 방식이 아니라, 문제의 prefix를 보존한 뒤 bottleneck 후보 step부터 schedule suffix를 다시 생성하는 guided re-rollout이다.

각 reflection pass에서 생성된 candidate schedule `tau'`는 현재 best schedule과 비교된다. Candidate schedule의 makespan이 더 작을 때만 best schedule을 교체한다.

```text
tau_best <- tau'      if M(tau') < M(tau_best)
tau_best <- tau_best  otherwise
```

따라서 reflection pass가 실패하거나 guidance가 오히려 나쁜 suffix를 만들더라도, 최종 반환 schedule은 해당 sample의 base rollout보다 악화되지 않는다. 최종 결과에는 base rollout의 makespan, reflection 이후 makespan, improvement delta, 개선 여부, pass별 guided step, replay prefix 위치가 함께 기록된다. 이 때문에 결과 CSV의 `Base_Makespan`은 self-reflection 이전의 1차 rollout 결과이고, `Final_Makespan`은 reflection pass 이후 acceptance rule을 통과한 최종 결과이다.

요약하면, 본 연구의 self-reflection prompting은 schedule 완성 후 hindsight signal을 추출하고, 이를 reflection memory와 step guidance로 변환한 뒤, 같은 candidate-scoring policy를 사용하여 suffix를 다시 풀어보는 test-time search 절차이다. 이 과정은 추가 학습 없이도 모델이 한 번의 rollout에서 놓친 bottleneck activation, excessive waiting, weak post-route choice를 재검토하게 만들며, 특히 long-horizon JSSP처럼 초기의 작은 dispatch 선택이 후반 makespan에 크게 증폭되는 문제에서 효과적이다.

## 기존 문단에서 수정해야 할 핵심

기존 표현:

```text
모델은 이를 바탕으로 해당 step에서 더 나은 feasible action code를 다시 선택하고, 그 이후 suffix를 재생성한다.
```

현재 구현에 맞춘 정확한 표현:

```text
모델은 reflection prompt를 조건으로 feasible action code들을 다시 candidate-scoring한다.
이 재평가에서 기존 선택과 다른 유효한 대체 action이 선택되면, deterministic controller가
해당 대체 action의 job을 preferred job으로, 초기 rollout에서 선택되었던 job을
discouraged job으로 설정하여 step-level guidance로 변환한다. 이후 가장 이른 guided step
이전의 prefix는 기존 선택을 replay하고, guided step부터는 reflection memory와 guidance가
포함된 prompt를 사용하여 suffix를 다시 rollout한다. 새 rollout이 기존 best rollout보다
더 작은 makespan을 달성할 때만 최종 결과로 채택한다.
```

## 구현 대응표

| 논문 개념 | 현재 구현 |
|---|---|
| 1차 rollout | `run_step_rollout()` 내부의 `base_result` |
| candidate scoring policy | `_sample_step_action(..., candidate_score_head=...)` |
| critical step selection | `_select_top_critical_steps()` |
| regret-like signal | `_critical_step_score()`의 projected makespan gap, makespan jump, start delay |
| step diagnostics | `_build_step_diagnostics()` |
| reflection memory | `_build_reflection_memory()` |
| reflection prompt | `build_step_improvement_prompt()` + episode summary |
| reflection candidate selection | reflection prompt를 넣은 `_sample_step_action()` |
| deterministic fallback | `_best_alternative_option()` |
| guidance | `guidance_map[step_idx]`: preferred job, discouraged previous job, reason |
| prefix replay | `replay_action_jobs_by_step`, `replay_prefix_until` |
| guided suffix regeneration | `_run_single_step_rollout(..., guidance_by_step=..., reflection_memory_text=...)` |
| acceptance rule | `candidate_result["makespan"] < best_result["makespan"]`일 때만 교체 |
| CSV trace | `DecisionSource = replay / guided / model`, `Base_Makespan`, `Final_Makespan`, `ImprovementDelta` |

## 1-step 예시

예를 들어 어떤 step `t=37`에서 feasible action code가 세 개 있다고 하자.

```text
C_t = {<A1042>, <A3371>, <A8029>}
```

초기 rollout에서 candidate-scoring 결과가 다음과 같았다고 하자.

```text
<A1042> -> Job 2, score=7.10, prob=0.52, projected Cmax=920, start=410
<A3371> -> Job 5, score=6.95, prob=0.45, projected Cmax=890, start=395
<A8029> -> Job 7, score=4.20, prob=0.03, projected Cmax=940, start=430
```

초기 policy는 score가 가장 높은 `<A1042>`를 선택한다. 그러나 rollout 종료 후 hindsight 분석에서 `<A3371>`이 projected Cmax를 더 작게 만들었고, 더 이른 start time을 가지며, 후속 route에서 bottleneck machine을 더 빨리 해소할 수 있었다는 정보가 발견될 수 있다. 이때 critical score는 대략 다음처럼 해석된다.

```text
Delta_t^proj = 920 - 890 = 30
J_t          = makespan_after - makespan_before
D_t          = 410 - 395 = 15
```

`Delta_t^proj`와 `D_t`가 양수이므로 이 step은 reflection 대상이 될 수 있다. Reflection memory에는 다음과 같은 형태의 정보가 들어간다.

```text
Current episode makespan=...
Rule: step 37.
Previous choice <A1042>/Job2 had est_Cmax=920, start=410, ...
Prefer <A3371>/Job5 instead when feasible because it gives est_Cmax=890,
start=395, lower idle gap, and stronger post_route=...
```

다음 pass에서는 step 37 이전까지는 기존 선택을 replay한다. step 37에 도달하면 prompt에 reflection memory와 guidance가 추가된다.

```text
Post-episode guidance:
This step was identified as a Cmax bottleneck.
Prefer Job5 if feasible.
Discourage Job2 if a stronger alternative is available.
Reason: chosen=<A1042>/Job2 showed high bottleneck risk;
prefer <A3371>/Job5.
```

그 뒤 모델은 다시 세 후보를 candidate scoring한다. 이때 action code 자체는 pass마다 seed가 달라질 수 있으므로, 논문에서는 action code 문자열보다 "feasible candidate/job" 단위의 guided reranking으로 설명하는 것이 더 정확하다. 새 rollout에서 Job 5를 선택하고 suffix를 다시 생성한 결과 makespan이 기존보다 작아지면 해당 rollout을 최종 결과로 채택한다. 반대로 suffix 재생성 결과가 더 나빠지면 기존 rollout을 유지한다.

## 논문에 쓰면 안 되는 부정확한 표현

아래 표현들은 현재 구현과 다르므로 피하는 것이 좋다.

```text
Self-reflection이 모델 파라미터를 업데이트한다.
```

현재 self-reflection은 gradient update를 수행하지 않는다.

```text
Reflection prompt가 직접 새로운 action token을 생성한다.
```

현재 구조에서는 reflection prompt를 조건으로 candidate score head가 feasible 후보들을 다시 점수화한다.

```text
Critical step 하나의 action만 바꾼 뒤 나머지 schedule은 그대로 둔다.
```

현재 구조는 prefix만 replay하고, guided step 이후 suffix는 다시 rollout한다.

```text
Reflection 결과가 항상 최종 schedule에 반영된다.
```

현재 구조는 candidate rollout이 기존 best보다 더 작은 makespan을 달성할 때만 반영한다.
