# Main Method: Action-Code Candidate-Scoring SFT

이 문서는 현재 canonical 학습 노트북인
`notebooks/[action_code_add][fix]colab_02_train_step_lora_full.ipynb` 기준의
supervised policy learning 문단이다.

중요한 점은 이 방법이 RL이 아니라 SFT이며, action token generation loss가 아니라
candidate-scoring loss를 사용한다는 것이다.

---

## 수정이 필요한 이유

기존 문단은 다음 구조를 설명하고 있었다.

```text
assistant span 전체 LM loss를 주지 않고,
teacher action code에 해당하는 action token 하나만 supervision 대상으로 두며,
현재 feasible action token set 위에서 masked softmax를 계산한다.
```

그러나 현재 `[action_code_add][fix]` SFT는 이 방식이 아니다.

현재 방식은 다음과 같다.

```text
action code는 prompt 안에서 candidate를 식별하는 label로 사용된다.
하지만 최종 supervision은 action token logit에 걸리지 않는다.
각 feasible candidate마다 별도의 candidate-scoring query를 만들고,
<CAND_SCORE> 위치 hidden vector를 추출한 뒤,
candidate_score_head가 scalar score를 계산한다.
그 score들에 대해 cross-entropy를 계산한다.
```

따라서 main method에서는 `feasible action token vocabulary`나
`supervised action position의 LM logit`이라는 표현을 쓰면 안 된다.
현재 방법은 **feasible candidate reranking/classification**으로 설명해야 한다.

---

## 논문용 수정 문단

본 연구의 supervised policy learning은 assistant response 전체에
language-modeling loss를 부여하지 않는다. 또한 teacher가 선택한 action code를
다음 token으로 직접 예측하도록 학습하지도 않는다. 대신 현재 step에서 가능한
feasible candidate set 내부에서 teacher가 선택한 candidate를 식별하는
candidate-scoring objective를 사용한다.

현재 step을 \(t\), 해당 step의 상태 및 문제 문맥을 \(x_t\)라 하자. 환경은 현재
상태에서 실행 가능한 후보 집합을 계산하며, 이를

\[
\mathcal{C}_t = \{c_{t,1}, c_{t,2}, \ldots, c_{t,K_t}\}
\]

로 나타낸다. 여기서 각 candidate \(c_{t,k}\)는 action code
\(\langle Axxxx\rangle\), 대응되는 job, 그리고 해당 action을 실행했을 때의
transition features를 포함한다. 예를 들어 candidate line에는 machine,
processing time, earliest start/end time, makespan 변화, remaining work 등의
정보가 포함된다.

각 feasible candidate \(c_{t,k}\)에 대해 모델 입력 query \(q_{t,k}\)를 별도로
구성한다. 이 query는 현재 상태 \(x_t\), 전체 feasible candidate 목록
\(\mathcal{C}_t\), 그리고 현재 평가 대상 candidate \(c_{t,k}\)를 포함한다. Query
끝에는 score extraction marker인 `<CAND_SCORE>` token을 배치한다.

\[
q_{t,k}
= \mathrm{Prompt}(x_t, \mathcal{C}_t, c_{t,k}, \texttt{<CAND\_SCORE>})
\]

Backbone language model \(f_\theta\)는 \(q_{t,k}\)를 인코딩한다. 이때 전체
assistant span의 token loss를 계산하지 않고, `<CAND_SCORE>` 위치의 final hidden
state만 추출한다.

\[
h_{t,k}
= f_\theta(q_{t,k})_{\texttt{<CAND\_SCORE>}}
\]

여기서 \(h_{t,k}\)는 candidate \(c_{t,k}\)를 현재 상태와 다른 후보들에 대한
문맥 속에서 평가한 hidden representation이다. 이후 별도의 candidate score head
\(g_\phi\)가 이 hidden vector를 scalar score로 변환한다.

\[
s_{t,k} = g_\phi(h_{t,k})
\]

현재 구현에서 \(g_\phi\)는 단일 linear layer이다.

\[
g_\phi(h) = w^\top h + b
\]

Teacher가 선택한 후보를 \(c_t^\ast\)라 하고, 그 후보의 index를 \(k^\ast\)라 하면,
정책 확률은 전체 vocabulary가 아니라 현재 feasible candidate set 내부에서의
softmax로 정의된다.

\[
p_{\theta,\phi}(c_t^\ast \mid x_t, \mathcal{C}_t)
=
\frac{\exp(s_{t,k^\ast})}
{\sum_{k=1}^{K_t} \exp(s_{t,k})}
\tag{10}
\]

이에 대한 supervised policy loss는 candidate scores 위의 cross-entropy이다.

\[
\mathcal{L}_{\mathrm{SFT}}(t)
=
-\log p_{\theta,\phi}(c_t^\ast \mid x_t, \mathcal{C}_t)
=
-\log
\frac{\exp(s_{t,k^\ast})}
{\sum_{k=1}^{K_t} \exp(s_{t,k})}
\tag{11}
\]

이 objective는 일반적인 language-modeling loss와 다르다. 일반적인 LM loss는
전체 vocabulary 위에서 다음 token을 예측하는 문제인 반면, 본 objective는 현재
환경이 제공한 feasible candidate set 내부에서 teacher-selected candidate를
식별하는 reranking/classification 문제에 가깝다. 따라서 action code는 직접적인
generation target이라기보다, feasible candidate를 안정적으로 식별하고
candidate-specific query를 구성하기 위한 symbolic label로 사용된다.

또한 특정 step에서 feasible candidate가 하나뿐인 경우
\(K_t = |\mathcal{C}_t| = 1\), 해당 step은 정책적으로 선택의 여지가 없다. 이러한
trivial step은 서로 다른 action 사이의 선호를 학습시키지 못하므로, 학습 데이터
구성 및 filtering 단계에서 policy loss의 주요 대상으로 사용하지 않는다. 현재
SFT 설정에서는 `min_train_feasible_actions=2`를 사용하여
\(|\mathcal{C}_t| \ge 2\)인 step을 학습 대상으로 삼는다. 이는 선택지가 하나뿐인
step에서 불필요한 gradient가 축적되는 것을 방지하고, 모델이 실제 의사결정
분기점에서 candidate 간 선호를 학습하도록 하기 위한 설계이다.

결과적으로 본 연구의 supervised policy learning은 자유 형식 텍스트 생성보다
**environment-grounded feasible candidate scoring**의 성격을 강하게 가진다.
Backbone LLM은 각 candidate를 문맥화한 hidden representation \(h_{t,k}\)를
생성하고, candidate score head는 이 representation을 scalar score로 투영한다.
학습은 teacher-selected candidate의 score가 같은 step의 다른 feasible
candidate보다 높아지도록 backbone parameters \(\theta\)와 score-head parameters
\(\phi\)를 함께 최적화한다.

---

## 기존 문단에서 반드시 바꿔야 하는 표현

기존 표현:

```text
feasible action token 집합 \mathcal{V}_t
supervised action position에서 모델이 산출한 logit z_{t,j}
teacher action code에 해당하는 action token 하나만 supervision
```

수정 표현:

```text
feasible candidate set \mathcal{C}_t
candidate-specific query q_{t,k}
<CAND_SCORE> 위치 hidden vector h_{t,k}
candidate score head score s_{t,k}
candidate scores 위의 cross-entropy
```

---

## 코드와의 대응

현재 코드 기준 대응은 다음과 같다.

- `build_candidate_scoring_example`
  - 각 feasible candidate별 query를 만든다.
  - 각 query 안의 `<CAND_SCORE>` 위치를 저장한다.
  - teacher action code가 candidate list에서 몇 번째인지 `target_candidate_index`로 저장한다.

- `CandidateScoringModel._score_flat_queries`
  - backbone forward를 수행한다.
  - `<CAND_SCORE>` 위치 hidden vector를 gather한다.
  - `candidate_score_head(hidden)`으로 candidate score를 계산한다.

- `CandidateScoringModel.forward`
  - candidate별 flat score를 step별 score matrix로 다시 묶는다.
  - `F.cross_entropy(candidate_scores.float(), target_candidate_index.long())`를 계산한다.

- `min_train_feasible_actions=2`
  - feasible candidate가 2개 이상인 step만 학습 대상으로 남긴다.

---

## 한 문장 요약

현재 main method는 action token을 직접 예측하는 masked LM objective가 아니라,
각 feasible candidate를 `<CAND_SCORE>` hidden vector로 표현하고
candidate score head로 점수화한 뒤, teacher-selected candidate를 맞히는
candidate-level cross-entropy objective이다.
