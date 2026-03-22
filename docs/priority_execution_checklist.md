# Priority Execution Checklist

이 문서는 현재 JSSP LLM 프레임워크의 우선순위 작업을 추적하기 위한 실행 체크리스트다.
이후 구현을 진행할 때마다 이 파일의 상태를 먼저 갱신한다.

상태 표기:
- `[ ]` 미착수
- `[-]` 진행 중
- `[x]` 완료
- `[!]` 보류 / 설계 결정 필요

기준일:
- 2026-03-20

---

## 0. 작업 원칙

- 실험 신뢰도를 해치는 항목을 먼저 수정한다.
- 논문 claim과 실제 코드 경로가 다르면 claim이 아니라 코드를 맞춘다.
- 데이터 포맷을 바꾸는 작업은 train/inference/RL 경로를 함께 맞춘다.
- 각 항목은 구현 후 최소 1회 로컬 검증 또는 샘플 검증을 남긴다.

---

## 1. 최우선 항목

### 1-1. Instance-level train/eval split
- 상태: `[-]`
- 목적:
  - 같은 `instance_id`의 서로 다른 step row가 train/eval에 동시에 들어가는 누수를 제거한다.
  - eval loss가 지나치게 낙관적으로 나오는 문제를 해결한다.
- 영향 범위:
  - `unified_trainer.py`
  - `notebooks/colab_02_train_step_lora_full.ipynb`
  - 필요 시 `llm_jssp/utils/data_preprocessing_english.py`는 확인만
- 해야 할 일:
  - [ ] 현재 row-wise `train_test_split` 지점을 찾는다.
  - [ ] `instance_id` 기준 unique split 로직을 설계한다.
  - [ ] train/eval row 수와 unique instance 수를 로그로 출력한다.
  - [ ] `max_train_samples`, `max_eval_samples`가 instance split 이후에도 의도대로 동작하게 맞춘다.
  - [ ] notebook/source 경로를 동일한 split 규칙으로 맞춘다.
- 완료 기준:
  - 같은 `instance_id`가 train/eval 양쪽에 동시에 존재하지 않는다.
  - split 결과 로그에 `train_instances`, `eval_instances`, `train_rows`, `eval_rows`가 출력된다.
  - source trainer와 `colab_02`가 동일한 split semantics를 가진다.
- 검증:
  - [ ] 작은 샘플 dataset으로 split overlap 검사
  - [ ] 실제 step dataset에서 overlap 0 확인

### 1-2. Dispatch parity in main inference / RL pipeline
- 상태: `[-]`
- 목적:
  - 메인 inference/RL 파이프라인이 serial뿐 아니라 dispatch도 직접 지원하도록 맞춘다.
  - 논문에서 serial+dispatch를 함께 주장할 수 있게 한다.
- 영향 범위:
  - `inference_jssp_fssp.py`
  - `RL_jssp_fssp.py`
  - `notebooks/colab_03_inference_step_full.ipynb`
  - `notebooks/colab_05_rl_full.ipynb`
  - `notebooks/colab_06_rl_compare.ipynb`
- 해야 할 일:
  - [ ] 현재 serial 경로와 dispatch 경로 분기 현황을 점검한다.
  - [ ] 메인 CLI 인자/CFG에서 `env_mode=serial|dispatch`가 일관되게 먹도록 맞춘다.
  - [ ] policy / reason / reflection / masking이 dispatch에서도 같은 인터페이스로 동작하게 맞춘다.
  - [ ] RL rollout, evaluation, logging에서 dispatch 상태 값(`current_time`, `idle_machines`, `num_running_ops`)이 깨지지 않는지 확인한다.
- 완료 기준:
  - source inference와 source RL에서 `env_mode='dispatch'`가 직접 실행된다.
  - notebook `03/05/06`도 같은 의미의 dispatch 실행 경로를 가진다.
- 검증:
  - [ ] serial smoke test
  - [ ] dispatch smoke test
  - [ ] policy + reflection + rationale on dispatch sample

### 1-3. Candidate ordering leakage 제거
- 상태: `[-]`
- 목적:
  - 후보가 `estimated_makespan_after` 순으로 정렬되어 생기는 순서 bias를 줄인다.
  - 모델이 feature reasoning 대신 “앞에 있는 후보 선택 습관”을 배우는 위험을 낮춘다.
- 영향 범위:
  - `llm_jssp/utils/step_prompting.py`
  - `llm_jssp/utils/step_prompting_dispatch.py`
  - `generate_jssp_step_dataset.py`
  - `inference_jssp_fssp.py`
  - `RL_jssp_fssp.py`
  - `notebooks/colab_00_generate_step_dataset.ipynb`
  - `notebooks/colab_03_inference_step_full.ipynb`
  - `notebooks/colab_05_rl_full.ipynb`
  - `notebooks/colab_06_rl_compare.ipynb`
- 해야 할 일:
  - [ ] 현재 candidate ordering 지점을 정확히 식별한다.
  - [ ] main experiment용 기본 ordering 정책을 정한다.
    - 후보 무작위화
    - 또는 deterministic random seed 기반 shuffle
  - [ ] chosen action lookup이 ordering 변화에도 안정적인지 확인한다.
  - [ ] reason prompt의 `Top alternatives` 로직은 유지하되, policy candidate listing과 혼동되지 않게 분리한다.
  - [ ] random-order ablation 가능하게 옵션을 남긴다.
- 완료 기준:
  - policy prompt에서 candidate 순서가 heuristic sort에 고정되지 않는다.
  - ordering 전략이 source/notebook에서 동일하게 맞아 있다.
- 검증:
  - [ ] 같은 state에서 candidate order가 바뀌는 샘플 확인
  - [ ] label/chosen code가 ordering 변경 후에도 일치 확인

---

## 2. 그다음 항목

### 2-1. Action-centric loss
- 상태: `[-]`
- 목적:
  - 이 문제를 “전체 LM next-token”보다 “feasible action set 위 분류”에 더 가깝게 학습한다.
  - `action_reason`에서 explanation token이 action learning을 희석하는 문제를 줄인다.
- 영향 범위:
  - `unified_trainer.py`
  - `notebooks/colab_02_train_step_lora_full.ipynb`
  - 필요 시 preprocessing helper
- 해야 할 일:
  - [ ] 현 구조에서 action token 위치를 명시적으로 추출할 수 있는지 확인한다.
  - [ ] 최소 구현안을 정한다.
    - feasible token normalized cross-entropy
    - 또는 first action token up-weighting
  - [ ] `action_only`, `reason_only`, `action_reason`에서 loss semantics를 정리한다.
- 완료 기준:
  - `system/user` 토큰은 loss에서 제외된다.
  - `action_only`는 assistant의 첫 `<Axxxx>` action token만 supervision한다.
  - `action_reason`는 assistant-only supervision을 유지하되, 첫 action token은 feasible action token set 위 normalized CE로 학습한다.
  - `reason_only`는 assistant rationale span만 supervision한다.
- 검증:
  - [ ] loss 계산 샘플 테스트
  - [ ] mixed/action_reason에서 action token feasible-set CE 확인

### 2-2. Prefix masking -> direct token-set masking
- 상태: `[-]`
- 목적:
  - 현재 single-token `<Axxxx>` 체계에 맞게 마스킹을 단순화하고 속도를 높인다.
  - 논문 설명도 “single-token feasible-set classification under masking”으로 단순화한다.
- 영향 범위:
  - `llm_jssp/utils/jssp_masking_hooks.py`
  - `llm_jssp/utils/jssp_step_masking_hooks.py`
  - `inference_jssp_fssp.py`
  - `RL_jssp_fssp.py`
  - notebooks `03/05/06`
- 해야 할 일:
  - [ ] 현재 prefix scan 경로와 token-id mask 경로를 정확히 파악한다.
  - [ ] feasible action code -> token id 집합 변환 경로를 정리한다.
  - [ ] one-token generation 경로에 direct mask를 적용한다.
  - [ ] round-trip validation과 collision 방어가 유지되는지 확인한다.
- 완료 기준:
  - inference/RL에서 feasible token id set만 남기는 direct mask 경로가 동작한다.
  - prefix-level scan 없이 같은 결과를 낼 수 있다.
- 검증:
  - [ ] serial sample action decode
  - [ ] dispatch sample action decode
  - [ ] masking 전/후 feasible set 일치 확인

---

## 3. 그다음 항목

### 3-1. Multiple teachers / preference supervision
- 상태: `[-]`
- 목적:
  - single-teacher imitation 한계를 줄인다.
  - 동일 makespan의 여러 좋은 선형화 또는 near-optimal teacher를 활용한다.
- 영향 범위:
  - dataset generation pipeline 전반
  - trainer / preference trainer 설계
- 해야 할 일:
  - [ ] multiple optimal / near-optimal teacher 생성 전략 조사
  - [ ] dataset schema 확장안 설계
  - [ ] pairwise preference supervision 가능 여부 정리
- 완료 기준:
  - 최소 설계 문서 또는 prototype 구현 존재

### 3-2. Long-horizon regret 강화 (reason/reflection)
- 상태: `[-]`
- 목적:
  - one-step projected feature 의존을 줄이고 downstream regret를 더 직접 반영한다.
- 해야 할 일:
  - [ ] short rollout 기반 regret 추정안
  - [ ] value model 도입 가능성
  - [ ] counterfactual suffix regeneration 설계
- 완료 기준:
  - reflection score 또는 rationale target에 one-step beyond signal이 추가된다.

### 3-3. BOPO pair construction refinement
- 상태: `[-]`
- 목적:
  - divergence 이후 같은 step index pair를 붙이는 노이즈를 줄인다.
- 해야 할 일:
  - [ ] shared-prefix pair만 쓰는 경로 설계
  - [ ] state matching 기반 pairing 가능성 검토
- 완료 기준:
  - pair construction 기준이 명시적으로 더 clean해진다.

### 3-4. Size-normalized RL reward
- 상태: `[-]`
- 목적:
  - raw negative makespan의 크기 의존성을 줄인다.
- 해야 할 일:
  - [ ] PRD / heuristic-relative improvement / normalized gap 후보 비교
  - [ ] mixed-size training 시 reward scale 안정화
- 완료 기준:
  - size-normalized reward 옵션이 추가된다.

### 3-5. Action-code augmentation 강화
- 상태: `[-]`
- 목적:
  - 같은 state에 대한 code identity bias를 더 줄인다.
- 해야 할 일:
  - [ ] offline multi-code augmentation 설계
  - [ ] epoch-level online remapping 설계
  - [ ] reproducibility/logging 방안 정리
- 완료 기준:
  - 최소 1개 augmentation 전략이 구현 또는 설계 확정된다.

---

## 4. 보류 / Future Work

### 4-1. Larger action vocabulary redesign
- 상태: `[ ]`
- 목적:
  - 큰 action space에서는 pointer / candidate-position decoding 같은 대안도 고려한다.
- 메모:
  - 당장 변경 대상은 아님
  - discussion/future work에 넣기 좋음

---

## 5. 진행 로그

- 2026-03-20: 체크리스트 초안 생성
- 2026-03-20: `unified_trainer.py`, `colab_02_train_step_lora_full.ipynb`에 instance-level split 로직 반영. `instance_id/source_index` 기반 split 구현 완료, 실제 dataset overlap 검증은 남음.
- 2026-03-20: `inference_jssp_fssp.py`, `RL_jssp_fssp.py`에 `env_mode=serial|dispatch` 분기 반영. source inference/RL에서 dispatch environment, dispatch prompt builder, dispatch action-effect 경로를 직접 사용하도록 수정. notebook `03/05/06`의 개념 경로는 이미 dispatch 포맷과 동기화되어 있으나, 런타임 smoke test는 남음.
- 2026-03-20: policy prompt의 candidate listing order를 heuristic sort 고정에서 deterministic random shuffle로 변경. source `step_prompting.py`, `step_prompting_dispatch.py` 및 notebook `00/03/05/06` inline helper에 반영. reason의 `Top alternatives` 정렬 로직은 유지.
- 2026-03-20: `unified_trainer.py`, `colab_02_train_step_lora_full.ipynb`의 action-centric loss를 재구성. plain-text `SFTTrainer` 경로를 버리고, `build_step_supervision_example` 기반 explicit tokenized supervision(`input_ids`, `labels`, `loss_weights`)으로 변경. `action_only`는 첫 `<Axxxx>`만 감독하고, `action_reason`는 assistant-only supervision 위에 첫 action token에 feasible-set normalized CE를 적용하도록 trainer/collator를 교체. source `py_compile`, notebook AST, helper smoke test 완료. 실제 train-step smoke test는 남음.
- 2026-03-20: `llm_jssp/utils/jssp_step_masking_hooks.py`를 direct token-id mask 기반으로 단순화. feasible action code를 tokenizer token id 집합으로 직접 변환해 허용하며, prefix scan은 제거. notebook `03/05/06` inline masking helper도 같은 방식으로 동기화함. 실제 generation equivalence 검증은 남음.

- 2026-03-20: `docs/multiple_teachers_preference_design.md` 작성. multi-teacher step schema, pairwise preference row schema, state alignment 원칙, phased rollout plan 정리.
- 2026-03-20: synthetic split 샘플로 instance-level split overlap 0 확인. 실제 step dataset 실검증은 남음.
- 2026-03-20: synthetic serial/dispatch prompt에서 candidate listing이 deterministic random order로 노출되는 것 확인. heuristic metric 정렬과 visible listing을 분리.
- 2026-03-20: `RL_jssp_fssp.py`, `colab_05_rl_full.ipynb`, `colab_06_rl_compare.ipynb`에 `reward_mode`(`raw_neg_makespan`, `neg_makespan_per_op`, `mwkr_relative`) 추가. reward normalization helper를 GRPO/REINFORCE 경로에 반영.
- 2026-03-20: `RL_jssp_fssp.py`, `colab_05_rl_full.ipynb`, `colab_06_rl_compare.ipynb`에 `bopo_pair_mode` 추가. shared-prefix first-divergence pair construction을 지원하도록 수정.
- 2026-03-20: `docs/long_horizon_regret_design.md` 작성. short rollout regret, counterfactual suffix regeneration, critic/value integration 경로와 reflection score 결합 방안을 정리.
- 2026-03-20: `docs/action_code_augmentation_design.md` 작성. offline multi-view augmentation과 epoch-level online remapping 경로, schema 확장안, 실험 비교안을 정리.
