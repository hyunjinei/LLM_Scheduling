# Step-by-Step JSSP Migration Board

Last updated: 2026-03-03
Project: `LLM_JSSP_masking`
Target: one-shot full schedule generation -> step-by-step action generation (`Action: Job j`) with feasible masking.

---

## 0) Current Status Snapshot

- [x] Existing codebase structure analysis completed
- [x] Current train/val format 확인 (`llm_jssp/train.json`, `validation_data/*.json`)
- [x] One-shot masking/FSM 동작 방식 파악
- [x] Migration 설계 문서 초안 완료
- [x] Step environment implementation
- [x] Step dataset generator implementation
- [x] Training pipeline step-mode integration
- [x] Inference pipeline step-mode integration
- [x] Step masking hook base implementation
- [x] GRPO base implementation (`RL_jssp_fssp.py`)
- [x] Step-policy GRPO path with env interaction (step 고정)
- [x] Step action-level improvement prompt path (`--enable_step_improvement`)
- [x] Step rationale generation path (`--emit_step_rationale`) without affecting action masking
- [x] Step LoRA supervision mode: `action_only` / `action_reason`
- [x] One-shot runtime path removed from trainer/inference CLI (step-only)
- [x] Inference entrypoint refactored to `main()` (no top-level run side effects)
- [x] RL training path cleaned to step-only (removed unreachable one-shot branches)
- [ ] End-to-end verification

---

## 1) Final Architecture (to build)

### 1.1 Policy format
- Input per step: current state + feasible job list
- Output per step: `Action: Job <id>`
- Horizon: `num_jobs * num_machines` steps (e.g. 10x10 = 100)

### 1.2 Runtime loop
1. Reset env with JSSP instance
2. Build step prompt from state
3. Apply feasible action mask
4. Decode 1 action
5. Validate + apply env transition
6. Repeat until done
7. Export schedule + makespan

### 1.3 Key principle
- Constraints are handled by env + decoding mask, not by free-form text post-hoc repair.

---

## 2) Data Schema (Target)

## 2.1 Step-level training row
```json
{
  "instance_id": "ta01",
  "num_jobs": 10,
  "num_machines": 10,
  "total_steps": 100,
  "step_idx": 37,
  "feasible_jobs": [0, 2, 3, 5, 6, 8],
  "state_json": {
    "job_next_op": [4,3,4,2,4,4,3,4,5,4],
    "job_ready_time": [390,401,405,388,412,399,410,403,411,406],
    "machine_ready_time": [398,412,404,410,407,402,395,409,401,406],
    "next_machine": [7,2,9,5,1,6,3,0,4,8],
    "next_proc_time": [15,23,11,19,9,14,20,8,16,10],
    "remaining_ops": [6,7,6,8,6,6,7,6,5,6],
    "remaining_work": [88,110,95,121,79,84,106,81,67,90]
  },
  "state_text": "...",
  "target_job": 3,
  "target_text": "Action: Job 3"
}
```

## 2.2 Hard checks
- `rows_per_instance == J*M`
- every `target_job in feasible_jobs`
- no missing/duplicate operation scheduling
- final `done=True` and valid makespan

---

## 3) File-by-File Task List

## 3.1 New files (must create)

1. `llm_jssp/utils/jssp_step_env.py`
- [x] Define `StaticJSSPStepEnv`
- [x] Implement `reset`, `get_feasible_jobs`, `step`, `is_done`
- [x] Add `get_state_json` and optional `get_event_log`
- [x] Add strict assertions for invalid actions

2. `llm_jssp/utils/step_prompting.py`
- [x] `build_step_prompt(state_json, feasible_jobs, step_idx, total_steps)`
- [x] Keep prompt deterministic and compact
- [x] Ensure prompt format is tokenizer-safe

3. `llm_jssp/utils/jssp_step_masking_hooks.py`
- [x] Implement step-action FSM
- [x] Allow only action format tokens (`Action`, `Job`, digits, space, newline, eos)
- [x] Apply feasible job ID mask at numeric slot
- [x] Expose helper for `prefix_allowed_tokens_fn`

4. `generate_jssp_step_dataset.py`
- [x] Read source dataset (`llm_jssp/train.json`)
- [x] Parse teacher schedule from one-shot `output`
- [x] Roll out env and emit step rows
- [x] Save JSONL to `train_data/jssp_step_train.jsonl`
- [x] Add CLI args (input/output/split/seed)

5. `tests/test_jssp_step_env.py`
- [x] Step count, state transitions, done condition
- [x] Invalid action rejection test

6. `tests/test_step_dataset_generator.py`
- [x] 10x10 -> 100 rows assertion
- [x] feasible target checks
- [x] teacher rollout consistency

7. `tests/test_jssp_step_masking_hooks.py`
- [x] infeasible IDs probability zero
- [x] format token restriction

## 3.2 Existing files (must modify)

1. `unified_trainer.py`
- [x] Step-only trainer path (`--step_dataset_path`, `--step_supervision_mode`)
- [x] Add `--step_dataset_path`
- [x] Remove one-shot branch from trainer runtime

2. `llm_jssp/utils/data_preprocessing_english.py`
- [x] Add step preprocessing function
- [x] Map step row -> training text pair
- [x] Keep backward compatibility with existing one-shot

3. `inference_jssp_fssp.py`
- [x] Step-only inference runtime (one-shot entry removed)
- [x] Add step loop and env transition integration
- [x] Add step mask application path
- [x] Add per-step logs and final schedule reconstruction
- [x] Add custom eval dataset override (`--eval_data_path`)
- [x] Refactor into `main()` execution flow

4. `llm_jssp/utils/solution_generation_english.py`
- [x] Improve reflection prompt constraints and output rules
- [x] Keep existing one-shot API untouched

5. `analysis_gantt_generator.py` (optional but recommended)
- [ ] Support step-mode metadata fields
- [ ] Keep existing plots compatible

---

## 4) Detailed Execution Plan (ordered)

## Phase A: Core engine + data
- [x] A1. Implement `StaticJSSPStepEnv`
- [x] A2. Implement teacher rollout parser
- [x] A3. Implement `generate_jssp_step_dataset.py`
- [x] A4. Unit tests for A1/A3

Acceptance:
- [x] Dataset generation works on sample set
- [x] 10x10 instance yields exactly 100 rows

## Phase B: Prompt + mask
- [x] B1. Implement step prompt builder
- [x] B2. Implement step masking hooks
- [x] B3. Unit tests for token/action validity

Acceptance:
- [x] Infeasible action decode blocked
- [x] Output format stable as `Action: Job j`

## Phase C: Training integration
- [x] C1. Add step dataset mode in trainer
- [x] C2. Add preprocessing branch
- [ ] C3. Tiny smoke training run

Acceptance:
- [ ] Step-mode LoRA checkpoint produced

## Phase D: Inference integration
- [x] D1. Step-mode inference loop
- [x] D2. Schedule reconstruction + makespan
- [x] D3. CSV/report field extensions

Acceptance:
- [ ] End-to-end inference returns feasible schedule on benchmark samples (GPU runtime test pending)

## Phase E: Evaluation and paper-ready outputs
- [ ] E1. one-shot vs step ablation script
- [ ] E2. table/figure export compatibility
- [ ] E3. reproducibility command sheet

Acceptance:
- [ ] Comparative results reproducible with fixed seeds

## Phase F: RL (GRPO)
- [x] F1. Add GRPO training mode selector (`--rl_algo grpo|reinforce`)
- [x] F2. Add group sampling (`--group_size`) and clipped objective
- [x] F3. Add GRPO optimization controls (`--grpo_epochs`, `--clip_epsilon`, `--kl_coef`)
- [ ] F4. Run GRPO smoke training on small benchmark subset

---

## 5) Progress Tracker (update while coding)

## Completed
- [x] Repository/code-path analysis
- [x] Migration architecture draft
- [x] File-level implementation board finalized
- [x] `llm_jssp/utils/jssp_step_env.py` 구현 완료
- [x] `llm_jssp` 패키지명으로 폴더 정리 완료 (`예전cde_llm` -> `llm_jssp`)
- [x] 루트 `utils/` 제거, 공통 유틸을 `llm_jssp/utils/common.py`로 통합
- [x] `llm_jssp/utils/step_prompting.py` 구현 완료
- [x] `llm_jssp/utils/jssp_step_masking_hooks.py` 구현 완료
- [x] `generate_jssp_step_dataset.py` 구현 완료
- [x] `tests/test_jssp_step_env.py` / `tests/test_step_dataset_generator.py` / `tests/test_jssp_step_masking_hooks.py` 작성 및 실행 완료
- [x] 샘플 데이터 생성 검증 (`--max_instances 3`)
- [x] `unified_trainer.py` step-only 학습 경로로 정리 완료
- [x] `data_preprocessing_english.py` step prompt 포맷 함수 구현 완료
- [x] `inference_jssp_fssp.py` step-only 롤아웃 경로로 정리 완료
- [x] `inference_jssp_fssp.py` step 액션 개선 옵션 연결 (`--enable_step_improvement`, `--step_reflection_passes`)
- [x] `inference_jssp_fssp.py` top-level 실행 제거, `main()` 구조로 리팩토링 완료
- [x] `inference_jssp_fssp.py` one-shot dead code 블록 제거 및 import 정리 완료
- [x] `RL_jssp_fssp.py`에 GRPO 학습 모드 추가 (`--rl_algo grpo`)
- [x] `RL_jssp_fssp.py` step-policy 경로 고정 (oneshot CLI 제거)
- [x] `RL_jssp_fssp.py` unreachable one-shot 학습 분기 제거 완료
- [x] `RL_jssp_fssp.py` `--model_path` 미지정 시 기본 모델 자동 매핑 추가 완료
- [x] `tests/run_rl_smoke_test.sh` step-GRPO 기준으로 동기화 완료

## In Progress
- [ ] step-mode GPU inference runtime 검증
- [ ] step-GRPO smoke runtime 검증 (실 GPU 환경)

## Next Immediate Actions
- [ ] Run tiny step-mode smoke train to validate C3
- [ ] Run step-mode inference smoke test on TA benchmark subset
- [ ] Run `tests/run_rl_smoke_test.sh` on GPU machine and log wall-clock / memory
- [ ] Remove/relocate hardcoded HF token from inference script for security hygiene

---

## 6) Suggested CLI (to finalize after code is implemented)

```bash
# 1) Generate step dataset
python generate_jssp_step_dataset.py \
  --input llm_jssp/train.json \
  --output train_data/jssp_step_train_with_reason.jsonl \
  --strict_makespan

# 2) Train (step mode)
python unified_trainer.py \
  --step_dataset_path train_data/jssp_step_train_with_reason.jsonl \
  --step_supervision_mode action_reason

# 3) Inference (step mode + mask)
python inference_jssp_fssp.py \
  --train_jssp \
  --model_path /mnt/c/Users/User/Desktop/LLM_JSSP_masking/finetuned_models/jssp_llama8b_default_step_action_reason_r64_ep2/checkpoint-200 \
  --eval_data_path validation_data/custom_step_test.json \
  --demo_index 0 \
  --enable_step_improvement \
  --step_reflection_passes 1

# 4) RL (step policy + GRPO)
bash tests/run_rl_smoke_test.sh
```

---

## 7) Design Decisions Frozen

- [x] Static deterministic env first (paper benchmark priority)
- [x] SimPy integration deferred to phase-2 (dynamic/event-driven track)
- [x] Action space is Job-only choice at each step
- [x] Feasibility guaranteed by mask + env transition checks

---

## 8) Current Transition-State Schema

- [x] `state_json` now includes richer global dynamic state:
  `scheduled_ratio`, `current_cmax`, `total_remaining_work`,
  `unfinished_jobs_count`, `unfinished_jobs_ratio`,
  `machine_ready_min/mean/max/std`,
  `machine_remaining_load`, `machine_remaining_ops`,
  `bottleneck_machine_id/load/ops_left`
- [x] Candidate prompt lines now expose action-conditioned transition features:
  `next_m`, `next2_m`, `est_start`, `est_end`,
  `cmax before->after`, `delta_cmax`,
  `job_ready before->after`, `machine_ready before->after`,
  `job_wait`, `machine_idle_gap`,
  `remaining_ops/work before->after`,
  `affected machine load/ops`,
  ratio features
- [x] Dataset rows now carry `feature_schema_version=jssp_step_v2_transition`
- [x] Because prompt/state schema changed, step dataset must be regenerated
- [x] Because LoRA input changed, LoRA model must be retrained on regenerated data
