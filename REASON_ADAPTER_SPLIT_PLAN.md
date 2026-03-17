# Reason Adapter Split Plan

## Goal

Split the current mixed `action + reason` supervision into:

1. `policy dataset / policy adapter`
2. `reason dataset / reason adapter`

This keeps scheduling quality stable while improving explanation quality separately.

## Why Split

- `policy` is a constrained action-selection task:
  - output is one line: `Action: <Sxxxx>`
  - decoding stability, masking, feasibility, and RL compatibility are critical
- `reason` is a free-form explanation task:
  - output is structured natural language
  - can be longer and more comparative
  - should not destabilize action decoding

Recommendation:

- `colab_05` / `colab_06`: policy only
- `colab_03`: policy first, then optional reason adapter switch for explanation

## Web-Backed Design References

### 1. FALCON / BOPO

Source:
- `Hard Constraints Meet Soft Generation: Guaranteed Feasibility for LLM-based Combinatorial Optimization`
- https://arxiv.org/abs/2602.01090

Relevant takeaway:
- BOPO trains from objective-gap-weighted preference pairs without human labels.
- This is directly relevant to `policy` improvement, not explanation text generation.

Implication for this repo:
- keep BOPO / REINFORCE / GRPO on the policy adapter only
- do not mix rationale generation into RL rollout

### 2. Preferential Rationale Tuning

Source:
- `Learning Together to Perform Better: Teaching Small-Scale LLMs to Collaborate via Preferential Rationale Tuning`
- https://arxiv.org/abs/2506.02519

Relevant takeaway:
- rationale quality can be improved as a separate optimization target
- rationale supervision should be treated separately from final answer optimization

Implication for this repo:
- reason adapter should be trained on rationale-only targets
- later, preference-style filtering can be added for better explanation quality

### 3. Post-hoc CO Explanation

Source:
- `RouteExplainer: An Explanation Framework for Vehicle Routing Problem`
- https://arxiv.org/abs/2403.03585

Relevant takeaway:
- explanations work better when tied to action influence / counterfactual alternatives
- explanation should compare chosen action against strong alternatives

Implication for this repo:
- reason dataset should include:
  - selected action
  - selected action transition features
  - top alternative actions
  - contrastive explanation target

## Frozen Dataset Names

### Policy Dataset

- output file:
  - `train_data/jssp_step_train_policy.jsonl`
- summary file:
  - `train_data/jssp_step_train_policy.summary.json`

### Reason Dataset

- output file:
  - `train_data/jssp_step_train_reason.jsonl`
- summary file:
  - `train_data/jssp_step_train_reason.summary.json`

### Legacy Mixed Dataset

- legacy file:
  - `train_data/jssp_step_train_with_reason.jsonl`
- status:
  - keep only for backward compatibility / comparison
  - not the recommended final training set

## Frozen Model Naming

### Policy Adapter

- local output example:
  - `finetuned_models/jssp_policy_llama8b_step_r64_ep2/`
- HF repo example:
  - `HYUNJINI/jssp_policy_llama8b_step_r64_ep2`

### Reason Adapter

- local output example:
  - `finetuned_models/jssp_reason_llama8b_step_r64_ep2/`
- HF repo example:
  - `HYUNJINI/jssp_reason_llama8b_step_r64_ep2`

## Frozen Dataset Roles

### Policy Dataset Role

- learn to choose feasible action code
- no explanation generation during training target
- RL-compatible

### Reason Dataset Role

- learn to justify a fixed chosen action
- no action selection during training target
- post-hoc explanation only

## Frozen Policy Dataset Fields

Each row should contain:

- `feature_schema_version`
  - `jssp_step_policy_v1`
- `instance_id`
- `source_index`
- `num_jobs`
- `num_machines`
- `total_steps`
- `step_idx`
- `state_json`
- `state_text`
- `problem_context_text`
- `feasible_jobs`
- `feasible_action_codes`
- `action_code_to_job`
- `target_job`
- `target_action_code`
- `target_text`
- `teacher_operation_idx`
- `teacher_machine`

Policy dataset should not require:

- `target_reason_text`
- `target_action_reason_text`

These may optionally be emitted only in compatibility mode.

## Frozen Reason Dataset Fields

Each row should contain:

- `feature_schema_version`
  - `jssp_step_reason_v1`
- `instance_id`
- `source_index`
- `num_jobs`
- `num_machines`
- `total_steps`
- `step_idx`
- `state_json`
- `state_text`
- `problem_context_text`
- `feasible_jobs`
- `feasible_action_codes`
- `action_code_to_job`
- `selected_job`
- `selected_action_code`
- `reason_input_text`
- `reason_target_text`
- `chosen_transition_features`
- `contrast_action_codes`
- `contrast_transition_features`
- `reason_source`
  - e.g. `deterministic_teacher_v1`

## Frozen Reason Input Format

`reason_input_text` should be separate from `state_text`.

Recommended format:

```text
You are analyzing an already-selected JSSP action.
Objective: explain why this action was selected and why strong alternatives were not selected.

{state_text}

Selected action: <S4507>
Chosen transition:
- cmax: 420->438
- delta_cmax: 18
- est_start: 415
- est_end: 438
- machine_idle_gap: 15
- remaining_work_after: 91

Top alternatives:
- <S3812>: cmax 420->420, delta_cmax=0, est_start=401, est_end=416, remaining_work_after=105
- <S9033>: cmax 420->451, delta_cmax=31, est_start=420, est_end=451, remaining_work_after=46

Output format:
Reason: ...
Not chosen:
- <Sxxxx>: ...
```

## Frozen Reason Target Format

`reason_target_text`:

```text
Reason: <S4507> is selected because it keeps the projected makespan increase limited while advancing a high-remaining-work job on a bottleneck-relevant route.
Not chosen:
- <S3812>: lower immediate Cmax, but weaker progress on the chosen job's remaining critical workload.
- <S9033>: larger immediate projected makespan increase at this step.
```

Important:
- do not train hidden chain-of-thought
- train concise, structured, contrastive explanations only

## Generator Interface Plan

Keep a single shared rollout engine, but expose role-based generation.

Recommended CLI:

```bash
python generate_jssp_step_dataset.py \
  --input llm_jssp/train.json \
  --output train_data/jssp_step_train_policy.jsonl \
  --dataset_role policy \
  --strict_makespan \
  --progress_every 500 \
  --action_code_cap 9999
```

```bash
python generate_jssp_step_dataset.py \
  --input llm_jssp/train.json \
  --output train_data/jssp_step_train_reason.jsonl \
  --dataset_role reason \
  --reason_topk 3 \
  --strict_makespan \
  --progress_every 500 \
  --action_code_cap 9999
```

Optional compatibility mode:

```bash
python generate_jssp_step_dataset.py \
  --input llm_jssp/train.json \
  --output train_data/jssp_step_train_with_reason.jsonl \
  --dataset_role both \
  --strict_makespan \
  --progress_every 500 \
  --action_code_cap 9999
```

## colab_02 Plan

Add:

- `CFG['adapter_role'] in {'policy', 'reason'}`

### policy mode

- dataset field used as user input:
  - `state_text`
- assistant target:
  - `target_text`
- system prompt:
  - `Return exactly one line: Action: <Sxxxx>`
- output dir:
  - policy adapter path

### reason mode

- dataset field used as user input:
  - `reason_input_text`
- assistant target:
  - `reason_target_text`
- system prompt:
  - `Output format: Reason: ... / Not chosen: ...`
- output dir:
  - reason adapter path

## colab_03 Plan

Add:

- `CFG['policy_model_repo_or_path']`
- `CFG['reason_model_repo_or_path']`
- `CFG['enable_reason_adapter']`

Inference flow:

1. load base model
2. load policy adapter as `policy`
3. optionally load reason adapter as `reason`
4. `model.set_adapter('policy')`
5. generate `Action: <Sxxxx>` step-by-step
6. if reason enabled:
   - build `reason_input_text`
   - `model.set_adapter('reason')`
   - generate explanation
   - switch back to `policy`

## colab_05 / colab_06 Plan

- unchanged in principle:
  - policy adapter only
- no reason generation in rollout
- no adapter switch in RL

## Implementation Order

1. extend `generate_jssp_step_dataset.py` with `--dataset_role`
2. add `reason_input_text` / `reason_target_text` generation
3. keep compatibility mode for old mixed dataset
4. update `colab_02` with `adapter_role`
5. update `colab_03` with adapter switching
6. leave `colab_05/06` as policy-only

## Short Recommendation

The clean final architecture is:

- `policy dataset` -> `policy adapter` -> action selection / RL
- `reason dataset` -> `reason adapter` -> explanation only

This is the best trade-off between:

- scheduling performance
- explanation quality
- decoding stability
- RL simplicity
