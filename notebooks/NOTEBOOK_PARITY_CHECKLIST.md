# Notebook Parity Checklist

## Entry Script -> Notebook
- `generate_jssp_step_dataset.py` -> `colab_00_generate_step_dataset.ipynb`
- `unified_trainer.py` -> `[action_code_add][fix]colab_02_train_step_lora_full.ipynb`
- `inference_jssp_fssp.py` -> `[action_code_add][fix]colab_03_inference_step_full.ipynb`
- `RL_jssp_fssp.py` -> `[action_code_add][fix]colab_05_rl_full.ipynb`

## Implemented
- All notebooks use inline Python code (no `python xxx.py` execution).
- HF dataset/model source path supported.
- Local path source options supported where relevant.
- Strict step action parsing (fallback 없음) 반영.
- Transition-state schema v2 반영:
  `current_cmax`, machine-load summary, candidate transition features,
  action-conditioned `cmax/job_ready/machine_ready` before->after.
- `colab_00`, canonical action-code candidate-scoring `02/03/05`, and `colab_06` utility cells synchronized with source parity overrides.
- Canonical `02` sample preview shows `feature_schema_version` when present.

## Notes
- The old `colab_02/03/05`, `[fix]`, and `[fix_candidate]` notebook variants are archived under `legacy/notebooks_previous/`.
- Current experiment notebooks are the `[action_code_add][fix]` files in `notebooks/`.
