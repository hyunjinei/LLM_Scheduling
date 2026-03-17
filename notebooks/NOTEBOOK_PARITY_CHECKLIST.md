# Notebook Parity Checklist

## Entry Script -> Notebook
- `generate_jssp_step_dataset.py` -> `colab_00_generate_step_dataset.ipynb`
- `unified_trainer.py` -> `colab_02_train_step_lora_full.ipynb`
- `inference_jssp_fssp.py` -> `colab_03_inference_step_full.ipynb`
- `RL_jssp_fssp.py` -> `colab_05_rl_full.ipynb`

## Implemented
- All notebooks use inline Python code (no `python xxx.py` execution).
- HF dataset/model source path supported.
- Local path source options supported where relevant.
- Strict step action parsing (fallback 없음) 반영.
- Transition-state schema v2 반영:
  `current_cmax`, machine-load summary, candidate transition features,
  action-conditioned `cmax/job_ready/machine_ready` before->after.
- `colab_00`, `colab_03`, `colab_05`, `colab_06` utility cells synchronized with source parity overrides.
- `colab_02` sample preview shows `feature_schema_version` when present.

## Notes
- Keep the full notebooks (`colab_02_train_step_lora_full.ipynb`, `colab_03_inference_step_full.ipynb`, `colab_05_rl_full.ipynb`) for script-level parity.
