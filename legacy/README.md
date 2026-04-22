# Legacy Archive

이 디렉터리는 현재 canonical 실행 경로에서 제외된 파일을 보관한다.

현재 표준은 `notebooks/`의 `[action_code_add][fix]` 노트북 3개와 Python 진입점이다.

- SFT: `notebooks/[action_code_add][fix]colab_02_train_step_lora_full.ipynb`
- Inference: `notebooks/[action_code_add][fix]colab_03_inference_step_full.ipynb`
- RL: `notebooks/[action_code_add][fix]colab_05_rl_full.ipynb`
- Python SFT: `unified_trainer.py`
- Python inference: `inference_jssp_fssp.py`
- Python RL: `RL_jssp_fssp.py`

`legacy/python_generation/`에는 token-generation 기반 이전 RL/training 파일이 있다.
예전 notebook variants, 실행 로그, 임시 데이터 조각은 로컬 보관용이며 GitHub 업로드 대상에서 제외한다.
