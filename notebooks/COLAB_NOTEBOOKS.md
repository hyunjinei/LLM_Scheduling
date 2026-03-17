# Notebook Guide

0. `colab_00_generate_step_dataset.ipynb`
   - Colab 실행용 inline step 데이터 생성 노트북
   - one-shot `train.json` -> step `jsonl` 변환
   - strict/strict_makespan/fail_fast/progress 옵션 포함
   - 생성 결과 HF dataset repo 업로드 옵션 포함

1. `colab_01_upload_datasets_to_hf.ipynb`
   - 로컬 실행용 dataset 업로드 노트북

2. `colab_02_train_step_lora_full.ipynb`
   - `unified_trainer.py` 대응 full 옵션 노트북
   - dataset source(hf/local), shuffle, resume, output_dir 자동명명 지원
   - 학습 완료 후 HF model repo 업로드 옵션 포함

3. `colab_03_inference_step_full.ipynb`
   - `inference_jssp_fssp.py` 대응 full 실행 노트북
   - demo/전체평가, step reflection, rationale, CSV 출력 포함
   - strict step parsing + decoding masking 포함

4. `colab_04_upload_model_to_hf.ipynb`
   - 로컬 실행용 모델 업로드 노트북

5. `colab_05_rl_full.ipynb`
   - `RL_jssp_fssp.py` 대응 full 실행 노트북
   - `rl_algo=reinforce/grpo` 둘 다 지원
   - random problem / dataset source(hf/local) 지원
   - 학습 완료 후 로컬 저장 + 선택적 HF model repo 업로드

추가 문서: `NOTEBOOK_PARITY_CHECKLIST.md`
