# Notebook Guide

0. `colab_00_generate_step_dataset.ipynb`
   - Colab 실행용 inline step 데이터 생성 노트북
   - one-shot `train.json` -> step `jsonl` 변환
   - strict/strict_makespan/fail_fast/progress 옵션 포함
   - 생성 결과 HF dataset repo 업로드 옵션 포함

1. `colab_01_upload_datasets_to_hf.ipynb`
   - 로컬 실행용 dataset 업로드 노트북

2. `[action_code_add][fix]colab_02_train_step_lora_full.ipynb`
   - canonical SFT 노트북
   - action code prompt + `<CAND_SCORE>` candidate-scoring reranker 학습
   - LoRA adapter와 `candidate_scorer.pt`를 함께 저장

3. `[action_code_add][fix]colab_03_inference_step_full.ipynb`
   - canonical inference 노트북
   - 후보 전체를 candidate score head로 rerank
   - greedy 첫 sample + sampling 나머지 sample, per-sample CSV 저장 지원

4. `colab_04_upload_model_to_hf.ipynb`
   - 로컬 실행용 모델 업로드 노트북

5. `[action_code_add][fix]colab_05_rl_full.ipynb`
   - canonical RL 노트북
   - GRPO / BOPO / REINFORCE 모두 candidate-scoring policy 기준
   - `score_head_only` 모드로 `candidate_score_head`만 업데이트 가능

추가 문서: `NOTEBOOK_PARITY_CHECKLIST.md`
