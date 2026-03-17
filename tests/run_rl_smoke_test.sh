#!/bin/bash
set -e

MODEL_PATH="${MODEL_PATH:-unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit}"

python3 RL_jssp_fssp.py \
  --model_path "$MODEL_PATH" \
  --rl_algo grpo \
  --group_size 2 \
  --grpo_epochs 1 \
  --use_random_problems \
  --random_jobs 3 \
  --random_machines 3 \
  --episodes_per_epoch 1 \
  --epochs 1 \
  --step_action_max_new_tokens 8 \
  --invalid_makespan_penalty 1000000 \
  --learning_rate 5e-5 \
  --load_in_4bit
