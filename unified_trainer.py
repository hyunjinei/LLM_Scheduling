import argparse
import torch
import wandb
from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
import csv
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import pandas as pd
from pathlib import Path
import json
from functools import partial
from llm_jssp.utils.helping_functions_korea import print_number_of_trainable_model_parameters
from llm_jssp.utils.action_token_utils import (
    ensure_action_special_tokens,
    validate_action_tokenizer_installation,
)

torch.cuda.empty_cache()
torch.cuda.ipc_collect()

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a FastLanguageModel with specified parameters.")
    
    # Model and data parameters
    parser.add_argument('--max_seq_length', type=int, default=40000, help='Maximum sequence length')  # 8192 -> 40000
    parser.add_argument('--model_type', type=str, default='llama8b', 
                        choices=['llama8b', 'llama1b', 'qwen25_7b','qwen25_7b_math', 'qwen25_14b', 'deepseek_8b'], 
                        help='Which model to use')
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'], 
                        help='Data type (bfloat16 or float16)')
    parser.add_argument('--load_in_4bit', action='store_true', default=True, 
                        help='Use 4-bit quantization to reduce memory usage')

    # LoRA hyperparameters - 메모리 절약을 위해 rank 줄임
    parser.add_argument('--lora_r', type=int, default=64, help='Rank of the LoRA decomposition')  # 128 -> 64
    parser.add_argument('--lora_alpha', type=int, default=64, help='Scaling factor for LoRA updates')  # 128 -> 64
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='Dropout rate for LoRA layers (0.0 for Unsloth fast patching)')  # 0.05 -> 0.0
    parser.add_argument('--bias', type=str, default='none', choices=['none', 'all', 'lora_only'], help='Bias type')

    # Additional configurations
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth', help='Use gradient checkpointing')
    parser.add_argument('--random_state', type=int, default=42, help='Random state for reproducibility')
    parser.add_argument('--use_rslora', action='store_true', default=True, help='Use RSLoRA')
    parser.add_argument('--loftq_config', type=str, default=None, help='LoFT-Q configuration')

    # Training hyperparameters
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Batch size per device during training')  # 1 -> 2 (복잡한 프롬프트용)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='Number of gradient accumulation steps')  # 16 -> 8 (메모리 안정성)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=1, help='Batch size per device during evaluation')
    parser.add_argument('--num_train_epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    # parser.add_argument('--max_steps', type=int, default=-1, help='Maximum number of training steps')

    parser.add_argument('--save_steps', type=int, default=50, help='Save checkpoint every X updates steps')
    parser.add_argument('--save_total_limit', type=int, default=100, help='Limit the total amount of checkpoints')
    parser.add_argument('--logging_steps', type=int, default=1, help='Log every X updates steps')
    parser.add_argument('--eval_steps', type=int, default=50, help='Evaluate every X updates steps')
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='Evaluation strategy')
    parser.add_argument('--load_best_model_at_end', type=bool, default=True, help='Load best model at end')
    parser.add_argument('--metric_for_best_model', type=str, default='eval_loss', help='Metric for best model')
    parser.add_argument('--greater_is_better', type=bool, default=False, help='Greater is better for metric')
    parser.add_argument('--dataloader_num_workers', type=int, default=0, help='Number of workers for data loading')
    parser.add_argument('--remove_unused_columns', type=bool, default=False, help='Remove unused columns')
    parser.add_argument('--optim', type=str, default='adamw_8bit', help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler_type', type=str, default='linear', help='Learning rate scheduler type')

    parser.add_argument('--warmup_steps', type=int, default=10, help='Number of warmup steps')

    parser.add_argument('--fp16', type=bool, default=not torch.cuda.is_bf16_supported(), help='Use FP16')
    parser.add_argument('--bf16', type=bool, default=torch.cuda.is_bf16_supported(), help='Use BF16')
    parser.add_argument('--group_by_length', type=bool, default=True, help='Group by length')
    parser.add_argument('--report_to', type=str, default='wandb', help='Report to')
    parser.add_argument('--run_name', type=str, default=None, help='Run name')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Model training options
    parser.add_argument('--train_lm_head', action='store_true', default=False, 
                        help='Whether to train the language model head or not')
    parser.add_argument('--train_embed_tokens', action='store_true', default=False, 
                        help='Whether to train the embed_tokens or not')
    
    # Output format options
    parser.add_argument('--output_accord', action='store_true', default=False, 
                        help='Whether to train using ACCORD-style output or not')
    parser.add_argument('--output_list_of_lists', action='store_true', default=False, 
                        help='Whether to train using only list of lists of index route as an output or not')
    
    # Task type options
    parser.add_argument('--train_vrp_tsp', action='store_true', default=False, 
                        help='Whether to train VRP-TSP model or not')
    parser.add_argument('--train_knapsack', action='store_true', default=False, 
                        help='Whether to train KNAPSACK model or not')
    parser.add_argument('--train_binpack', action='store_true', default=False, 
                        help='Whether to train BINPACK model or not')
    parser.add_argument('--train_jssp', action='store_true', default=False, 
                        help='Whether to train JSSP model or not')
    parser.add_argument('--train_fssp', action='store_true', default=False, 
                        help='Whether to train FSSP model or not')
    
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory name')
    parser.add_argument('--wandb_project', type=str, default=None, 
                        help='WandB project name (default: derived from task type)')
    parser.add_argument(
        '--step_dataset_path',
        type=str,
        default='train_data/jssp_step_train.jsonl',
        help='Path to step dataset JSONL',
    )
    parser.add_argument(
        '--step_supervision_mode',
        type=str,
        default='action_only',
        choices=['action_only', 'action_reason', 'reason_only'],
        help='Step SFT target style: action only, action + rationale text, or rationale only.',
    )
    parser.add_argument('--action_code_width', type=int, default=4, help='Fixed digit width for action tokens (e.g., <A0001>).')
    parser.add_argument('--action_code_cap', type=int, default=9999, help='Upper bound of action token pool (sampled sparsely per step from [1, cap]).')
    
    # 🔥 NEW: Checkpoint 재시작 및 데이터 셔플링 옵션
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Specific checkpoint path to resume from (e.g., ~/finetuned_models/model/checkpoint-1072)')
    parser.add_argument('--shuffle_data', action='store_true', default=False,
                        help='Shuffle training data order (default: False, keeps 2x2 to 10x10 order)')
    parser.add_argument('--shuffle_seed', type=int, default=42,
                        help='Random seed for data shuffling (only used when --shuffle_data is True)')

    args = parser.parse_args()

    # Print output style information
    if args.output_accord:
        print("=="*60)
        print("Training with ACCORD style output")
        print("=="*60)
    else:
        print("=="*60)
        print("Training with list of lists style output")
        print("=="*60)
    
    # Determine the task type
    task_type = None
    if args.train_vrp_tsp:
        task_type = "vrp_tsp"
    elif args.train_knapsack:
        task_type = "knapsack"
    elif args.train_binpack:
        task_type = "binpack"
    elif args.train_jssp:
        task_type = "jssp"
    elif args.train_fssp:
        task_type = "fssp"
    else:
        raise ValueError("No task type selected. Please specify a training task.")

    if args.train_jssp and args.step_supervision_mode in {'action_only', 'action_reason', 'reason_only'}:
        if not args.train_embed_tokens:
            print("⚠️ Enabling train_embed_tokens for single-token action policy.")
            args.train_embed_tokens = True
        if not args.train_lm_head:
            print("⚠️ Enabling train_lm_head for single-token action policy.")
            args.train_lm_head = True

    # =========================
    # Load Model and Tokenizer
    # =========================

    # Correct the model names
    if args.model_type == 'llama8b':
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    elif args.model_type == 'llama1b':
        model_name = "unsloth/Llama-3.2-1B-Instruct-bnb-4bit"
    elif args.model_type == 'qwen25_7b':
        # model_name = "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit" # 이게원인인듯
        model_name = "unsloth/Qwen2.5-7B-Instruct-bnb-4bit"
    elif args.model_type == 'qwen25_7b_math':
        model_name = "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit"
    elif args.model_type == 'qwen25_14b':
        model_name = "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit"
    elif args.model_type == 'deepseek_8b':
        model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit"
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    base_model = os.path.basename(model_name)
    print("base_model: ", base_model)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16,
        local_files_only=False,
    ) 
    
    print("Model loaded successfully.")
    print("Model dtype: ", model.dtype)
    print(f"args.max_seq_length {args.max_seq_length}")
    print("Model max_seq_length: ", model.config.max_position_embeddings)
    token_install = ensure_action_special_tokens(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
    )
    validate_action_tokenizer_installation(
        tokenizer=tokenizer,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
    )
    print("Action token install:", token_install)

    # Define modules to train
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj", 
    ]
    if args.train_lm_head:
        target_modules.append("lm_head")
    if args.train_embed_tokens:
        target_modules.append("embed_tokens")

    print("Target modules: ", target_modules)

    # Configure the model with PEFT (Parameter-Efficient Fine-Tuning) - Korea style
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,  # 64 사용
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,  # 64 사용
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_rslora=True,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        loftq_config=args.loftq_config,
    )

    # English style 파라미터 출력
    print(print_number_of_trainable_model_parameters(model))

    from datasets import load_dataset

    step_dataset_path = os.path.expanduser(args.step_dataset_path)
    if not os.path.exists(step_dataset_path):
        print(f"❌ step dataset 파일이 없습니다: {step_dataset_path}")
        print("   먼저 generate_jssp_step_dataset.py를 실행하세요.")
        return

    dataset = load_dataset("json", data_files=step_dataset_path, split="train")
    print(f"✅ step 데이터 로딩 완료: {len(dataset):,}개")

    # 🔍 데이터 순서 확인 (첫 5개 문제 크기)
    print("\n📊 현재 데이터 순서 (첫 5개):")
    for i in range(min(5, len(dataset))):
        try:
            n = dataset[i]["num_jobs"]
            m = dataset[i]["num_machines"]
            print(f"   {i+1}: {n}x{m} 문제")
        except Exception:
            print(f"   {i+1}: 크기 정보 읽기 실패")

    # 🔥 NEW: 데이터 셔플링 옵션
    if args.shuffle_data:
        print(f"\n🔀 데이터 셔플링 중... (seed: {args.shuffle_seed})")
        dataset = dataset.shuffle(seed=args.shuffle_seed)
        print("✅ 데이터 셔플링 완료!")

        # 셔플링 후 순서 확인
        print("\n📊 셔플링 후 데이터 순서 (첫 5개):")
        for i in range(min(5, len(dataset))):
            try:
                n = dataset[i]["num_jobs"]
                m = dataset[i]["num_machines"]
                print(f"   {i+1}: {n}x{m} 문제")
            except Exception:
                print(f"   {i+1}: 크기 정보 읽기 실패")
    else:
        print("\n📋 데이터 셔플링 비활성화 (step dataset 원본 순서 유지)")

    test_size = 0.05
    pre_map_cap_candidates = []
    if args.max_train_samples is not None:
        pre_map_cap_candidates.append(math.ceil(int(args.max_train_samples) / (1.0 - test_size)))
    if args.max_eval_samples is not None and args.evaluation_strategy != "no":
        pre_map_cap_candidates.append(math.ceil(int(args.max_eval_samples) / test_size))
    pre_map_cap = max(pre_map_cap_candidates) if pre_map_cap_candidates else None
    if pre_map_cap is not None:
        pre_map_cap = min(int(pre_map_cap), len(dataset))
        print(f"📉 pre-map sample cap applied: {pre_map_cap:,} rows")
        dataset = dataset.select(range(pre_map_cap))

    # step 프롬프트 생성 (토큰화는 SFTTrainer가 담당)
    from llm_jssp.utils.data_preprocessing_english import create_step_prompt_formats
    print("🔄 step 프롬프트 시스템 적용중...")
    print(f"   - step supervision mode: {args.step_supervision_mode}")
    _create_prompt_formats = partial(
        create_step_prompt_formats,
        tokenizer=tokenizer,
        step_supervision_mode=args.step_supervision_mode,
    )
    dataset = dataset.map(
        _create_prompt_formats,
        num_proc=max(1, int(args.dataset_num_proc)),
    )
    print("✅ 프롬프트 생성 완료!")
    
    # 간단한 train_test_split
    split_dataset = dataset.train_test_split(test_size=test_size, seed=args.seed)
    
    # SFTTrainer가 토큰화 담당 (text 필드 사용)
    train_dataset = split_dataset['train']
    eval_dataset = split_dataset['test']

    print("train_dataset length : ", len(train_dataset))
    print("eval_dataset length : ", len(eval_dataset))

    # 프롬프트 샘플 확인
    print("\n🔍 프롬프트 샘플 (첫 500자):")
    if "text" in train_dataset[0]:
        sample_text = train_dataset[0]["text"][:500]
        print(f"{sample_text}...")
        print(f"📏 프롬프트 길이: {len(train_dataset[0]['text'])} 문자")
    else:
        print("❌ 'text' 필드가 없습니다!")
        print(f"🔍 사용 가능한 필드: {list(train_dataset[0].keys())}")

    # 메모리 절약을 위해 긴 출력 주석처리
    # print("train_dataset: ", train_dataset[0])
    # print("train_dataset: ", eval_dataset[0])

    
    # # Analyze token lengths
    # texts = train_dataset["text"]
    # tokenized_lengths = [len(tokenizer(text, truncation=False, add_special_tokens=False)["input_ids"]) for text in texts]

    # # Calculate statistics
    # avg_length = sum(tokenized_lengths) / len(tokenized_lengths)
    # min_length = min(tokenized_lengths)
    # max_length = max(tokenized_lengths)

    # print("Average prompt length:", avg_length)
    # print("Minimum prompt length:", min_length)
    # print("Maximum prompt length:", max_length)

    # =========================
    # Generate Output Directory Name
    # =========================
    
    # Create a concise but descriptive output directory name
    output_style = "accord" if args.output_accord else "list_of_lists" if args.output_list_of_lists else "default"
    
    # Define model_short_name before any usage
    if args.model_type == 'llama8b':
        model_short_name = "llama8b"
    elif args.model_type == 'llama1b':
        model_short_name = "llama1b"
    elif args.model_type == 'qwen2.5_7b':
        model_short_name = "qwen25_7b"
    elif args.model_type == 'qwen2.5_14b':
        model_short_name = "qwen25_14b"
    elif args.model_type == 'deepseek_8b':
        model_short_name = "deepseek_8b"
    elif args.model_type == 'qwen25_7b_math':
        model_short_name = "qwen25_7b_math"
    else:
        model_short_name = base_model.replace("-", "_")

        
    if args.output_dir is None:
        dataset_tag = f"step_{args.step_supervision_mode}"
        # Create a clean and informative directory name
        dir_out = os.path.join(
            "finetuned_models",
            f"{task_type}_{model_short_name}_{output_style}_{dataset_tag}_r{args.lora_r}_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    os.makedirs(dir_out, exist_ok=True)
    print("Output directory: ", dir_out)
    
    # =========================
    # Initialize WandB
    # =========================
    
    # Create a descriptive wandb run name
    wandb_run_name = f"{task_type}_{model_short_name}_{output_style}_r{args.lora_r}"
    
    # Set the project name based on task or user preference
    if args.wandb_project:
        project_name = args.wandb_project
    else:
        project_name = f"{task_type}_optimization"
    
    wandb.init(
        project=project_name,
        name=wandb_run_name,
        config=vars(args)  # Log all args to wandb
    )

    # Save hyperparameters to CSV
    with open(os.path.join(dir_out, 'training_hyperparams_args.csv'), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for key, value in vars(args).items():
            writer.writerow([key, value])

    # =========================
    # Initialize the Trainer
    # =========================

    if args.train_lm_head and args.train_embed_tokens:
        from unsloth import UnslothTrainer, UnslothTrainingArguments
        Trainer = UnslothTrainer
        TrainingArguments = UnslothTrainingArguments
        print("Training with UnslothTrainer")
    else:
        from trl import SFTTrainer
        from transformers import TrainingArguments
        Trainer = SFTTrainer
        TrainingArguments = TrainingArguments
        print("Training with SFTTrainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=40,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            bf16=True if args.dtype == "bfloat16" else False,
            fp16=True if args.dtype == "float16" else False,
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="wandb",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            group_by_length=True,  # 동적 패딩을 위해 명시적으로 설정
        ),
    )

    # =========================
    # Monitor GPU Memory Usage
    # =========================
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    # =========================
    # Start Training
    # =========================
    
    # 🔥 NEW: Checkpoint 재시작 로직 개선
    if args.resume_from_checkpoint:
        # 사용자가 특정 checkpoint 경로를 지정한 경우
        checkpoint_path = os.path.expanduser(args.resume_from_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"🔄 지정된 Checkpoint에서 재시작: {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            print(f"❌ 지정된 Checkpoint 경로가 존재하지 않습니다: {checkpoint_path}")
            print("새로운 모델로 학습을 시작합니다.")
            trainer.train()
    else:
        # 기존 로직: output_dir에서 checkpoint 자동 탐지
        contains_checkpoints = any(
            "checkpoint" in name for name in os.listdir(dir_out) if os.path.isdir(os.path.join(dir_out, name))
        )

        if not contains_checkpoints:
            print("Checkpoint dir is empty: Training new model")
            trainer.train()
        else:
            print("Checkpoint dir is NOT empty: Continuing training")
            trainer.train(resume_from_checkpoint=True)

if __name__ == "__main__":
    main()
