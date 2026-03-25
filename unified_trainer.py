import argparse
import torch
import wandb
import torch.nn as nn
import numpy as np
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


class StepSupervisionCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = (
            getattr(tokenizer, "pad_token_id", None)
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(tokenizer, "eos_token_id", 0)
        )

    def __call__(self, features):
        max_len = max(len(feature["input_ids"]) for feature in features)
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": [],
            "loss_weights": [],
            "action_target_mask": [],
            "feasible_action_ids": [],
        }
        max_action_count = max(len(feature.get("feasible_action_ids", [])) for feature in features)
        for feature in features:
            seq_len = len(feature["input_ids"])
            pad_len = max_len - seq_len
            batch["input_ids"].append(
                list(feature["input_ids"]) + [int(self.pad_token_id)] * pad_len
            )
            batch["attention_mask"].append(
                list(feature["attention_mask"]) + [0] * pad_len
            )
            batch["labels"].append(
                list(feature["labels"]) + [-100] * pad_len
            )
            batch["loss_weights"].append(
                list(feature["loss_weights"]) + [0.0] * pad_len
            )
            batch["action_target_mask"].append(
                list(feature.get("action_target_mask", [])) + [0] * pad_len
            )
            feasible_action_ids = list(feature.get("feasible_action_ids", []))
            batch["feasible_action_ids"].append(
                feasible_action_ids + [-1] * (max_action_count - len(feasible_action_ids))
            )
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long),
            "loss_weights": torch.tensor(batch["loss_weights"], dtype=torch.float32),
            "action_target_mask": torch.tensor(batch["action_target_mask"], dtype=torch.long),
            "feasible_action_ids": torch.tensor(batch["feasible_action_ids"], dtype=torch.long),
        }


def _build_step_supervision_trainer(base_cls):
    class StepSupervisionTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            model_inputs = dict(inputs)
            labels = model_inputs.pop("labels")
            loss_weights = model_inputs.pop("loss_weights", None)
            action_target_mask = model_inputs.pop("action_target_mask", None)
            feasible_action_ids = model_inputs.pop("feasible_action_ids", None)
            outputs = model(**model_inputs)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).view_as(shift_labels)

            valid_mask = shift_labels.ne(-100)
            if loss_weights is None:
                weights = valid_mask.to(token_loss.dtype)
            else:
                shift_weights = loss_weights[..., 1:].contiguous().to(token_loss.dtype)
                weights = shift_weights * valid_mask.to(token_loss.dtype)

            if action_target_mask is not None:
                shift_action_mask = action_target_mask[..., 1:].contiguous().bool()
            else:
                shift_action_mask = torch.zeros_like(valid_mask, dtype=torch.bool)

            base_weights = weights * (~shift_action_mask).to(weights.dtype)
            loss_num = (token_loss * base_weights).sum()
            denom = base_weights.sum()

            if feasible_action_ids is not None and bool(shift_action_mask.any().item()):
                matched_action_targets = 0
                action_rows, action_cols = torch.nonzero(shift_action_mask, as_tuple=True)
                for row_idx, col_idx in zip(action_rows.tolist(), action_cols.tolist()):
                    row_feasible_ids = feasible_action_ids[row_idx]
                    row_feasible_ids = row_feasible_ids[row_feasible_ids.ge(0)]
                    if row_feasible_ids.numel() <= 0:
                        raise ValueError(
                            "Empty feasible_action_ids encountered at an action-supervised position."
                        )
                    if row_feasible_ids.numel() < 2:
                        continue
                    target_id = int(shift_labels[row_idx, col_idx].item())
                    target_matches = torch.nonzero(
                        row_feasible_ids.eq(target_id),
                        as_tuple=False,
                    ).view(-1)
                    if target_matches.numel() <= 0:
                        raise ValueError(
                            "Action target token id was not found inside feasible_action_ids. "
                            f"target_id={target_id}, feasible_count={int(row_feasible_ids.numel())}, "
                            f"feasible_head={row_feasible_ids[:8].tolist()}"
                        )
                    candidate_logits = shift_logits[row_idx, col_idx].index_select(
                        0,
                        row_feasible_ids.to(shift_logits.device, dtype=torch.long),
                    )
                    target_index = target_matches[0].to(
                        device=shift_logits.device,
                        dtype=torch.long,
                    ).view(1)
                    action_loss = nn.functional.cross_entropy(
                        candidate_logits.view(1, -1),
                        target_index,
                        reduction="sum",
                    )
                    action_weight = weights[row_idx, col_idx].to(action_loss.dtype)
                    loss_num = loss_num + (action_loss * action_weight)
                    denom = denom + action_weight
                    matched_action_targets += 1

                if matched_action_targets <= 0:
                    raise ValueError(
                        "No matched action targets with feasible_count >= 2 contributed "
                        "to the action-centric loss."
                    )

            denom = denom.clamp_min(1.0)
            loss = loss_num / denom
            return (loss, outputs) if return_outputs else loss

    return StepSupervisionTrainer

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
    parser.add_argument(
        '--eval_split_ratio',
        type=float,
        default=0.05,
        help='Instance-level eval split ratio used when evaluation is enabled.',
    )
    parser.add_argument(
        '--eval_split_mode',
        type=str,
        default='fixed_per_size',
        choices=['fixed_per_size', 'ratio'],
        help='How to select eval instances: fixed count per problem size or global ratio.',
    )
    parser.add_argument(
        '--eval_instances_per_size',
        type=int,
        default=3,
        help='When eval_split_mode=fixed_per_size, hold out this many instances per (num_jobs, num_machines).',
    )
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
    parser.add_argument(
        '--action_loss_weight',
        type=float,
        default=4.0,
        help='Weight multiplier applied to the first supervised <Axxxx> assistant token.',
    )
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
        '--max_train_samples',
        type=int,
        default=None,
        help='Optional cap on train rows after split.',
    )
    parser.add_argument(
        '--max_eval_samples',
        type=int,
        default=None,
        help='Optional cap on eval rows after split.',
    )
    parser.add_argument(
        '--dataset_num_proc',
        type=int,
        default=16,
        help='Number of worker processes used for dataset preprocessing.',
    )
    parser.add_argument(
        '--min_train_feasible_actions',
        type=int,
        default=1,
        help='Keep only train rows whose feasible action count is at least this value.',
    )
    parser.add_argument(
        '--min_eval_feasible_actions',
        type=int,
        default=1,
        help='Keep only eval rows whose feasible action count is at least this value.',
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

    def _try_get_source_index_array(hf_dataset):
        if "source_index" not in set(hf_dataset.column_names):
            return None
        arrow_table = getattr(hf_dataset, "_data", None)
        if arrow_table is not None and hasattr(arrow_table, "column"):
            try:
                column = arrow_table.column("source_index")
                chunks = [
                    np.asarray(chunk.to_numpy(zero_copy_only=False), dtype=np.int64)
                    for chunk in getattr(column, "chunks", [])
                ]
                if chunks:
                    return np.concatenate(chunks)
            except Exception:
                pass
        return np.asarray(hf_dataset["source_index"], dtype=np.int64)

    def _try_get_int_array(hf_dataset, column_name):
        if column_name not in set(hf_dataset.column_names):
            return None
        arrow_table = getattr(hf_dataset, "_data", None)
        if arrow_table is not None and hasattr(arrow_table, "column"):
            try:
                column = arrow_table.column(column_name)
                chunks = [
                    np.asarray(chunk.to_numpy(zero_copy_only=False), dtype=np.int64)
                    for chunk in getattr(column, "chunks", [])
                ]
                if chunks:
                    return np.concatenate(chunks)
            except Exception:
                pass
        return np.asarray(hf_dataset[column_name], dtype=np.int64)

    def _count_feasible_actions(example):
        codes = example.get("feasible_action_codes")
        if not isinstance(codes, list):
            raise ValueError(
                "Step dataset row is missing 'feasible_action_codes' list required for "
                "min_feasible_actions filtering."
            )
        return len(codes)

    def _filter_by_min_feasible_actions(hf_dataset, min_count, split_name):
        min_count = max(1, int(min_count))
        if min_count <= 1 or hf_dataset is None:
            return hf_dataset
        before_count = len(hf_dataset)
        hf_dataset = hf_dataset.filter(
            lambda example: _count_feasible_actions(example) >= min_count,
            num_proc=max(1, int(args.dataset_num_proc)),
        )
        after_count = len(hf_dataset)
        print(
            f"🧹 {split_name} feasible-action filter:"
            f" min_feasible_actions>={min_count},"
            f" kept={after_count:,}/{before_count:,}"
        )
        return hf_dataset

    def _resolve_instance_keys(hf_dataset):
        columns = set(hf_dataset.column_names)
        if "source_index" in columns:
            source_values = _try_get_source_index_array(hf_dataset)
            if source_values is not None:
                return [f"source_{int(x)}" for x in source_values.tolist()]

        instance_values = hf_dataset["instance_id"] if "instance_id" in columns else None
        keys = []
        if instance_values is None:
            raise ValueError(
                "Step dataset must contain either 'instance_id' or 'source_index' for instance-level split."
            )
        for raw_instance_id in instance_values:
            instance_id = "" if raw_instance_id is None else str(raw_instance_id).strip()
            if not instance_id:
                raise ValueError(
                    "Empty instance_id encountered and source_index is unavailable for instance-level split."
                )
            keys.append(instance_id)
        return keys

    def _ordered_unique(keys):
        seen = set()
        ordered = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            ordered.append(key)
        return ordered

    def _split_indices_by_instance(
        hf_dataset,
        test_ratio,
        split_seed,
        enable_eval_split,
        split_mode="fixed_per_size",
        eval_instances_per_size=3,
    ):
        source_values = _try_get_source_index_array(hf_dataset)
        if source_values is not None and source_values.size > 0:
            ordered_pos = np.sort(np.unique(source_values, return_index=True)[1])
            ordered_sources = source_values[ordered_pos].tolist()
            ordered_instances = [f"source_{int(x)}" for x in ordered_sources]
            if not enable_eval_split:
                train_indices = np.arange(len(source_values), dtype=np.int64).tolist()
                return train_indices, [], ordered_instances, []

            resolved_split_mode = str(split_mode).lower()
            rng = random.Random(split_seed)
            if resolved_split_mode == "fixed_per_size":
                num_jobs_values = _try_get_int_array(hf_dataset, "num_jobs")
                num_machines_values = _try_get_int_array(hf_dataset, "num_machines")
                if num_jobs_values is None or num_machines_values is None:
                    raise ValueError(
                        "fixed_per_size eval split requires 'num_jobs' and 'num_machines' columns."
                    )
                ordered_num_jobs = num_jobs_values[ordered_pos].tolist()
                ordered_num_machines = num_machines_values[ordered_pos].tolist()
                size_to_sources = defaultdict(list)
                ordered_sizes = []
                seen_sizes = set()
                for source_idx, n_jobs, n_machines in zip(
                    ordered_sources,
                    ordered_num_jobs,
                    ordered_num_machines,
                ):
                    size_key = (int(n_jobs), int(n_machines))
                    if size_key not in seen_sizes:
                        seen_sizes.add(size_key)
                        ordered_sizes.append(size_key)
                    size_to_sources[size_key].append(int(source_idx))
                eval_sources = []
                per_size = max(1, int(eval_instances_per_size))
                for size_key in ordered_sizes:
                    candidates = list(size_to_sources[size_key])
                    rng.shuffle(candidates)
                    eval_sources.extend(candidates[: min(per_size, len(candidates))])
            else:
                shuffled_sources = ordered_sources[:]
                rng.shuffle(shuffled_sources)
                eval_instance_count = max(1, int(round(len(shuffled_sources) * float(test_ratio))))
                eval_instance_count = min(eval_instance_count, max(1, len(shuffled_sources) - 1))
                eval_sources = shuffled_sources[:eval_instance_count]

            eval_source_set = {int(x) for x in eval_sources}
            eval_mask = np.isin(source_values, np.asarray(eval_sources, dtype=np.int64))
            train_indices = np.flatnonzero(~eval_mask).astype(np.int64).tolist()
            eval_indices = np.flatnonzero(eval_mask).astype(np.int64).tolist()
            train_instance_ids = [
                f"source_{int(x)}" for x in ordered_sources if int(x) not in eval_source_set
            ]
            eval_instance_ids = [
                f"source_{int(x)}" for x in ordered_sources if int(x) in eval_source_set
            ]
            return train_indices, eval_indices, train_instance_ids, eval_instance_ids

        instance_keys = _resolve_instance_keys(hf_dataset)
        ordered_instances = _ordered_unique(instance_keys)
        if not ordered_instances:
            raise ValueError("No instances found for instance-level split.")

        if not enable_eval_split:
            train_indices = list(range(len(hf_dataset)))
            return train_indices, [], ordered_instances, []

        shuffled_instances = ordered_instances[:]
        rng = random.Random(split_seed)
        rng.shuffle(shuffled_instances)
        eval_instance_count = max(1, int(round(len(shuffled_instances) * float(test_ratio))))
        eval_instance_count = min(eval_instance_count, max(1, len(shuffled_instances) - 1))
        eval_instance_set = set(shuffled_instances[:eval_instance_count])

        train_indices = []
        eval_indices = []
        for row_idx, instance_key in enumerate(instance_keys):
            if instance_key in eval_instance_set:
                eval_indices.append(row_idx)
            else:
                train_indices.append(row_idx)

        train_instance_ids = [iid for iid in ordered_instances if iid not in eval_instance_set]
        eval_instance_ids = [iid for iid in ordered_instances if iid in eval_instance_set]
        return train_indices, eval_indices, train_instance_ids, eval_instance_ids

    # 🔍 데이터 순서 확인 (첫 5개 문제 크기)
    print("\n📊 현재 데이터 순서 (첫 5개):")
    for i in range(min(5, len(dataset))):
        try:
            n = dataset[i]["num_jobs"]
            m = dataset[i]["num_machines"]
            print(f"   {i+1}: {n}x{m} 문제")
        except Exception:
            print(f"   {i+1}: 크기 정보 읽기 실패")

    test_size = max(0.0, min(0.99, float(args.eval_split_ratio)))
    enable_eval = args.evaluation_strategy != "no"
    train_indices, eval_indices, train_instance_ids, eval_instance_ids = _split_indices_by_instance(
        dataset,
        test_ratio=test_size,
        split_seed=args.seed,
        enable_eval_split=enable_eval,
        split_mode=args.eval_split_mode,
        eval_instances_per_size=args.eval_instances_per_size,
    )

    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices) if enable_eval else None

    overlap_count = len(set(train_instance_ids) & set(eval_instance_ids))
    print(
        f"\n🧩 instance split:"
        f" mode={args.eval_split_mode},"
        f" train_instances={len(train_instance_ids):,},"
        f" eval_instances={len(eval_instance_ids):,},"
        f" overlap={overlap_count}"
    )
    print(
        f"🧩 row counts before map:"
        f" train_rows={len(train_dataset):,},"
        f" eval_rows={(len(eval_dataset) if eval_dataset is not None else 0):,}"
    )

    train_dataset = _filter_by_min_feasible_actions(
        train_dataset,
        args.min_train_feasible_actions,
        "train",
    )
    if eval_dataset is not None:
        eval_dataset = _filter_by_min_feasible_actions(
            eval_dataset,
            args.min_eval_feasible_actions,
            "eval",
        )

    if args.shuffle_data:
        print(f"\n🔀 train 데이터 셔플링 중... (seed: {args.shuffle_seed})")
        train_dataset = train_dataset.shuffle(seed=args.shuffle_seed)
        print("✅ train 데이터 셔플링 완료!")
    else:
        print("\n📋 train 데이터 셔플링 비활성화 (instance split 후 원본 순서 유지)")

    if args.max_train_samples is not None:
        train_cap = min(int(args.max_train_samples), len(train_dataset))
        print(f"📉 train sample cap applied: {train_cap:,} rows")
        train_dataset = train_dataset.select(range(train_cap))
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_cap = min(int(args.max_eval_samples), len(eval_dataset))
        print(f"📉 eval sample cap applied: {eval_cap:,} rows")
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(eval_cap))

    # step 프롬프트 + explicit supervision 생성
    from llm_jssp.utils.data_preprocessing_english import build_step_supervision_example
    print("🔄 step supervision tokenization 적용중...")
    print(f"   - step supervision mode: {args.step_supervision_mode}")
    _create_prompt_formats = partial(
        build_step_supervision_example,
        tokenizer=tokenizer,
        step_supervision_mode=args.step_supervision_mode,
        max_length=args.max_seq_length,
        action_loss_weight=args.action_loss_weight,
    )
    train_dataset = train_dataset.map(
        _create_prompt_formats,
        num_proc=max(1, int(args.dataset_num_proc)),
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            _create_prompt_formats,
            num_proc=max(1, int(args.dataset_num_proc)),
        )
    print("✅ supervision tokenization 완료!")

    print("train_dataset length : ", len(train_dataset))
    print("eval_dataset length : ", len(eval_dataset) if eval_dataset is not None else 0)

    # 프롬프트 샘플 확인
    print("\n🔍 프롬프트 샘플 (첫 500자):")
    if "text" in train_dataset[0]:
        sample_text = train_dataset[0]["text"][:500]
        print(f"{sample_text}...")
        print(f"📏 프롬프트 길이: {len(train_dataset[0]['text'])} 문자")
        print(
            "🎯 supervision stats:"
            f" prompt_tokens={train_dataset[0].get('prompt_token_count', 'n/a')},"
            f" assistant_tokens={train_dataset[0].get('assistant_token_count', 'n/a')},"
            f" supervised_tokens={train_dataset[0].get('supervised_token_count', 'n/a')}"
        )
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
        from transformers import Trainer
        from transformers import TrainingArguments
        TrainingArguments = TrainingArguments
        print("Training with Trainer")

    Trainer = _build_step_supervision_trainer(Trainer)
    data_collator = StepSupervisionCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
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
            eval_strategy=args.evaluation_strategy,
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            group_by_length=args.group_by_length,
        ),
    )
    print(
        "🎯 action-centric loss:"
        f" mode={args.step_supervision_mode},"
        f" action_weight={float(args.action_loss_weight)},"
        " supervision=assistant-only/action-targeted+feasible-set-ce"
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
