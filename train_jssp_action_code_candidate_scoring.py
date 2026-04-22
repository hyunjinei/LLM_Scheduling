from __future__ import annotations

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import Dataset
from huggingface_hub import hf_hub_download, login
from transformers import Trainer as HFTrainer
from transformers import TrainerCallback, TrainingArguments
from unsloth import FastLanguageModel

from llm_jssp.utils.action_token_utils import (
    ensure_action_special_tokens,
    validate_action_tokenizer_installation,
)
from llm_jssp.utils.action_code_candidate_scoring import (
    CandidateScoringCollator,
    CandidateScoringModel,
    build_candidate_scoring_example,
    ensure_candidate_score_token,
    maybe_reinitialize_action_token_rows,
    summarize_action_token_row_geometry,
)
from llm_jssp.utils.helping_functions_korea import print_number_of_trainable_model_parameters


MODEL_MAP = {
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "qwen25_7b": "unsloth/Qwen3.5-9B-Base",
    "qwen25_7b_math": "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
    "qwen25_14b": "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
    "deepseek_8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
}


def _load_unsloth_model_with_chat_template_fallback(**load_kwargs):
    try:
        return FastLanguageModel.from_pretrained(**load_kwargs)
    except Exception as exc:
        if "additional_chat_templates" not in str(exc):
            raise
        retry_kwargs = dict(load_kwargs)
        retry_kwargs["local_files_only"] = True
        print("[Warn] additional_chat_templates 404; retrying with local_files_only=True using cached files.")
        return FastLanguageModel.from_pretrained(**retry_kwargs)


REQUIRED_STEP_KEYS = [
    "instance_id",
    "source_index",
    "state_text",
    "target_text",
    "num_jobs",
    "num_machines",
    "total_steps",
    "step_idx",
    "feasible_action_codes",
]


class CandidateScoringTrainer(HFTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        model_inputs = dict(inputs)
        target_candidate_index = model_inputs.pop("target_candidate_index")
        model_inputs.pop("candidate_action_codes", None)
        outputs = model(**model_inputs, target_candidate_index=target_candidate_index)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def save_model(self, output_dir=None, _internal_call=False):
        save_dir = Path(output_dir or self.args.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_checkpoint_bundle(save_dir)
        tok = getattr(self, "tokenizer", None)
        if tok is not None:
            tok.save_pretrained(str(save_dir))


def _infer_model_device(model):
    try:
        emb = model.backbone.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _print_candidate_scoring_probe(model, batch, max_rows=8, topk=5, prefix="CANDIDATE PROBE"):
    model_device = _infer_model_device(model)
    device_batch = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    model_inputs = {k: v for k, v in device_batch.items() if k not in {"target_candidate_index", "candidate_action_codes"}}
    target_candidate_index = device_batch["target_candidate_index"]
    candidate_action_codes = batch["candidate_action_codes"]

    was_training = bool(model.training)
    model.eval()
    try:
        with torch.no_grad():
            outputs = model(**model_inputs)
            candidate_scores = outputs["candidate_scores"]
            probs = torch.softmax(candidate_scores.float(), dim=-1)
            print(f"\n[{prefix}]")
            shown_rows = min(int(max_rows), int(candidate_scores.shape[0]))
            target_probs_summary = []
            uniform_probs_summary = []
            row_ce_summary = []
            random_ce_summary = []
            top1_hits = []
            for row_idx in range(shown_rows):
                row_codes = list(candidate_action_codes[row_idx])
                row_probs = probs[row_idx, : len(row_codes)]
                row_scores = candidate_scores[row_idx, : len(row_codes)].float()
                target_idx = int(target_candidate_index[row_idx].item())
                target_prob = float(row_probs[target_idx].detach().cpu().item())
                uniform_prob = 1.0 / max(1, len(row_codes))
                entropy = float((-(row_probs * row_probs.clamp_min(1e-12).log())).sum().detach().cpu().item())
                random_ce = float(torch.log(torch.tensor(float(len(row_codes)))).item()) if len(row_codes) > 0 else 0.0
                row_ce = float((-row_probs[target_idx].clamp_min(1e-12).log()).detach().cpu().item())
                topk_row = min(int(topk), len(row_codes))
                top_vals, top_idx = torch.topk(row_probs, k=topk_row)
                top1_idx = int(torch.argmax(row_probs).item())
                target_rank = 1 + int((row_probs > row_probs[target_idx]).sum().item())
                top1_hit = int(top1_idx == target_idx)
                target_probs_summary.append(target_prob)
                uniform_probs_summary.append(uniform_prob)
                row_ce_summary.append(row_ce)
                random_ce_summary.append(random_ce)
                top1_hits.append(top1_hit)
                print(
                    {
                        "row_idx": int(row_idx),
                        "feasible_count": int(len(row_codes)),
                        "target_idx": int(target_idx),
                        "target_rank": int(target_rank),
                        "target_action_code": str(row_codes[target_idx]),
                        "top1_action_code": str(row_codes[top1_idx]),
                        "top1_hit": int(top1_hit),
                        "target_prob": float(target_prob),
                        "uniform_prob": float(uniform_prob),
                        "target_prob_delta_vs_uniform": float(target_prob - uniform_prob),
                        "row_ce": float(row_ce),
                        "random_ce": float(random_ce),
                        "score_min": float(row_scores.min().detach().cpu().item()),
                        "score_max": float(row_scores.max().detach().cpu().item()),
                        "score_std": float(row_scores.std().detach().cpu().item()) if len(row_codes) > 1 else 0.0,
                        "entropy": float(entropy),
                        "top_probs": [
                            (str(row_codes[int(j)]), float(p))
                            for p, j in zip(top_vals.tolist(), top_idx.tolist())
                        ],
                    }
                )
            if shown_rows > 0:
                print(
                    {
                        "summary_rows": int(shown_rows),
                        "avg_target_prob": float(sum(target_probs_summary) / max(1, len(target_probs_summary))),
                        "avg_uniform_prob": float(sum(uniform_probs_summary) / max(1, len(uniform_probs_summary))),
                        "avg_target_prob_delta_vs_uniform": float((sum(target_probs_summary) - sum(uniform_probs_summary)) / max(1, len(target_probs_summary))),
                        "avg_row_ce": float(sum(row_ce_summary) / max(1, len(row_ce_summary))),
                        "avg_random_ce": float(sum(random_ce_summary) / max(1, len(random_ce_summary))),
                        "top1_acc": float(sum(top1_hits) / max(1, len(top1_hits))),
                    }
                )
    finally:
        if was_training:
            model.train()


class CandidateProbeCallback(TrainerCallback):
    def __init__(self, probe_batch, probe_steps=20, probe_rows=5, probe_topk=8):
        self.probe_batch = probe_batch
        self.probe_steps = max(1, int(probe_steps))
        self.probe_rows = max(1, int(probe_rows))
        self.probe_topk = max(1, int(probe_topk))

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if model is None or state.global_step <= 0:
            return
        if int(state.global_step) % int(self.probe_steps) != 0:
            return
        _print_candidate_scoring_probe(
            model=model,
            batch=self.probe_batch,
            max_rows=self.probe_rows,
            topk=self.probe_topk,
        )


def _normalize_step_row(row):
    out = {}
    out["instance_id"] = str(row.get("instance_id", "") or "")
    source_index = row.get("source_index", -1)
    out["source_index"] = int(source_index) if source_index is not None else -1
    out["state_text"] = str(row.get("state_text", ""))
    out["target_text"] = str(row.get("target_text", ""))
    out["feature_schema_version"] = str(row.get("feature_schema_version", "unknown"))
    out["num_jobs"] = int(row.get("num_jobs", 0))
    out["num_machines"] = int(row.get("num_machines", 0))
    out["total_steps"] = int(row.get("total_steps", 0))
    out["step_idx"] = int(row.get("step_idx", 0))

    raw_action_codes = row.get("feasible_action_codes")
    if not isinstance(raw_action_codes, list) or not raw_action_codes:
        raise ValueError("Missing non-empty 'feasible_action_codes' in step row.")
    out["action_codes"] = [str(x) for x in raw_action_codes if str(x).strip()]
    if not out["action_codes"]:
        raise ValueError("Normalized 'feasible_action_codes' became empty in step row.")
    out["feasible_action_codes"] = list(out["action_codes"])
    out["num_feasible_actions"] = int(len(out["action_codes"]))
    return out


def _iter_step_rows(path, row_cap=None, min_feasible_actions=1):
    path = os.path.expanduser(str(path))
    emitted = 0
    row_cap = None if row_cap is None else int(row_cap)
    min_feasible_actions = max(1, int(min_feasible_actions))
    with open(path, "r", encoding="utf-8") as f:
        first_non_ws = None
        while True:
            ch = f.read(1)
            if ch == "":
                break
            if not ch.isspace():
                first_non_ws = ch
                break
        f.seek(0)

        if first_non_ws == "[":
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON input must be a list when using array format.")
            for row in data:
                normalized = _normalize_step_row(row)
                if int(normalized.get("num_feasible_actions", 0)) < min_feasible_actions:
                    continue
                yield normalized
                emitted += 1
                if row_cap is not None and emitted >= row_cap:
                    break
        else:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception as e:
                    raise ValueError(f"Invalid JSONL at line {line_idx}: {e}")
                normalized = _normalize_step_row(row)
                if int(normalized.get("num_feasible_actions", 0)) < min_feasible_actions:
                    continue
                yield normalized
                emitted += 1
                if row_cap is not None and emitted >= row_cap:
                    break


def _try_get_source_index_array(ds):
    if "source_index" not in set(ds.column_names):
        return None
    arrow_table = getattr(ds, "_data", None)
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
    return np.asarray(ds["source_index"], dtype=np.int64)


def _try_get_int_array(ds, column_name):
    if column_name not in set(ds.column_names):
        return None
    arrow_table = getattr(ds, "_data", None)
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
    return np.asarray(ds[column_name], dtype=np.int64)


def _resolve_instance_keys(ds):
    columns = set(ds.column_names)
    if "source_index" in columns:
        source_values = _try_get_source_index_array(ds)
        if source_values is not None:
            return [f"source_{int(x)}" for x in source_values.tolist()]

    instance_values = ds["instance_id"] if "instance_id" in columns else None
    if instance_values is None:
        raise ValueError("Step dataset must contain either 'instance_id' or 'source_index' for instance-level split.")
    keys = []
    for raw_instance_id in instance_values:
        instance_id = "" if raw_instance_id is None else str(raw_instance_id).strip()
        if not instance_id:
            raise ValueError("Empty instance_id encountered and source_index is unavailable for instance-level split.")
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
    ds,
    test_ratio,
    split_seed,
    enable_eval_split,
    split_mode="fixed_per_size",
    eval_instances_per_size=3,
):
    source_values = _try_get_source_index_array(ds)
    if source_values is not None and source_values.size > 0:
        ordered_pos = np.sort(np.unique(source_values, return_index=True)[1])
        ordered_sources = source_values[ordered_pos].tolist()
        ordered_instances = [f"source_{int(x)}" for x in ordered_sources]
        if not enable_eval_split:
            train_indices = np.arange(len(source_values), dtype=np.int64).tolist()
            return train_indices, [], ordered_instances, []

        resolved_split_mode = str(split_mode).lower()
        rng = random.Random(int(split_seed))
        if resolved_split_mode == "fixed_per_size":
            num_jobs_values = _try_get_int_array(ds, "num_jobs")
            num_machines_values = _try_get_int_array(ds, "num_machines")
            if num_jobs_values is None or num_machines_values is None:
                raise ValueError("fixed_per_size eval split requires 'num_jobs' and 'num_machines' columns.")
            ordered_num_jobs = num_jobs_values[ordered_pos].tolist()
            ordered_num_machines = num_machines_values[ordered_pos].tolist()
            size_to_sources = defaultdict(list)
            ordered_sizes = []
            seen_sizes = set()
            for source_idx, n_jobs, n_machines in zip(ordered_sources, ordered_num_jobs, ordered_num_machines):
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
        train_instance_ids = [f"source_{int(x)}" for x in ordered_sources if int(x) not in eval_source_set]
        eval_instance_ids = [f"source_{int(x)}" for x in ordered_sources if int(x) in eval_source_set]
        return train_indices, eval_indices, train_instance_ids, eval_instance_ids

    instance_keys = _resolve_instance_keys(ds)
    ordered_instances = _ordered_unique(instance_keys)
    if not ordered_instances:
        raise ValueError("No instances found for instance-level split.")

    if not enable_eval_split:
        return list(range(len(ds))), [], ordered_instances, []

    shuffled_instances = ordered_instances[:]
    rng = random.Random(int(split_seed))
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


def _filter_by_min_feasible_actions(ds, min_count, split_name, dataset_num_proc):
    min_count = max(1, int(min_count))
    if ds is None or min_count <= 1:
        return ds
    before_count = len(ds)
    ds = ds.filter(
        lambda example: int(example.get("num_feasible_actions", len(example.get("action_codes", [])))) >= min_count,
        num_proc=max(1, int(dataset_num_proc)),
    )
    after_count = len(ds)
    print(f"{split_name} feasible-action filter: min_feasible_actions>={min_count}, kept={after_count:,}/{before_count:,}")
    return ds


def _resolve_output_dir(args):
    base_dir = Path(os.path.expanduser(str(args.output_dir)))
    if args.resume_from_checkpoint or not args.auto_unique_output_dir:
        return base_dir
    suffix = args.output_dir_suffix
    if not suffix:
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    return base_dir.parent / f"{base_dir.name}_{suffix}"


def _build_parser():
    parser = argparse.ArgumentParser(description="Train JSSP action-code candidate-scoring LoRA policy.")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--model_type", type=str, default="llama8b", choices=sorted(MODEL_MAP))
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--load_in_4bit", action="store_true", default=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"])
    parser.add_argument("--use_gradient_checkpointing", type=str, default="unsloth")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--use_rslora", action="store_true", default=True)
    parser.add_argument("--loftq_config", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--save_total_limit", type=int, default=20)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--warmup_steps", type=int, default=20)
    parser.add_argument("--optim", type=str, default="adamw_8bit")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--group_by_length", action="store_true", default=False)
    parser.add_argument("--dataset_num_proc", type=int, default=16)
    parser.add_argument("--report_to", type=str, default="none")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--bf16", action="store_true", default=False)
    parser.add_argument("--load_best_model_at_end", action="store_true", default=False)
    parser.add_argument("--metric_for_best_model", type=str, default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true", default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--remove_unused_columns", action="store_true", default=False)
    parser.add_argument("--action_loss_weight", type=float, default=1.0)
    parser.add_argument("--enable_eval", action="store_true", default=False)
    parser.add_argument("--eval_split_mode", type=str, default="fixed_per_size", choices=["fixed_per_size", "ratio"])
    parser.add_argument("--eval_instances_per_size", type=int, default=3)
    parser.add_argument("--eval_split_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=100000)
    parser.add_argument("--max_eval_samples", type=int, default=20)
    parser.add_argument("--min_train_feasible_actions", type=int, default=2)
    parser.add_argument("--min_eval_feasible_actions", type=int, default=1)
    parser.add_argument("--step_supervision_mode", type=str, default="candidate_scoring")
    parser.add_argument("--dataset_source", type=str, default="local", choices=["local", "hf"])
    parser.add_argument("--step_dataset_path", type=str, default="train_data/jssp_step_train_policy_dispatch.jsonl")
    parser.add_argument("--step_dataset_repo", type=str, default="HYUNJINI/AXXXX_jssp_policy_step_train_dispatch_v1")
    parser.add_argument("--step_dataset_file", type=str, default="train_data/jssp_step_train_policy_dispatch.jsonl")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--policy_head_type", type=str, default="candidate_scoring", choices=["candidate_scoring"])
    parser.add_argument("--candidate_score_token", type=str, default="<CAND_SCORE>")
    parser.add_argument("--candidate_score_head_init_std", type=float, default=0.02)
    parser.add_argument("--candidate_scoring_query_forward_batch_size", type=int, default=16)
    parser.add_argument("--action_code_width", type=int, default=4)
    parser.add_argument("--action_code_cap", type=int, default=9999)
    parser.add_argument("--train_vrp_tsp", action="store_true", default=False)
    parser.add_argument("--train_knapsack", action="store_true", default=False)
    parser.add_argument("--train_binpack", action="store_true", default=False)
    parser.add_argument("--train_jssp", action="store_true", default=True)
    parser.add_argument("--train_fssp", action="store_true", default=False)
    parser.add_argument("--train_lm_head", action="store_true", default=False)
    parser.add_argument("--train_embed_tokens", action="store_true", default=False)
    parser.add_argument("--reinit_action_token_rows_when_frozen", action="store_true", default=True)
    parser.add_argument("--action_token_reinit_scale", type=float, default=0.02)
    parser.add_argument("--action_token_reinit_seed", type=int, default=42)
    parser.add_argument("--action_token_reinit_share_input_output_rows", action="store_true", default=True)
    parser.add_argument("--shuffle_data", action="store_true", default=True)
    parser.add_argument("--shuffle_seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="finetuned_models/fix_jssp_policy_dispatch_llama8b_r128")
    parser.add_argument("--output_accord", action="store_true", default=False)
    parser.add_argument("--output_list_of_lists", action="store_true", default=False)
    parser.add_argument("--auto_unique_output_dir", action="store_true", default=True)
    parser.add_argument("--output_dir_suffix", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--enable_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--log_candidate_probe", action="store_true", default=True)
    parser.add_argument("--candidate_probe_steps", type=int, default=20)
    parser.add_argument("--candidate_probe_rows", type=int, default=5)
    parser.add_argument("--candidate_probe_topk", type=int, default=8)
    return parser


def main():
    args = _build_parser().parse_args()

    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)
        print("HF login ready")

    output_dir = _resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("output_dir:", output_dir)

    step_dataset_path = os.path.expanduser(args.step_dataset_path)
    if args.dataset_source == "hf":
        step_dataset_path = hf_hub_download(
            repo_id=args.step_dataset_repo,
            repo_type="dataset",
            filename=args.step_dataset_file,
            token=args.hf_token or None,
        )
    if not os.path.exists(step_dataset_path):
        raise FileNotFoundError(f"step dataset not found: {step_dataset_path}")
    print("step dataset:", step_dataset_path)

    base_model_name = MODEL_MAP[args.model_type]
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    model, tokenizer = _load_unsloth_model_with_chat_template_fallback(
        model_name=base_model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=dtype,
        local_files_only=False,
    )
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
    print("action token install:", token_install)
    candidate_score_token_install = ensure_candidate_score_token(
        tokenizer=tokenizer,
        model=model,
        token=args.candidate_score_token,
    )
    print("candidate score token install:", candidate_score_token_install)
    geometry_before = summarize_action_token_row_geometry(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
    )
    print("action token row geometry before reinit:", geometry_before)
    reinit_info = maybe_reinitialize_action_token_rows(
        tokenizer=tokenizer,
        model=model,
        train_lm_head=args.train_lm_head,
        train_embed_tokens=args.train_embed_tokens,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
        enabled=args.reinit_action_token_rows_when_frozen,
        scale=args.action_token_reinit_scale,
        seed=args.action_token_reinit_seed,
        share_input_output_rows=args.action_token_reinit_share_input_output_rows,
    )
    print("action token row reinit:", reinit_info)
    geometry_after = summarize_action_token_row_geometry(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
    )
    print("action token row geometry after reinit:", geometry_after)

    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    if args.train_lm_head:
        target_modules.append("lm_head")
    if args.train_embed_tokens:
        target_modules.append("embed_tokens")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        target_modules=target_modules,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.bias,
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        random_state=args.random_state,
        loftq_config=args.loftq_config,
    )
    wrapped_model = CandidateScoringModel(
        backbone_model=model,
        score_token=args.candidate_score_token,
        head_init_std=args.candidate_score_head_init_std,
        query_forward_batch_size=args.candidate_scoring_query_forward_batch_size,
    )
    print(print_number_of_trainable_model_parameters(wrapped_model))

    raw_load_cap = None if args.enable_eval else int(args.max_train_samples) if args.max_train_samples is not None else None
    raw_min_feasible = max(1, int(args.min_train_feasible_actions if not args.enable_eval else 1))
    dataset = Dataset.from_generator(
        _iter_step_rows,
        gen_kwargs={
            "path": step_dataset_path,
            "row_cap": raw_load_cap,
            "min_feasible_actions": raw_min_feasible,
        },
    )
    print(f"raw dataset rows: {len(dataset):,}")

    train_indices, eval_indices, train_instance_ids, eval_instance_ids = _split_indices_by_instance(
        dataset,
        test_ratio=max(0.0, min(0.99, float(args.eval_split_ratio))),
        split_seed=args.seed,
        enable_eval_split=bool(args.enable_eval),
        split_mode=args.eval_split_mode,
        eval_instances_per_size=args.eval_instances_per_size,
    )
    train_dataset = dataset.select(train_indices)
    eval_dataset = dataset.select(eval_indices) if args.enable_eval else None
    print(
        "instance split:",
        f"mode={args.eval_split_mode}",
        f"train_instances={len(train_instance_ids):,}",
        f"eval_instances={len(eval_instance_ids):,}",
        f"overlap={len(set(train_instance_ids) & set(eval_instance_ids))}",
    )
    print(
        "row counts before map:",
        f"train_rows={len(train_dataset):,}",
        f"eval_rows={(len(eval_dataset) if eval_dataset is not None else 0):,}",
    )

    if not args.enable_eval and raw_load_cap is not None:
        print(f"raw train feasible filter enabled (no eval): min_feasible_actions>={raw_min_feasible}")
    else:
        train_dataset = _filter_by_min_feasible_actions(
            train_dataset,
            args.min_train_feasible_actions,
            "train",
            args.dataset_num_proc,
        )
    if eval_dataset is not None:
        eval_dataset = _filter_by_min_feasible_actions(
            eval_dataset,
            args.min_eval_feasible_actions,
            "eval",
            args.dataset_num_proc,
        )

    if args.shuffle_data:
        print(f"shuffle train split enabled (seed={args.shuffle_seed})")
        train_dataset = train_dataset.shuffle(seed=args.shuffle_seed)
    else:
        print("shuffle train split disabled")

    if args.max_train_samples is not None and raw_load_cap is None:
        train_cap = min(int(args.max_train_samples), len(train_dataset))
        print("train sample cap:", train_cap)
        train_dataset = train_dataset.select(range(train_cap))
    if eval_dataset is not None and args.max_eval_samples is not None:
        eval_cap = min(int(args.max_eval_samples), len(eval_dataset))
        print("eval sample cap:", eval_cap)
        eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(eval_cap))

    fmt = partial(
        build_candidate_scoring_example,
        tokenizer=tokenizer,
        max_length=args.max_seq_length,
        score_token=args.candidate_score_token,
        code_width=args.action_code_width,
    )
    print("dataset formatter: candidate_scoring")
    train_dataset = train_dataset.map(fmt, num_proc=max(1, int(args.dataset_num_proc)))
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(fmt, num_proc=max(1, int(args.dataset_num_proc)))

    print("train_dataset length:", len(train_dataset))
    print("eval_dataset length:", len(eval_dataset) if eval_dataset is not None else 0)
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(
            "sample candidate scoring stats:",
            {
                "candidate_count": len(sample.get("candidate_action_codes_in_order", [])),
                "target_candidate_index": sample.get("target_candidate_index"),
                "total_query_tokens": sample.get("prompt_token_count"),
            },
        )

    callbacks = []
    if args.log_candidate_probe and len(train_dataset) > 0:
        probe_rows = [train_dataset[i] for i in range(min(max(1, args.candidate_probe_rows), len(train_dataset)))]
        probe_batch = CandidateScoringCollator(tokenizer)(probe_rows)
        callbacks.append(
            CandidateProbeCallback(
                probe_batch=probe_batch,
                probe_steps=args.candidate_probe_steps,
                probe_rows=args.candidate_probe_rows,
                probe_topk=args.candidate_probe_topk,
            )
        )

    trainer = CandidateScoringTrainer(
        model=wrapped_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CandidateScoringCollator(tokenizer),
        callbacks=callbacks,
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
            output_dir=str(output_dir),
            report_to="wandb" if args.enable_wandb else "none",
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            eval_strategy="steps" if args.enable_eval else "no",
            eval_steps=args.eval_steps,
            per_device_eval_batch_size=args.per_device_eval_batch_size,
            group_by_length=args.group_by_length,
            remove_unused_columns=False,
        ),
    )

    with open(output_dir / "training_hyperparams_args.csv", "w", newline="", encoding="utf-8-sig") as csvfile:
        writer = csv.writer(csvfile)
        for key, value in vars(args).items():
            writer.writerow([key, value])

    if args.resume_from_checkpoint:
        checkpoint_path = os.path.expanduser(args.resume_from_checkpoint)
        if os.path.exists(checkpoint_path):
            print(f"resume_from_checkpoint: {checkpoint_path}")
            trainer.train(resume_from_checkpoint=checkpoint_path)
        else:
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")
    else:
        trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print("final model saved:", final_dir)


if __name__ == "__main__":
    main()
