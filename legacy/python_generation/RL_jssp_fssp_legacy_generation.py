import argparse
import contextlib
import csv
import json
import math
import os
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
from torch.optim import AdamW
from tqdm import trange
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported


@dataclass
class EpisodeBatch:
    log_probs: torch.Tensor
    advantages: torch.Tensor


@dataclass
class TrajectorySample:
    sequence_ids: torch.Tensor
    prompt_len: int
    action_token_id: int
    reward: float
    advantage: float
    old_log_prob: torch.Tensor
    feasible: bool
    makespan: float


@dataclass
class BOPOStepPair:
    winner_sequence_ids: torch.Tensor
    winner_prompt_len: int
    winner_action_token_id: int
    loser_sequence_ids: torch.Tensor
    loser_prompt_len: int
    loser_action_token_id: int
    relative_gap: float
    winner_makespan: float
    loser_makespan: float


@dataclass
class StepActionTrace:
    sequence_ids: torch.Tensor
    prompt_len: int
    action_token_id: int
    chosen_job: int
    step_idx: int

from llm_jssp.utils.solution_generation_english import (
    read_matrix_form_jssp,
)
from llm_jssp.utils.jssp_step_env import StaticJSSPStepEnv
from llm_jssp.utils.jssp_dispatch_env import DispatchJSSPStepEnv
from llm_jssp.utils.jssp_step_masking_hooks import (
    build_step_prefix_allowed_tokens_fn,
    StepActionParseError,
)
from llm_jssp.utils.action_token_utils import (
    ensure_action_special_tokens,
    parse_action_code,
    token_id_to_action_code,
)
from llm_jssp.utils.random_jssp import generate_random_instance
from llm_jssp.utils.step_prompting import (
    build_problem_context_text,
    build_randomized_action_code_map,
    build_step_improvement_prompt,
    build_step_prompt,
    compute_action_transition_features,
)
from llm_jssp.utils.step_prompting_dispatch import (
    build_step_prompt as build_dispatch_step_prompt,
    compute_action_transition_features as compute_dispatch_action_transition_features,
)

# ---------------------------------------------------------------------------
# Global defaults / parsing helpers
# ---------------------------------------------------------------------------


DEFAULT_MODEL_BY_TYPE = {
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "qwen2.5_7b": "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
    "qwen2.5_14b": "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
    "deepseek_8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "qwen25_7b_math": "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
}

HEADER_PATTERN = re.compile(
    r"JSSP\s+with\s+(\d+)\s+Jobs,\s*(\d+)\s+Machines",
    re.IGNORECASE,
)
JOB_HEADER_PATTERN = re.compile(
    r"^\s*Job\s+(\d+)\s+consists\s+of\s+Operations:\s*$",
    re.IGNORECASE,
)
OPERATION_PATTERN = re.compile(
    r"^\s*Operation\s+(\d+):\s*M(\d+),\s*(\d+)\s*$",
    re.IGNORECASE,
)

GRPO_CMAX_PATTERN = re.compile(
    r"cmax\s*:\s*([-+]?\d+(?:\.\d+)?)\s*->\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
GRPO_DELTA_CMAX_PATTERN = re.compile(
    r"delta_cmax\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
GRPO_EST_END_PATTERN = re.compile(
    r"est_end\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
GRPO_PROC_TIME_PATTERN = re.compile(
    r"(?:processing\s*time|proc_time)\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
GRPO_REM_RATIO_PATTERN = re.compile(
    r"rem_work_after_ratio\s*=\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Utility structures
# ---------------------------------------------------------------------------


class ExponentialBaseline:
    """Simple running baseline for REINFORCE style updates."""

    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self.value: Optional[float] = None

    def update(self, reward: float) -> float:
        if self.value is None:
            self.value = reward
        else:
            self.value = self.beta * self.value + (1 - self.beta) * reward
        return self.value


def summarize_trainable_parameters(model) -> Tuple[int, int, float]:
    trainable = 0
    total = 0
    for param in model.parameters():
        n = int(param.numel())
        total += n
        if param.requires_grad:
            trainable += n
    ratio = (float(trainable) / float(total)) if total > 0 else 0.0
    return trainable, total, ratio


def validate_rl_update_mode(model, update_mode: str, model_path: str) -> Tuple[int, int, float]:
    trainable, total, ratio = summarize_trainable_parameters(model)
    print(f"[Info] RL trainable params: {trainable:,} / {total:,} ({ratio * 100:.2f}%)")

    if update_mode == "adapter_only":
        if trainable <= 0:
            raise ValueError(
                "RL update mode is adapter_only, but no trainable parameters were found. "
                f"Check model_path={model_path!r} and adapter loading."
            )
    elif update_mode == "full":
        if trainable <= 0:
            raise ValueError(
                "RL update mode is full, but no trainable parameters were found. "
                f"Check model_path={model_path!r}."
            )
    else:
        raise ValueError(f"Unsupported rl_update_mode={update_mode}")

    return trainable, total, ratio


def resolve_checkpoint_tag(repo_id: str, checkpoint_tag: Optional[str], token: Optional[str] = None) -> Optional[str]:
    if not checkpoint_tag:
        return None
    expanded = os.path.expanduser(str(repo_id))
    if os.path.exists(expanded):
        return None
    api = HfApi(token=token)
    files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
    tag = str(checkpoint_tag)
    if any(path.startswith(f"{tag}/") for path in files):
        return tag
    raise FileNotFoundError(f"checkpoint tag not found in repo {repo_id!r}: {tag!r}")


def resolve_model_source(model_path: str, checkpoint_tag: Optional[str] = None, token: Optional[str] = None) -> str:
    expanded = os.path.expanduser(str(model_path))
    if os.path.exists(expanded):
        return expanded
    resolved_checkpoint = resolve_checkpoint_tag(expanded, checkpoint_tag, token=token)
    if resolved_checkpoint is None:
        return expanded

    snapshot_dir = snapshot_download(
        repo_id=expanded,
        repo_type="model",
        allow_patterns=[f"{resolved_checkpoint}/*"],
        token=token,
    )
    local_checkpoint_dir = os.path.join(snapshot_dir, resolved_checkpoint)
    if os.path.exists(local_checkpoint_dir):
        return local_checkpoint_dir

    for probe_file in ("adapter_config.json", "adapter_model.safetensors"):
        try:
            local_file = hf_hub_download(
                repo_id=expanded,
                repo_type="model",
                filename=f"{resolved_checkpoint}/{probe_file}",
                token=token,
            )
            return os.path.dirname(local_file)
        except Exception:
            pass

    raise FileNotFoundError(
        f"checkpoint folder not found after download: {local_checkpoint_dir}"
    )


def is_adapter_source(path_or_repo: Optional[str], token: Optional[str] = None) -> bool:
    if not path_or_repo:
        return False
    if os.path.exists(path_or_repo):
        return os.path.exists(os.path.join(path_or_repo, "adapter_config.json"))
    try:
        api = HfApi(token=token)
        files = set(api.list_repo_files(repo_id=path_or_repo, repo_type="model"))
    except Exception:
        return False
    return "adapter_config.json" in files


def maybe_load_policy_adapter(model, policy_model_path: Optional[str], token: Optional[str] = None) -> bool:
    if not policy_model_path:
        return False
    if not hasattr(model, "load_adapter"):
        raise RuntimeError(
            "Policy adapter path was provided, but the loaded model does not support load_adapter()."
        )
    if os.path.exists(policy_model_path):
        model.load_adapter(
            policy_model_path,
            adapter_name="default",
            adapter_kwargs={"local_files_only": True},
        )
    else:
        model.load_adapter(policy_model_path, adapter_name="default", token=token)
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    return True


# ---------------------------------------------------------------------------
# Heuristic baseline (MWKR)
# ---------------------------------------------------------------------------


def mwkr_schedule(inst_for_ortools: List[List[List[int]]]) -> Tuple[List[Dict], float]:
    """Most Work Remaining heuristic schedule."""

    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0])

    job_next = [0] * num_jobs
    job_time = [0.0] * num_jobs
    machine_time = [0.0] * num_machines
    total_remaining = [
        sum(op[1] for op in job_ops) for job_ops in inst_for_ortools
    ]

    schedule = []

    while any(idx < len(inst_for_ortools[j]) for j, idx in enumerate(job_next)):
        # Jobs ready to schedule are those with remaining operations
        ready_jobs = [
            j for j in range(num_jobs) if job_next[j] < len(inst_for_ortools[j])
        ]
        if not ready_jobs:
            break

        # Choose job with maximum remaining work (MWKR)
        job = max(ready_jobs, key=lambda j: total_remaining[j])
        op_idx = job_next[job]
        machine, duration = inst_for_ortools[job][op_idx]

        start = max(job_time[job], machine_time[machine])
        end = start + duration

        schedule.append(
            {
                "Job": job,
                "Operation": op_idx,
                "Machine": machine,
                "Start Time": start,
                "Duration": duration,
                "End Time": end,
            }
        )

        job_next[job] += 1
        job_time[job] = end
        machine_time[machine] = end
        total_remaining[job] -= duration

    makespan = max(job_time) if schedule else float("inf")
    return schedule, makespan


def _count_total_ops(inst_for_ortools: List[List[List[int]]]) -> int:
    return int(sum(len(job_ops) for job_ops in inst_for_ortools))


def compute_episode_reward(
    makespan: float,
    feasible: bool,
    inst_for_ortools: List[List[List[int]]],
    invalid_makespan_penalty: float,
    reward_mode: str = "raw_neg_makespan",
    heuristic_makespan: float | None = None,
) -> float:
    if not feasible or not math.isfinite(float(makespan)):
        if reward_mode == "neg_makespan_per_op":
            total_ops = max(_count_total_ops(inst_for_ortools), 1)
            return -float(invalid_makespan_penalty) / float(total_ops)
        if reward_mode == "mwkr_relative":
            return -1.0
        return -float(invalid_makespan_penalty)

    makespan = float(makespan)
    if reward_mode == "raw_neg_makespan":
        return -makespan
    if reward_mode == "neg_makespan_per_op":
        total_ops = max(_count_total_ops(inst_for_ortools), 1)
        return -(makespan / float(total_ops))
    if reward_mode == "mwkr_relative":
        baseline_ms = float(heuristic_makespan) if heuristic_makespan is not None else float(mwkr_schedule(inst_for_ortools)[1])
        if not math.isfinite(baseline_ms) or baseline_ms <= 0:
            return -makespan
        return (baseline_ms - makespan) / baseline_ms
    raise ValueError(f"Unsupported reward_mode={reward_mode}")


def force_safe_train_attention_backend(model) -> Dict[str, Optional[float]]:
    """
    Best-effort switch to an eager / math-only attention backend for legacy custom RL
    updates. Official Unsloth GRPO uses its own trainer path and does not need this.
    """
    info = {
        "changed_configs": 0,
        "flash_sdp": None,
        "mem_efficient_sdp": None,
        "math_sdp": None,
    }
    config = getattr(model, "config", None)
    if config is not None:
        current_impl = getattr(config, "_attn_implementation", None)
        if current_impl != "eager":
            try:
                setattr(config, "_attn_implementation", "eager")
                info["changed_configs"] += 1
            except Exception:
                pass
        if hasattr(config, "attn_implementation"):
            current_public = getattr(config, "attn_implementation", None)
            if current_public != "eager":
                try:
                    setattr(config, "attn_implementation", "eager")
                    info["changed_configs"] += 1
                except Exception:
                    pass

    if torch.cuda.is_available() and hasattr(torch.backends, "cuda"):
        cuda_backends = torch.backends.cuda
        try:
            if hasattr(cuda_backends, "enable_flash_sdp"):
                cuda_backends.enable_flash_sdp(False)
            if hasattr(cuda_backends, "enable_mem_efficient_sdp"):
                cuda_backends.enable_mem_efficient_sdp(False)
            if hasattr(cuda_backends, "enable_math_sdp"):
                cuda_backends.enable_math_sdp(True)
        except Exception:
            pass

        if hasattr(cuda_backends, "flash_sdp_enabled"):
            info["flash_sdp"] = bool(cuda_backends.flash_sdp_enabled())
        if hasattr(cuda_backends, "mem_efficient_sdp_enabled"):
            info["mem_efficient_sdp"] = bool(cuda_backends.mem_efficient_sdp_enabled())
        if hasattr(cuda_backends, "math_sdp_enabled"):
            info["math_sdp"] = bool(cuda_backends.math_sdp_enabled())

    try:
        import unsloth.models.llama as unsloth_llama
        import unsloth.utils.attention_dispatch as unsloth_attention_dispatch

        def _force_sdpa_backend(*args, **kwargs):
            return unsloth_attention_dispatch.SDPA

        if hasattr(unsloth_attention_dispatch, "HAS_XFORMERS"):
            unsloth_attention_dispatch.HAS_XFORMERS = False
            info["unsloth_has_xformers"] = False
        if hasattr(unsloth_attention_dispatch, "xformers_attention"):
            unsloth_attention_dispatch.xformers_attention = None
            info["unsloth_xformers_attention"] = "disabled"
        if hasattr(unsloth_attention_dispatch, "select_attention_backend"):
            unsloth_attention_dispatch.select_attention_backend = _force_sdpa_backend
            info["unsloth_backend_selector"] = "sdpa"
        if hasattr(unsloth_llama, "select_attention_backend"):
            unsloth_llama.select_attention_backend = _force_sdpa_backend
            info["unsloth_llama_selector"] = "sdpa"
        if hasattr(unsloth_llama, "HAS_XFORMERS"):
            unsloth_llama.HAS_XFORMERS = False
        if hasattr(unsloth_llama, "xformers_attention"):
            unsloth_llama.xformers_attention = None
    except Exception as exc:
        info["unsloth_patch_note"] = str(exc)
    return info


def parse_prompt_jobs_first(prompt_text: str, strict: bool = True) -> List[List[List[int]]]:
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string.")

    header_match = HEADER_PATTERN.search(prompt_text)
    expected_jobs = int(header_match.group(1)) if header_match else None
    expected_machines = int(header_match.group(2)) if header_match else None

    jobs: Dict[int, List[List[int]]] = {}
    current_job: Optional[int] = None
    for raw_line in prompt_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        job_match = JOB_HEADER_PATTERN.match(line)
        if job_match:
            current_job = int(job_match.group(1))
            jobs.setdefault(current_job, [])
            continue

        op_match = OPERATION_PATTERN.match(line)
        if op_match:
            if current_job is None:
                raise ValueError(f"Operation line appears before job header: {raw_line!r}")
            op_idx = int(op_match.group(1))
            machine = int(op_match.group(2))
            duration = int(op_match.group(3))
            expected_op_idx = len(jobs[current_job])
            if strict and op_idx != expected_op_idx:
                raise ValueError(
                    f"Operation index mismatch in Job {current_job}: expected {expected_op_idx}, got {op_idx}."
                )
            jobs[current_job].append([machine, duration])

    if not jobs:
        raise ValueError("Failed to parse any jobs from prompt_jobs_first content.")

    ordered_job_ids = sorted(jobs)
    if strict:
        for expected_job_id, actual_job_id in enumerate(ordered_job_ids):
            if expected_job_id != actual_job_id:
                raise ValueError(
                    f"Job ids are not contiguous from 0. expected {expected_job_id}, got {actual_job_id}."
                )

    inst_for_ortools = [jobs[job_id] for job_id in ordered_job_ids]
    if expected_jobs is not None and strict and len(inst_for_ortools) != expected_jobs:
        raise ValueError(
            f"Header declared {expected_jobs} jobs, parsed {len(inst_for_ortools)} jobs."
        )
    if expected_machines is not None and strict:
        for job_id, ops in enumerate(inst_for_ortools):
            if len(ops) != expected_machines:
                raise ValueError(
                    f"Header declared {expected_machines} machines, but Job {job_id} has {len(ops)} operations."
                )
    return inst_for_ortools


def extract_problem_instance_from_example(example: Dict) -> List[List[List[int]]]:
    if not isinstance(example, dict):
        raise ValueError("Dataset example must be a mapping.")
    if "inst_for_ortools" in example and example["inst_for_ortools"] is not None:
        return example["inst_for_ortools"]
    prompt_jobs_first = example.get("prompt_jobs_first")
    if isinstance(prompt_jobs_first, str) and prompt_jobs_first.strip():
        return parse_prompt_jobs_first(prompt_jobs_first)
    matrix_content = example.get("matrix")
    if isinstance(matrix_content, str) and matrix_content.strip():
        _, _, inst, _ = read_matrix_form_jssp(matrix_content)
        return inst
    raise ValueError("Dataset example is missing both prompt_jobs_first and matrix fields.")


def _completion_to_text(completion) -> str:
    if completion is None:
        return ""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", "")))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def _normalize_action_code_list(action_codes, code_width: int = 4) -> List[str]:
    out: List[str] = []
    for code in list(action_codes or []):
        text = str(code).strip()
        parsed = parse_action_code(text, code_width=code_width)
        out.append(str(parsed or text))
    return out


def _extract_target_action_code(target_text: str, code_width: int = 4) -> Optional[str]:
    parsed = parse_action_code(str(target_text or ""), code_width=code_width)
    if parsed is not None:
        return str(parsed)
    stripped = str(target_text or "").strip()
    return stripped if stripped else None


def extract_proxy_action_metrics_from_state_text(
    state_text: str,
    code_width: int = 4,
) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    for raw_line in str(state_text or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("<"):
            continue
        code = parse_action_code(line, code_width=code_width)
        if code is None:
            continue
        cmax_match = GRPO_CMAX_PATTERN.search(line)
        delta_match = GRPO_DELTA_CMAX_PATTERN.search(line)
        est_end_match = GRPO_EST_END_PATTERN.search(line)
        proc_match = GRPO_PROC_TIME_PATTERN.search(line)
        rem_ratio_match = GRPO_REM_RATIO_PATTERN.search(line)
        metrics[str(code)] = {
            "action_code": str(code),
            "cmax_after": float(int(cmax_match.group(2))) if cmax_match else float(10**12),
            "delta_cmax": float(int(delta_match.group(1))) if delta_match else float(10**12),
            "est_end": float(int(est_end_match.group(1))) if est_end_match else float(10**12),
            "proc_time": float(int(proc_match.group(1))) if proc_match else float(10**12),
            "rem_work_after_ratio": float(rem_ratio_match.group(1)) if rem_ratio_match else 1e9,
        }
    ordered = sorted(
        metrics.values(),
        key=lambda x: (
            float(x["cmax_after"]),
            float(x["delta_cmax"]),
            float(x["est_end"]),
            float(x["proc_time"]),
            str(x["action_code"]),
        ),
    )
    if not ordered:
        return metrics
    best_after = float(ordered[0]["cmax_after"])
    best_delta = float(ordered[0]["delta_cmax"])
    max_after_gap = max(float(item["cmax_after"]) - best_after for item in ordered)
    max_delta_gap = max(float(item["delta_cmax"]) - best_delta for item in ordered)
    denom_rank = max(len(ordered) - 1, 1)
    for rank, item in enumerate(ordered):
        after_gap = max(0.0, float(item["cmax_after"]) - best_after)
        delta_gap = max(0.0, float(item["delta_cmax"]) - best_delta)
        item["rank"] = float(rank)
        item["rank_score"] = 1.0 if len(ordered) == 1 else 1.0 - float(rank) / float(denom_rank)
        item["cmax_gap_score"] = 1.0 - (after_gap / max(max_after_gap, 1.0))
        item["delta_gap_score"] = 1.0 - (delta_gap / max(max_delta_gap, 1.0))
    return {str(item["action_code"]): dict(item) for item in ordered}


def build_unsloth_grpo_prompt(state_text: str):
    return [
        {
            "role": "system",
            "content": (
                "You are solving JSSP step-by-step. "
                "Output exactly one feasible action code like <A1234>. "
                "Do not explain your answer."
            ),
        },
        {
            "role": "user",
            "content": str(state_text),
        },
    ]


def resolve_grpo_step_dataset_path(args, hf_token: str = "") -> str:
    if getattr(args, "grpo_dataset_path", None):
        return os.path.expanduser(str(args.grpo_dataset_path))
    source = str(getattr(args, "grpo_dataset_source", "hf")).strip().lower()
    if source == "local":
        if str(getattr(args, "env_mode", "serial")).strip().lower() == "dispatch":
            return os.path.expanduser(str(args.grpo_step_dataset_local_path_dispatch))
        return os.path.expanduser(str(args.grpo_step_dataset_local_path))
    if source != "hf":
        raise ValueError("grpo_dataset_source must be 'hf' or 'local'.")
    if str(getattr(args, "env_mode", "serial")).strip().lower() == "dispatch":
        repo_id = str(args.grpo_step_dataset_hf_repo_dispatch)
        filename = str(args.grpo_step_dataset_hf_file_dispatch)
    else:
        repo_id = str(args.grpo_step_dataset_hf_repo)
        filename = str(args.grpo_step_dataset_hf_file)
    return hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, token=hf_token or None)


def build_unsloth_grpo_step_dataset(args, hf_token: str = ""):
    dataset_path = resolve_grpo_step_dataset_path(args, hf_token=hf_token)
    ds = load_dataset("json", data_files=dataset_path)["train"]
    raw_rows = len(ds)
    exact_jobs = getattr(args, "rl_num_jobs", None)
    exact_machines = getattr(args, "rl_num_machines", None)
    min_jobs = getattr(args, "min_rl_num_jobs", None)
    min_machines = getattr(args, "min_rl_num_machines", None)
    min_feasible = int(getattr(args, "grpo_min_feasible_actions", 2) or 2)
    code_width = int(getattr(args, "action_code_width", 4) or 4)
    raw_size_counter: Counter = Counter()
    filtered_size_counter: Counter = Counter()
    records = []
    for row in ds:
        n_jobs = int(row.get("num_jobs", 0) or 0)
        n_machines = int(row.get("num_machines", 0) or 0)
        num_feasible = int(row.get("num_feasible_actions", len(row.get("feasible_action_codes", []) or [])) or 0)
        raw_size_counter[(n_jobs, n_machines)] += 1
        keep = True
        if exact_jobs is not None and n_jobs != int(exact_jobs):
            keep = False
        if exact_machines is not None and n_machines != int(exact_machines):
            keep = False
        if min_jobs is not None and n_jobs < int(min_jobs):
            keep = False
        if min_machines is not None and n_machines < int(min_machines):
            keep = False
        if num_feasible < min_feasible:
            keep = False
        if not keep:
            continue

        state_text = str(row.get("state_text", "")).strip()
        feasible_action_codes = _normalize_action_code_list(row.get("feasible_action_codes", []) or [], code_width=code_width)
        if not state_text or not feasible_action_codes:
            continue
        proxy_metrics = extract_proxy_action_metrics_from_state_text(state_text, code_width=code_width)
        target_action_code = _extract_target_action_code(row.get("target_text", ""), code_width=code_width)
        records.append(
            {
                "prompt": build_unsloth_grpo_prompt(state_text),
                "state_text": state_text,
                "feasible_action_codes": list(feasible_action_codes),
                "proxy_metrics_json": json.dumps(proxy_metrics, ensure_ascii=False),
                "target_action_code": target_action_code,
                "instance_id": str(row.get("instance_id", "")),
                "step_idx": int(row.get("step_idx", 0) or 0),
                "num_jobs": n_jobs,
                "num_machines": n_machines,
                "num_feasible_actions": num_feasible,
            }
        )
        filtered_size_counter[(n_jobs, n_machines)] += 1

    shuffle_seed = getattr(args, "grpo_shuffle_seed", None)
    if shuffle_seed is not None:
        rng_local = random.Random(int(shuffle_seed))
        rng_local.shuffle(records)

    max_rows = getattr(args, "grpo_max_dataset_rows", None)
    if max_rows is not None:
        max_rows = int(max_rows)
        if max_rows > 0:
            records = records[:max_rows]

    print("[Info] official grpo dataset path:", dataset_path)
    print("[Info] official grpo raw rows:", raw_rows)
    print("[Info] official grpo filtered rows:", len(records))
    print("[Info] official grpo size_dist_top_raw:", raw_size_counter.most_common(10))
    print("[Info] official grpo size_dist_top_filtered:", filtered_size_counter.most_common(10))
    print(
        "[Info] official grpo filter config:",
        {
            "rl_num_jobs": exact_jobs,
            "rl_num_machines": exact_machines,
            "min_rl_num_jobs": min_jobs,
            "min_rl_num_machines": min_machines,
            "grpo_min_feasible_actions": min_feasible,
            "grpo_max_dataset_rows": max_rows,
        },
    )
    if not records:
        raise ValueError("No GRPO step rows remain after filtering.")
    return Dataset.from_list(records), dataset_path


def build_unsloth_grpo_reward_functions(args):
    code_width = int(getattr(args, "action_code_width", 4) or 4)
    valid_weight = float(getattr(args, "grpo_reward_valid_weight", 1.0))
    proxy_weight = float(getattr(args, "grpo_reward_proxy_weight", 1.0))
    teacher_weight = float(getattr(args, "grpo_reward_teacher_weight", 0.25))

    def reward_valid_action(prompts=None, completions=None, feasible_action_codes=None, **kwargs):
        feasible_lists = feasible_action_codes or kwargs.get("feasible_action_codes") or []
        rewards = []
        for completion, feasible in zip(completions or [], feasible_lists):
            code = parse_action_code(_completion_to_text(completion), code_width=code_width)
            feasible_set = set(_normalize_action_code_list(feasible, code_width=code_width))
            if code is None:
                rewards.append(-1.0 * valid_weight)
            elif str(code) in feasible_set:
                rewards.append(1.0 * valid_weight)
            else:
                rewards.append(-0.5 * valid_weight)
        return rewards

    def reward_proxy_quality(prompts=None, completions=None, proxy_metrics_json=None, feasible_action_codes=None, **kwargs):
        metric_rows = proxy_metrics_json or kwargs.get("proxy_metrics_json") or []
        rewards = []
        for completion, metrics_blob in zip(completions or [], metric_rows):
            code = parse_action_code(_completion_to_text(completion), code_width=code_width)
            try:
                metrics = json.loads(metrics_blob or "{}")
            except Exception:
                metrics = {}
            if code is None or str(code) not in metrics:
                rewards.append(-1.0 * proxy_weight)
                continue
            chosen = metrics[str(code)]
            score = (
                0.60 * float(chosen.get("cmax_gap_score", 0.0))
                + 0.25 * float(chosen.get("delta_gap_score", 0.0))
                + 0.15 * float(chosen.get("rank_score", 0.0))
            )
            rewards.append(float(score) * proxy_weight)
        return rewards

    def reward_teacher_action(prompts=None, completions=None, target_action_code=None, **kwargs):
        targets = target_action_code or kwargs.get("target_action_code") or []
        rewards = []
        for completion, target in zip(completions or [], targets):
            code = parse_action_code(_completion_to_text(completion), code_width=code_width)
            rewards.append(float(teacher_weight) if code is not None and target is not None and str(code) == str(target) else 0.0)
        return rewards

    reward_valid_action.__name__ = "reward_valid_action"
    reward_proxy_quality.__name__ = "reward_proxy_quality"
    reward_teacher_action.__name__ = "reward_teacher_action"
    return [reward_valid_action, reward_proxy_quality, reward_teacher_action]


def write_history_csv(path_obj: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path_obj.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_step_chat_prompt(tokenizer, state_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert JSSP scheduler. "
                "Primary objective: minimize final makespan (Cmax). "
                "Choose exactly one feasible action token for this step. "
                "Output exactly one token such as <A3812> and nothing else."
            ),
        },
        {"role": "user", "content": state_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_step_improvement_chat_prompt(tokenizer, improvement_prompt_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are refining one JSSP step action. "
                "Primary objective: minimize final makespan (Cmax). "
                "Return exactly one feasible action token such as <A3812>."
            ),
        },
        {"role": "user", "content": improvement_prompt_text},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def generate_step_action(
    model,
    tokenizer,
    prompt_text: str,
    feasible_action_codes: List[str],
    device: torch.device,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    use_masking: bool = True,
    action_code_width: int = 4,
    action_code_cap: int = 9999,
) -> Tuple[str, torch.Tensor, int, str]:
    if not feasible_action_codes:
        raise StepActionParseError("No feasible action codes available at this step.")

    prompt_inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
    prompt_len = int(prompt_inputs["input_ids"].size(1))

    was_training = model.training
    model.eval()
    try:
        generation_kwargs = dict(
            **prompt_inputs,
            max_new_tokens=1 if use_masking else int(max_new_tokens),
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            top_k=int(top_k),
            return_dict_in_generate=False,
        )

        if use_masking:
            generation_kwargs["prefix_allowed_tokens_fn"] = build_step_prefix_allowed_tokens_fn(
                tokenizer=tokenizer,
                feasible_action_codes_provider=list(feasible_action_codes),
                code_width=action_code_width,
                code_cap=action_code_cap,
            )

        with torch.no_grad():
            generated = model.generate(**generation_kwargs)

        sequence_ids = generated[0].detach().cpu()
        new_ids = sequence_ids[prompt_len:]
        if new_ids.numel() <= 0:
            raise StepActionParseError("No action token was generated.")

        chosen_action_code = token_id_to_action_code(
            tokenizer,
            int(new_ids[0].item()),
            code_width=action_code_width,
        )
        if chosen_action_code is None:
            generated_text = tokenizer.decode(new_ids, skip_special_tokens=False).strip()
            raise StepActionParseError(
                f"Failed to map generated token to action token. output={generated_text!r}"
            )
        if str(chosen_action_code) not in feasible_action_codes:
            raise StepActionParseError(
                f"Generated action token is not feasible. parsed={chosen_action_code}, feasible={list(feasible_action_codes)}"
            )
        return str(chosen_action_code), sequence_ids, prompt_len, str(chosen_action_code)
    finally:
        if was_training:
            model.train()


def _normalize_env_mode(env_mode: str) -> str:
    resolved = str(env_mode).lower()
    if resolved not in {"serial", "dispatch"}:
        raise ValueError(f"Unsupported env_mode={env_mode}")
    return resolved


def _make_step_env(inst_for_ortools, env_mode: str):
    if _normalize_env_mode(env_mode) == "dispatch":
        return DispatchJSSPStepEnv(inst_for_ortools)
    return StaticJSSPStepEnv(inst_for_ortools)


def _estimate_action_effects(state_json, action_code_to_job, env_mode: str):
    if _normalize_env_mode(env_mode) == "dispatch":
        return compute_dispatch_action_transition_features(state_json, action_code_to_job)
    return compute_action_transition_features(state_json, action_code_to_job)


def _build_state_text(
    state_json,
    feasible_jobs,
    step_idx,
    total_steps,
    problem_context_text,
    action_code_to_job,
    env_mode: str,
):
    if _normalize_env_mode(env_mode) == "dispatch":
        return build_dispatch_step_prompt(
            state_json=state_json,
            feasible_jobs=feasible_jobs,
            step_idx=step_idx,
            total_steps=total_steps,
            problem_context_text=problem_context_text,
            action_code_to_job=action_code_to_job,
        )
    return build_step_prompt(
        state_json=state_json,
        feasible_jobs=feasible_jobs,
        step_idx=step_idx,
        total_steps=total_steps,
        problem_context_text=problem_context_text,
        action_code_to_job=action_code_to_job,
    )


def _best_alternative_option(step):
    chosen_code = str(step.get("chosen_action_code"))
    alternatives = [
        opt
        for opt in step.get("all_options", [])
        if str(opt.get("action_code")) != chosen_code
    ]
    if not alternatives:
        return None
    return min(
        alternatives,
        key=lambda x: (
            int(x.get("estimated_makespan_after", 10**12)),
            int(x.get("estimated_start", 10**12)),
            int(x.get("proc_time", 10**12)),
        ),
    )


def _critical_step_score(step):
    best_alt = _best_alternative_option(step)
    if best_alt is None:
        return None
    chosen_ms = int(step.get("chosen_estimated_makespan_after", step.get("makespan_after", 0)))
    best_alt_ms = int(best_alt.get("estimated_makespan_after", chosen_ms))
    immediate_gap = int(chosen_ms - best_alt_ms)
    makespan_jump = int(step.get("makespan_after", 0) - step.get("makespan_before", 0))
    chosen_start = int(step.get("chosen_start_time", 0))
    best_alt_start = int(best_alt.get("estimated_start", chosen_start))
    start_delay = int(chosen_start - best_alt_start)
    if immediate_gap <= 0 and makespan_jump <= 0 and start_delay <= 0:
        return None
    return (immediate_gap, makespan_jump, start_delay, int(step.get("step_idx", -1)))


def _select_top_critical_steps(step_records, top_k: int = 3):
    scored = []
    for step in step_records:
        score = _critical_step_score(step)
        if score is None:
            continue
        scored.append((score, step))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [step for _, step in scored[: max(1, int(top_k))]]


def _select_critical_step(step_records):
    top = _select_top_critical_steps(step_records, top_k=1)
    return top[0] if top else None


def _build_step_diagnostics(step):
    chosen_code = str(step.get("chosen_action_code"))
    chosen_job = int(step.get("chosen_job", -1))
    chosen_ms = int(step.get("chosen_estimated_makespan_after", step.get("makespan_after", 0)))
    chosen_start = int(step.get("chosen_start_time", 0))
    alternatives = [
        opt
        for opt in step.get("all_options", [])
        if str(opt.get("action_code")) != chosen_code
    ]
    alternatives = sorted(
        alternatives,
        key=lambda x: (
            int(x.get("estimated_makespan_after", 10**12)),
            int(x.get("estimated_start", 10**12)),
        ),
    )
    lines = [
        (
            f"Chosen={chosen_code}/Job{chosen_job}, est_Cmax_after={chosen_ms}, "
            f"start={chosen_start}, machine=M{int(step.get('machine_id', -1))}"
        )
    ]
    for rank, opt in enumerate(alternatives[:3], start=1):
        lines.append(
            (
                f"Alt{rank}={opt['action_code']}/Job{int(opt['job_id'])}, "
                f"est_Cmax_after={int(opt['estimated_makespan_after'])}, "
                f"est_start={int(opt['estimated_start'])}, "
                f"proc={int(opt['proc_time'])}"
            )
        )
    return "\n".join(lines)


def _build_reflection_memory(current_makespan: float, critical_steps):
    lines = [
        f"Current episode makespan={float(current_makespan):.1f}.",
        "Reflection rules (avoid bottleneck choices, prefer lower estimated Cmax/start):",
    ]
    for rank, step in enumerate(critical_steps, start=1):
        best_alt = _best_alternative_option(step)
        if best_alt is None:
            continue
        chosen_code = str(step.get("chosen_action_code"))
        chosen_job = int(step.get("chosen_job", -1))
        chosen_ms = int(step.get("chosen_estimated_makespan_after", step.get("makespan_after", 0)))
        chosen_start = int(step.get("chosen_start_time", 0))
        alt_code = str(best_alt.get("action_code"))
        alt_job = int(best_alt.get("job_id", -1))
        alt_ms = int(best_alt.get("estimated_makespan_after", chosen_ms))
        alt_start = int(best_alt.get("estimated_start", chosen_start))
        lines.append(
            (
                f"Rule {rank}: step {int(step.get('step_idx', -1))}, avoid {chosen_code}/Job{chosen_job} "
                f"(est_Cmax={chosen_ms}, start={chosen_start}); "
                f"prefer {alt_code}/Job{alt_job} (est_Cmax={alt_ms}, start={alt_start})."
            )
        )
    return "\n".join(lines)


def _print_step_trace(step_records):
    for step in step_records:
        print(
            f"[step {int(step['step_idx']):03d}] ms {int(step['makespan_before'])}->{int(step['makespan_after'])} "
            f"chosen={step['chosen_action_code']}->Job{step['chosen_job']} "
            f"(M{step['machine_id']},t={step['chosen_proc_time']},"
            f"start={step['chosen_start_time']},end={step['chosen_end_time']})"
        )
        not_chosen = step.get("not_chosen_options", [])
        if not_chosen:
            parts = []
            for opt in not_chosen:
                parts.append(
                    f"{opt['action_code']}->Job{opt['job_id']}"
                    f"(M{opt['machine_id']},t={opt['proc_time']},ms={opt['estimated_makespan_after']},"
                    f"dC={opt['delta_cmax']})"
                )
            print("  not_chosen:", "; ".join(parts))


def rollout_step_episode(
    model,
    tokenizer,
    inst_for_ortools: List[List[List[int]]],
    device: torch.device,
    step_action_max_new_tokens: int,
    env_mode: str,
    temperature: float = 1.0,
    top_p: float = 0.95,
    top_k: int = 50,
    use_masking: bool = True,
    include_problem_context: bool = True,
    enable_step_improvement: bool = False,
    step_reflection_passes: int = 1,
    step_reflection_topk: int = 3,
    action_code_width: int = 4,
    action_code_seed: int = 42,
    action_code_cap: int = 9999,
    print_step_trace: bool = False,
) -> Tuple[float, bool, List[StepActionTrace]]:
    problem_context_text = (
        build_problem_context_text(inst_for_ortools) if include_problem_context else None
    )

    def _rollout_once(code_seed: int, guidance_by_step=None, reflection_memory_text: str | None = None):
        env = _make_step_env(inst_for_ortools, env_mode)
        env.reset()
        traces: List[StepActionTrace] = []
        step_records = []
        action_rng = random.Random(int(code_seed))
        guidance_map = guidance_by_step or {}

        try:
            while not env.is_done():
                state = env.get_state_json()
                feasible_jobs = [int(j) for j in state["feasible_jobs"]]
                step_idx = int(state["step_idx"])
                action_code_to_job = build_randomized_action_code_map(
                    feasible_jobs=feasible_jobs,
                    rng=action_rng,
                    code_width=action_code_width,
                    code_start=1,
                    code_cap=action_code_cap,
                )
                makespan_before, action_effects = _estimate_action_effects(
                    state_json=state,
                    action_code_to_job=action_code_to_job,
                    env_mode=env_mode,
                )
                effect_by_code = {x["action_code"]: x for x in action_effects}
                feasible_action_codes = list(action_code_to_job.keys())
                state_text = _build_state_text(
                    state_json=state,
                    feasible_jobs=feasible_jobs,
                    step_idx=step_idx,
                    total_steps=int(state["total_steps"]),
                    problem_context_text=problem_context_text,
                    action_code_to_job=action_code_to_job,
                    env_mode=env_mode,
                )
                if reflection_memory_text:
                    state_text = (
                        f"{state_text}\n"
                        f"Episode reflection memory:\n{reflection_memory_text}\n"
                        "Apply these reflection rules while selecting this step action."
                    )
                if step_idx in guidance_map:
                    step_guidance = dict(guidance_map[step_idx])
                    preferred_job = int(step_guidance.get("preferred_job", -1))
                    avoid_jobs = [
                        int(x) for x in step_guidance.get("avoid_jobs", [])
                        if int(x) >= 0
                    ]
                    reason_text = str(step_guidance.get("reason", "")).strip()
                    guide_lines = [
                        "Post-episode guidance: This step was identified as a Cmax bottleneck.",
                    ]
                    if preferred_job >= 0:
                        guide_lines.append(f"If feasible, prefer Job {preferred_job}.")
                    if avoid_jobs:
                        guide_lines.append(
                            "Avoid these jobs if strong alternatives exist: "
                            + ", ".join(f"Job {j}" for j in avoid_jobs)
                        )
                    if reason_text:
                        guide_lines.append(f"Why: {reason_text}")
                    state_text = f"{state_text}\n" + "\n".join(guide_lines)

                base_prompt = _build_step_chat_prompt(tokenizer, state_text)
                chosen_action_code, sequence_ids, prompt_len, _ = generate_step_action(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=base_prompt,
                    feasible_action_codes=feasible_action_codes,
                    device=device,
                    max_new_tokens=step_action_max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    use_masking=use_masking,
                    action_code_width=action_code_width,
                    action_code_cap=action_code_cap,
                )
                chosen_job = int(action_code_to_job[chosen_action_code])
                chosen_effect = effect_by_code.get(chosen_action_code)
                if chosen_effect is None:
                    raise RuntimeError(
                        "Internal mismatch: chosen action code is missing in step option effects. "
                        f"chosen_action_code={chosen_action_code}, available={list(effect_by_code.keys())}"
                    )

                _, _, _, info = env.step(chosen_job)
                traces.append(
                    StepActionTrace(
                        sequence_ids=sequence_ids,
                        prompt_len=prompt_len,
                        action_token_id=int(sequence_ids[prompt_len].item()),
                        chosen_job=chosen_job,
                        step_idx=step_idx,
                    )
                )
                step_records.append(
                    {
                        "step_idx": int(step_idx),
                        "state_text": state_text,
                        "feasible_action_codes": feasible_action_codes,
                        "action_code_to_job": action_code_to_job,
                        "chosen_action_code": str(chosen_action_code),
                        "chosen_job": int(chosen_job),
                        "machine_id": int(info["machine_id"]),
                        "chosen_start_time": int(info["start_time"]),
                        "chosen_end_time": int(info["end_time"]),
                        "chosen_proc_time": int(info["duration"]),
                        "makespan_before": int(makespan_before),
                        "makespan_after": int(info["makespan_so_far"]),
                        "chosen_estimated_makespan_after": int(chosen_effect["estimated_makespan_after"]),
                        "all_options": action_effects,
                        "not_chosen_options": [
                            x for x in action_effects
                            if str(x["action_code"]) != str(chosen_action_code)
                        ],
                        "guidance_applied": bool(step_idx in guidance_map),
                    }
                )
        except StepActionParseError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Unexpected rollout error at step {step_idx}: {exc}"
            ) from exc

        return float(env.get_makespan()), True, traces, step_records, []

    makespan, feasible, traces, step_records, notes = _rollout_once(
        code_seed=int(action_code_seed),
        guidance_by_step=None,
        reflection_memory_text=None,
    )
    if not feasible:
        return makespan, feasible, traces

    best_makespan = float(makespan)
    best_traces = traces
    best_step_records = step_records
    improvement_notes = list(notes)

    if enable_step_improvement and int(step_reflection_passes) > 0:
        if print_step_trace:
            print(
                f"[Episode Improvement] start: passes={int(step_reflection_passes)}, "
                f"topk={max(1, int(step_reflection_topk))}, "
                f"baseline_makespan={float(best_makespan):.1f}"
            )
        for pass_idx in range(int(step_reflection_passes)):
            critical_steps = _select_top_critical_steps(
                best_step_records,
                top_k=max(1, int(step_reflection_topk)),
            )
            if not critical_steps:
                improvement_notes.append(f"pass {pass_idx + 1}: no critical step found")
                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                        "no critical step found"
                    )
                break

            reflection_memory = _build_reflection_memory(best_makespan, critical_steps)
            guidance_map = {}
            for critical_step in critical_steps:
                step_idx = int(critical_step["step_idx"])
                feasible_action_codes = list(critical_step["feasible_action_codes"])
                action_code_to_job = dict(critical_step["action_code_to_job"])
                chosen_code = str(critical_step["chosen_action_code"])
                chosen_job = int(critical_step["chosen_job"])
                best_alt = _best_alternative_option(critical_step)

                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                        f"critical_step={step_idx}, chosen={chosen_code}"
                    )

                suggested_code = None
                if feasible_action_codes:
                    improvement_prompt = build_step_improvement_prompt(
                        state_text=critical_step["state_text"],
                        candidate_action_text=f"{chosen_code}",
                        feasible_jobs=feasible_action_codes,
                        reflection_memory=reflection_memory,
                        step_diagnostics=_build_step_diagnostics(critical_step),
                    )
                    improvement_prompt = (
                        f"{improvement_prompt}\n"
                        f"Episode summary: final makespan={best_makespan}.\n"
                        f"Target step={step_idx}, chosen={chosen_code}.\n"
                        "Select an action that reduces bottleneck risk and final makespan."
                    )
                    reflection_prompt = _build_step_improvement_chat_prompt(
                        tokenizer, improvement_prompt
                    )
                    try:
                        suggested_code, _, _, _ = generate_step_action(
                            model=model,
                            tokenizer=tokenizer,
                            prompt_text=reflection_prompt,
                            feasible_action_codes=feasible_action_codes,
                            device=device,
                            max_new_tokens=step_action_max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            use_masking=use_masking,
                            action_code_width=action_code_width,
                            action_code_cap=action_code_cap,
                        )
                    except StepActionParseError as exc:
                        improvement_notes.append(
                            f"pass {pass_idx + 1} step {step_idx}: improvement parse failed ({exc})"
                        )
                        if print_step_trace:
                            print(
                                f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)} "
                                f"step {step_idx}: parse failed ({exc})"
                            )

                if (not suggested_code or str(suggested_code) == chosen_code) and best_alt is not None:
                    alt_code = str(best_alt["action_code"])
                    if alt_code in action_code_to_job and alt_code != chosen_code:
                        suggested_code = alt_code
                        improvement_notes.append(
                            f"pass {pass_idx + 1} step {step_idx}: deterministic critic suggested {alt_code}"
                        )
                        if print_step_trace:
                            print(
                                f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)} "
                                f"step {step_idx}: critic suggested {alt_code}"
                            )

                if not suggested_code:
                    continue
                suggested_job = int(action_code_to_job[suggested_code])
                if suggested_job == chosen_job:
                    continue
                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)} "
                        f"step {step_idx}: suggested={suggested_code} -> Job {suggested_job}"
                    )
                guidance_map[step_idx] = {
                    "preferred_job": int(suggested_job),
                    "avoid_jobs": [int(chosen_job)],
                    "reason": (
                        f"chosen={chosen_code}/Job{chosen_job} showed high bottleneck risk; "
                        f"prefer {suggested_code}/Job{suggested_job}"
                    ),
                    "suggested_action_code": str(suggested_code),
                }

            if not guidance_map:
                improvement_notes.append(f"pass {pass_idx + 1}: no actionable guidance generated")
                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                        "no actionable guidance generated"
                    )
                continue

            cand_ms, cand_feasible, cand_traces, cand_steps, _ = _rollout_once(
                code_seed=int(action_code_seed) + (pass_idx + 1) * 9973,
                guidance_by_step=guidance_map,
                reflection_memory_text=reflection_memory,
            )
            if cand_feasible and math.isfinite(cand_ms) and cand_ms < best_makespan:
                improvement_notes.append(
                    f"pass {pass_idx + 1}: improved {best_makespan:.1f} -> {cand_ms:.1f} "
                    f"with guided_steps={sorted(guidance_map.keys())}"
                )
                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                        f"improved {best_makespan:.1f} -> {cand_ms:.1f}"
                    )
                best_makespan = float(cand_ms)
                best_traces = cand_traces
                best_step_records = cand_steps
            else:
                improvement_notes.append(
                    f"pass {pass_idx + 1}: no improvement ({best_makespan:.1f} vs {float(cand_ms):.1f})"
                )
                if print_step_trace:
                    print(
                        f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                        f"no improvement ({best_makespan:.1f} vs {float(cand_ms):.1f})"
                    )

    if print_step_trace:
        _print_step_trace(best_step_records)
        if improvement_notes:
            print("episode_improvement_notes:")
            for note in improvement_notes:
                print(" -", note)

    return float(best_makespan), True, best_traces
def compute_log_prob_mean(
    model,
    sequence_ids: torch.Tensor,
    prompt_len: int,
    action_token_id: int,
    device: torch.device,
    require_grad: bool,
) -> torch.Tensor:
    """
    Compute log-probability of the first generated action token only.
    """
    seq = sequence_ids[:prompt_len].unsqueeze(0).to(device)
    action_token_id = int(action_token_id)
    forward_kwargs = {
        "input_ids": seq,
        "use_cache": False,
        "num_logits_to_keep": 1,
    }
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel") and hasattr(torch.nn.attention, "SDPBackend"):
        sdp_ctx = torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH])
    else:
        sdp_ctx = (
            torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
            if torch.cuda.is_available() and hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "sdp_kernel")
            else contextlib.nullcontext()
        )

    if require_grad:
        with sdp_ctx:
            outputs = model(**forward_kwargs)
    else:
        with torch.no_grad():
            with sdp_ctx:
                outputs = model(**forward_kwargs)

    logits = outputs.logits[:, -1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs[0, action_token_id]


# ---------------------------------------------------------------------------
# RL training loop
# ---------------------------------------------------------------------------


def reinforce_step(
    batch: EpisodeBatch,
    optimizer: AdamW,
) -> float:
    """Perform a single REINFORCE update over collected samples."""

    advantages = batch.advantages
    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    advantages = advantages.detach()
    loss = -(batch.log_probs * advantages).sum()

    optimizer.zero_grad()
    loss.backward()
    params = []
    for group in optimizer.param_groups:
        params.extend(group["params"])
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizer.step()

    return loss.item()


def grpo_step(
    samples: List[TrajectorySample],
    model,
    optimizer: AdamW,
    device: torch.device,
    clip_epsilon: float = 0.2,
    grpo_epochs: int = 1,
    kl_coef: float = 0.0,
    update_batch_size: int = 4,
) -> Tuple[float, float]:
    """
    PPO-style clipped policy update with group-normalized advantages.
    """
    if not samples:
        return 0.0, 0.0

    old_log_probs = torch.stack([s.old_log_prob for s in samples]).to(device).detach()
    advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32, device=device)
    total_samples = int(len(samples))
    micro_batch_size = max(1, int(update_batch_size))

    total_loss = 0.0
    total_kl = 0.0
    for _ in range(grpo_epochs):
        optimizer.zero_grad()
        epoch_loss_value = 0.0
        epoch_kl_value = 0.0
        for start in range(0, total_samples, micro_batch_size):
            end = min(start + micro_batch_size, total_samples)
            chunk = samples[start:end]
            chunk_old = old_log_probs[start:end]
            chunk_adv = advantages[start:end]
            current_log_probs = []
            for s in chunk:
                log_prob = compute_log_prob_mean(
                    model=model,
                    sequence_ids=s.sequence_ids,
                    prompt_len=s.prompt_len,
                    action_token_id=s.action_token_id,
                    device=device,
                    require_grad=True,
                )
                current_log_probs.append(log_prob)
            current_log_probs = torch.stack(current_log_probs)

            log_ratio = current_log_probs - chunk_old
            ratio = torch.exp(torch.clamp(log_ratio, -20, 20))
            unclipped_obj = ratio * chunk_adv
            clipped_obj = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * chunk_adv
            policy_loss = -torch.mean(torch.min(unclipped_obj, clipped_obj))

            approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio + 1e-8))
            loss = policy_loss + kl_coef * approx_kl
            scaled_loss = loss * (float(end - start) / float(total_samples))
            scaled_loss.backward()

            weight = float(end - start) / float(total_samples)
            epoch_loss_value += float(loss.item()) * weight
            epoch_kl_value += float(approx_kl.item()) * weight
            del current_log_probs, log_ratio, ratio, unclipped_obj, clipped_obj, policy_loss, approx_kl, loss, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        params = []
        for group in optimizer.param_groups:
            params.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        total_loss += float(epoch_loss_value)
        total_kl += float(epoch_kl_value)

    return total_loss / grpo_epochs, total_kl / grpo_epochs


def build_bopo_step_pairs(
    group_rollouts: List[Dict],
    rng: np.random.Generator,
    min_relative_gap: float = 0.0,
    max_pairs_per_group: int = 256,
    max_step_pairs_per_pair: int = 32,
    pair_mode: str = "divergent_suffix",
) -> List[BOPOStepPair]:
    """
    Build BOPO preference pairs from a rollout group.

    Strategy:
      1. Keep feasible rollouts only.
      2. Sort by makespan (smaller is better), anchor best rollout.
      3. Pair best vs each loser if relative gap >= threshold.
      4. For each episode pair, either:
         - sample aligned step indices (`aligned`)
         - keep only the first divergence step (`shared_prefix`)
         - or use multiple divergent steps across the suffix (`divergent_suffix`)
    """
    feasible_rollouts = [
        r
        for r in group_rollouts
        if bool(r.get("feasible"))
        and math.isfinite(float(r.get("makespan", float("inf"))))
        and len(r.get("traces", [])) > 0
    ]
    if len(feasible_rollouts) < 2:
        return []

    feasible_rollouts.sort(key=lambda x: float(x["makespan"]))
    winner = feasible_rollouts[0]
    winner_ms = float(winner["makespan"])
    winner_traces = winner["traces"]

    resolved_pair_mode = str(pair_mode).strip().lower()
    if resolved_pair_mode not in {"aligned", "shared_prefix", "divergent_suffix"}:
        raise ValueError(f"Unsupported pair_mode={pair_mode}")

    def _first_divergence_index(w_traces, l_traces):
        n_steps_local = min(len(w_traces), len(l_traces))
        for idx in range(n_steps_local):
            if int(w_traces[idx].chosen_job) != int(l_traces[idx].chosen_job):
                return idx
        return None

    def _divergence_indices(w_traces, l_traces):
        n_steps_local = min(len(w_traces), len(l_traces))
        return [
            int(idx)
            for idx in range(n_steps_local)
            if int(w_traces[idx].chosen_job) != int(l_traces[idx].chosen_job)
        ]

    def _subsample_indices(indices, max_steps):
        if max_steps <= 0 or len(indices) <= max_steps:
            return [int(x) for x in indices]
        positions = np.linspace(0, len(indices) - 1, num=max_steps)
        picked = sorted({int(round(float(p))) for p in positions.tolist()})
        return [int(indices[i]) for i in picked]

    pairs: List[BOPOStepPair] = []
    for loser in feasible_rollouts[1:]:
        loser_ms = float(loser["makespan"])
        if loser_ms <= winner_ms:
            continue
        rel_gap = (loser_ms - winner_ms) / max(loser_ms, 1.0)
        if rel_gap < float(min_relative_gap):
            continue

        loser_traces = loser["traces"]
        n_steps = min(len(winner_traces), len(loser_traces))
        if n_steps <= 0:
            continue

        if resolved_pair_mode == "shared_prefix":
            div_idx = _first_divergence_index(winner_traces, loser_traces)
            if div_idx is None:
                continue
            indices = [int(div_idx)]
        elif resolved_pair_mode == "divergent_suffix":
            divergence_indices = _divergence_indices(winner_traces, loser_traces)
            if not divergence_indices:
                continue
            indices = _subsample_indices(divergence_indices, int(max_step_pairs_per_pair))
        else:
            indices = list(range(n_steps))
            max_steps = int(max_step_pairs_per_pair)
            if max_steps > 0 and n_steps > max_steps:
                picked = rng.choice(n_steps, size=max_steps, replace=False)
                indices = [int(x) for x in picked.tolist()]

        for step_idx in indices:
            w_trace = winner_traces[step_idx]
            l_trace = loser_traces[step_idx]
            pairs.append(
                BOPOStepPair(
                    winner_sequence_ids=w_trace.sequence_ids,
                    winner_prompt_len=int(w_trace.prompt_len),
                    winner_action_token_id=int(w_trace.action_token_id),
                    loser_sequence_ids=l_trace.sequence_ids,
                    loser_prompt_len=int(l_trace.prompt_len),
                    loser_action_token_id=int(l_trace.action_token_id),
                    relative_gap=float(rel_gap),
                    winner_makespan=float(winner_ms),
                    loser_makespan=float(loser_ms),
                )
            )

    max_pairs = int(max_pairs_per_group)
    if max_pairs > 0 and len(pairs) > max_pairs:
        picked = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[int(i)] for i in picked.tolist()]

    return pairs


def bopo_step(
    pairs: List[BOPOStepPair],
    model,
    optimizer: AdamW,
    device: torch.device,
    beta: float = 2.0,
    gap_scale: float = 3.0,
    margin: float = 0.0,
    update_batch_size: int = 8,
) -> Tuple[float, float, int]:
    """
    BOPO-style pairwise preference optimization.

    loss = -log sigma( beta * (1 + gap_scale * rel_gap) * (logp_win - logp_lose - margin) )
    """
    if not pairs:
        return 0.0, 0.0, 0

    total_loss = 0.0
    total_gap = 0.0
    updates = 0
    batch_size = max(1, int(update_batch_size))

    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        losses = []
        gaps = []
        try:
            optimizer.zero_grad()
            for p in chunk:
                lp_w = compute_log_prob_mean(
                    model=model,
                    sequence_ids=p.winner_sequence_ids,
                    prompt_len=p.winner_prompt_len,
                    action_token_id=p.winner_action_token_id,
                    device=device,
                    require_grad=True,
                )
                lp_l = compute_log_prob_mean(
                    model=model,
                    sequence_ids=p.loser_sequence_ids,
                    prompt_len=p.loser_prompt_len,
                    action_token_id=p.loser_action_token_id,
                    device=device,
                    require_grad=True,
                )

                rel_gap = max(0.0, float(p.relative_gap))
                scaled_beta = float(beta) * (1.0 + float(gap_scale) * rel_gap)
                pref_logit = scaled_beta * (lp_w - lp_l - float(margin))
                losses.append(-F.logsigmoid(pref_logit))
                gaps.append(rel_gap)
                del lp_w, lp_l, pref_logit

            if not losses:
                continue

            loss = torch.stack(losses).mean()
            loss.backward()
            params = []
            for group in optimizer.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += float(loss.item()) * len(losses)
            total_gap += float(sum(gaps))
            updates += int(len(losses))
            del loss, losses, gaps
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise

    return total_loss / max(updates, 1), total_gap / max(updates, 1), updates


def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)

    if args.rl_algo == "unsloth_grpo":
        PatchFastRL("GRPO", FastLanguageModel)
        print("[Info] official RL backend: unsloth GRPO")

    if not args.model_path:
        args.model_path = DEFAULT_MODEL_BY_TYPE[args.model_type]
        print(f"[Info] --model_path not provided. Using default: {args.model_path}")

    resolved_model_path = resolve_model_source(
        args.model_path,
        checkpoint_tag=getattr(args, "model_checkpoint_tag", None),
        token=args.hf_token,
    )
    print(f"[Info] loading model from: {resolved_model_path}")
    is_adapter = is_adapter_source(resolved_model_path, token=args.hf_token)
    base_load_path = DEFAULT_MODEL_BY_TYPE[args.model_type] if is_adapter else resolved_model_path
    print(f"[Info] base load path: {base_load_path}")

    load_kwargs = {
        "model_name": base_load_path,
        "max_seq_length": int(args.max_seq_length),
        "load_in_4bit": bool(args.load_in_4bit),
        "dtype": torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        "local_files_only": False,
    }
    if args.rl_algo == "unsloth_grpo":
        load_kwargs["fast_inference"] = bool(getattr(args, "grpo_use_fast_inference", False))
        load_kwargs["gpu_memory_utilization"] = float(getattr(args, "grpo_gpu_memory_utilization", 0.6))

    model, tokenizer = FastLanguageModel.from_pretrained(**load_kwargs)
    token_install = ensure_action_special_tokens(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
    )
    print("[Info] action token install:", token_install)
    if is_adapter:
        maybe_load_policy_adapter(model, resolved_model_path, token=args.hf_token)
        print("[Info] policy adapter loaded")
    if hasattr(model, "for_training"):
        model.for_training()
    if args.rl_algo == "unsloth_grpo":
        print("[Info] RL attention backend: official unsloth grpo trainer path")
    else:
        attn_backend_info = force_safe_train_attention_backend(model)
        print("[Info] RL attention backend:", attn_backend_info)
    model.train()
    validate_rl_update_mode(
        model=model,
        update_mode=args.rl_update_mode,
        model_path=resolved_model_path,
    )

    output_dir = Path(args.output_dir or f"outputs/{args.rl_algo}_{args.env_mode}")
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.rl_algo == "unsloth_grpo":
        try:
            from unsloth import UnslothGRPOConfig as GRPOConfig
        except Exception:
            from trl import GRPOConfig
        from trl import GRPOTrainer

        grpo_dataset, grpo_dataset_path = build_unsloth_grpo_step_dataset(args, hf_token=args.hf_token or "")
        reward_funcs = build_unsloth_grpo_reward_functions(args)
        max_completion_length = int(getattr(args, "grpo_max_completion_length", args.step_action_max_new_tokens) or args.step_action_max_new_tokens)
        default_prompt_length = max(256, int(args.max_seq_length) - max_completion_length - 32)
        configured_prompt_length = getattr(args, "grpo_max_prompt_length", None)
        max_prompt_length = int(configured_prompt_length or default_prompt_length)
        max_prompt_length = min(max_prompt_length, int(args.max_seq_length) - max_completion_length - 8)
        if max_prompt_length <= 0:
            raise ValueError("grpo_max_prompt_length must be smaller than max_seq_length - max_completion_length")

        grpo_num_train_epochs = getattr(args, "grpo_num_train_epochs", None)
        resolved_num_train_epochs = float(args.epochs if grpo_num_train_epochs in (None, "") else grpo_num_train_epochs)

        grpo_args = GRPOConfig(
            output_dir=str(output_dir),
            learning_rate=float(args.learning_rate),
            per_device_train_batch_size=int(args.grpo_per_device_train_batch_size),
            gradient_accumulation_steps=int(args.grpo_gradient_accumulation_steps),
            num_generations=int(args.grpo_num_generations),
            max_prompt_length=int(max_prompt_length),
            max_completion_length=int(max_completion_length),
            num_train_epochs=resolved_num_train_epochs,
            max_steps=int(args.grpo_max_steps),
            logging_steps=int(args.grpo_logging_steps),
            save_steps=int(args.grpo_save_steps),
            bf16=bool(args.dtype == "bfloat16" and torch.cuda.is_available() and is_bfloat16_supported()),
            fp16=bool(args.dtype != "bfloat16"),
            remove_unused_columns=False,
            report_to=str(args.grpo_report_to),
            optim=str(args.grpo_optim),
            weight_decay=float(args.grpo_weight_decay),
            warmup_ratio=float(args.grpo_warmup_ratio),
            lr_scheduler_type=str(args.grpo_lr_scheduler_type),
            max_grad_norm=float(args.grpo_max_grad_norm),
            seed=int(args.seed),
        )

        print(
            "[Info] official grpo config:",
            {
                "train_rows": len(grpo_dataset),
                "num_generations": int(args.grpo_num_generations),
                "num_train_epochs": resolved_num_train_epochs,
                "max_steps": int(args.grpo_max_steps),
                "max_prompt_length": int(max_prompt_length),
                "max_completion_length": int(max_completion_length),
            },
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=reward_funcs,
            args=grpo_args,
            train_dataset=grpo_dataset,
        )
        resume_ckpt = getattr(args, "resume_from_checkpoint", None) or None
        trainer.train(resume_from_checkpoint=resume_ckpt)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))

        grpo_log_history = list(getattr(trainer.state, "log_history", []) or [])
        grpo_log_path = output_dir / "grpo_log_history.csv"
        if grpo_log_history:
            write_history_csv(grpo_log_path, grpo_log_history)
            print("[Info] grpo_log_history_path:", grpo_log_path)
        else:
            print("[Info] No GRPO log history found.")
        print("[Info] official grpo dataset path:", grpo_dataset_path)
        print("[Info] official grpo train rows:", len(grpo_dataset))
        print("[Info] training done")
        print("[Info] output_dir:", output_dir)
        return

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    rng = np.random.default_rng(args.seed)
    use_random = args.use_random_problems
    if use_random:
        data_split = None
    else:
        dataset = load_dataset("json", data_files=args.dataset_path)
        data_split = dataset["train"]
        raw_rows = len(data_split)
        raw_size_counter: Counter = Counter()
        filtered_size_counter: Counter = Counter()
        keep_indices = []
        exact_jobs = getattr(args, "rl_num_jobs", None)
        exact_machines = getattr(args, "rl_num_machines", None)
        min_jobs = getattr(args, "min_rl_num_jobs", None)
        min_machines = getattr(args, "min_rl_num_machines", None)

        for idx, example in enumerate(data_split):
            inst = extract_problem_instance_from_example(example)
            n_jobs = int(len(inst))
            n_machines = int(len(inst[0]) if inst else 0)
            size_key = (n_jobs, n_machines)
            raw_size_counter[size_key] += 1

            keep = True
            if exact_jobs is not None and n_jobs != int(exact_jobs):
                keep = False
            if exact_machines is not None and n_machines != int(exact_machines):
                keep = False
            if min_jobs is not None and n_jobs < int(min_jobs):
                keep = False
            if min_machines is not None and n_machines < int(min_machines):
                keep = False
            if keep:
                keep_indices.append(int(idx))
                filtered_size_counter[size_key] += 1

        if len(keep_indices) != raw_rows:
            data_split = data_split.select(keep_indices)

        print("[Info] rl dataset raw rows:", raw_rows)
        print("[Info] rl dataset filtered rows:", len(data_split))
        print("[Info] rl dataset kept ratio:", (len(data_split) / raw_rows) if raw_rows else 0.0)
        print("[Info] rl dataset size_dist_top_raw:", raw_size_counter.most_common(10))
        print("[Info] rl dataset size_dist_top_filtered:", filtered_size_counter.most_common(10))
        print(
            "[Info] rl dataset filter config:",
            {
                "rl_num_jobs": exact_jobs,
                "rl_num_machines": exact_machines,
                "min_rl_num_jobs": min_jobs,
                "min_rl_num_machines": min_machines,
            },
        )
        if len(data_split) == 0:
            raise ValueError("No RL training problems remain after applying size filters.")

    baseline_tracker = ExponentialBaseline(beta=args.baseline_beta)
    for epoch in range(args.epochs):
        episode_log_probs = []
        episode_advantages = []
        grpo_samples: List[TrajectorySample] = []
        bopo_pairs_epoch: List[BOPOStepPair] = []
        bopo_updates = 0
        bopo_loss_total = 0.0
        bopo_gap_total = 0.0

        with trange(args.episodes_per_epoch, desc=f"Epoch {epoch+1}/{args.epochs}") as t:
            for _ in t:
                if use_random:
                    instance = generate_random_instance(
                        num_jobs=args.random_jobs,
                        num_machines=args.random_machines,
                        process_time_range=(args.random_time_low, args.random_time_high),
                        rng=rng,
                    )
                    inst = instance["inst_for_ortools"]
                else:
                    example = random.choice(data_split)
                    inst = extract_problem_instance_from_example(example)

                heuristic_makespan = None
                if args.reward_mode == "mwkr_relative":
                    _, heuristic_makespan = mwkr_schedule(inst)

                if args.rl_algo in {"grpo", "grpo_manual", "grpo_episode"}:
                    group_rollouts = []
                    rewards = []
                    for _group_idx in range(max(1, args.group_size)):
                        rollout_code_seed = int(rng.integers(0, 2**31 - 1))
                        try:
                            makespan, feasible, traces = rollout_step_episode(
                                model=model,
                                tokenizer=tokenizer,
                                inst_for_ortools=inst,
                                device=device,
                                step_action_max_new_tokens=args.step_action_max_new_tokens,
                                env_mode=args.env_mode,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                use_masking=not args.disable_masking,
                                include_problem_context=not args.disable_step_problem_context,
                                enable_step_improvement=args.enable_step_improvement,
                                step_reflection_passes=args.step_reflection_passes,
                                step_reflection_topk=args.step_reflection_topk,
                                action_code_width=args.action_code_width,
                                action_code_seed=rollout_code_seed,
                                action_code_cap=args.action_code_cap,
                                print_step_trace=args.print_step_trace,
                            )
                        except StepActionParseError as exc:
                            makespan = float(args.invalid_makespan_penalty)
                            feasible = False
                            traces = []
                            if args.print_step_trace:
                                print(f"[warn] GRPO rollout parse failed -> penalty applied: {exc}")
                        reward = compute_episode_reward(
                            makespan=makespan,
                            feasible=bool(feasible),
                            inst_for_ortools=inst,
                            invalid_makespan_penalty=float(args.invalid_makespan_penalty),
                            reward_mode=args.reward_mode,
                            heuristic_makespan=heuristic_makespan,
                        )

                        step_samples = []
                        for tr in traces:
                            old_log_prob = compute_log_prob_mean(
                                model=model,
                                sequence_ids=tr.sequence_ids,
                                prompt_len=tr.prompt_len,
                                action_token_id=tr.action_token_id,
                                device=device,
                                require_grad=False,
                            ).detach().cpu()
                            step_samples.append(
                                {
                                    "sequence_ids": tr.sequence_ids.detach().cpu(),
                                    "prompt_len": tr.prompt_len,
                                    "action_token_id": int(tr.action_token_id),
                                    "old_log_prob": old_log_prob,
                                }
                            )

                        group_rollouts.append(
                            {
                                "reward": float(reward),
                                "feasible": bool(feasible),
                                "makespan": float(makespan),
                                "step_samples": step_samples,
                            }
                        )
                        rewards.append(float(reward))

                    rewards_t = torch.tensor(rewards, dtype=torch.float32)
                    mean_r = float(rewards_t.mean().item())
                    std_r = float(rewards_t.std(unbiased=False).item())
                    denom = std_r + 1e-8

                    for rollout in group_rollouts:
                        advantage = (rollout["reward"] - mean_r) / denom
                        for s in rollout["step_samples"]:
                            grpo_samples.append(
                                TrajectorySample(
                                    sequence_ids=s["sequence_ids"],
                                    prompt_len=s["prompt_len"],
                                    action_token_id=int(s["action_token_id"]),
                                    reward=rollout["reward"],
                                    advantage=float(advantage),
                                    old_log_prob=s["old_log_prob"],
                                    feasible=rollout["feasible"],
                                    makespan=rollout["makespan"],
                                )
                            )

                    feasible_makespans = [
                        c["makespan"] for c in group_rollouts if c["feasible"]
                    ]
                    best_ms = (
                        min(feasible_makespans)
                        if feasible_makespans
                        else float(args.invalid_makespan_penalty)
                    )
                    t.set_postfix(
                        algo="grpo-episode",
                        best_makespan=f"{best_ms:.1f}",
                        group_reward=f"{mean_r:.1f}",
                    )
                elif args.rl_algo == "bopo":
                    group_rollouts = []
                    for _group_idx in range(max(2, args.group_size)):
                        rollout_code_seed = int(rng.integers(0, 2**31 - 1))
                        try:
                            makespan, feasible, traces = rollout_step_episode(
                                model=model,
                                tokenizer=tokenizer,
                                inst_for_ortools=inst,
                                device=device,
                                step_action_max_new_tokens=args.step_action_max_new_tokens,
                                env_mode=args.env_mode,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                top_k=args.top_k,
                                use_masking=not args.disable_masking,
                                include_problem_context=not args.disable_step_problem_context,
                                enable_step_improvement=args.enable_step_improvement,
                                step_reflection_passes=args.step_reflection_passes,
                                step_reflection_topk=args.step_reflection_topk,
                                action_code_width=args.action_code_width,
                                action_code_seed=rollout_code_seed,
                                action_code_cap=args.action_code_cap,
                                print_step_trace=args.print_step_trace,
                            )
                        except StepActionParseError as exc:
                            makespan = float(args.invalid_makespan_penalty)
                            feasible = False
                            traces = []
                            if args.print_step_trace:
                                print(f"[warn] BOPO rollout parse failed -> penalty applied: {exc}")

                        group_rollouts.append(
                            {
                                "feasible": bool(feasible),
                                "makespan": float(makespan),
                                "traces": traces,
                            }
                        )

                    feasible_makespans = [
                        c["makespan"] for c in group_rollouts if c["feasible"]
                    ]
                    best_ms = (
                        min(feasible_makespans)
                        if feasible_makespans
                        else float(args.invalid_makespan_penalty)
                    )

                    bopo_pairs = build_bopo_step_pairs(
                        group_rollouts=group_rollouts,
                        rng=rng,
                        min_relative_gap=args.bopo_min_relative_gap,
                        max_pairs_per_group=args.bopo_max_pairs_per_group,
                        max_step_pairs_per_pair=args.bopo_max_step_pairs_per_pair,
                        pair_mode=args.bopo_pair_mode,
                    )
                    if not bopo_pairs:
                        t.set_postfix(
                            algo="bopo-step",
                            best_makespan=f"{best_ms:.1f}",
                            pairs=0,
                        )
                        continue
                    bopo_pairs_epoch.extend(bopo_pairs)

                    t.set_postfix(
                        algo="bopo-collect",
                        best_makespan=f"{best_ms:.1f}",
                        pairs=len(bopo_pairs),
                        epoch_pairs=len(bopo_pairs_epoch),
                    )
                else:
                    rollout_code_seed = int(rng.integers(0, 2**31 - 1))
                    try:
                        makespan, feasible, traces = rollout_step_episode(
                            model=model,
                            tokenizer=tokenizer,
                            inst_for_ortools=inst,
                            device=device,
                            step_action_max_new_tokens=args.step_action_max_new_tokens,
                            env_mode=args.env_mode,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            top_k=args.top_k,
                            use_masking=not args.disable_masking,
                            include_problem_context=not args.disable_step_problem_context,
                            enable_step_improvement=args.enable_step_improvement,
                            step_reflection_passes=args.step_reflection_passes,
                            step_reflection_topk=args.step_reflection_topk,
                            action_code_width=args.action_code_width,
                            action_code_seed=rollout_code_seed,
                            action_code_cap=args.action_code_cap,
                            print_step_trace=args.print_step_trace,
                        )
                    except StepActionParseError as exc:
                        if args.print_step_trace:
                            print(f"[warn] REINFORCE rollout parse failed -> skip: {exc}")
                        t.set_postfix_str("parse-failed rollout, skipping")
                        continue
                    if not feasible or not traces or not math.isfinite(makespan):
                        t.set_postfix_str("invalid step rollout, skipping")
                        continue

                    reward = compute_episode_reward(
                        makespan=makespan,
                        feasible=bool(feasible),
                        inst_for_ortools=inst,
                        invalid_makespan_penalty=float(args.invalid_makespan_penalty),
                        reward_mode=args.reward_mode,
                        heuristic_makespan=heuristic_makespan,
                    )
                    mwkr_makespan = (
                        float(heuristic_makespan)
                        if heuristic_makespan is not None
                        else float(mwkr_schedule(inst)[1])
                    )
                    baseline = compute_episode_reward(
                        makespan=mwkr_makespan,
                        feasible=math.isfinite(mwkr_makespan),
                        inst_for_ortools=inst,
                        invalid_makespan_penalty=float(args.invalid_makespan_penalty),
                        reward_mode=args.reward_mode,
                        heuristic_makespan=mwkr_makespan,
                    )
                    if args.use_running_baseline:
                        baseline = baseline_tracker.update(reward)
                    advantage = reward - baseline

                    for tr in traces:
                        log_prob = compute_log_prob_mean(
                            model=model,
                            sequence_ids=tr.sequence_ids,
                            prompt_len=tr.prompt_len,
                            action_token_id=tr.action_token_id,
                            device=device,
                            require_grad=True,
                        )
                        episode_log_probs.append(log_prob)
                        episode_advantages.append(torch.tensor(advantage, device=device))

                    t.set_postfix(
                        algo="reinforce-step",
                        makespan=f"{makespan:.1f}",
                        reward=f"{reward:.1f}",
                        advantage=f"{advantage:.1f}",
                        steps=len(traces),
                    )

        if args.rl_algo in {"grpo", "grpo_manual", "grpo_episode"}:
            if not grpo_samples:
                print("No GRPO samples collected this epoch; skipping update.")
                continue
            loss_value, approx_kl = grpo_step(
                samples=grpo_samples,
                model=model,
                optimizer=optimizer,
                device=device,
                clip_epsilon=args.clip_epsilon,
                grpo_epochs=args.grpo_epochs,
                kl_coef=args.kl_coef,
                update_batch_size=args.grpo_update_batch_size,
            )
            print(
                f"[Epoch {epoch+1}] GRPO loss: {loss_value:.4f}, "
                f"approx_kl: {approx_kl:.6f}, samples: {len(grpo_samples)}"
            )
        elif args.rl_algo == "bopo":
            if not bopo_pairs_epoch:
                print("No valid BOPO preference pairs collected this epoch; skipping summary.")
                continue
            bopo_loss, bopo_gap, pair_updates = bopo_step(
                pairs=bopo_pairs_epoch,
                model=model,
                optimizer=optimizer,
                device=device,
                beta=args.bopo_beta,
                gap_scale=args.bopo_gap_scale,
                margin=args.bopo_margin,
                update_batch_size=args.bopo_update_batch_size,
            )
            bopo_updates += int(pair_updates)
            bopo_loss_total += float(bopo_loss) * max(int(pair_updates), 1)
            bopo_gap_total += float(bopo_gap) * max(int(pair_updates), 1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            avg_loss = float(bopo_loss_total) / float(max(bopo_updates, 1))
            avg_gap = float(bopo_gap_total) / float(max(bopo_updates, 1))
            print(
                f"[Epoch {epoch+1}] BOPO pair-updates: {bopo_updates}, "
                f"avg_pair_loss: {avg_loss:.6f}, avg_rel_gap: {avg_gap:.6f}"
            )
        else:
            if not episode_log_probs:
                print("No valid episodes collected this epoch; skipping update.")
                continue

            batch = EpisodeBatch(
                log_probs=torch.stack(episode_log_probs),
                advantages=torch.stack(episode_advantages),
            )
            loss_value = reinforce_step(batch, optimizer)
            print(f"[Epoch {epoch+1}] REINFORCE policy loss: {loss_value:.4f}")

        save_interval = max(1, int(getattr(args, "save_every_n_epochs", 1) or 1))
        if ((epoch + 1) % save_interval) == 0:
            ckpt_dir = output_dir / f"checkpoint-epoch-{epoch+1}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[Epoch {epoch+1}] saved: {ckpt_dir}")

    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    print("[Info] training done")
    print("[Info] output_dir:", output_dir)
    print("[Info] final_dir:", final_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Policy optimization for JSSP (official Unsloth GRPO + legacy manual RL).")
    parser.add_argument("--max_seq_length", type=int, default=40000, help="Maximum sequence length")
    parser.add_argument("--model_type", type=str, default="llama8b",
                        choices=["llama8b", "llama1b", "qwen2.5_7b", "qwen2.5_14b", "deepseek_8b", "qwen25_7b_math"],
                        help="Base model family (inference-compatible defaults).")
    parser.add_argument("--model_path", type=str, default=None, help="LoRA checkpoint path (defaults to inference model).")
    parser.add_argument("--model_checkpoint_tag", type=str, default=None, help="Optional checkpoint subfolder tag inside an HF model repo.")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"],
                        help="Computation dtype.")
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use 4-bit quantization (recommended). Pass --no-load_in_4bit to disable.",
    )

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episodes_per_epoch", type=int, default=2)
    parser.add_argument("--save_every_n_epochs", type=int, default=1, help="Save checkpoint every N epochs for manual RL loops.")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument(
        "--rl_update_mode",
        type=str,
        default="adapter_only",
        choices=["adapter_only", "full"],
        help="Which parameters RL is allowed to update. Defaults to adapter_only.",
    )
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=1, help="Legacy decode bound; step action decoding uses --step_action_max_new_tokens.")
    parser.add_argument("--baseline_beta", type=float, default=0.9)
    parser.add_argument("--use_running_baseline", action="store_true")
    parser.add_argument(
        "--rl_algo",
        type=str,
        default="grpo_episode",
        choices=["grpo_episode", "unsloth_grpo", "reinforce", "grpo", "grpo_manual", "bopo"],
    )
    parser.add_argument("--group_size", type=int, default=4, help="Number of sampled rollouts per prompt (GRPO).")
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Number of optimization epochs per GRPO batch.")
    parser.add_argument("--grpo_update_batch_size", type=int, default=1, help="Micro-batch size for manual whole-episode GRPO updates.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO/GRPO clip epsilon.")
    parser.add_argument("--kl_coef", type=float, default=0.0, help="Approx-KL regularization coefficient.")
    parser.add_argument("--bopo_beta", type=float, default=2.0, help="BOPO pairwise inverse temperature.")
    parser.add_argument("--bopo_gap_scale", type=float, default=3.0, help="Scale factor for objective-gap weighted BOPO logits.")
    parser.add_argument("--bopo_margin", type=float, default=0.0, help="Optional margin in BOPO preference logit.")
    parser.add_argument("--bopo_min_relative_gap", type=float, default=0.0, help="Minimum relative makespan gap to keep a BOPO pair.")
    parser.add_argument("--bopo_max_pairs_per_group", type=int, default=256, help="Maximum BOPO step-pairs generated per rollout group.")
    parser.add_argument("--bopo_max_step_pairs_per_pair", type=int, default=32, help="Maximum aligned step pairs used between winner/loser rollouts.")
    parser.add_argument("--bopo_update_batch_size", type=int, default=8, help="Micro-batch size for BOPO pair updates.")
    parser.add_argument("--bopo_pair_mode", type=str, default="divergent_suffix", choices=["aligned", "shared_prefix", "divergent_suffix"], help="BOPO pair construction mode.")
    parser.add_argument("--step_action_max_new_tokens", type=int, default=1, help="Max decode length for one step action.")
    parser.add_argument("--env_mode", type=str, required=True, choices=["serial", "dispatch"], help="Step environment mode used during rollout.")
    parser.add_argument("--action_code_width", type=int, default=4, help="Fixed digit width in action token (e.g., <A0001>).")
    parser.add_argument("--action_code_seed", type=int, default=42, help="Base seed for randomized action-code mapping.")
    parser.add_argument("--action_code_cap", type=int, default=9999, help="Upper bound of action token pool (sampled sparsely per step from [1, cap]).")
    parser.add_argument("--disable_step_problem_context", action="store_true", help="Do not include static problem context in step prompts.")
    parser.add_argument("--enable_step_improvement", action="store_true", help="Enable post-episode critical-step improvement passes.")
    parser.add_argument("--step_reflection_passes", type=int, default=1, help="Number of post-episode improvement passes.")
    parser.add_argument("--step_reflection_topk", type=int, default=3, help="Number of critical steps to guide per improvement pass.")
    parser.add_argument(
        "--invalid_makespan_penalty",
        type=float,
        default=1e6,
        help="Penalty makespan used for infeasible outputs in legacy manual RL.",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="raw_neg_makespan",
        choices=["raw_neg_makespan", "neg_makespan_per_op", "mwkr_relative"],
        help="Reward normalization mode for RL rollouts.",
    )
    parser.add_argument("--disable_masking", action="store_true", help="Disable decoding-time feasibility masking.")
    parser.add_argument("--print_step_trace", action="store_true", help="Print chosen/not-chosen options and makespan per step.")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--dataset_path", type=str, default="llm_jssp/train.json", help="Legacy manual RL raw-problem dataset path.")
    parser.add_argument("--rl_num_jobs", type=int, default=None)
    parser.add_argument("--rl_num_machines", type=int, default=None)
    parser.add_argument("--min_rl_num_jobs", type=int, default=None)
    parser.add_argument("--min_rl_num_machines", type=int, default=None)
    parser.add_argument("--use_random_problems", action="store_true",
                        help="Generate random JSSP instances instead of using dataset.")
    parser.add_argument("--random_jobs", type=int, default=10)
    parser.add_argument("--random_machines", type=int, default=10)
    parser.add_argument("--random_time_low", type=int, default=1)
    parser.add_argument("--random_time_high", type=int, default=100)

    parser.add_argument("--grpo_dataset_source", type=str, default="hf", choices=["hf", "local"])
    parser.add_argument("--grpo_dataset_path", type=str, default=None)
    parser.add_argument("--grpo_dataset_role", type=str, default="policy")
    parser.add_argument("--grpo_step_dataset_hf_repo", type=str, default="HYUNJINI/jssp_policy_step_train_all_v1")
    parser.add_argument("--grpo_step_dataset_hf_file", type=str, default="train_data/jssp_step_train_policy.jsonl")
    parser.add_argument("--grpo_step_dataset_local_path", type=str, default="/content/jssp_step_train_policy.jsonl")
    parser.add_argument("--grpo_step_dataset_hf_repo_dispatch", type=str, default="HYUNJINI/jssp_policy_step_train_dispatch_v1")
    parser.add_argument("--grpo_step_dataset_hf_file_dispatch", type=str, default="train_data/jssp_step_train_policy_dispatch.jsonl")
    parser.add_argument("--grpo_step_dataset_local_path_dispatch", type=str, default="/content/jssp_step_train_policy_dispatch.jsonl")
    parser.add_argument("--grpo_max_dataset_rows", type=int, default=5000)
    parser.add_argument("--grpo_min_feasible_actions", type=int, default=2)
    parser.add_argument("--grpo_shuffle_seed", type=int, default=42)
    parser.add_argument("--grpo_per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--grpo_gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--grpo_num_generations", type=int, default=6)
    parser.add_argument("--grpo_num_train_epochs", type=float, default=None)
    parser.add_argument("--grpo_max_steps", type=int, default=-1)
    parser.add_argument("--grpo_logging_steps", type=int, default=5)
    parser.add_argument("--grpo_save_steps", type=int, default=100)
    parser.add_argument("--grpo_max_prompt_length", type=int, default=4096)
    parser.add_argument("--grpo_max_completion_length", type=int, default=8)
    parser.add_argument("--grpo_report_to", type=str, default="none")
    parser.add_argument("--grpo_use_fast_inference", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--grpo_gpu_memory_utilization", type=float, default=0.6)
    parser.add_argument("--grpo_reward_valid_weight", type=float, default=1.0)
    parser.add_argument("--grpo_reward_proxy_weight", type=float, default=1.0)
    parser.add_argument("--grpo_reward_teacher_weight", type=float, default=0.25)
    parser.add_argument("--grpo_optim", type=str, default="adamw_8bit")
    parser.add_argument("--grpo_weight_decay", type=float, default=0.01)
    parser.add_argument("--grpo_warmup_ratio", type=float, default=0.05)
    parser.add_argument("--grpo_lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--grpo_max_grad_norm", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
