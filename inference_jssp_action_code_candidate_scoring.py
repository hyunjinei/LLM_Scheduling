from __future__ import annotations

import argparse
import copy
import csv
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
from unsloth import FastLanguageModel

from llm_jssp.utils import inference_step_common as base
from llm_jssp.utils.action_code_candidate_scoring import (
    ensure_candidate_score_token,
    infer_module_device,
    load_candidate_score_head,
    score_candidate_actions,
)
from llm_jssp.utils.action_token_utils import ensure_action_special_tokens
from llm_jssp.utils.jssp_step_stack import resolve_step_stack
from llm_jssp.utils.solution_generation_english import read_matrix_form_jssp


MODEL_BASE = {
    "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "llama1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "qwen2.5_7b": "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
    "qwen2.5_14b": "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
    "deepseek_8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "qwen25_7b_math": "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
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


def _build_parser():
    parser = argparse.ArgumentParser(description="Inference for JSSP action-code candidate-scoring policy.")
    parser.add_argument("--model_type", type=str, default="llama8b", choices=sorted(MODEL_BASE))
    parser.add_argument("--model_repo_or_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--model_checkpoint_tag", type=str, default=None)
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN", ""))
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument("--load_in_4bit", action="store_true", default=True)

    parser.add_argument("--eval_source", type=str, default="local", choices=["local", "hf"])
    parser.add_argument("--eval_dataset_repo", type=str, default="HYUNJINI/jssp_validation_all_v1")
    parser.add_argument("--eval_dataset_file", type=str, default="validation_data/ta.json")
    parser.add_argument("--eval_data_path", type=str, default="validation_data/ta.json")
    parser.add_argument("--eval_start_index", type=int, default=0)
    parser.add_argument("--eval_end_index", type=int, default=None)

    parser.add_argument("--env_mode", type=str, default="dispatch", choices=["serial", "dispatch"])
    parser.add_argument("--policy_head_type", type=str, default="candidate_scoring", choices=["candidate_scoring"])
    parser.add_argument("--candidate_score_token", type=str, default="<CAND_SCORE>")
    parser.add_argument("--candidate_scoring_do_sample", action="store_true", default=True)
    parser.add_argument("--greedy_first_solution", action="store_true", default=True)
    parser.add_argument("--candidate_scoring_temperature", type=float, default=1.0)
    parser.add_argument("--candidate_scoring_query_forward_batch_size", type=int, default=16)
    parser.add_argument("--print_candidate_probs_during_inference", action="store_true", default=True)
    parser.add_argument("--candidate_prob_trace_topk", type=int, default=5)

    parser.add_argument("--num_return_sequences", type=int, default=10)
    parser.add_argument("--disable_masking", action="store_true", default=False)
    parser.add_argument("--step_action_max_new_tokens", type=int, default=8)
    parser.add_argument("--disable_step_problem_context", action="store_true", default=False)
    parser.add_argument("--enable_step_improvement", action="store_true", default=False)
    parser.add_argument("--step_reflection_passes", type=int, default=2)
    parser.add_argument("--step_reflection_topk", type=int, default=5)
    parser.add_argument("--emit_step_rationale", action="store_true", default=False)
    parser.add_argument("--step_rationale_max_new_tokens", type=int, default=96)
    parser.add_argument("--reason_model_path", type=str, default=None)
    parser.add_argument("--reason_topk", type=int, default=3)
    parser.add_argument("--action_code_width", type=int, default=4)
    parser.add_argument("--action_code_seed", type=int, default=42)
    parser.add_argument("--action_code_cap", type=int, default=9999)
    parser.add_argument("--demo_index", type=int, default=None)
    parser.add_argument("--demo_num_solutions", type=int, default=None)
    parser.add_argument("--output_accord", action="store_true", default=False)
    parser.add_argument("--output_list_of_lists", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="val_results/jssp_val")
    parser.add_argument("--inspect_masking", action="store_true", default=False)
    parser.add_argument("--print_step_trace", action="store_true", default=False)
    parser.add_argument("--max_new_tokens", type=int, default=40000)
    parser.add_argument("--use_distance", action="store_true", default=True)
    parser.add_argument("--infer_fssp", action="store_true", default=False)
    parser.add_argument("--train_vrp_tsp", action="store_true", default=False)
    parser.add_argument("--train_knapsack", action="store_true", default=False)
    parser.add_argument("--train_binpack", action="store_true", default=False)
    parser.add_argument("--train_jssp", action="store_true", default=True)
    parser.add_argument("--train_fssp", action="store_true", default=False)
    return parser


def _resolve_checkpoint_tag(repo_id: str, checkpoint_tag: str | None, token: str | None = None):
    if not checkpoint_tag:
        return None
    repo_id = os.path.expanduser(str(repo_id))
    if os.path.exists(repo_id):
        return None
    checkpoint_tag = str(checkpoint_tag)
    if checkpoint_tag != "latest_checkpoint":
        return checkpoint_tag
    try:
        api = HfApi(token=token)
        files = api.list_repo_files(repo_id=repo_id, repo_type="model")
    except Exception:
        return None
    checkpoint_dirs = sorted(
        {
            path.split("/")[0]
            for path in files
            if path.startswith("checkpoint-") and "/" in path
        },
        key=lambda name: int(name.split("-")[-1]),
    )
    return checkpoint_dirs[-1] if checkpoint_dirs else None


def _resolve_model_source(repo_or_path: str | None, local_path: str | None = None, checkpoint_tag: str | None = None, token: str | None = None):
    if local_path:
        return os.path.expanduser(local_path)
    if not repo_or_path:
        return None
    repo_or_path = os.path.expanduser(repo_or_path)
    if os.path.exists(repo_or_path):
        return repo_or_path
    resolved_checkpoint = _resolve_checkpoint_tag(repo_or_path, checkpoint_tag, token=token)
    if resolved_checkpoint is None:
        return repo_or_path

    snapshot_dir = snapshot_download(
        repo_id=repo_or_path,
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
                repo_id=repo_or_path,
                repo_type="model",
                filename=f"{resolved_checkpoint}/{probe_file}",
                token=token,
            )
            return os.path.dirname(local_file)
        except Exception:
            pass

    raise FileNotFoundError(f"checkpoint folder not found after download: {local_checkpoint_dir}")


def _is_adapter_source(path_or_repo: str | None, token: str | None = None) -> bool:
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


def _maybe_load_policy_adapter(model, policy_model_path: str | None, token: str | None = None):
    if not policy_model_path:
        return False
    if not hasattr(model, "load_adapter"):
        raise RuntimeError("Policy adapter path was provided, but the loaded model does not support load_adapter().")
    if os.path.exists(policy_model_path):
        model.load_adapter(policy_model_path, adapter_name="default", adapter_kwargs={"local_files_only": True})
    else:
        model.load_adapter(policy_model_path, adapter_name="default", token=token)
    if hasattr(model, "set_adapter"):
        model.set_adapter("default")
    return True


def _resolve_candidate_scorer_path(model_source: str | None, token: str | None = None):
    if not model_source:
        return None
    if os.path.exists(model_source):
        candidates = [
            os.path.join(model_source, "candidate_scorer.pt"),
            os.path.join(model_source, "final", "candidate_scorer.pt"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None
    for filename in ("candidate_scorer.pt",):
        try:
            return hf_hub_download(
                repo_id=model_source,
                repo_type="model",
                filename=filename,
                token=token,
            )
        except Exception:
            pass
    return None


def _load_eval_dataset(args):
    if str(args.eval_source).lower() == "hf":
        eval_path = hf_hub_download(
            repo_id=args.eval_dataset_repo,
            repo_type="dataset",
            filename=args.eval_dataset_file,
            token=args.hf_token or None,
        )
    else:
        eval_path = os.path.expanduser(args.eval_data_path)
    print("eval dataset path:", eval_path)
    return load_dataset("json", data_files=eval_path)


def _sample_step_action(
    model,
    tokenizer,
    candidate_score_head,
    prompt_text: str,
    feasible_action_codes,
    args,
    candidate_scoring_do_sample: bool | None = None,
    candidate_scoring_temperature: float | None = None,
):
    if candidate_scoring_do_sample is None:
        candidate_scoring_do_sample = bool(args.candidate_scoring_do_sample)
    if candidate_scoring_temperature is None:
        candidate_scoring_temperature = float(args.candidate_scoring_temperature)

    result = score_candidate_actions(
        model=model,
        tokenizer=tokenizer,
        candidate_score_head=candidate_score_head,
        prompt_text=prompt_text,
        feasible_action_codes=list(feasible_action_codes),
        do_sample=bool(candidate_scoring_do_sample),
        temperature=float(candidate_scoring_temperature),
        code_width=int(args.action_code_width),
        score_token=str(args.candidate_score_token),
        max_length=int(args.max_seq_length),
        query_forward_batch_size=int(args.candidate_scoring_query_forward_batch_size),
        topk=int(args.candidate_prob_trace_topk),
    )
    return str(result["chosen_action_code"]), result["debug_payload"]


def _run_single_step_rollout(
    model,
    tokenizer,
    candidate_score_head,
    inst_for_ortools,
    args,
    guidance_by_step=None,
    reflection_memory_text: str | None = None,
    replay_action_jobs_by_step=None,
    replay_prefix_until: int | None = None,
    candidate_scoring_do_sample: bool | None = None,
    candidate_scoring_temperature: float | None = None,
):
    env = base._make_step_env(inst_for_ortools, args.env_mode)
    env.reset()
    raw_step_outputs = []
    problem_context_text = None if args.disable_step_problem_context else base.build_problem_context_text(inst_for_ortools)
    action_rng = random.Random(int(args.action_code_seed))
    guidance_map = guidance_by_step or {}

    while not env.is_done():
        state = env.get_state_json()
        feasible_jobs = list(state["feasible_jobs"])
        step_idx = int(state["step_idx"])
        action_code_to_job = base.build_randomized_action_code_map(
            feasible_jobs=feasible_jobs,
            rng=action_rng,
            code_width=args.action_code_width,
            code_start=1,
            code_cap=args.action_code_cap,
        )
        makespan_before, action_effects = base._estimate_action_effects(
            state_json=state,
            action_code_to_job=action_code_to_job,
            env_mode=args.env_mode,
        )
        effect_by_code = {x["action_code"]: x for x in action_effects}
        job_to_action_code = base.invert_action_code_map(action_code_to_job)
        feasible_action_codes = list(action_code_to_job.keys())
        prompt_state_text = base._build_state_text(
            state_json=state,
            feasible_jobs=feasible_jobs,
            step_idx=step_idx,
            total_steps=state["total_steps"],
            problem_context_text=problem_context_text,
            action_code_to_job=action_code_to_job,
            env_mode=args.env_mode,
        )
        if reflection_memory_text:
            prompt_state_text = (
                f"{prompt_state_text}\n"
                f"Episode reflection memory:\n{reflection_memory_text}\n"
                "Apply these reflection rules while selecting this step action."
            )
        if step_idx in guidance_map:
            step_guidance = dict(guidance_map[step_idx])
            preferred_job = int(step_guidance.get("preferred_job", -1))
            avoid_jobs = [int(x) for x in step_guidance.get("avoid_jobs", []) if int(x) >= 0]
            reason_text = str(step_guidance.get("reason", "")).strip()
            guide_lines = ["Post-episode guidance: This step was identified as a Cmax bottleneck."]
            if preferred_job >= 0:
                guide_lines.append(f"If feasible, prefer Job {preferred_job}.")
            if avoid_jobs:
                guide_lines.append("Avoid these jobs if strong alternatives exist: " + ", ".join(f"Job {j}" for j in avoid_jobs))
            if reason_text:
                guide_lines.append(f"Why: {reason_text}")
            prompt_state_text = f"{prompt_state_text}\n" + "\n".join(guide_lines)

        is_replay_prefix = (
            replay_action_jobs_by_step is not None
            and replay_prefix_until is not None
            and int(step_idx) < int(replay_prefix_until)
        )
        if is_replay_prefix:
            chosen_job = int(replay_action_jobs_by_step[int(step_idx)])
            if chosen_job not in feasible_jobs:
                raise RuntimeError(
                    "Prefix replay became infeasible before the guided step. "
                    f"step_idx={step_idx}, replay_job={chosen_job}, feasible={feasible_jobs}"
                )
            chosen_action_code = str(job_to_action_code[int(chosen_job)])
            debug_payload = {"replay": True, "chosen_action_code": chosen_action_code}
            initial_model_text = f"[REPLAY] {chosen_action_code}"
        else:
            prompt_text = base._build_step_chat_prompt(tokenizer, prompt_state_text)
            chosen_action_code, debug_payload = _sample_step_action(
                model=model,
                tokenizer=tokenizer,
                candidate_score_head=candidate_score_head,
                prompt_text=prompt_text,
                feasible_action_codes=feasible_action_codes,
                args=args,
                candidate_scoring_do_sample=candidate_scoring_do_sample,
                candidate_scoring_temperature=candidate_scoring_temperature,
            )
            initial_model_text = str(debug_payload)
            chosen_job = int(action_code_to_job[chosen_action_code])

        chosen_effect = effect_by_code.get(chosen_action_code)
        if chosen_effect is None:
            raise RuntimeError(
                "Internal mismatch: chosen action code is missing in step option effects. "
                f"chosen_action_code={chosen_action_code}, available={list(effect_by_code.keys())}"
            )

        rationale_text = None
        if args.emit_step_rationale:
            if is_replay_prefix:
                rationale_text = base._deterministic_step_rationale(
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                )
            else:
                rationale_prompt = base.build_reason_input_text(
                    state_text=prompt_state_text,
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                    top_k=int(args.reason_topk),
                    state_json=state,
                )
                rationale_chat_prompt = base._build_step_rationale_chat_prompt(tokenizer, rationale_prompt)
                rationale_raw = base._sample_step_rationale(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_text=rationale_chat_prompt,
                    max_new_tokens=args.step_rationale_max_new_tokens,
                )
                rationale_text = base._clean_step_rationale(
                    rationale_raw,
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                )

        _, _, _, info = env.step(chosen_job)
        not_chosen = [x for x in action_effects if str(x["action_code"]) != str(chosen_action_code)]
        step_record = {
            "step_idx": int(info["step_idx"]),
            "feasible_jobs": feasible_jobs,
            "feasible_action_codes": feasible_action_codes,
            "action_code_to_job": action_code_to_job,
            "state_text": prompt_state_text,
            "model_output": initial_model_text,
            "rationale_text": rationale_text,
            "chosen_action_code": chosen_action_code,
            "chosen_job": int(chosen_job),
            "op_idx": int(info["op_idx"]),
            "machine_id": int(info["machine_id"]),
            "chosen_start_time": int(info["start_time"]),
            "chosen_end_time": int(info["end_time"]),
            "chosen_proc_time": int(info.get("duration", chosen_effect["proc_time"])),
            "makespan_before": int(makespan_before),
            "makespan_after": int(info["makespan_so_far"]),
            "chosen_estimated_makespan_after": int(chosen_effect["estimated_makespan_after"]),
            "all_options": action_effects,
            "not_chosen_options": not_chosen,
            "guidance_applied": bool(step_idx in guidance_map),
            "decision_source": "replay" if is_replay_prefix else ("guided" if step_idx in guidance_map else "model"),
        }
        raw_step_outputs.append(step_record)
        if args.print_step_trace:
            base._print_step_trace([step_record])
        if (
            args.print_candidate_probs_during_inference
            and not is_replay_prefix
            and int(debug_payload.get("feasible_count", 0)) > 1
        ):
            print(f"[CANDIDATE PROBE step={int(step_idx) + 1}] {debug_payload}")

    event_log = env.get_event_log()
    solution_lines = ["Solution:", ""]
    for op in event_log:
        solution_lines.append(f"Job {op['job_id']} Operation {op['op_idx']}, M{op['machine_id']}")
    solution_lines.append("")
    solution_lines.append(f"Makespan: {env.get_makespan()}")
    solution_text = "\n".join(solution_lines)

    return {
        "solution_text": solution_text,
        "makespan": env.get_makespan(),
        "raw_step_outputs": raw_step_outputs,
    }


def run_step_rollout(
    model,
    tokenizer,
    candidate_score_head,
    inst_for_ortools,
    args,
    candidate_scoring_do_sample: bool | None = None,
    candidate_scoring_temperature: float | None = None,
):
    best_result = _run_single_step_rollout(
        model=model,
        tokenizer=tokenizer,
        candidate_score_head=candidate_score_head,
        inst_for_ortools=inst_for_ortools,
        args=args,
        guidance_by_step=None,
        reflection_memory_text=None,
        candidate_scoring_do_sample=candidate_scoring_do_sample,
        candidate_scoring_temperature=candidate_scoring_temperature,
    )
    base_result = copy.deepcopy(best_result)
    best_result["base_result"] = base_result
    best_result["base_makespan"] = int(base_result["makespan"])
    best_result["final_makespan"] = int(best_result["makespan"])
    best_result["improvement_delta"] = 0
    best_result["improvement_enabled"] = bool(args.enable_step_improvement)
    best_result["improved_over_base"] = False
    best_result["improvement_history"] = []

    if not args.enable_step_improvement or int(args.step_reflection_passes) <= 0:
        return best_result

    notes = []
    improvement_history = []
    if args.print_step_trace:
        print(
            f"[Episode Improvement] start: passes={int(args.step_reflection_passes)}, "
            f"topk={max(1, int(args.step_reflection_topk))}, "
            f"baseline_makespan={int(best_result['makespan'])}"
        )
    for pass_idx in range(int(args.step_reflection_passes)):
        critical_steps = base._select_top_critical_steps(
            best_result["raw_step_outputs"],
            top_k=max(1, int(args.step_reflection_topk)),
        )
        if not critical_steps:
            notes.append(f"pass {pass_idx + 1}: no critical step found")
            improvement_history.append(
                {
                    "pass_idx": int(pass_idx + 1),
                    "baseline_makespan": int(best_result["makespan"]),
                    "candidate_makespan": None,
                    "improved": False,
                    "guided_steps": [],
                    "reason": "no critical step found",
                }
            )
            break

        reflection_memory = base._build_reflection_memory(
            current_makespan=int(best_result["makespan"]),
            critical_steps=critical_steps,
        )
        guidance_map = {}
        step_stack = resolve_step_stack(base._normalize_env_mode(args.env_mode))
        for critical_step in critical_steps:
            step_idx = int(critical_step["step_idx"])
            feasible_action_codes = list(critical_step["feasible_action_codes"])
            action_code_to_job = dict(critical_step["action_code_to_job"])
            chosen_code = str(critical_step["chosen_action_code"])
            chosen_job = int(critical_step["chosen_job"])
            best_alt = base._best_alternative_option(critical_step)

            suggested_code = None
            if feasible_action_codes:
                improvement_prompt = step_stack.build_step_improvement_prompt(
                    state_text=critical_step["state_text"],
                    candidate_action_text=f"{chosen_code}",
                    feasible_jobs=feasible_action_codes,
                    reflection_memory=reflection_memory,
                    step_diagnostics=base._build_step_diagnostics(critical_step),
                )
                improvement_prompt = (
                    f"{improvement_prompt}\n"
                    f"Episode summary: final makespan={best_result['makespan']}.\n"
                    f"Target step={step_idx}, chosen={chosen_code}.\n"
                    "Select an action that reduces bottleneck risk and final makespan."
                )
                reflection_prompt = base._build_step_improvement_chat_prompt(tokenizer, improvement_prompt)
                try:
                    suggested_code, _ = _sample_step_action(
                        model=model,
                        tokenizer=tokenizer,
                        candidate_score_head=candidate_score_head,
                        prompt_text=reflection_prompt,
                        feasible_action_codes=feasible_action_codes,
                        args=args,
                    )
                except Exception as exc:
                    notes.append(f"pass {pass_idx + 1} step {step_idx}: improvement parse failed ({exc})")

            if (not suggested_code or str(suggested_code) == chosen_code) and best_alt is not None:
                alt_code = str(best_alt["action_code"])
                if alt_code in action_code_to_job and alt_code != chosen_code:
                    suggested_code = alt_code
                    notes.append(f"pass {pass_idx + 1} step {step_idx}: deterministic critic suggested {alt_code}")

            if not suggested_code:
                continue
            suggested_job = int(action_code_to_job[suggested_code])
            if suggested_job == chosen_job:
                continue

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
            notes.append(f"pass {pass_idx + 1}: no actionable guidance generated")
            improvement_history.append(
                {
                    "pass_idx": int(pass_idx + 1),
                    "baseline_makespan": int(best_result["makespan"]),
                    "candidate_makespan": None,
                    "improved": False,
                    "guided_steps": [],
                    "reason": "no actionable guidance generated",
                }
            )
            continue

        replay_action_jobs_by_step = {
            int(step["step_idx"]): int(step["chosen_job"])
            for step in best_result["raw_step_outputs"]
        }
        replay_prefix_until = min(int(step_idx) for step_idx in guidance_map.keys())
        candidate_result = _run_single_step_rollout(
            model=model,
            tokenizer=tokenizer,
            candidate_score_head=candidate_score_head,
            inst_for_ortools=inst_for_ortools,
            args=SimpleNamespace(**{**vars(args), "action_code_seed": int(args.action_code_seed) + (pass_idx + 1) * 9973}),
            guidance_by_step=guidance_map,
            reflection_memory_text=reflection_memory,
            replay_action_jobs_by_step=replay_action_jobs_by_step,
            replay_prefix_until=replay_prefix_until,
            candidate_scoring_do_sample=candidate_scoring_do_sample,
            candidate_scoring_temperature=candidate_scoring_temperature,
        )
        improvement_history.append(
            {
                "pass_idx": int(pass_idx + 1),
                "baseline_makespan": int(best_result["makespan"]),
                "candidate_makespan": int(candidate_result["makespan"]),
                "improved": int(candidate_result["makespan"]) < int(best_result["makespan"]),
                "guided_steps": sorted(int(x) for x in guidance_map.keys()),
                "replay_prefix_until": int(replay_prefix_until),
                "reflection_memory": reflection_memory,
            }
        )
        if int(candidate_result["makespan"]) < int(best_result["makespan"]):
            notes.append(
                f"pass {pass_idx + 1}: improved {best_result['makespan']} -> {candidate_result['makespan']} "
                f"with guided_steps={sorted(guidance_map.keys())}"
            )
            best_result = candidate_result
        else:
            notes.append(
                f"pass {pass_idx + 1}: no improvement ({best_result['makespan']} vs {candidate_result['makespan']})"
            )

    best_result["episode_improvement_notes"] = list(notes)
    best_result["improvement_history"] = improvement_history
    best_result["final_makespan"] = int(best_result["makespan"])
    best_result["improvement_delta"] = int(base_result["makespan"]) - int(best_result["makespan"])
    best_result["improved_over_base"] = int(best_result["makespan"]) < int(base_result["makespan"])
    return best_result


def evaluate_model_step(
    model,
    tokenizer,
    candidate_score_head,
    dataset,
    args,
    csv_filename,
    aggregate_csv_filename,
    sequence_trace_csv_filename,
):
    all_predictions = []
    all_references = []

    write_header = not os.path.exists(csv_filename)
    with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(
                [
                    "Index",
                    "Num_J",
                    "Num_M",
                    "Declared_makespan",
                    "Actual_Makespan",
                    "Gap",
                    "Total_Sampling_Attempts",
                    "Num_Feasible_Solutions",
                    "Feasible_Sol_Makespans",
                    "Validation_Message",
                    "Path",
                    "Best_Solution",
                    "Avg_Time",
                ]
            )

    sample_count = max(int(args.num_return_sequences), 1)
    aggregate_fieldnames = [
        "Index",
        "Num_J",
        "Num_M",
        "Benchmark_Makespan",
        "Gap",
        "Best_Gap",
        "Best_Makespan",
        "Mean_Makespan",
        "Var_Makespan",
        "Std_Makespan",
        "Num_Attempts",
        "Num_Feasible",
        "Feasible_Rate",
        "Path",
    ]
    for k in range(sample_count):
        aggregate_fieldnames.extend(
            [
                f"Sample{k + 1}_Feasible",
                f"Sample{k + 1}_Makespan",
                f"Sample{k + 1}_BaseMakespan",
                f"Sample{k + 1}_Improved",
                f"Sample{k + 1}_ImprovementDelta",
            ]
        )
    aggregate_write_header = not os.path.exists(aggregate_csv_filename)
    with open(aggregate_csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
        if aggregate_write_header:
            writer.writeheader()

    dataset_split = dataset["train"]
    dataset_len = len(dataset_split)
    start_index = max(0, int(args.eval_start_index or 0))
    end_index = dataset_len if args.eval_end_index is None else min(dataset_len, int(args.eval_end_index))
    if start_index >= end_index:
        print(f"No evaluation items to process: start_index={start_index}, end_index={end_index}, dataset_len={dataset_len}")
        return all_predictions, all_references
    print(f"evaluation index range: [{start_index}, {end_index}) out of {dataset_len}")

    for idx in range(start_index, end_index):
        example = dataset_split[idx]
        matrix_content = example["matrix"]
        n, m, inst_for_ortools, real_makespan = read_matrix_form_jssp(matrix_content)
        ms = real_makespan if real_makespan is not None else example.get("makespan")

        rollout_results = []
        sample_feasible = []
        sample_makespans = []
        total_time = 0.0
        for sample_idx in range(sample_count):
            sample_do_sample = bool(args.candidate_scoring_do_sample)
            if bool(args.greedy_first_solution) and sample_idx == 0:
                sample_do_sample = False
            start = time.time()
            result = run_step_rollout(
                model=model,
                tokenizer=tokenizer,
                candidate_score_head=candidate_score_head,
                inst_for_ortools=inst_for_ortools,
                args=SimpleNamespace(**{**vars(args), "action_code_seed": int(args.action_code_seed) + idx + sample_idx}),
                candidate_scoring_do_sample=sample_do_sample,
                candidate_scoring_temperature=args.candidate_scoring_temperature,
            )
            total_time += time.time() - start
            rollout_results.append(result)
            sample_feasible.append(True)
            sample_makespans.append(int(result["makespan"]))

        feasible_results = [r for r in rollout_results if r is not None]
        feasible_makespans = [int(r["makespan"]) for r in feasible_results]

        if feasible_results:
            best_idx = int(np.argmin(feasible_makespans))
            best_result = feasible_results[best_idx]
            declared_makespan = int(best_result["makespan"])
            best_solution = best_result["solution_text"]
        else:
            best_result = None
            declared_makespan = None
            best_solution = "NO_FEASIBLE_SOLUTION"

        gap = None
        if declared_makespan is not None and ms and ms > 0:
            gap = abs((declared_makespan - ms) / ms)

        all_predictions.append(best_solution)
        avg_time = total_time / sample_count
        validation_message = (
            f"Step rollout feasible with makespan {declared_makespan}"
            if declared_makespan is not None
            else "No feasible rollout"
        )

        with open(csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    idx,
                    n,
                    m,
                    declared_makespan,
                    ms,
                    gap,
                    len(rollout_results),
                    len(feasible_results),
                    feasible_makespans,
                    validation_message,
                    example.get("path", ""),
                    " ",
                    avg_time,
                ]
            )

        output_root = args.output_dir or str(Path(csv_filename).parent)
        if best_result is not None:
            saved_paths = base.save_result_csv_artifacts(
                result=best_result,
                output_dir=output_root,
                idx=idx,
                n=n,
                m=m,
                benchmark_makespan=ms,
                example_path=example.get("path", ""),
                prefix="eval",
                variant="best",
                extra_summary={"WasBest": True},
            )
            print(f"[saved csv] idx={idx}: {saved_paths['step_csv']}")

        feasible_values = [v for v in sample_makespans if v is not None]
        mean_ms = float(np.mean(feasible_values)) if feasible_values else None
        var_ms = float(np.var(feasible_values)) if feasible_values else None
        std_ms = float(np.std(feasible_values)) if feasible_values else None

        agg_row = {
            "Index": idx,
            "Num_J": int(n),
            "Num_M": int(m),
            "Benchmark_Makespan": ms,
            "Gap": gap,
            "Best_Gap": gap,
            "Best_Makespan": declared_makespan,
            "Mean_Makespan": mean_ms,
            "Var_Makespan": var_ms,
            "Std_Makespan": std_ms,
            "Num_Attempts": sample_count,
            "Num_Feasible": len(feasible_values),
            "Feasible_Rate": float(len(feasible_values) / sample_count),
            "Path": example.get("path", ""),
        }
        for k in range(sample_count):
            agg_row[f"Sample{k + 1}_Feasible"] = int(bool(sample_feasible[k]))
            agg_row[f"Sample{k + 1}_Makespan"] = sample_makespans[k]
            sample_result = rollout_results[k]
            base_makespan = None if sample_result is None else sample_result.get("base_makespan")
            agg_row[f"Sample{k + 1}_BaseMakespan"] = base_makespan
            agg_row[f"Sample{k + 1}_Improved"] = None if sample_result is None else int(bool(sample_result.get("improved_over_base", False)))
            agg_row[f"Sample{k + 1}_ImprovementDelta"] = None if sample_result is None else sample_result.get("improvement_delta")

        with open(aggregate_csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
            writer.writerow(agg_row)

        for sample_idx, result in enumerate(rollout_results):
            if result is None:
                continue
            is_best_sample = bool(best_result is not None and int(result["makespan"]) == int(best_result["makespan"]))
            sample_saved_paths = base.save_result_csv_artifacts(
                result=result,
                output_dir=output_root,
                idx=idx,
                n=n,
                m=m,
                benchmark_makespan=ms,
                example_path=example.get("path", ""),
                prefix="eval",
                sample_idx=sample_idx,
                variant="final",
                extra_summary={"WasBest": is_best_sample},
            )
            print(f"[saved csv] idx={idx} sample={sample_idx + 1}/{sample_count} final: {sample_saved_paths['step_csv']}")
            base.append_sequence_trace_rows(
                sequence_trace_csv_filename,
                base.build_sequence_trace_rows(
                    result=result,
                    idx=idx,
                    n=n,
                    m=m,
                    benchmark_makespan=ms,
                    example_path=example.get("path", ""),
                    sample_idx=sample_idx,
                    variant="final",
                    was_best=is_best_sample,
                ),
            )
            base_result = result.get("base_result")
            if base_result is not None and bool(result.get("improvement_enabled", False)):
                base.save_result_csv_artifacts(
                    result=base_result,
                    output_dir=output_root,
                    idx=idx,
                    n=n,
                    m=m,
                    benchmark_makespan=ms,
                    example_path=example.get("path", ""),
                    prefix="eval",
                    sample_idx=sample_idx,
                    variant="base",
                    extra_summary={"WasBest": False},
                )
                base.append_sequence_trace_rows(
                    sequence_trace_csv_filename,
                    base.build_sequence_trace_rows(
                        result=base_result,
                        idx=idx,
                        n=n,
                        m=m,
                        benchmark_makespan=ms,
                        example_path=example.get("path", ""),
                        sample_idx=sample_idx,
                        variant="base",
                        was_best=False,
                    ),
                )

    return all_predictions, all_references


def run_demo_instance_step(
    model,
    tokenizer,
    candidate_score_head,
    dataset,
    args,
    index: int,
    num_solutions: int,
):
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split for demo runs.")
    dataset_length = len(dataset["train"])
    if index < 0 or index >= dataset_length:
        raise ValueError(f"demo_index {index} out of range (dataset has {dataset_length} items).")

    example = dataset["train"][index]
    matrix_content = example["matrix"]
    n, m, inst_for_ortools, real_makespan = read_matrix_form_jssp(matrix_content)
    ms = real_makespan if real_makespan is not None else example.get("makespan")

    print("=" * 80)
    print(
        f"Step-demo instance {index} — Jobs={n}, Machines={m}, "
        f"EnvMode={args.env_mode}, EpisodeImprovement={args.enable_step_improvement}, "
        f"Rationale={args.emit_step_rationale}"
    )
    print("=" * 80)

    best_result = None
    best_makespan = None
    demo_results = []
    for sample_idx in range(max(num_solutions, 1)):
        sample_do_sample = bool(args.candidate_scoring_do_sample)
        if bool(args.greedy_first_solution) and sample_idx == 0:
            sample_do_sample = False
        result = run_step_rollout(
            model=model,
            tokenizer=tokenizer,
            candidate_score_head=candidate_score_head,
            inst_for_ortools=inst_for_ortools,
            args=SimpleNamespace(**{**vars(args), "action_code_seed": int(args.action_code_seed) + sample_idx}),
            candidate_scoring_do_sample=sample_do_sample,
            candidate_scoring_temperature=args.candidate_scoring_temperature,
        )
        demo_results.append(result)
        makespan = int(result["makespan"])
        print(f"\n--- Step rollout {sample_idx + 1}/{num_solutions} ---")
        print(result["solution_text"])
        print(f"makespan={makespan}")
        if not args.print_step_trace:
            base._print_step_trace(result.get("raw_step_outputs", []))

        if best_makespan is None or makespan < best_makespan:
            best_makespan = makespan
            best_result = result

    if ms:
        print(f"\nBenchmark makespan: {ms}")
        if best_makespan is not None and ms > 0:
            print(f"Best gap: {abs((best_makespan - ms) / ms):.4f}")

    if best_result is not None:
        output_root = args.output_dir or "val_results/jssp_val"
        demo_saved_paths = base.save_result_csv_artifacts(
            result=best_result,
            output_dir=output_root,
            idx=index,
            n=n,
            m=m,
            benchmark_makespan=ms,
            example_path=example.get("path", ""),
            prefix="demo",
            variant="best",
            extra_summary={"WasBest": True},
        )
        print("demo csv saved:", demo_saved_paths)
        for sample_idx, result in enumerate(demo_results):
            is_best_sample = bool(int(result["makespan"]) == int(best_result["makespan"]))
            base.save_result_csv_artifacts(
                result=result,
                output_dir=output_root,
                idx=index,
                n=n,
                m=m,
                benchmark_makespan=ms,
                example_path=example.get("path", ""),
                prefix="demo",
                sample_idx=sample_idx,
                variant="final",
                extra_summary={"WasBest": is_best_sample},
            )
            base_result = result.get("base_result")
            if base_result is not None and bool(result.get("improvement_enabled", False)):
                base.save_result_csv_artifacts(
                    result=base_result,
                    output_dir=output_root,
                    idx=index,
                    n=n,
                    m=m,
                    benchmark_makespan=ms,
                    example_path=example.get("path", ""),
                    prefix="demo",
                    sample_idx=sample_idx,
                    variant="base",
                    extra_summary={"WasBest": False},
                )

    return best_result


def main():
    args = _build_parser().parse_args()
    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)
        print("HF login ready")

    model_path = _resolve_model_source(
        args.model_repo_or_path,
        local_path=args.model_path,
        checkpoint_tag=args.model_checkpoint_tag,
        token=args.hf_token or None,
    )
    print("loading model from:", model_path)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    base_load_name = MODEL_BASE[args.model_type] if _is_adapter_source(model_path, token=args.hf_token or None) else model_path
    print("base load path:", base_load_name)
    model, tokenizer = _load_unsloth_model_with_chat_template_fallback(
        model_name=base_load_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=dtype,
        local_files_only=False,
    )
    action_token_install = ensure_action_special_tokens(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
    )
    print("action token install:", action_token_install)
    candidate_score_token_install = ensure_candidate_score_token(
        tokenizer=tokenizer,
        model=model,
        token=args.candidate_score_token,
    )
    print("candidate score token id:", candidate_score_token_install["token_id"])
    if _is_adapter_source(model_path, token=args.hf_token or None):
        _maybe_load_policy_adapter(model, model_path, token=args.hf_token or None)
        print("policy adapter loaded:", model_path)
    FastLanguageModel.for_inference(model)
    scorer_path = _resolve_candidate_scorer_path(model_path, token=args.hf_token or None)
    if scorer_path is None:
        raise FileNotFoundError(
            f"candidate_scorer.pt not found for model_path={model_path}. "
            "Use a checkpoint/final dir produced by the action-code candidate-scoring trainer."
        )
    model_device = infer_module_device(model)
    hidden_size = int(getattr(model.config, "hidden_size"))
    backbone_param = next(model.parameters())
    candidate_score_head, _ = load_candidate_score_head(
        scorer_path=scorer_path,
        hidden_size=hidden_size,
        device=model_device,
        dtype=backbone_param.dtype,
    )
    print("candidate score head loaded:", scorer_path)

    eval_dataset = _load_eval_dataset(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_base_name = os.path.basename(str(model_path)) if model_path else "model"
    output_style = "accord" if args.output_accord else "list_of_lists"
    csv_filename = output_dir / f"output_step_{output_style}_{model_base_name}_num_return_sequences{args.num_return_sequences}.csv"
    aggregate_csv_filename = output_dir / f"evaluate_all_problems_{model_base_name}_num_return_sequences{args.num_return_sequences}.csv"
    sequence_trace_csv_filename = output_dir / f"evaluate_sequence_trace_{model_base_name}_num_return_sequences{args.num_return_sequences}.csv"

    print("step-by-step inference start")
    print("masking enabled:", not args.disable_masking)
    print("csv path:", csv_filename)
    print("aggregate csv path:", aggregate_csv_filename)
    print("sequence trace csv path:", sequence_trace_csv_filename)

    if args.demo_index is not None:
        demo_num_solutions = args.demo_num_solutions if args.demo_num_solutions is not None else args.num_return_sequences
        run_demo_instance_step(
            model=model,
            tokenizer=tokenizer,
            candidate_score_head=candidate_score_head,
            dataset=eval_dataset,
            args=args,
            index=int(args.demo_index),
            num_solutions=int(demo_num_solutions),
        )
        return

    evaluate_model_step(
        model=model,
        tokenizer=tokenizer,
        candidate_score_head=candidate_score_head,
        dataset=eval_dataset,
        args=args,
        csv_filename=str(csv_filename),
        aggregate_csv_filename=str(aggregate_csv_filename),
        sequence_trace_csv_filename=str(sequence_trace_csv_filename),
    )
    print("inference complete")
    print("csv saved:", csv_filename)
    print("aggregate csv saved:", aggregate_csv_filename)
    print("sequence trace csv saved:", sequence_trace_csv_filename)


if __name__ == "__main__":
    main()
