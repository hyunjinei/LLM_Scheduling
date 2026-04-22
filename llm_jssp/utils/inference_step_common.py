"""Shared inference helpers used by the canonical candidate-scoring runner."""

import copy
import re
import random
import os
import torch
from datasets import load_dataset
import csv
from tqdm import tqdm
import numpy as np
import time
from pathlib import Path

from llm_jssp.utils.solution_generation_english import read_matrix_form_jssp
from llm_jssp.utils.jssp_step_env import StaticJSSPStepEnv
from llm_jssp.utils.jssp_dispatch_env import DispatchJSSPStepEnv
from llm_jssp.utils.jssp_masking_hooks import build_prefix_allowed_tokens_fn_from_instance
from llm_jssp.utils.step_prompting import (
    build_problem_context_text,
    build_randomized_action_code_map,
    build_step_improvement_prompt,
    build_step_prompt,
    compute_action_transition_features,
    invert_action_code_map,
)
from llm_jssp.utils.step_prompting_dispatch import (
    build_step_prompt as build_dispatch_step_prompt,
    compute_action_transition_features as compute_dispatch_action_transition_features,
)
from llm_jssp.utils.step_reasoning import build_reason_input_text
from llm_jssp.utils.action_token_utils import token_id_to_action_code
from llm_jssp.utils.jssp_step_masking_hooks import (
    build_step_prefix_allowed_tokens_fn,
    StepActionParseError,
)

def _format_route_tokens(route_tokens) -> str:
    if not route_tokens:
        return "[]"
    return "[" + ", ".join(str(token) for token in route_tokens) + "]"


def _load_eval_dataset(run_args):
    if run_args.eval_data_path:
        eval_data_path = os.path.expanduser(run_args.eval_data_path)
        print(f"Using custom eval dataset: {eval_data_path}")
        return load_dataset("json", data_files=eval_data_path)
    if run_args.infer_fssp:
        return load_dataset("json", data_files="validation_data/fssp_val_data.json")
    return load_dataset("json", data_files="validation_data/ta.json")


def _build_step_chat_prompt(tokenizer, state_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert JSSP scheduler. "
                "Primary objective: minimize final makespan (Cmax). "
                "Choose exactly one feasible action token for the current step. "
                "Output exactly one token such as <A3812> and nothing else."
            ),
        },
        {
            "role": "user",
            "content": state_text,
        },
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
        {
            "role": "user",
            "content": improvement_prompt_text,
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _build_step_rationale_chat_prompt(tokenizer, rationale_prompt_text: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You explain a fixed scheduling decision. "
                "Focus on makespan minimization (Cmax). "
                "Do not output a new action. Follow the requested explanation format only."
            ),
        },
        {
            "role": "user",
            "content": rationale_prompt_text,
        },
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def _activate_adapter_if_available(model, adapter_name: str):
    if hasattr(model, "set_adapter"):
        model.set_adapter(adapter_name)


def _maybe_load_reason_adapter(model, reason_model_path: str | None):
    if not reason_model_path:
        return False
    if not hasattr(model, "load_adapter"):
        raise RuntimeError(
            "Reason adapter path was provided, but the loaded model does not support load_adapter()."
        )
    model.load_adapter(reason_model_path, adapter_name="reason")
    _activate_adapter_if_available(model, "default")
    return True


def _sample_step_action(
    model,
    tokenizer,
    prompt_text: str,
    feasible_action_codes,
    use_masking: bool = True,
    max_new_tokens: int = 1,
    action_code_width: int = 4,
    action_code_cap: int = 9999,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    generation_kwargs = {
        "max_new_tokens": 1 if use_masking else max_new_tokens,
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.95,
        "num_return_sequences": 1,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if use_masking:
        generation_kwargs["prefix_allowed_tokens_fn"] = build_step_prefix_allowed_tokens_fn(
            tokenizer=tokenizer,
            feasible_action_codes_provider=list(feasible_action_codes),
            code_width=action_code_width,
            code_cap=action_code_cap,
        )

    outputs = model.generate(**inputs, **generation_kwargs)
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]

    if not feasible_action_codes:
        raise StepActionParseError("No feasible action codes available at this step.")
    if generated_ids.numel() <= 0:
        raise StepActionParseError("No action token was generated.")

    chosen_action_code = token_id_to_action_code(
        tokenizer,
        int(generated_ids[0].item()),
        code_width=action_code_width,
    )
    if chosen_action_code is None:
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
        raise StepActionParseError(
            f"Failed to parse generated token into action token. output={generated_text!r}"
        )
    generated_text = str(chosen_action_code)

    if str(chosen_action_code) not in feasible_action_codes:
        raise StepActionParseError(
            "Parsed action code is not feasible. "
            f"parsed={chosen_action_code}, feasible={list(feasible_action_codes)}, output={generated_text!r}"
        )

    return str(chosen_action_code), generated_text


def _sample_step_rationale(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 96,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer([prompt_text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_k=40,
        top_p=0.9,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = inputs["input_ids"].shape[1]
    generated_ids = outputs[0][prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return generated_text


def _deterministic_step_rationale(
    chosen_action_code: str,
    chosen_effect: dict,
    action_effects: list,
) -> str:
    chosen_ms = int(chosen_effect["estimated_makespan_after"])
    best_alt = None
    for opt in action_effects:
        if str(opt.get("action_code")) == str(chosen_action_code):
            continue
        ms = int(opt.get("estimated_makespan_after", chosen_ms))
        if best_alt is None or ms < best_alt:
            best_alt = ms
    if best_alt is None:
        compare_text = "no alternative feasible option at this step."
    else:
        gap = int(chosen_ms - best_alt)
        if gap <= 0:
            compare_text = "best estimated Cmax among feasible options."
        else:
            compare_text = f"estimated Cmax is +{gap} versus the best alternative."
    return (
        f"{chosen_action_code}: feasible on M{int(chosen_effect['machine_id'])} "
        f"(est_start={int(chosen_effect['estimated_start'])}, est_end={int(chosen_effect['estimated_end'])}, "
        f"proc={int(chosen_effect['proc_time'])}), "
        f"Cmax {int(chosen_effect['current_cmax_before'])}->{int(chosen_effect['current_cmax_after'])} "
        f"(delta={int(chosen_effect['delta_cmax'])}), "
        f"machine_idle_gap={int(chosen_effect['machine_idle_gap'])}; {compare_text}"
    )


def _clean_step_rationale(
    raw_text: str,
    chosen_action_code: str,
    chosen_effect: dict,
    action_effects: list,
) -> str:
    text = (raw_text or "").strip()
    if not text:
        return _deterministic_step_rationale(
            chosen_action_code, chosen_effect, action_effects
        )

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln for ln in lines if not ln.lower().startswith("action:")]

    reason = None
    for ln in lines:
        if ln.lower().startswith("reason:"):
            reason = ln.split(":", 1)[1].strip()
            break

    if reason is None:
        for ln in lines:
            low = ln.lower()
            if low.startswith("not chosen"):
                continue
            if low.startswith("- "):
                continue
            reason = ln
            break

    if not reason:
        return _deterministic_step_rationale(
            chosen_action_code, chosen_effect, action_effects
        )

    if "Not chosen:" in reason:
        reason = reason.split("Not chosen:", 1)[0].strip()

    found_codes = re.findall(r"<\s*[aA]\s*\d+\s*>", reason)
    normalized_codes = {re.sub(r"\s+", "", code).upper() for code in found_codes}
    chosen_norm = re.sub(r"\s+", "", str(chosen_action_code)).upper()
    if any(code != chosen_norm for code in normalized_codes):
        return _deterministic_step_rationale(
            chosen_action_code, chosen_effect, action_effects
        )

    reason = re.sub(r"<\s*[aA]\s*\d+\s*>", str(chosen_action_code), reason)
    if len(reason) > 220:
        reason = reason[:217].rstrip() + "..."
    return f"{chosen_action_code}: {reason}"


def _normalize_env_mode(env_mode: str) -> str:
    resolved = str(env_mode).lower()
    if resolved not in {"serial", "dispatch"}:
        raise ValueError(f"Unsupported env_mode={env_mode}")
    return resolved


def _make_step_env(inst_for_ortools, env_mode: str):
    if _normalize_env_mode(env_mode) == "dispatch":
        return DispatchJSSPStepEnv(inst_for_ortools)
    return StaticJSSPStepEnv(inst_for_ortools)


def _estimate_action_effects(state_json, action_code_to_job, env_mode: str = "serial"):
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
    env_mode: str = "serial",
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


def _print_step_trace(raw_step_outputs):
    for step in raw_step_outputs:
        chosen = (
            f"{step['chosen_action_code']} -> Job {step['chosen_job']} "
            f"(M{step['machine_id']}, t={step['chosen_proc_time']}, "
            f"start={step['chosen_start_time']}, end={step['chosen_end_time']})"
        )
        print(
            f"[Step {int(step['step_idx']):03d}] "
            f"makespan {int(step['makespan_before'])} -> {int(step['makespan_after'])} | chosen: {chosen}"
        )
        rationale = step.get("rationale_text")
        if rationale:
            print(f"  reason: {rationale}")
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


def _select_top_critical_steps(raw_step_outputs, top_k: int = 3):
    scored = []
    for step in raw_step_outputs:
        score = _critical_step_score(step)
        if score is None:
            continue
        scored.append((score, step))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [step for _, step in scored[: max(1, int(top_k))]]


def _select_critical_step(raw_step_outputs):
    top = _select_top_critical_steps(raw_step_outputs, top_k=1)
    return top[0] if top else None


def _build_step_diagnostics(step):
    chosen_code = str(step.get("chosen_action_code"))
    chosen_job = int(step.get("chosen_job", -1))
    chosen_ms = int(step.get("chosen_estimated_makespan_after", step.get("makespan_after", 0)))
    chosen_start = int(step.get("chosen_start_time", 0))
    chosen_machine = int(step.get("machine_id", -1))
    chosen_end = int(step.get("chosen_end_time", step.get("makespan_after", 0)))
    chosen_effect = None
    alternatives = [
        opt
        for opt in step.get("all_options", [])
        if str(opt.get("action_code")) != chosen_code
    ]
    for opt in step.get("all_options", []):
        if str(opt.get("action_code")) == chosen_code:
            chosen_effect = opt
            break
    alternatives = sorted(
        alternatives,
        key=lambda x: (
            int(x.get("estimated_makespan_after", 10**12)),
            int(x.get("estimated_start", 10**12)),
        ),
    )
    lines = [
        (
            f"Chosen={chosen_code}/Job{chosen_job}, machine=M{chosen_machine}, "
            f"start={chosen_start}, end={chosen_end}, est_Cmax_after={chosen_ms}"
        )
    ]
    if chosen_effect is not None:
        chosen_post_route = _format_route_tokens(chosen_effect.get("post_route_tokens", []))
        lines.append(
            (
                "Chosen details: "
                f"delta_cmax={int(chosen_effect.get('delta_cmax', 0))}, "
                f"job_wait={int(chosen_effect.get('job_wait', 0))}, "
                f"machine_idle_gap={int(chosen_effect.get('machine_idle_gap', 0))}, "
                f"post_route={chosen_post_route}, "
                f"machine_load={int(chosen_effect.get('affected_machine_load', 0))}"
            )
        )
    for rank, opt in enumerate(alternatives[:3], start=1):
        alt_post_route = _format_route_tokens(opt.get("post_route_tokens", []))
        lines.append(
            (
                f"Alt{rank}={opt['action_code']}/Job{int(opt['job_id'])}, "
                f"est_Cmax_after={int(opt['estimated_makespan_after'])}, "
                f"est_start={int(opt['estimated_start'])}, "
                f"proc={int(opt['proc_time'])}, "
                f"delta_cmax={int(opt.get('delta_cmax', 0))}, "
                f"idle_gap={int(opt.get('machine_idle_gap', 0))}, "
                f"job_wait={int(opt.get('job_wait', 0))}, "
                f"post_route={alt_post_route}, "
                f"machine_load={int(opt.get('affected_machine_load', 0))}"
            )
        )
    return "\n".join(lines)


def _build_reflection_memory(current_makespan: int, critical_steps):
    lines = [f"Current episode makespan={int(current_makespan)}."]
    if not critical_steps:
        lines.append("No critical-step hindsight available.")
        return "\n".join(lines)

    increase_count = sum(
        1 for step in critical_steps
        if int(step.get("makespan_after", 0)) > int(step.get("makespan_before", 0))
    )
    bottleneck_machine_counts = {}
    for step in critical_steps:
        for opt in step.get("all_options", []):
            machine_id = int(opt.get("machine_id", -1))
            bottleneck_machine_counts[machine_id] = bottleneck_machine_counts.get(machine_id, 0) + 1
    dominant_machine = None
    if bottleneck_machine_counts:
        dominant_machine = max(
            bottleneck_machine_counts.items(),
            key=lambda x: (x[1], -x[0]),
        )[0]

    lines.extend(
        [
            (
                "Episode postmortem: the schedule likely lost quality through one or more of the "
                "following patterns: delaying bottleneck activation, accepting larger idle/wait gaps "
                "without structural payoff, or choosing weaker post-routes when immediate Cmax "
                "signals were close."
            ),
            (
                f"Critical-step count={len(critical_steps)}, "
                f"makespan-increase steps among them={increase_count}."
            ),
        ]
    )
    if dominant_machine is not None and dominant_machine >= 0:
        lines.append(
            f"Dominant machine appearing in critical alternatives: M{dominant_machine}. Treat this as a likely bottleneck focus."
        )
    lines.extend(
        [
            "Reflection rules:",
            "1. Prefer actions that activate or release the bottleneck route earlier.",
            "2. If immediate projected Cmax is tied or close, prefer lower regret in post-route progression.",
            "3. Avoid larger machine idle gaps or waits unless they clearly unlock stronger bottleneck progress.",
            "4. Do not repeat previously identified bad choices when a strong feasible alternative exists.",
        ]
    )
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
        alt_delta = int(best_alt.get("delta_cmax", 0))
        alt_idle_gap = int(best_alt.get("machine_idle_gap", 0))
        alt_job_wait = int(best_alt.get("job_wait", 0))
        alt_post_route = _format_route_tokens(best_alt.get("post_route_tokens", []))
        chosen_idle_gap = int(
            next(
                (
                    opt.get("machine_idle_gap", 0)
                    for opt in step.get("all_options", [])
                    if str(opt.get("action_code")) == chosen_code
                ),
                0,
            )
        )
        chosen_job_wait = int(
            next(
                (
                    opt.get("job_wait", 0)
                    for opt in step.get("all_options", [])
                    if str(opt.get("action_code")) == chosen_code
                ),
                0,
            )
        )
        lines.append(
            (
                f"Rule {rank}: step {int(step.get('step_idx', -1))}. "
                f"Previous choice {chosen_code}/Job{chosen_job} had est_Cmax={chosen_ms}, "
                f"start={chosen_start}, idle_gap={chosen_idle_gap}, job_wait={chosen_job_wait}. "
                f"Prefer {alt_code}/Job{alt_job} instead when feasible because it gives "
                f"est_Cmax={alt_ms}, start={alt_start}, delta_cmax={alt_delta}, "
                f"idle_gap={alt_idle_gap}, job_wait={alt_job_wait}, and post_route={alt_post_route}."
            )
        )
    return "\n".join(lines)


def _run_single_step_rollout(
    model,
    tokenizer,
    inst_for_ortools,
    env_mode: str = "serial",
    use_masking: bool = True,
    step_action_max_new_tokens: int = 1,
    include_problem_context: bool = True,
    enable_step_improvement: bool = False,
    step_reflection_passes: int = 1,
    emit_step_rationale: bool = False,
    step_rationale_max_new_tokens: int = 96,
    reason_topk: int = 3,
    use_reason_adapter: bool = False,
    action_code_width: int = 4,
    action_code_seed: int = 42,
    action_code_cap: int = 9999,
    guidance_by_step=None,
    reflection_memory_text: str | None = None,
    replay_action_jobs_by_step=None,
    replay_prefix_until: int | None = None,
    print_step_trace: bool = False,
):
    env = _make_step_env(inst_for_ortools, env_mode)
    env.reset()
    raw_step_outputs = []
    problem_context_text = (
        build_problem_context_text(inst_for_ortools) if include_problem_context else None
    )
    action_rng = random.Random(int(action_code_seed))
    guidance_map = guidance_by_step or {}

    while not env.is_done():
        state = env.get_state_json()
        feasible_jobs = list(state["feasible_jobs"])
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
        job_to_action_code = invert_action_code_map(action_code_to_job)
        feasible_action_codes = list(action_code_to_job.keys())
        prompt_state_text = _build_state_text(
            state_json=state,
            feasible_jobs=feasible_jobs,
            step_idx=step_idx,
            total_steps=state["total_steps"],
            problem_context_text=problem_context_text,
            action_code_to_job=action_code_to_job,
            env_mode=env_mode,
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
            initial_model_text = f"[REPLAY] {chosen_action_code}"
        else:
            prompt_text = _build_step_chat_prompt(tokenizer, prompt_state_text)
            if use_reason_adapter:
                _activate_adapter_if_available(model, "default")
            chosen_action_code, initial_model_text = _sample_step_action(
                model=model,
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                feasible_action_codes=feasible_action_codes,
                use_masking=use_masking,
                max_new_tokens=step_action_max_new_tokens,
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

        rationale_text = None
        if emit_step_rationale:
            if is_replay_prefix:
                rationale_text = _deterministic_step_rationale(
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                )
            else:
                rationale_prompt = build_reason_input_text(
                    state_text=prompt_state_text,
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                    top_k=int(reason_topk),
                    state_json=state,
                )
                rationale_chat_prompt = _build_step_rationale_chat_prompt(
                    tokenizer, rationale_prompt
                )
                try:
                    if use_reason_adapter:
                        _activate_adapter_if_available(model, "reason")
                    rationale_raw = _sample_step_rationale(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=rationale_chat_prompt,
                        max_new_tokens=step_rationale_max_new_tokens,
                    )
                finally:
                    if use_reason_adapter:
                        _activate_adapter_if_available(model, "default")
                rationale_text = _clean_step_rationale(
                    rationale_raw,
                    chosen_action_code=chosen_action_code,
                    chosen_effect=chosen_effect,
                    action_effects=action_effects,
                )

        _, _, _, info = env.step(chosen_job)
        not_chosen = [
            x for x in action_effects
            if str(x["action_code"]) != str(chosen_action_code)
        ]
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
            "decision_source": (
                "replay" if is_replay_prefix else ("guided" if step_idx in guidance_map else "model")
            ),
        }
        raw_step_outputs.append(
            step_record
        )
        if print_step_trace:
            _print_step_trace([step_record])

    event_log = env.get_event_log()
    solution_lines = ["Solution:", ""]
    for op in event_log:
        solution_lines.append(
            f"Job {op['job_id']} Operation {op['op_idx']}, M{op['machine_id']}"
        )
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
    inst_for_ortools,
    env_mode: str = "serial",
    use_masking: bool = True,
    step_action_max_new_tokens: int = 1,
    include_problem_context: bool = True,
    enable_step_improvement: bool = False,
    step_reflection_passes: int = 1,
    step_reflection_topk: int = 3,
    emit_step_rationale: bool = False,
    step_rationale_max_new_tokens: int = 96,
    reason_topk: int = 3,
    use_reason_adapter: bool = False,
    action_code_width: int = 4,
    action_code_seed: int = 42,
    action_code_cap: int = 9999,
    print_step_trace: bool = False,
):
    best_result = _run_single_step_rollout(
        model=model,
        tokenizer=tokenizer,
        inst_for_ortools=inst_for_ortools,
        env_mode=env_mode,
        use_masking=use_masking,
        step_action_max_new_tokens=step_action_max_new_tokens,
        include_problem_context=include_problem_context,
        enable_step_improvement=False,
        step_reflection_passes=0,
        emit_step_rationale=emit_step_rationale,
        step_rationale_max_new_tokens=step_rationale_max_new_tokens,
        reason_topk=reason_topk,
        use_reason_adapter=use_reason_adapter,
        action_code_width=action_code_width,
        action_code_seed=action_code_seed,
        action_code_cap=action_code_cap,
        guidance_by_step=None,
        reflection_memory_text=None,
        print_step_trace=print_step_trace,
    )
    base_result = copy.deepcopy(best_result)
    best_result["base_result"] = base_result
    best_result["base_makespan"] = int(base_result["makespan"])
    best_result["final_makespan"] = int(best_result["makespan"])
    best_result["improvement_delta"] = 0
    best_result["improvement_enabled"] = bool(enable_step_improvement)
    best_result["improved_over_base"] = False
    best_result["improvement_history"] = []

    if not enable_step_improvement or int(step_reflection_passes) <= 0:
        return best_result

    notes = []
    improvement_history = []
    if print_step_trace:
        print(
            f"[Episode Improvement] start: passes={int(step_reflection_passes)}, "
            f"topk={max(1, int(step_reflection_topk))}, "
            f"baseline_makespan={int(best_result['makespan'])}"
        )
    for pass_idx in range(int(step_reflection_passes)):
        critical_steps = _select_top_critical_steps(
            best_result["raw_step_outputs"],
            top_k=max(1, int(step_reflection_topk)),
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
            if print_step_trace:
                print(
                    f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                    "no critical step found"
                )
            break

        reflection_memory = _build_reflection_memory(
            current_makespan=int(best_result["makespan"]),
            critical_steps=critical_steps,
        )
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
                    f"Episode summary: final makespan={best_result['makespan']}.\n"
                    f"Target step={step_idx}, chosen={chosen_code}.\n"
                    "Select an action that reduces bottleneck risk and final makespan."
                )
                reflection_prompt = _build_step_improvement_chat_prompt(
                    tokenizer, improvement_prompt
                )
                try:
                    suggested_code, _ = _sample_step_action(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=reflection_prompt,
                        feasible_action_codes=feasible_action_codes,
                        use_masking=use_masking,
                        max_new_tokens=step_action_max_new_tokens,
                        action_code_width=action_code_width,
                        action_code_cap=action_code_cap,
                    )
                except StepActionParseError as exc:
                    notes.append(
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
                    notes.append(
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
            if print_step_trace:
                print(
                    f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                    "no actionable guidance generated"
                )
            continue

        replay_action_jobs_by_step = {
            int(step["step_idx"]): int(step["chosen_job"])
            for step in best_result["raw_step_outputs"]
        }
        replay_prefix_until = min(int(step_idx) for step_idx in guidance_map.keys())
        if print_step_trace:
            print(
                f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                f"replay_prefix_until={int(replay_prefix_until)} "
                "(prefix actions replayed, tail regenerated)"
            )

        candidate_result = _run_single_step_rollout(
            model=model,
            tokenizer=tokenizer,
            inst_for_ortools=inst_for_ortools,
            env_mode=env_mode,
            use_masking=use_masking,
            step_action_max_new_tokens=step_action_max_new_tokens,
            include_problem_context=include_problem_context,
            enable_step_improvement=False,
            step_reflection_passes=0,
            emit_step_rationale=emit_step_rationale,
            step_rationale_max_new_tokens=step_rationale_max_new_tokens,
            reason_topk=reason_topk,
            use_reason_adapter=use_reason_adapter,
            action_code_width=action_code_width,
            action_code_seed=action_code_seed + (pass_idx + 1) * 9973,
            action_code_cap=action_code_cap,
            guidance_by_step=guidance_map,
            reflection_memory_text=reflection_memory,
            replay_action_jobs_by_step=replay_action_jobs_by_step,
            replay_prefix_until=replay_prefix_until,
            print_step_trace=False,
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
            if print_step_trace:
                print(
                    f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                    f"improved {int(best_result['makespan'])} -> {int(candidate_result['makespan'])}"
                )
            best_result = candidate_result
        else:
            notes.append(
                f"pass {pass_idx + 1}: no improvement ({best_result['makespan']} vs {candidate_result['makespan']})"
            )
            if print_step_trace:
                print(
                    f"[Episode Improvement] pass {pass_idx + 1}/{int(step_reflection_passes)}: "
                    f"no improvement ({int(best_result['makespan'])} vs {int(candidate_result['makespan'])})"
                )

    best_result["base_result"] = base_result
    best_result["base_makespan"] = int(base_result["makespan"])
    best_result["final_makespan"] = int(best_result["makespan"])
    best_result["improvement_delta"] = int(base_result["makespan"]) - int(best_result["makespan"])
    best_result["improved_over_base"] = int(best_result["makespan"]) < int(base_result["makespan"])
    best_result["improvement_enabled"] = True
    best_result["episode_improvement_notes"] = notes
    best_result["improvement_history"] = improvement_history
    if print_step_trace:
        _print_step_trace(best_result["raw_step_outputs"])
        if notes:
            print("episode_improvement_notes:")
            for x in notes:
                print(" -", x)
    return best_result


def build_machine_log_from_step_outputs(raw_step_outputs):
    rows = []
    for step in raw_step_outputs or []:
        st = int(step.get("chosen_start_time", 0))
        ed = int(step.get("chosen_end_time", st))
        rows.append(
            {
                "Machine": int(step.get("machine_id", -1)),
                "Start": st,
                "End": ed,
                "Delta": int(max(0, ed - st)),
                "JobNum": int(step.get("chosen_job", -1)),
                "Operation": int(step.get("op_idx", -1)),
                "ActionCode": str(step.get("chosen_action_code", "")),
                "Reason": str(step.get("rationale_text", "")),
            }
        )
    return rows


def save_result_csv_artifacts(
    result,
    output_dir,
    idx,
    n,
    m,
    benchmark_makespan=None,
    example_path="",
    prefix="eval",
    sample_idx=None,
    variant="final",
    extra_summary=None,
):
    artifact_dir = Path(output_dir) / "per_instance_csv"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    stem_parts = [f"{prefix}_{int(idx):04d}"]
    if sample_idx is not None:
        stem_parts.append(f"sample_{int(sample_idx) + 1:02d}")
    if variant:
        stem_parts.append(str(variant))
    stem = "_".join(stem_parts)
    step_csv_path = artifact_dir / f"{stem}_steps.csv"
    machine_csv_path = artifact_dir / f"{stem}_machine_log.csv"
    summary_csv_path = artifact_dir / f"{stem}_summary.csv"

    raw_steps = list(result.get("raw_step_outputs", []) or [])
    step_fields = sorted({k for row in raw_steps for k in row.keys()})
    with open(step_csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        if step_fields:
            w = csv.DictWriter(f, fieldnames=step_fields)
            w.writeheader()
            for row in raw_steps:
                w.writerow({k: row.get(k) for k in step_fields})

    machine_rows = build_machine_log_from_step_outputs(raw_steps)
    machine_fields = [
        "Machine",
        "Start",
        "End",
        "Delta",
        "JobNum",
        "Operation",
        "ActionCode",
        "Reason",
    ]
    with open(machine_csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=machine_fields)
        w.writeheader()
        for row in machine_rows:
            w.writerow(row)

    makespan = int(result.get("makespan", 0))
    gap = None
    if benchmark_makespan not in (None, 0):
        gap = abs((makespan - float(benchmark_makespan)) / float(benchmark_makespan))

    summary_fields = [
        "Index",
        "Num_J",
        "Num_M",
        "SampleIndex",
        "Variant",
        "Actual_Makespan",
        "Benchmark_Makespan",
        "Gap",
        "Num_Steps",
        "ImprovementEnabled",
        "ImprovedOverBase",
        "Base_Makespan",
        "Final_Makespan",
        "ImprovementDelta",
        "WasBest",
        "Path",
        "Prefix",
    ]
    if extra_summary:
        for key in extra_summary.keys():
            if key not in summary_fields:
                summary_fields.append(key)
    with open(summary_csv_path, mode="w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        summary_row = {
            "Index": int(idx),
            "Num_J": int(n),
            "Num_M": int(m),
            "SampleIndex": None if sample_idx is None else int(sample_idx),
            "Variant": str(variant),
            "Actual_Makespan": makespan,
            "Benchmark_Makespan": benchmark_makespan,
            "Gap": gap,
            "Num_Steps": len(raw_steps),
            "ImprovementEnabled": int(bool(result.get("improvement_enabled", False))),
            "ImprovedOverBase": int(bool(result.get("improved_over_base", False))),
            "Base_Makespan": result.get("base_makespan"),
            "Final_Makespan": result.get("final_makespan", makespan),
            "ImprovementDelta": result.get("improvement_delta"),
            "WasBest": int(bool((extra_summary or {}).get("WasBest", False))),
            "Path": example_path,
            "Prefix": prefix,
        }
        if extra_summary:
            summary_row.update(extra_summary)
        w.writerow(summary_row)

    return {
        "step_csv": str(step_csv_path),
        "machine_csv": str(machine_csv_path),
        "summary_csv": str(summary_csv_path),
    }


def build_sequence_trace_rows(
    result,
    idx,
    n,
    m,
    benchmark_makespan=None,
    example_path="",
    sample_idx=None,
    variant="final",
    was_best=False,
):
    rows = []
    raw_steps = list(result.get("raw_step_outputs", []) or [])
    for step in raw_steps:
        rows.append(
            {
                "Index": int(idx),
                "SampleIndex": None if sample_idx is None else int(sample_idx),
                "Variant": str(variant),
                "WasBest": int(bool(was_best)),
                "Num_J": int(n),
                "Num_M": int(m),
                "Benchmark_Makespan": benchmark_makespan,
                "Actual_Makespan": int(result.get("makespan", 0)),
                "ImprovementEnabled": int(bool(result.get("improvement_enabled", False))),
                "ImprovedOverBase": int(bool(result.get("improved_over_base", False))),
                "Base_Makespan": result.get("base_makespan"),
                "Final_Makespan": result.get("final_makespan", result.get("makespan")),
                "ImprovementDelta": result.get("improvement_delta"),
                "Path": example_path,
                "StepIdx": int(step.get("step_idx", -1)),
                "DecisionSource": str(step.get("decision_source", "")),
                "GuidanceApplied": int(bool(step.get("guidance_applied", False))),
                "ChosenAction": str(step.get("chosen_action_code", "")),
                "ChosenJob": int(step.get("chosen_job", -1)),
                "Operation": int(step.get("op_idx", -1)),
                "Machine": int(step.get("machine_id", -1)),
                "Start": int(step.get("chosen_start_time", 0)),
                "End": int(step.get("chosen_end_time", 0)),
                "ProcTime": int(step.get("chosen_proc_time", 0)),
                "MakespanBefore": int(step.get("makespan_before", 0)),
                "MakespanAfter": int(step.get("makespan_after", 0)),
                "EstimatedMakespanAfter": int(step.get("chosen_estimated_makespan_after", 0)),
                "NumFeasibleActions": len(list(step.get("feasible_action_codes", []) or [])),
                "ModelOutput": str(step.get("model_output", "")),
                "Rationale": str(step.get("rationale_text", "")),
            }
        )
    return rows


def append_sequence_trace_rows(sequence_trace_csv_filename, rows):
    if not sequence_trace_csv_filename or not rows:
        return
    fieldnames = [
        "Index",
        "SampleIndex",
        "Variant",
        "WasBest",
        "Num_J",
        "Num_M",
        "Benchmark_Makespan",
        "Actual_Makespan",
        "ImprovementEnabled",
        "ImprovedOverBase",
        "Base_Makespan",
        "Final_Makespan",
        "ImprovementDelta",
        "Path",
        "StepIdx",
        "DecisionSource",
        "GuidanceApplied",
        "ChosenAction",
        "ChosenJob",
        "Operation",
        "Machine",
        "Start",
        "End",
        "ProcTime",
        "MakespanBefore",
        "MakespanAfter",
        "EstimatedMakespanAfter",
        "NumFeasibleActions",
        "ModelOutput",
        "Rationale",
    ]
    write_header = not os.path.exists(sequence_trace_csv_filename)
    with open(sequence_trace_csv_filename, mode="a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def evaluate_model_step(
    model,
    tokenizer,
    dataset,
    run_args,
    csv_filename="evaluation_results_step.csv",
    aggregate_csv_filename=None,
    sequence_trace_csv_filename=None,
    num_return_sequences=1,
    use_masking=True,
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

    if aggregate_csv_filename is None:
        aggregate_csv_filename = str(
            Path(csv_filename).with_name("evaluate_all_problems.csv")
        )
    sample_count = max(int(num_return_sequences), 1)
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
        aggregate_fieldnames.append(f"Sample{k + 1}_Feasible")
        aggregate_fieldnames.append(f"Sample{k + 1}_Makespan")
        aggregate_fieldnames.append(f"Sample{k + 1}_BaseMakespan")
        aggregate_fieldnames.append(f"Sample{k + 1}_Improved")
        aggregate_fieldnames.append(f"Sample{k + 1}_ImprovementDelta")

    aggregate_write_header = not os.path.exists(aggregate_csv_filename)
    with open(aggregate_csv_filename, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
        if aggregate_write_header:
            writer.writeheader()

    for idx, example in tqdm(enumerate(dataset["train"])):
        matrix_content = example["matrix"]
        n, m, inst_for_ortools, real_makespan = read_matrix_form_jssp(matrix_content)
        ms = real_makespan if real_makespan is not None else example.get("makespan")

        rollout_results = []
        sample_feasible = []
        sample_makespans = []
        total_time = 0.0
        for _ in range(sample_count):
            start = time.time()
            result = run_step_rollout(
                model=model,
                tokenizer=tokenizer,
                inst_for_ortools=inst_for_ortools,
                use_masking=use_masking,
                step_action_max_new_tokens=run_args.step_action_max_new_tokens,
                env_mode=run_args.env_mode,
                include_problem_context=not run_args.disable_step_problem_context,
                enable_step_improvement=run_args.enable_step_improvement,
                step_reflection_passes=run_args.step_reflection_passes,
                step_reflection_topk=run_args.step_reflection_topk,
                emit_step_rationale=run_args.emit_step_rationale,
                step_rationale_max_new_tokens=run_args.step_rationale_max_new_tokens,
                reason_topk=run_args.reason_topk,
                use_reason_adapter=bool(run_args.reason_model_path),
                action_code_width=run_args.action_code_width,
                action_code_seed=run_args.action_code_seed + idx,
                action_code_cap=run_args.action_code_cap,
                print_step_trace=run_args.print_step_trace,
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
        if declared_makespan is not None:
            validation_message = f"Step rollout feasible with makespan {declared_makespan}"
        else:
            validation_message = "No feasible rollout"

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

        output_root = run_args.output_dir or str(Path(csv_filename).parent)
        if best_result is not None:
            saved_paths = save_result_csv_artifacts(
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
        if feasible_values:
            mean_ms = float(np.mean(feasible_values))
            var_ms = float(np.var(feasible_values))
            std_ms = float(np.std(feasible_values))
        else:
            mean_ms = None
            var_ms = None
            std_ms = None

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
            agg_row[f"Sample{k + 1}_Improved"] = (
                None if sample_result is None else int(bool(sample_result.get("improved_over_base", False)))
            )
            agg_row[f"Sample{k + 1}_ImprovementDelta"] = (
                None if sample_result is None else sample_result.get("improvement_delta")
            )

        with open(aggregate_csv_filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=aggregate_fieldnames)
            writer.writerow(agg_row)

        for sample_idx, result in enumerate(rollout_results):
            if result is None:
                continue
            is_best_sample = bool(
                best_result is not None and int(result["makespan"]) == int(best_result["makespan"])
            )
            save_result_csv_artifacts(
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
            append_sequence_trace_rows(
                sequence_trace_csv_filename,
                build_sequence_trace_rows(
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
                save_result_csv_artifacts(
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
                append_sequence_trace_rows(
                    sequence_trace_csv_filename,
                    build_sequence_trace_rows(
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
    dataset,
    run_args,
    index: int,
    num_solutions: int,
    use_masking: bool = True,
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
        f"🎯 Step-demo instance {index} — Jobs={n}, Machines={m}, "
        f"EnvMode={run_args.env_mode}, Masking={use_masking}, Context={not run_args.disable_step_problem_context}, "
        f"EpisodeImprovement={run_args.enable_step_improvement}, "
        f"Rationale={run_args.emit_step_rationale}"
    )
    print("=" * 80)

    best_result = None
    best_makespan = None
    demo_results = []
    for sample_idx in range(max(num_solutions, 1)):
        result = run_step_rollout(
            model=model,
            tokenizer=tokenizer,
            inst_for_ortools=inst_for_ortools,
            use_masking=use_masking,
            step_action_max_new_tokens=run_args.step_action_max_new_tokens,
            env_mode=run_args.env_mode,
            include_problem_context=not run_args.disable_step_problem_context,
            enable_step_improvement=run_args.enable_step_improvement,
            step_reflection_passes=run_args.step_reflection_passes,
            step_reflection_topk=run_args.step_reflection_topk,
            emit_step_rationale=run_args.emit_step_rationale,
            step_rationale_max_new_tokens=run_args.step_rationale_max_new_tokens,
            reason_topk=run_args.reason_topk,
            use_reason_adapter=bool(run_args.reason_model_path),
            action_code_width=run_args.action_code_width,
            action_code_seed=run_args.action_code_seed + sample_idx,
            action_code_cap=run_args.action_code_cap,
            print_step_trace=run_args.print_step_trace,
        )
        demo_results.append(result)
        makespan = int(result["makespan"])
        print(f"\n--- Step rollout {sample_idx + 1}/{num_solutions} ---")
        print(result["solution_text"])
        print(f"makespan={makespan}")
        if not run_args.print_step_trace:
            _print_step_trace(result.get("raw_step_outputs", []))
        notes = result.get("episode_improvement_notes")
        if notes and not run_args.print_step_trace:
            print("episode_improvement_notes:")
            for x in notes:
                print(" -", x)
        if run_args.emit_step_rationale:
            print("step rationales:")
            for step_info in result.get("raw_step_outputs", []):
                rationale = step_info.get("rationale_text")
                if rationale:
                    print(
                        f"  - step {step_info['step_idx']}: "
                        f"{step_info.get('chosen_action_code')} (job {step_info['chosen_job']}) -> {rationale}"
                    )

        if best_makespan is None or makespan < best_makespan:
            best_makespan = makespan
            best_result = result

    if ms:
        print(f"\nBenchmark makespan: {ms}")
        if best_makespan is not None and ms > 0:
            print(f"Best gap: {abs((best_makespan - ms) / ms):.4f}")

    if best_result is not None:
        output_root = run_args.output_dir or "val_results/jssp_val"
        demo_saved_paths = save_result_csv_artifacts(
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
            save_result_csv_artifacts(
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
                save_result_csv_artifacts(
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

    print("✅ Step demo run complete.")
    return best_result
def _unique_tokens(tokenizer, token_ids):
    seen = set()
    tokens = []
    for tid in token_ids:
        token_str = tokenizer.decode([tid], skip_special_tokens=True)
        normalized = token_str
        if normalized not in seen:
            seen.add(normalized)
            tokens.append(normalized)
    return tokens


def _summarize_allowed(tokenizer, allowed_ids, vocab_size):
    unique_tokens = _unique_tokens(tokenizer, allowed_ids)
    digits = sorted({tok.strip() for tok in unique_tokens if tok.strip().isdigit()})
    machines = sorted({tok[1:] for tok in unique_tokens if tok.startswith("M") and tok[1:].isdigit()})
    blocked = vocab_size - len(set(allowed_ids))
    return unique_tokens, digits, machines, blocked


def _collect_prefix_positions(text: str) -> list:
    def _body_start(s: str) -> int:
        lowered = s.lower()
        solution_pos = lowered.find("solution:")
        if solution_pos != -1:
            return solution_pos + len("solution:")
        for job_match in re.finditer(r"(?:^|\n)\s*Job\b", s, re.IGNORECASE):
            after = s[job_match.end():]
            after_stripped = after.lstrip()
            if not after_stripped:
                return job_match.start()
            first = after_stripped[0]
            if first.isdigit():
                return job_match.start()
        return -1

    body_start = _body_start(text)
    positions = set()

    if body_start == -1:
        return []

    for match in re.finditer(r"Solution:", text, re.IGNORECASE):
        if match.end() >= body_start:
            positions.add(match.end())

    for pattern in [r"Job ", r"Operation ", r", M", r"Makespan"]:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            if match.end() >= body_start:
                positions.add(match.end())

    return sorted(pos for pos in positions if pos >= body_start)


def _make_tensor(ids):
    if not ids:
        return torch.tensor([], dtype=torch.long)
    return torch.tensor(ids, dtype=torch.long)


def print_masking_breakdown(solution_text, tokenizer, inst_for_ortools, header="[Masking Trace]"):
    positions = _collect_prefix_positions(solution_text)
    if not positions:
        print(f"{header}: 해당 솔루션에서 분석할 프리픽스가 없습니다.")
        return

    print("\n" + header)
    fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst_for_ortools)
    vocab_size = len(tokenizer.get_vocab())

    for pos in positions:
        prefix = solution_text[:pos]
        try:
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        except Exception as exc:
            print(f"  [SKIP] prefix encoding 실패: {exc}")
            continue

        tensor_prefix = _make_tensor(prefix_ids)
        state = fsm.update_from_input(0, tensor_prefix)
        allowed_ids = fsm(0, tensor_prefix)
        tokens, digits, machines, blocked = _summarize_allowed(tokenizer, allowed_ids, vocab_size)
        next_fragment = solution_text[pos: pos + 20].replace("\n", "\\n")

        tail_preview = prefix[-40:].replace("\n", "\\n")
        print(f"\nPrefix tail: '{tail_preview}'")
        print(f"  remaining_ops={state['remaining_ops']}  available_jobs={sorted(state.get('available_jobs', []))}  makespan_started={state['makespan_started']}")
        print(f"  허용 토큰 ({len(tokens)}개): {tokens}")
        if digits:
            print(f"    • 숫자 옵션: {digits}")
        if machines:
            print(f"    • 기계 옵션: {machines}")
        print(f"  차단된 토큰 수: {blocked}")
        print(f"  실제 다음 출력: '{next_fragment}'")
