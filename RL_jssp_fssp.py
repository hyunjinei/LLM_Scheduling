import argparse
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import login
from torch.optim import AdamW
from tqdm import trange
from unsloth import FastLanguageModel


@dataclass
class EpisodeBatch:
    log_probs: torch.Tensor
    advantages: torch.Tensor


@dataclass
class TrajectorySample:
    sequence_ids: torch.Tensor
    prompt_len: int
    reward: float
    advantage: float
    old_log_prob: torch.Tensor
    feasible: bool
    makespan: float


@dataclass
class BOPOStepPair:
    winner_sequence_ids: torch.Tensor
    winner_prompt_len: int
    loser_sequence_ids: torch.Tensor
    loser_prompt_len: int
    relative_gap: float
    winner_makespan: float
    loser_makespan: float


@dataclass
class StepActionTrace:
    sequence_ids: torch.Tensor
    prompt_len: int
    chosen_job: int
    step_idx: int

from llm_jssp.utils.solution_generation_english import (
    read_matrix_form_jssp,
)
from llm_jssp.utils.jssp_step_env import StaticJSSPStepEnv
from llm_jssp.utils.jssp_step_masking_hooks import (
    build_step_prefix_allowed_tokens_fn,
    StepActionParseError,
)
from llm_jssp.utils.action_token_utils import (
    ensure_action_special_tokens,
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


def _estimate_action_effects(state_json, action_code_to_job):
    return compute_action_transition_features(state_json, action_code_to_job)


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
        env = StaticJSSPStepEnv(inst_for_ortools)
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
                )
                effect_by_code = {x["action_code"]: x for x in action_effects}
                feasible_action_codes = list(action_code_to_job.keys())
                state_text = build_step_prompt(
                    state_json=state,
                    feasible_jobs=feasible_jobs,
                    step_idx=step_idx,
                    total_steps=int(state["total_steps"]),
                    problem_context_text=problem_context_text,
                    action_code_to_job=action_code_to_job,
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
    device: torch.device,
    require_grad: bool,
) -> torch.Tensor:
    """
    Compute mean log-probability over generated tokens only.
    """
    seq = sequence_ids.unsqueeze(0).to(device)
    input_ids = seq[:, :-1]
    labels = seq[:, 1:].clone()
    labels[:, : max(prompt_len - 1, 0)] = -100

    if require_grad:
        outputs = model(input_ids=input_ids, labels=labels)
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)

    token_mask = (labels != -100)
    token_count = int(token_mask.sum().item())
    if token_count == 0:
        return torch.tensor(0.0, device=device)

    return -outputs.loss


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
) -> Tuple[float, float]:
    """
    PPO-style clipped policy update with group-normalized advantages.
    """
    if not samples:
        return 0.0, 0.0

    old_log_probs = torch.stack([s.old_log_prob for s in samples]).to(device).detach()
    advantages = torch.tensor([s.advantage for s in samples], dtype=torch.float32, device=device)

    total_loss = 0.0
    total_kl = 0.0
    for _ in range(grpo_epochs):
        current_log_probs = []
        for s in samples:
            log_prob = compute_log_prob_mean(
                model=model,
                sequence_ids=s.sequence_ids,
                prompt_len=s.prompt_len,
                device=device,
                require_grad=True,
            )
            current_log_probs.append(log_prob)
        current_log_probs = torch.stack(current_log_probs)

        log_ratio = current_log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, -20, 20))
        unclipped_obj = ratio * advantages
        clipped_obj = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
        policy_loss = -torch.mean(torch.min(unclipped_obj, clipped_obj))

        approx_kl = torch.mean((ratio - 1.0) - torch.log(ratio + 1e-8))
        loss = policy_loss + kl_coef * approx_kl

        optimizer.zero_grad()
        loss.backward()
        params = []
        for group in optimizer.param_groups:
            params.extend(group["params"])
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        total_loss += float(loss.item())
        total_kl += float(approx_kl.item())

    return total_loss / grpo_epochs, total_kl / grpo_epochs


def build_bopo_step_pairs(
    group_rollouts: List[Dict],
    rng: np.random.Generator,
    min_relative_gap: float = 0.0,
    max_pairs_per_group: int = 256,
    max_step_pairs_per_pair: int = 32,
) -> List[BOPOStepPair]:
    """
    Build BOPO preference pairs from a rollout group.

    Strategy:
      1. Keep feasible rollouts only.
      2. Sort by makespan (smaller is better), anchor best rollout.
      3. Pair best vs each loser if relative gap >= threshold.
      4. For each episode pair, sample aligned step indices to limit cost.
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
                    loser_sequence_ids=l_trace.sequence_ids,
                    loser_prompt_len=int(l_trace.prompt_len),
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

    for p in pairs:
        try:
            lp_w = compute_log_prob_mean(
                model=model,
                sequence_ids=p.winner_sequence_ids,
                prompt_len=p.winner_prompt_len,
                device=device,
                require_grad=True,
            )
            lp_l = compute_log_prob_mean(
                model=model,
                sequence_ids=p.loser_sequence_ids,
                prompt_len=p.loser_prompt_len,
                device=device,
                require_grad=True,
            )

            rel_gap = max(0.0, float(p.relative_gap))
            scaled_beta = float(beta) * (1.0 + float(gap_scale) * rel_gap)
            pref_logit = scaled_beta * (lp_w - lp_l - float(margin))
            loss = -F.logsigmoid(pref_logit)

            optimizer.zero_grad()
            loss.backward()
            params = []
            for group in optimizer.param_groups:
                params.extend(group["params"])
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_gap += rel_gap
            updates += 1

            del lp_w, lp_l, loss, pref_logit
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

    # Hugging Face login (optional)
    if args.hf_token:
        login(token=args.hf_token, add_to_git_credential=False)

    if not args.model_path:
        default_model_by_type = {
            "llama8b": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
            "llama1b": "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
            "qwen2.5_7b": "unsloth/Qwen2.5-7B-Instruct-unsloth-bnb-4bit",
            "qwen2.5_14b": "unsloth/Qwen2.5-14B-Instruct-unsloth-bnb-4bit",
            "deepseek_8b": "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
            "qwen25_7b_math": "unsloth/Qwen2.5-Math-7B-Instruct-bnb-4bit",
        }
        args.model_path = default_model_by_type[args.model_type]
        print(f"[Info] --model_path not provided. Using default: {args.model_path}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit,
        dtype=torch.bfloat16 if args.dtype == "bfloat16" else torch.float16,
        local_files_only=False,
    )
    token_install = ensure_action_special_tokens(
        tokenizer=tokenizer,
        model=model,
        code_width=args.action_code_width,
        code_cap=args.action_code_cap,
    )
    print("[Info] action token install:", token_install)
    if hasattr(model, "for_training"):
        model.for_training()
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    rng = np.random.default_rng(args.seed)
    use_random = args.use_random_problems
    if use_random:
        data_split = None
    else:
        dataset = load_dataset("json", data_files=args.dataset_path)
        data_split = dataset["train"]

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
                    matrix_content = example["matrix"]
                    _, _, inst, _ = read_matrix_form_jssp(matrix_content)

                if args.rl_algo == "grpo":
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
                        reward = (
                            -makespan
                            if feasible and math.isfinite(makespan)
                            else -float(args.invalid_makespan_penalty)
                        )

                        step_samples = []
                        for tr in traces:
                            old_log_prob = compute_log_prob_mean(
                                model=model,
                                sequence_ids=tr.sequence_ids,
                                prompt_len=tr.prompt_len,
                                device=device,
                                require_grad=False,
                            ).detach()
                            step_samples.append(
                                {
                                    "sequence_ids": tr.sequence_ids,
                                    "prompt_len": tr.prompt_len,
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
                        algo="grpo-step",
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

                    reward = -makespan
                    _, mwkr_makespan = mwkr_schedule(inst)
                    baseline = -mwkr_makespan if math.isfinite(mwkr_makespan) else reward
                    if args.use_running_baseline:
                        baseline = baseline_tracker.update(reward)
                    advantage = reward - baseline

                    for tr in traces:
                        log_prob = compute_log_prob_mean(
                            model=model,
                            sequence_ids=tr.sequence_ids,
                            prompt_len=tr.prompt_len,
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

        if args.rl_algo == "grpo":
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

        if args.output_dir:
            save_dir = Path(args.output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(description="Policy optimization for JSSP (REINFORCE / GRPO / BOPO).")
    parser.add_argument("--max_seq_length", type=int, default=40000, help="Maximum sequence length")
    parser.add_argument("--model_type", type=str, default="llama8b",
                        choices=["llama8b", "llama1b", "qwen2.5_7b", "qwen2.5_14b", "deepseek_8b", "qwen25_7b_math"],
                        help="Base model family (inference-compatible defaults).")
    parser.add_argument("--model_path", type=str, default=None, help="LoRA checkpoint path (defaults to inference model).")
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
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--max_new_tokens", type=int, default=1, help="Legacy decode bound; step action decoding uses --step_action_max_new_tokens.")
    parser.add_argument("--baseline_beta", type=float, default=0.9)
    parser.add_argument("--use_running_baseline", action="store_true")
    parser.add_argument("--rl_algo", type=str, default="grpo", choices=["reinforce", "grpo", "bopo"])
    parser.add_argument("--group_size", type=int, default=4, help="Number of sampled rollouts per prompt (GRPO).")
    parser.add_argument("--grpo_epochs", type=int, default=2, help="Number of optimization epochs per GRPO batch.")
    parser.add_argument("--clip_epsilon", type=float, default=0.2, help="PPO/GRPO clip epsilon.")
    parser.add_argument("--kl_coef", type=float, default=0.0, help="Approx-KL regularization coefficient.")
    parser.add_argument("--bopo_beta", type=float, default=2.0, help="BOPO pairwise inverse temperature.")
    parser.add_argument("--bopo_gap_scale", type=float, default=3.0, help="Scale factor for objective-gap weighted BOPO logits.")
    parser.add_argument("--bopo_margin", type=float, default=0.0, help="Optional margin in BOPO preference logit.")
    parser.add_argument("--bopo_min_relative_gap", type=float, default=0.0, help="Minimum relative makespan gap to keep a BOPO pair.")
    parser.add_argument("--bopo_max_pairs_per_group", type=int, default=256, help="Maximum BOPO step-pairs generated per rollout group.")
    parser.add_argument("--bopo_max_step_pairs_per_pair", type=int, default=32, help="Maximum aligned step pairs used between winner/loser rollouts.")
    parser.add_argument("--step_action_max_new_tokens", type=int, default=1, help="Max decode length for one step action.")
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
        help="Penalty makespan used for infeasible outputs in GRPO reward.",
    )
    parser.add_argument("--disable_masking", action="store_true", help="Disable decoding-time feasibility masking.")
    parser.add_argument("--print_step_trace", action="store_true", help="Print chosen/not-chosen options and makespan per step.")
    parser.add_argument("--hf_token", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)

    parser.add_argument("--dataset_path", type=str, default="llm_jssp/train.json")
    parser.add_argument("--use_random_problems", action="store_true",
                        help="Generate random JSSP instances instead of using dataset.")
    parser.add_argument("--random_jobs", type=int, default=10)
    parser.add_argument("--random_machines", type=int, default=10)
    parser.add_argument("--random_time_low", type=int, default=1)
    parser.add_argument("--random_time_high", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
