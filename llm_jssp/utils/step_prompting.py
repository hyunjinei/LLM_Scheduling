"""Prompt builders and transition-feature helpers for step-by-step JSSP."""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from llm_jssp.utils.action_token_utils import format_action_code


def _format_machine_ready(machine_ready_time: Sequence[int]) -> str:
    return " ".join(f"M{i}={t}" for i, t in enumerate(machine_ready_time))


def _format_feasible_jobs(feasible_jobs: Sequence[int]) -> str:
    if not feasible_jobs:
        return "[]"
    return "[" + ", ".join(f"Job {j}" for j in feasible_jobs) + "]"


def _format_action_codes(action_codes: Sequence[str]) -> str:
    if not action_codes:
        return "[]"
    return "[" + ", ".join(str(code) for code in action_codes) + "]"


def _format_route_tokens(route_tokens: Sequence[str]) -> str:
    if not route_tokens:
        return "[]"
    return "[" + ", ".join(str(token) for token in route_tokens) + "]"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if float(denominator) == 0.0:
        return 0.0
    return float(numerator) / float(denominator)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _machine_token(machine_id: int) -> str:
    return f"M{int(machine_id)}" if int(machine_id) >= 0 else "M-"


def build_randomized_action_code_map(
    feasible_jobs: Sequence[int],
    rng: Optional[random.Random] = None,
    code_width: int = 4,
    code_start: int = 1,
    code_cap: int = 9999,
) -> Dict[str, int]:
    """
    Create randomized action-code mapping for one step.

    Example output:
        {"<A0102>": 7, "<A4377>": 3, ...}
    """
    jobs = [int(j) for j in feasible_jobs]
    if len(set(jobs)) != len(jobs):
        raise ValueError(f"feasible_jobs contains duplicates: {jobs}")

    if code_start < 0:
        raise ValueError(f"code_start must be non-negative, got {code_start}")
    if code_cap < code_start:
        raise ValueError(
            f"code_cap must be >= code_start, got code_start={code_start}, code_cap={code_cap}"
        )
    if len(jobs) > (int(code_cap) - int(code_start) + 1):
        raise ValueError(
            "Not enough action-code slots for this step: "
            f"jobs={len(jobs)}, available={int(code_cap) - int(code_start) + 1}, "
            f"range=[{code_start}, {code_cap}]"
        )

    if not jobs:
        return {}

    shuffled_jobs = list(jobs)
    if rng is None:
        random.shuffle(shuffled_jobs)
        sampled_code_indices = random.sample(
            range(int(code_start), int(code_cap) + 1), k=len(shuffled_jobs)
        )
    else:
        rng.shuffle(shuffled_jobs)
        sampled_code_indices = rng.sample(
            range(int(code_start), int(code_cap) + 1), k=len(shuffled_jobs)
        )

    action_code_to_job: Dict[str, int] = {}
    for code_idx, job_id in zip(sampled_code_indices, shuffled_jobs):
        code = format_action_code(int(code_idx), code_width=code_width)
        action_code_to_job[code] = int(job_id)
    return action_code_to_job


def invert_action_code_map(action_code_to_job: Dict[str, int]) -> Dict[int, str]:
    job_to_action_code: Dict[int, str] = {}
    for code, job_id in action_code_to_job.items():
        if job_id in job_to_action_code:
            raise ValueError(f"Duplicate job id in action_code_to_job: {job_id}")
        job_to_action_code[int(job_id)] = str(code)
    return job_to_action_code


def build_problem_context_text(inst_for_ortools: Sequence[Sequence[Sequence[int]]]) -> str:
    """
    Build minimal static problem context text.
    """
    num_jobs = len(inst_for_ortools)
    total_ops = sum(len(job_ops) for job_ops in inst_for_ortools)
    max_machine = -1
    for job_ops in inst_for_ortools:
        for machine_id, _ in job_ops:
            max_machine = max(max_machine, int(machine_id))
    num_machines = max_machine + 1 if max_machine >= 0 else 0

    return f"Problem: {num_jobs} jobs x {num_machines} machines (total_ops={total_ops})"


def summarize_global_dynamic_state(state_json: Dict[str, object]) -> Dict[str, object]:
    machine_ready_time: List[int] = state_json["machine_ready_time"]  # type: ignore[assignment]
    current_cmax = int(
        state_json.get("current_cmax", max(machine_ready_time) if machine_ready_time else 0)
    )
    total_remaining_work = int(
        state_json.get("total_remaining_work", sum(state_json.get("remaining_work", [])))  # type: ignore[arg-type]
    )
    summary = {
        "step_idx": int(state_json["step_idx"]),
        "total_steps": int(state_json["total_steps"]),
        "scheduled_ratio": float(state_json.get("scheduled_ratio", 0.0)),
        "current_cmax": current_cmax,
        "total_remaining_work": total_remaining_work,
        "unfinished_jobs_count": int(state_json.get("unfinished_jobs_count", 0)),
        "unfinished_jobs_ratio": float(state_json.get("unfinished_jobs_ratio", 0.0)),
        "machine_ready_min": int(state_json.get("machine_ready_min", 0)),
        "machine_ready_mean": float(state_json.get("machine_ready_mean", 0.0)),
        "machine_ready_max": int(state_json.get("machine_ready_max", 0)),
        "machine_ready_std": float(state_json.get("machine_ready_std", 0.0)),
        "bottleneck_machine_id": int(state_json.get("bottleneck_machine_id", -1)),
        "bottleneck_machine_load": int(state_json.get("bottleneck_machine_load", 0)),
        "bottleneck_machine_ops_left": int(state_json.get("bottleneck_machine_ops_left", 0)),
    }
    return summary


def compute_action_transition_features(
    state_json: Dict[str, object],
    action_code_to_job: Dict[str, int],
) -> Tuple[int, List[Dict[str, object]]]:
    job_ready_time: List[int] = state_json["job_ready_time"]  # type: ignore[assignment]
    machine_ready_time: List[int] = state_json["machine_ready_time"]  # type: ignore[assignment]
    next_machine: List[int] = state_json["next_machine"]  # type: ignore[assignment]
    next_proc_time: List[int] = state_json["next_proc_time"]  # type: ignore[assignment]
    next2_machine: List[int] = state_json.get("next2_machine", [-1] * len(next_machine))  # type: ignore[assignment]
    next2_proc_time: List[int] = state_json.get("next2_proc_time", [0] * len(next_machine))  # type: ignore[assignment]
    remaining_ops: List[int] = state_json["remaining_ops"]  # type: ignore[assignment]
    remaining_work: List[int] = state_json["remaining_work"]  # type: ignore[assignment]
    job_total_ops: List[int] = state_json.get("job_total_ops", [1] * len(next_machine))  # type: ignore[assignment]
    job_total_work: List[int] = state_json.get("job_total_work", [1] * len(next_machine))  # type: ignore[assignment]
    machine_remaining_load: List[int] = state_json.get("machine_remaining_load", [0] * len(machine_ready_time))  # type: ignore[assignment]
    machine_remaining_ops: List[int] = state_json.get("machine_remaining_ops", [0] * len(machine_ready_time))  # type: ignore[assignment]
    post_route_tokens_by_job: List[List[str]] = state_json.get("post_route_tokens", [[] for _ in next_machine])  # type: ignore[assignment]

    current_cmax = int(
        state_json.get("current_cmax", max(machine_ready_time) if machine_ready_time else 0)
    )
    total_remaining_work = int(
        state_json.get("total_remaining_work", sum(int(x) for x in remaining_work))
    )

    effects: List[Dict[str, object]] = []
    for action_code, job_id in action_code_to_job.items():
        job = int(job_id)
        machine_id = int(next_machine[job])
        proc_time = int(next_proc_time[job])
        if machine_id < 0 or proc_time <= 0:
            continue

        job_ready_before = int(job_ready_time[job])
        machine_ready_before = int(machine_ready_time[machine_id])
        est_start = max(job_ready_before, machine_ready_before)
        est_end = est_start + proc_time
        current_cmax_after = max(current_cmax, est_end)
        delta_cmax = current_cmax_after - current_cmax
        rem_ops_before = int(remaining_ops[job])
        rem_ops_after = max(0, rem_ops_before - 1)
        rem_work_before = int(remaining_work[job])
        rem_work_after = max(0, rem_work_before - proc_time)
        total_ops = max(int(job_total_ops[job]), 1)
        total_work = max(int(job_total_work[job]), 1)
        job_progress_ratio_before = float(total_ops - rem_ops_before) / float(total_ops)
        job_progress_ratio_after = float(total_ops - rem_ops_after) / float(total_ops)
        affected_machine_load = int(machine_remaining_load[machine_id])
        affected_machine_ops_left = int(machine_remaining_ops[machine_id])

        effect = {
            "action_code": str(action_code),
            "job_id": job,
            "machine_id": machine_id,
            "machine_token": _machine_token(machine_id),
            "proc_time": proc_time,
            "next_machine": machine_id,
            "next_proc_time": proc_time,
            "next2_machine": int(next2_machine[job]),
            "next2_proc_time": int(next2_proc_time[job]),
            "remaining_ops_before": rem_ops_before,
            "remaining_ops_after": rem_ops_after,
            "remaining_work_before": rem_work_before,
            "remaining_work_after": rem_work_after,
            "job_progress_ratio_before": float(job_progress_ratio_before),
            "job_progress_ratio_after": float(job_progress_ratio_after),
            "job_ready_before": job_ready_before,
            "job_ready_after": int(est_end),
            "machine_ready_before": machine_ready_before,
            "machine_ready_after": int(est_end),
            "estimated_start": int(est_start),
            "estimated_end": int(est_end),
            "est_start": int(est_start),
            "est_end": int(est_end),
            "current_cmax_before": int(current_cmax),
            "current_cmax_after": int(current_cmax_after),
            "estimated_makespan_after": int(current_cmax_after),
            "delta_cmax": int(delta_cmax),
            "delta_cmax_ratio": float(_safe_ratio(delta_cmax, max(current_cmax_after, 1))),
            "job_wait": int(max(0, machine_ready_before - job_ready_before)),
            "machine_idle_gap": int(max(0, job_ready_before - machine_ready_before)),
            "slack_to_current_cmax": int(current_cmax - est_end),
            "affected_machine_load": affected_machine_load,
            "affected_machine_ops_left": affected_machine_ops_left,
            "affected_machine_load_ratio": float(
                _safe_ratio(affected_machine_load, max(total_remaining_work, 1))
            ),
            "remaining_work_after_ratio": float(_safe_ratio(rem_work_after, total_work)),
            "post_route_tokens": list(post_route_tokens_by_job[job]),
            "post_route_len": int(len(post_route_tokens_by_job[job])),
        }
        effects.append(effect)

    effects.sort(
        key=lambda x: (
            int(x["estimated_makespan_after"]),
            int(x["estimated_start"]),
            int(x["proc_time"]),
            int(x["job_id"]),
        )
    )
    return int(current_cmax), effects


def render_action_transition_line(effect: Dict[str, object]) -> str:
    return (
        f"{effect['action_code']} | "
        f"operation machine={_machine_token(int(effect['next_machine']))} | "
        f"processing time={effect['next_proc_time']} | "
        f"rem_ops:{effect['remaining_ops_before']}->{effect['remaining_ops_after']} | "
        f"rem_work:{effect['remaining_work_before']}->{effect['remaining_work_after']} | "
        f"job_prog:{_format_value(effect['job_progress_ratio_before'])}->{_format_value(effect['job_progress_ratio_after'])} | "
        f"job_ready:{effect['job_ready_before']}->{effect['job_ready_after']} | "
        f"machine_ready:{effect['machine_ready_before']}->{effect['machine_ready_after']} | "
        f"est_start={effect['estimated_start']} | "
        f"est_end={effect['estimated_end']} | "
        f"cmax:{effect['current_cmax_before']}->{effect['current_cmax_after']} | "
        f"delta_cmax={effect['delta_cmax']} | "
        f"delta_cmax_ratio={_format_value(effect['delta_cmax_ratio'])} | "
        f"job_wait={effect['job_wait']} | "
        f"machine_idle_gap={effect['machine_idle_gap']} | "
        f"slack_to_cmax={effect['slack_to_current_cmax']} | "
        f"machine_load={effect['affected_machine_load']} | "
        f"machine_ops_left={effect['affected_machine_ops_left']} | "
        f"machine_load_ratio={_format_value(effect['affected_machine_load_ratio'])} | "
        f"rem_work_after_ratio={_format_value(effect['remaining_work_after_ratio'])} | "
        f"post_route={_format_route_tokens(effect.get('post_route_tokens', []))}"
    )


def build_step_prompt(
    state_json: Dict[str, object],
    feasible_jobs: Sequence[int],
    step_idx: int,
    total_steps: int,
    problem_context_text: Optional[str] = None,
    action_code_to_job: Optional[Dict[str, int]] = None,
) -> str:
    """
    Build deterministic compact prompt for one-step action selection.

    Output format is always one line.
    - Default (legacy): Action: Job <id>
    - Action-token mode: <Axxxx>
    """
    lines = [
        "You are solving JSSP step-by-step.",
        "Objective: Minimize final makespan (Cmax) while keeping every action feasible.",
    ]
    if problem_context_text:
        lines.extend(
            [
                "Static problem context:",
                problem_context_text,
            ]
        )

    global_summary = summarize_global_dynamic_state(state_json)
    lines.extend(
        [
            "Global dynamic state:",
            (
                f"step={int(global_summary['step_idx']) + 1}/{int(global_summary['total_steps'])} "
                f"scheduled_ratio={_format_value(global_summary['scheduled_ratio'])}"
            ),
            f"current_cmax={global_summary['current_cmax']}",
            f"total_remaining_work={global_summary['total_remaining_work']}",
            (
                f"unfinished_jobs_count={global_summary['unfinished_jobs_count']} "
                f"unfinished_jobs_ratio={_format_value(global_summary['unfinished_jobs_ratio'])}"
            ),
            (
                f"machine_ready_min={global_summary['machine_ready_min']} "
                f"machine_ready_mean={_format_value(global_summary['machine_ready_mean'])} "
                f"machine_ready_max={global_summary['machine_ready_max']} "
                f"machine_ready_std={_format_value(global_summary['machine_ready_std'])}"
            ),
            (
                f"bottleneck_machine={_machine_token(int(global_summary['bottleneck_machine_id']))} "
                f"bottleneck_load={global_summary['bottleneck_machine_load']} "
                f"bottleneck_ops_left={global_summary['bottleneck_machine_ops_left']}"
            ),
        ]
    )

    if action_code_to_job:
        action_codes = list(action_code_to_job.keys())
        _, action_effects = compute_action_transition_features(
            state_json=state_json,
            action_code_to_job=action_code_to_job,
        )
        lines.extend(
            [
                f"Feasible action codes: {_format_action_codes(action_codes)}",
                "Candidate transition features:",
            ]
        )
        for effect in action_effects:
            lines.append(render_action_transition_line(effect))
        lines.extend(
            [
                "Action codes are randomized at each step. Do not assume persistent code identity.",
                "Choose exactly one feasible action code.",
                "Return exactly one code from the feasible action set, for example <A3812>.",
            ]
        )
    else:
        job_next_op: List[int] = state_json["job_next_op"]  # type: ignore[assignment]
        next_machine: List[int] = state_json["next_machine"]  # type: ignore[assignment]
        next_proc_time: List[int] = state_json["next_proc_time"]  # type: ignore[assignment]
        job_ready_time: List[int] = state_json["job_ready_time"]  # type: ignore[assignment]
        remaining_ops: List[int] = state_json["remaining_ops"]  # type: ignore[assignment]
        remaining_work: List[int] = state_json["remaining_work"]  # type: ignore[assignment]
        machine_ready_time: List[int] = state_json["machine_ready_time"]  # type: ignore[assignment]
        lines.extend(
            [
                f"Feasible jobs: {_format_feasible_jobs(feasible_jobs)}",
                "Job state summary:",
            ]
        )
        for job_id in range(len(job_next_op)):
            lines.append(
                (
                    f"Job {job_id}: "
                    f"next_op={job_next_op[job_id]}, "
                    f"next_machine={next_machine[job_id]}, "
                    f"next_proc={next_proc_time[job_id]}, "
                    f"job_ready={job_ready_time[job_id]}, "
                    f"remaining_ops={remaining_ops[job_id]}, "
                    f"remaining_work={remaining_work[job_id]}"
                )
            )
        lines.extend(
            [
                f"Machine ready times: {_format_machine_ready(machine_ready_time)}",
                "Choose exactly one job from feasible jobs to minimize final makespan (Cmax).",
                "Return exactly one line: Action: Job <id>",
            ]
        )

    return "\n".join(lines)


def build_step_improvement_prompt(
    state_text: str,
    candidate_action_text: str,
    feasible_jobs: Sequence[object],
    reflection_memory: Optional[str] = None,
    step_diagnostics: Optional[str] = None,
) -> str:
    """Build a hindsight-aware step-improvement prompt."""
    feasible_str: str
    if feasible_jobs and isinstance(feasible_jobs[0], str):
        feasible_str = _format_action_codes([str(x) for x in feasible_jobs])
        output_format = "<Axxxx>"
    else:
        feasible_str = _format_feasible_jobs([int(x) for x in feasible_jobs])
        output_format = "Action: Job <id>"
    prompt = (
        "You are revising one decision inside a completed JSSP schedule.\n"
        "Use hindsight from the full episode, not only local greedy signals.\n"
        "If a small short-term sacrifice helps earlier bottleneck activation, lower idle loss, "
        "or better downstream route progression, prefer it.\n\n"
        "Current step state:\n"
        f"{state_text}\n\n"
        "Previously chosen action in the failed/improvable rollout:\n"
        f"{candidate_action_text}\n\n"
        "Rules:\n"
        "- Objective: minimize final makespan (Cmax).\n"
        "- Think in hindsight: ask which choice would have reduced downstream bottleneck delay, idle gaps, or route blocking.\n"
        "- Prefer the action that best aligns with bottleneck-machine usage, downstream route progression, and lower regret against strong alternatives.\n"
        f"- Final action must be one of feasible options: {feasible_str}\n"
        f"- Return exactly one output in this format: {output_format}\n"
        "- Do not output explanation.\n"
        "- Do not repeat the previous action if the reflection evidence says an alternative is structurally better.\n"
    )
    if reflection_memory:
        prompt += f"\nEpisode hindsight reflection:\n{reflection_memory}\n"
    if step_diagnostics:
        prompt += f"\nCritical-step diagnostics:\n{step_diagnostics}\n"
    return prompt


def build_step_rationale_prompt(
    state_text: str,
    chosen_job: Optional[int] = None,
    feasible_jobs: Optional[Sequence[int]] = None,
    chosen_action_code: Optional[str] = None,
    feasible_action_codes: Optional[Sequence[str]] = None,
) -> str:
    """
    Build explanation prompt after action selection.

    This prompt is intentionally separated from action decoding so that
    feasibility/format masking of action is unaffected.
    """
    using_codes = chosen_action_code is not None
    if using_codes:
        chosen_label = str(chosen_action_code)
        if feasible_action_codes is None:
            raise ValueError("feasible_action_codes is required when chosen_action_code is used.")
        other_labels = [str(code) for code in feasible_action_codes if str(code) != chosen_label]
        others = ", ".join(other_labels) if other_labels else "(none)"
    else:
        if chosen_job is None or feasible_jobs is None:
            raise ValueError("chosen_job and feasible_jobs are required in legacy rationale mode.")
        chosen_label = f"Job {int(chosen_job)}"
        other_jobs = [int(j) for j in feasible_jobs if int(j) != int(chosen_job)]
        others = ", ".join(f"Job {j}" for j in other_jobs) if other_jobs else "(none)"

    return (
        "Explain why the already-selected action is reasonable for this JSSP step.\n"
        "Focus on final makespan (Cmax) and feasibility.\n\n"
        f"{state_text}\n\n"
        f"Selected action (fixed, do not change): {chosen_label}\n"
        f"Other feasible options: {others}\n\n"
        "Output format:\n"
        "Reason: <one concise sentence>\n"
        "Rules:\n"
        "- Output exactly one line starting with 'Reason:'.\n"
        "- Keep it concise (<= 25 words).\n"
        "- Do not change the selected action.\n"
        "- Do not output any 'Action:' line.\n"
        "- Do not output 'Not chosen:'.\n"
    )
