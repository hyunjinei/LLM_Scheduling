"""Dispatch-oriented prompt builders for event-driven JSSP."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .step_prompting import (
    build_problem_context_text,
    build_randomized_action_code_map,
    build_step_improvement_prompt,
    build_step_rationale_prompt,
    invert_action_code_map,
)


def _format_action_codes(action_codes: Sequence[str]) -> str:
    if not action_codes:
        return "[]"
    return "[" + ", ".join(str(code) for code in action_codes) + "]"


def _format_route_tokens(route_tokens: Sequence[str]) -> str:
    if not route_tokens:
        return "[]"
    return "[" + ", ".join(str(token) for token in route_tokens) + "]"


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _machine_token(machine_id: int) -> str:
    return f"M{int(machine_id)}" if int(machine_id) >= 0 else "M-"


def summarize_dispatch_state(state_json: Dict[str, object]) -> Dict[str, object]:
    machine_ready_time: List[int] = state_json["machine_ready_time"]  # type: ignore[assignment]
    current_cmax = int(
        state_json.get("current_cmax", max(machine_ready_time) if machine_ready_time else 0)
    )
    return {
        "step_idx": int(state_json["step_idx"]),
        "total_steps": int(state_json["total_steps"]),
        "scheduled_ratio": float(state_json.get("scheduled_ratio", 0.0)),
        "current_time": int(state_json.get("current_time", 0)),
        "current_cmax": current_cmax,
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

    current_time = int(state_json.get("current_time", 0))
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
        est_start = max(current_time, job_ready_before, machine_ready_before)
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

        effects.append(
            {
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
                "decision_time": int(current_time),
                "current_cmax_before": int(current_cmax),
                "current_cmax_after": int(current_cmax_after),
                "estimated_makespan_after": int(current_cmax_after),
                "delta_cmax": int(delta_cmax),
                "delta_cmax_ratio": (
                    float(delta_cmax) / float(max(current_cmax_after, 1))
                    if float(current_cmax_after) != 0.0
                    else 0.0
                ),
                "job_wait": int(max(0, current_time - job_ready_before)),
                "machine_idle_gap": int(max(0, current_time - machine_ready_before)),
                "slack_to_current_cmax": int(current_cmax - est_end),
                "affected_machine_load": affected_machine_load,
                "affected_machine_ops_left": affected_machine_ops_left,
                "affected_machine_load_ratio": (
                    float(affected_machine_load) / float(max(total_remaining_work, 1))
                ),
                "remaining_work_after_ratio": float(rem_work_after) / float(total_work),
                "post_route_tokens": list(post_route_tokens_by_job[job]),
                "post_route_len": int(len(post_route_tokens_by_job[job])),
            }
        )

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
        f"decision_t={effect['decision_time']} | "
        f"est_start={effect['estimated_start']} | "
        f"est_end={effect['estimated_end']} | "
        f"cmax:{effect['current_cmax_before']}->{effect['current_cmax_after']} | "
        f"delta_cmax={effect['delta_cmax']} | "
        f"job_ready:{effect['job_ready_before']}->{effect['job_ready_after']} | "
        f"machine_ready:{effect['machine_ready_before']}->{effect['machine_ready_after']} | "
        f"rem_ops:{effect['remaining_ops_before']}->{effect['remaining_ops_after']} | "
        f"rem_work:{effect['remaining_work_before']}->{effect['remaining_work_after']} | "
        f"machine_load={effect['affected_machine_load']} | "
        f"machine_ops_left={effect['affected_machine_ops_left']} | "
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
    lines = [
        "You are solving JSSP with event-driven dispatching.",
        "Objective: minimize final makespan (Cmax) while respecting precedence and machine availability.",
    ]
    if problem_context_text:
        lines.extend(["Static problem context:", problem_context_text])

    summary = summarize_dispatch_state(state_json)
    idle_machines = [int(x) for x in state_json.get("idle_machines", [])]  # type: ignore[arg-type]
    running_operations = list(state_json.get("running_operations", []) or [])
    lines.extend(
        [
            "Dispatch state:",
            (
                f"decision_step={int(summary['step_idx']) + 1}/{int(summary['total_steps'])} "
                f"scheduled_ratio={_format_value(summary['scheduled_ratio'])}"
            ),
            f"current_time={summary['current_time']}",
            f"current_cmax={summary['current_cmax']}",
            (
                f"unfinished_jobs_count={summary['unfinished_jobs_count']} "
                f"unfinished_jobs_ratio={_format_value(summary['unfinished_jobs_ratio'])}"
            ),
            (
                f"idle_machines={[ _machine_token(m) for m in idle_machines ]} "
                f"num_running_ops={len(running_operations)}"
            ),
            (
                f"machine_ready_min={summary['machine_ready_min']} "
                f"machine_ready_mean={_format_value(summary['machine_ready_mean'])} "
                f"machine_ready_max={summary['machine_ready_max']} "
                f"machine_ready_std={_format_value(summary['machine_ready_std'])}"
            ),
            (
                f"bottleneck_machine={_machine_token(int(summary['bottleneck_machine_id']))} "
                f"bottleneck_load={summary['bottleneck_machine_load']} "
                f"bottleneck_ops_left={summary['bottleneck_machine_ops_left']}"
            ),
        ]
    )
    if running_operations:
        lines.append("Running operations:")
        for op in running_operations:
            lines.append(
                f"Job {int(op['job_id'])} Op {int(op['op_idx'])} on "
                f"{_machine_token(int(op['machine_id']))}: "
                f"{int(op['start_time'])}->{int(op['end_time'])}"
            )

    if action_code_to_job:
        action_codes = list(action_code_to_job.keys())
        _, action_effects = compute_action_transition_features(
            state_json=state_json,
            action_code_to_job=action_code_to_job,
        )
        lines.extend(
            [
                f"Dispatchable action codes now: {_format_action_codes(action_codes)}",
                "Candidate dispatch effects:",
            ]
        )
        for effect in action_effects:
            lines.append(render_action_transition_line(effect))
        lines.extend(
            [
                "Action codes are randomized at each decision epoch. Do not assume persistent code identity.",
                "Choose exactly one dispatchable action code.",
                "Return exactly one code from the dispatchable action set, for example <A3812>.",
            ]
        )
    else:
        lines.extend(
            [
                f"Dispatchable jobs: {list(int(j) for j in feasible_jobs)}",
                "Choose exactly one dispatchable job.",
                "Return exactly one line: Action: Job <id>",
            ]
        )
    return "\n".join(lines)


__all__ = [
    "build_problem_context_text",
    "build_randomized_action_code_map",
    "invert_action_code_map",
    "build_step_prompt",
    "build_step_improvement_prompt",
    "build_step_rationale_prompt",
    "compute_action_transition_features",
    "render_action_transition_line",
]
