"""Deterministic rationale helpers and label builders for step-by-step JSSP data."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

try:
    from .step_prompting import (
        compute_action_transition_features,
        invert_action_code_map,
    )
except ImportError:  # pragma: no cover
    from step_prompting import compute_action_transition_features, invert_action_code_map  # type: ignore


def _sorted_reason_alternatives(
    action_effects: Sequence[Dict[str, object]],
    chosen_action_code: str,
    top_k: int = 3,
) -> List[Dict[str, object]]:
    alternatives = [
        dict(effect)
        for effect in action_effects
        if str(effect.get("action_code")) != str(chosen_action_code)
    ]
    alternatives.sort(
        key=lambda x: (
            int(x.get("estimated_makespan_after", 10**12)),
            int(x.get("delta_cmax", 10**12)),
            int(x.get("estimated_start", 10**12)),
            int(x.get("proc_time", 10**12)),
            int(x.get("job_id", 10**12)),
        )
    )
    return alternatives[: max(0, int(top_k))]


def _machine_label(machine_id: object) -> str:
    machine_int = int(machine_id)
    return f"M{machine_int}" if machine_int >= 0 else "M-"


def _effect_bottleneck_relation(
    effect: Dict[str, object],
    bottleneck_machine_id: int,
) -> str:
    machine_id = int(effect.get("machine_id", -1))
    next2_machine = int(effect.get("next2_machine", -1))
    remaining_ops_after = int(effect.get("remaining_ops_after", 0))
    if bottleneck_machine_id < 0:
        return "unknown"
    if machine_id == bottleneck_machine_id:
        return "direct"
    if remaining_ops_after > 0 and next2_machine == bottleneck_machine_id:
        return "releases_future_bottleneck"
    return "indirect"


def _effect_summary_lines(
    effect: Dict[str, object],
    state_json: Optional[Dict[str, object]] = None,
) -> List[str]:
    bottleneck_machine_id = (
        int(state_json.get("bottleneck_machine_id", -1))
        if state_json is not None
        else -1
    )
    return [
        f"- next_machine: {_machine_label(effect['machine_id'])}",
        f"- next2_machine: {_machine_label(effect.get('next2_machine', -1))}",
        f"- bottleneck_relation: {_effect_bottleneck_relation(effect, bottleneck_machine_id)}",
        f"- cmax: {int(effect['current_cmax_before'])}->{int(effect['current_cmax_after'])}",
        f"- delta_cmax: {int(effect['delta_cmax'])}",
        f"- est_start: {int(effect['estimated_start'])}",
        f"- est_end: {int(effect['estimated_end'])}",
        f"- machine_idle_gap: {int(effect['machine_idle_gap'])}",
        f"- job_wait: {int(effect['job_wait'])}",
        f"- remaining_ops_after: {int(effect['remaining_ops_after'])}",
        f"- remaining_work_after: {int(effect['remaining_work_after'])}",
        f"- machine_load: {int(effect['affected_machine_load'])}",
        f"- downstream_route: {_machine_label(effect.get('next2_machine', -1))}, next2_p={int(effect.get('next2_proc_time', 0))}",
    ]


def build_reason_input_text(
    state_text: str,
    chosen_action_code: str,
    chosen_effect: Dict[str, object],
    action_effects: Sequence[Dict[str, object]],
    top_k: int = 3,
    state_json: Optional[Dict[str, object]] = None,
) -> str:
    alternatives = _sorted_reason_alternatives(
        action_effects=action_effects,
        chosen_action_code=chosen_action_code,
        top_k=top_k,
    )
    bottleneck_machine_id = (
        int(state_json.get("bottleneck_machine_id", -1))
        if state_json is not None
        else -1
    )
    bottleneck_load = (
        int(state_json.get("bottleneck_machine_load", 0))
        if state_json is not None
        else 0
    )
    bottleneck_ops_left = (
        int(state_json.get("bottleneck_machine_ops_left", 0))
        if state_json is not None
        else 0
    )
    lines = [
        "You are analyzing an already-selected JSSP action.",
        (
            "Objective: explain why this action was selected and why strong alternatives "
            "were not selected."
        ),
        (
            "Ground the explanation in explicit evidence only: immediate timing/Cmax impact, "
            "bottleneck-machine use or release, downstream route exposure (next2), "
            "waiting/idle trade-offs, and contrast against top alternatives."
        ),
        "",
        state_text,
        "",
        f"Selected action: {chosen_action_code}",
        "Critical context:",
        (
            f"- bottleneck_machine: {_machine_label(bottleneck_machine_id)} "
            f"(remaining_load={bottleneck_load}, ops_left={bottleneck_ops_left})"
        ),
        "Chosen transition:",
        *_effect_summary_lines(chosen_effect, state_json=state_json),
        "",
        "Top alternatives:",
    ]
    if alternatives:
        for alt in alternatives:
            lines.append(
                (
                    f"- {alt['action_code']}: "
                    f"machine={_machine_label(alt['machine_id'])}, "
                    f"next2={_machine_label(alt.get('next2_machine', -1))}, "
                    f"cmax {int(alt['current_cmax_before'])}->{int(alt['current_cmax_after'])}, "
                    f"delta_cmax={int(alt['delta_cmax'])}, "
                    f"est_start={int(alt['estimated_start'])}, "
                    f"est_end={int(alt['estimated_end'])}, "
                    f"job_wait={int(alt['job_wait'])}, "
                    f"machine_idle_gap={int(alt['machine_idle_gap'])}, "
                    f"remaining_work_after={int(alt['remaining_work_after'])}, "
                    f"machine_load={int(alt['affected_machine_load'])}, "
                    f"bottleneck_relation={_effect_bottleneck_relation(alt, bottleneck_machine_id)}"
                )
            )
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "Output format:",
            "Reason: ...",
            "Not chosen:",
            "- <Axxxx>: ...",
        ]
    )
    return "\n".join(lines)


def _chosen_bottleneck_clause(
    effect: Dict[str, object],
    bottleneck_machine_id: int,
    bottleneck_load: int,
    bottleneck_ops_left: int,
) -> str:
    machine_id = int(effect.get("machine_id", -1))
    next2_machine = int(effect.get("next2_machine", -1))
    next2_proc_time = int(effect.get("next2_proc_time", 0))
    affected_machine_load = int(effect.get("affected_machine_load", 0))
    if bottleneck_machine_id >= 0 and machine_id == bottleneck_machine_id:
        return (
            f"It directly activates the current bottleneck machine {_machine_label(machine_id)} "
            f"(remaining_load={bottleneck_load}, ops_left={bottleneck_ops_left}), so delaying this move "
            "would postpone work on the heaviest unresolved machine frontier."
        )
    if bottleneck_machine_id >= 0 and next2_machine == bottleneck_machine_id:
        return (
            f"It also exposes a downstream step on bottleneck {_machine_label(bottleneck_machine_id)} "
            f"(next2_p={next2_proc_time}), pulling future critical work closer."
        )
    if bottleneck_machine_id >= 0 and affected_machine_load >= max(1, int(0.75 * bottleneck_load)):
        return (
            f"It works on high-load machine {_machine_label(machine_id)} "
            f"(remaining_load={affected_machine_load}), which is close to the current bottleneck pressure."
        )
    return (
        f"It advances the route on {_machine_label(machine_id)} without adding unnecessary "
        "timing friction to the current schedule state."
    )


def _chosen_progress_clause(effect: Dict[str, object]) -> str:
    rem_ops_before = int(effect.get("remaining_ops_before", 0))
    rem_ops_after = int(effect.get("remaining_ops_after", 0))
    rem_work_before = int(effect.get("remaining_work_before", 0))
    rem_work_after = int(effect.get("remaining_work_after", 0))
    next2_machine = int(effect.get("next2_machine", -1))
    next2_proc_time = int(effect.get("next2_proc_time", 0))
    if rem_ops_after <= 0:
        return (
            f"It completes this job, eliminating the remaining work from {rem_work_before} to 0."
        )
    if next2_machine >= 0:
        return (
            f"It reduces this job from remaining_work {rem_work_before}->{rem_work_after} "
            f"and remaining_ops {rem_ops_before}->{rem_ops_after}, while exposing the next route "
            f"on {_machine_label(next2_machine)} (t={next2_proc_time})."
        )
    return (
        f"It reduces this job from remaining_work {rem_work_before}->{rem_work_after} "
        f"and remaining_ops {rem_ops_before}->{rem_ops_after}."
    )


def _chosen_wait_clause(effect: Dict[str, object]) -> Optional[str]:
    machine_idle_gap = int(effect.get("machine_idle_gap", 0))
    job_wait = int(effect.get("job_wait", 0))
    machine_id = int(effect.get("machine_id", -1))
    if machine_idle_gap > 0:
        return (
            f"It fills machine idle gap={machine_idle_gap} on {_machine_label(machine_id)}, "
            "which helps avoid leaving that machine unused."
        )
    if job_wait > 0:
        return (
            f"It intentionally waits {job_wait} time units for {_machine_label(machine_id)}, "
            "indicating that accessing this machine/route is worth the queue delay."
        )
    return "It can start immediately with no extra machine idle gap and no queue wait."


def _chosen_tradeoff_clause(
    chosen: Dict[str, object],
    best_alternative: Optional[Dict[str, object]],
) -> str:
    chosen_after = int(chosen.get("current_cmax_after", chosen.get("estimated_makespan_after", 0)))
    chosen_delta = int(chosen.get("delta_cmax", 0))
    if best_alternative is None:
        if chosen_delta == 0:
            return "It fits under the current Cmax and does not increase makespan immediately."
        return (
            f"It changes projected Cmax {int(chosen['current_cmax_before'])}->{chosen_after} "
            f"(delta={chosen_delta})."
        )

    best_after = int(
        best_alternative.get("current_cmax_after", best_alternative.get("estimated_makespan_after", 0))
    )
    gap = chosen_after - best_after
    if gap < 0:
        return (
            f"Among the strongest alternatives, it gives the smallest immediate projected Cmax "
            f"({chosen_after} vs {best_after})."
        )
    if gap == 0:
        if chosen_delta == 0:
            return "Its immediate projected Cmax matches the strongest alternatives while staying under the current makespan frontier."
        return (
            f"Its immediate projected Cmax matches the strongest alternative "
            f"({chosen_after}), so the tie is broken by route and bottleneck considerations."
        )
    if gap <= max(3, int(chosen.get("proc_time", 0)) // 2):
        return (
            f"It accepts only a small immediate Cmax penalty (+{gap} vs the strongest alternative) "
            "in exchange for better bottleneck/route progression."
        )
    return (
        f"Although its immediate projected Cmax is worse by +{gap} than the strongest alternative, "
        "the preference comes from stronger bottleneck alignment and downstream progression rather "
        "than short-term Cmax alone."
    )


def _alt_reason_line(
    alt: Dict[str, object],
    chosen: Dict[str, object],
    bottleneck_machine_id: int,
) -> str:
    alt_after = int(alt.get("current_cmax_after", alt.get("estimated_makespan_after", 0)))
    chosen_after = int(chosen.get("current_cmax_after", chosen.get("estimated_makespan_after", 0)))
    alt_machine = int(alt.get("machine_id", -1))
    chosen_machine = int(chosen.get("machine_id", -1))
    alt_next2 = int(alt.get("next2_machine", -1))
    chosen_next2 = int(chosen.get("next2_machine", -1))

    clauses: List[str] = []
    if alt_after > chosen_after:
        clauses.append(
            f"higher immediate projected Cmax ({alt_after} vs {chosen_after})"
        )
    elif alt_after < chosen_after:
        clauses.append(
            f"smaller immediate projected Cmax ({alt_after} vs {chosen_after}), but weaker structural progression"
        )
    else:
        clauses.append(
            f"same immediate projected Cmax ({alt_after})"
        )

    if bottleneck_machine_id >= 0 and chosen_machine == bottleneck_machine_id and alt_machine != bottleneck_machine_id:
        clauses.append(
            f"does not activate bottleneck {_machine_label(bottleneck_machine_id)}"
        )
    elif bottleneck_machine_id >= 0 and chosen_next2 == bottleneck_machine_id and alt_next2 != bottleneck_machine_id:
        clauses.append(
            f"does not pull the future bottleneck step on {_machine_label(bottleneck_machine_id)} forward"
        )
    elif int(alt.get("affected_machine_load", 0)) + 1 < int(chosen.get("affected_machine_load", 0)):
        clauses.append(
            f"works on a lighter machine (load={int(alt.get('affected_machine_load', 0))} vs {int(chosen.get('affected_machine_load', 0))})"
        )

    if int(alt.get("machine_idle_gap", 0)) > int(chosen.get("machine_idle_gap", 0)):
        clauses.append(
            f"larger machine idle gap ({int(alt.get('machine_idle_gap', 0))} vs {int(chosen.get('machine_idle_gap', 0))})"
        )
    elif int(alt.get("job_wait", 0)) > int(chosen.get("job_wait", 0)):
        clauses.append(
            f"more waiting before start ({int(alt.get('job_wait', 0))} vs {int(chosen.get('job_wait', 0))})"
        )

    rem_work_after = int(alt.get("remaining_work_after", 0))
    rem_ops_after = int(alt.get("remaining_ops_after", 0))
    if rem_ops_after > 0 and alt_next2 >= 0:
        clauses.append(
            f"after this move it still leaves {rem_work_after} work and {rem_ops_after} ops, with next route on {_machine_label(alt_next2)} (t={int(alt.get('next2_proc_time', 0))})"
        )
    else:
        clauses.append(
            f"after this move it still leaves {rem_work_after} work and {rem_ops_after} ops on that job"
        )

    clauses = clauses[:4]
    return "; ".join(clauses) + "."


def build_teacher_step_rationale(
    state_json: Dict[str, object],
    feasible_jobs: Sequence[int],
    chosen_job: int,
    action_code_to_job: Optional[Dict[str, int]] = None,
    max_not_chosen: int = 6,
    compute_action_effects_fn=None,
) -> str:
    """
    Build compact deterministic rationale text for supervision.

    Output format:
        Reason: ...
        Not chosen:
        - Job k: ...
    """
    feasible = [int(j) for j in feasible_jobs]
    chosen = int(chosen_job)
    if chosen not in feasible:
        feasible = feasible + [chosen]

    label_by_job: Dict[int, str]
    action_code_by_job: Dict[int, str]
    if action_code_to_job:
        label_by_job = invert_action_code_map(action_code_to_job)
        if chosen not in label_by_job:
            raise ValueError(
                f"chosen_job={chosen} is not found in action_code_to_job={action_code_to_job}"
            )
        action_code_by_job = dict(label_by_job)
    else:
        label_by_job = {j: f"Job {j}" for j in feasible}
        action_code_by_job = {j: f"Job {j}" for j in feasible}

    compute_fn = compute_action_effects_fn or compute_action_transition_features
    _, action_effects = compute_fn(
        state_json=state_json,
        action_code_to_job={action_code_by_job[j]: j for j in feasible},
    )
    feats = {int(effect["job_id"]): effect for effect in action_effects}
    c = feats[chosen]
    chosen_label = label_by_job[chosen]
    bottleneck_machine_id = int(state_json.get("bottleneck_machine_id", -1))
    bottleneck_load = int(state_json.get("bottleneck_machine_load", 0))
    bottleneck_ops_left = int(state_json.get("bottleneck_machine_ops_left", 0))

    best_alternative: Optional[Dict[str, object]] = None
    if len(feasible) > 1:
        alt_candidates = [feats[j] for j in feasible if j != chosen and j in feats]
        if alt_candidates:
            alt_candidates.sort(
                key=lambda f: (
                    int(f["estimated_makespan_after"]),
                    int(f["delta_cmax"]),
                    int(f["estimated_start"]),
                    int(f["proc_time"]),
                    int(f["job_id"]),
                )
            )
            best_alternative = alt_candidates[0]

    clauses = [
        (
            f"{chosen_label} is feasible on {_machine_label(c['machine_id'])} "
            f"(est_start={int(c['estimated_start'])}, est_end={int(c['estimated_end'])})."
        ),
        (
            f"Projected Cmax changes {int(c['current_cmax_before'])}->{int(c['current_cmax_after'])} "
            f"(delta={int(c['delta_cmax'])})."
        ),
        _chosen_progress_clause(c),
        _chosen_bottleneck_clause(
            c,
            bottleneck_machine_id=bottleneck_machine_id,
            bottleneck_load=bottleneck_load,
            bottleneck_ops_left=bottleneck_ops_left,
        ),
        _chosen_wait_clause(c),
        _chosen_tradeoff_clause(c, best_alternative),
    ]

    reason_line = "Reason: " + " ".join(clause for clause in clauses if clause)

    candidates = [j for j in feasible if j != chosen]
    candidates = [j for j in candidates if j in feats]
    candidates.sort(
        key=lambda j: (
            int(feats[j]["estimated_makespan_after"]),
            int(feats[j]["estimated_start"]),
            int(feats[j]["proc_time"]),
            j,
        )
    )
    limited = candidates[: max(0, int(max_not_chosen))]

    lines = [reason_line, "Not chosen:"]
    if not limited:
        lines.append("- (none)")
        return "\n".join(lines)

    for j in limited:
        f = feats[j]
        label = label_by_job[j]
        lines.append(f"- {label}: {_alt_reason_line(f, c, bottleneck_machine_id)}")
    return "\n".join(lines)
