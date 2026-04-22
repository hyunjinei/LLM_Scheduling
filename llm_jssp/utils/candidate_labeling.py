import random
from typing import Dict, Iterable, List, Sequence


def _format_value(value: object, digits: int = 4) -> str:
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.{int(digits)}f}".rstrip("0").rstrip(".")


def _machine_token(machine_id: int) -> str:
    return f"M{int(machine_id)}"


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _summarize_next_machines(post_route_tokens: Sequence[object], max_items: int = 3) -> str:
    machine_tokens: List[str] = []
    for token in list(post_route_tokens or []):
        text = str(token).strip()
        if not text:
            continue
        if ":" in text:
            text = text.split(":", 1)[0].strip()
        machine_tokens.append(text)
        if len(machine_tokens) >= int(max_items):
            break
    return "[" + ",".join(machine_tokens) + "]" if machine_tokens else "[]"


def _build_environment_lines(
    state_json: Dict[str, object],
    feasible_count: int,
) -> List[str]:
    current_time = state_json.get("current_time", None)
    current_cmax = _safe_int(state_json.get("current_cmax", 0))
    unfinished_jobs_count = _safe_int(state_json.get("unfinished_jobs_count", 0))
    machine_ready_time = [_safe_int(x) for x in state_json.get("machine_ready_time", [])]  # type: ignore[arg-type]
    machine_remaining_load = [_safe_int(x) for x in state_json.get("machine_remaining_load", [])]  # type: ignore[arg-type]
    machine_remaining_ops = [_safe_int(x) for x in state_json.get("machine_remaining_ops", [])]  # type: ignore[arg-type]
    idle_machines = [_safe_int(x) for x in state_json.get("idle_machines", [])]  # type: ignore[arg-type]
    running_operations = list(state_json.get("running_operations", []) or [])

    machine_ready_min = _safe_int(state_json.get("machine_ready_min", min(machine_ready_time) if machine_ready_time else 0))
    machine_ready_mean = _safe_float(
        state_json.get(
            "machine_ready_mean",
            (sum(machine_ready_time) / float(len(machine_ready_time))) if machine_ready_time else 0.0,
        )
    )
    machine_ready_max = _safe_int(state_json.get("machine_ready_max", max(machine_ready_time) if machine_ready_time else 0))
    machine_ready_std = _safe_float(state_json.get("machine_ready_std", 0.0))

    load_mean = (sum(machine_remaining_load) / float(len(machine_remaining_load))) if machine_remaining_load else 0.0
    load_max = max(machine_remaining_load) if machine_remaining_load else 0
    if machine_remaining_load:
        variance = sum((float(x) - load_mean) ** 2 for x in machine_remaining_load) / float(len(machine_remaining_load))
        load_std = variance ** 0.5
    else:
        load_std = 0.0

    lines = ["Environment state:"]
    if current_time is not None:
        lines.append(f"current_time={_safe_int(current_time)}")
    lines.append(f"current_cmax={current_cmax}")
    lines.append(f"unfinished_jobs_count={unfinished_jobs_count}")
    lines.append(f"feasible_candidates_count={int(feasible_count)}")
    if current_time is not None:
        lines.append(f"idle_machine_count={len(idle_machines)}")
        lines.append(f"running_ops_count={len(running_operations)}")
    lines.append(
        " | ".join(
            [
                f"machine_ready_min={machine_ready_min}",
                f"machine_ready_mean={_format_value(machine_ready_mean)}",
                f"machine_ready_max={machine_ready_max}",
                f"machine_ready_std={_format_value(machine_ready_std)}",
            ]
        )
    )
    lines.append(
        " | ".join(
            [
                f"machine_load_mean={_format_value(load_mean)}",
                f"machine_load_max={load_max}",
                f"machine_load_std={_format_value(load_std)}",
            ]
        )
    )
    lines.append("Machine snapshot:")
    idle_set = {int(x) for x in idle_machines}
    reference_time = _safe_int(current_time, current_cmax)
    for machine_id in range(len(machine_ready_time)):
        ready_at = machine_ready_time[machine_id]
        rem_load = machine_remaining_load[machine_id] if machine_id < len(machine_remaining_load) else 0
        rem_ops = machine_remaining_ops[machine_id] if machine_id < len(machine_remaining_ops) else 0
        if current_time is None:
            status = f"ready_at={ready_at}"
        elif machine_id in idle_set and ready_at <= reference_time:
            status = "idle"
        elif ready_at > reference_time:
            status = f"busy_until={ready_at}"
        else:
            status = "ready"
        lines.append(
            f"{_machine_token(machine_id)} | ready_at={ready_at} | rem_load={rem_load} | rem_ops={rem_ops} | status={status}"
        )
    lines.append("Feasible candidates:")
    return lines


def _build_candidate_display_line(
    candidate_label: str,
    effect: Dict[str, object],
    state_json: Dict[str, object],
) -> str:
    machine_id = _safe_int(effect.get("machine_id", effect.get("next_machine", 0)))
    proc_time = _safe_int(effect.get("proc_time", effect.get("next_proc_time", 0)))
    est_start = _safe_int(effect.get("est_start", effect.get("estimated_start", 0)))
    est_end = _safe_int(effect.get("est_end", effect.get("estimated_end", 0)))
    decision_time = _safe_int(effect.get("decision_time", state_json.get("current_time", state_json.get("current_cmax", 0))))
    wait_from_now = max(0, est_start - decision_time)
    remaining_ops_after = _safe_int(effect.get("remaining_ops_after", 0))
    remaining_work_after = _safe_int(effect.get("remaining_work_after", 0))
    remaining_work_after_ratio = _safe_float(effect.get("remaining_work_after_ratio", 0.0))
    machine_load_before = _safe_int(effect.get("affected_machine_load", 0))
    machine_ops_left_before = _safe_int(effect.get("affected_machine_ops_left", 0))
    next_route_len = _safe_int(effect.get("post_route_len", 0))
    next_machines = _summarize_next_machines(effect.get("post_route_tokens", []))
    delta_cmax = _safe_int(effect.get("delta_cmax", 0))
    return (
        f"{str(candidate_label)} | "
        f"machine={_machine_token(machine_id)} | "
        f"proc_time={proc_time} | "
        f"est_start={est_start} | "
        f"est_end={est_end} | "
        f"wait_from_now={wait_from_now} | "
        f"delta_cmax={delta_cmax} | "
        f"remaining_ops_after={remaining_ops_after} | "
        f"remaining_work_after={remaining_work_after} | "
        f"remaining_work_after_ratio={_format_value(remaining_work_after_ratio)} | "
        f"machine_load_before={machine_load_before} | "
        f"machine_ops_left_before={machine_ops_left_before} | "
        f"next_route_len={next_route_len} | "
        f"next_machines={next_machines}"
    )


def build_randomized_candidate_label_map(
    feasible_jobs: Sequence[int],
    rng: random.Random | None = None,
    *,
    label_mode: str = "random_id",
    id_min: int = 1,
    id_max: int = 9999,
) -> Dict[str, int]:
    jobs = [int(job_id) for job_id in feasible_jobs]
    if len(set(jobs)) != len(jobs):
        raise ValueError(f"feasible_jobs contains duplicates: {jobs}")
    if not jobs:
        return {}

    normalized_mode = str(label_mode or "random_id").strip().lower()
    if normalized_mode not in {"random_id", "ordinal"}:
        raise ValueError(f"Unsupported label_mode={label_mode}")

    local_rng = rng or random.Random()
    shuffled_jobs = list(jobs)
    local_rng.shuffle(shuffled_jobs)

    if normalized_mode == "ordinal":
        labels = [f"Candidate {idx}" for idx in range(1, len(shuffled_jobs) + 1)]
    else:
        id_min = int(id_min)
        id_max = int(id_max)
        if id_max < id_min:
            raise ValueError(
                f"id_max must be >= id_min, got id_min={id_min}, id_max={id_max}"
            )
        if len(shuffled_jobs) > (id_max - id_min + 1):
            raise ValueError(
                "Not enough candidate-id slots for this step: "
                f"jobs={len(shuffled_jobs)}, available={id_max - id_min + 1}, "
                f"range=[{id_min}, {id_max}]"
            )
        sampled_ids = local_rng.sample(range(id_min, id_max + 1), k=len(shuffled_jobs))
        labels = [f"Candidate ID {candidate_id}" for candidate_id in sampled_ids]

    return {str(label): int(job_id) for label, job_id in zip(labels, shuffled_jobs)}


def rewrite_prompt_for_candidate_scoring(
    prompt_text: str,
    candidate_labels: Sequence[str] | None = None,
) -> str:
    label_list = [str(label) for label in (candidate_labels or [])]
    rewritten_lines: List[str] = []
    for line in str(prompt_text).splitlines():
        stripped = line.strip()
        if stripped.startswith("Dispatchable action codes now:"):
            rewritten_lines.append(
                f"Feasible candidates: {label_list}"
                if label_list
                else "Feasible candidates are listed below."
            )
            continue
        if stripped.startswith("Feasible action codes:"):
            rewritten_lines.append(
                f"Feasible candidates: {label_list}"
                if label_list
                else "Feasible candidates are listed below."
            )
            continue
        if "Action codes are randomized at each decision epoch." in stripped:
            rewritten_lines.append(
                "Candidate labels are step-local identifiers. Do not assume persistent identity across steps."
            )
            continue
        if stripped == "Choose exactly one dispatchable action code.":
            rewritten_lines.append("Choose exactly one feasible candidate.")
            continue
        if stripped == "Choose exactly one feasible action code.":
            rewritten_lines.append("Choose exactly one feasible candidate.")
            continue
        if stripped.startswith("Return exactly one code from the dispatchable action set"):
            rewritten_lines.append(
                "Select the strongest candidate from the feasible candidate set."
            )
            continue
        if stripped.startswith("Return exactly one code from the feasible action set"):
            rewritten_lines.append(
                "Select the strongest candidate from the feasible candidate set."
            )
            continue
        rewritten_lines.append(line)
    return "\n".join(rewritten_lines)


def prepare_candidate_scoring_prompt(
    prompt_text: str | None,
    candidate_labels: Sequence[str],
    *,
    state_json: Dict[str, object] | None = None,
    action_effects: Sequence[Dict[str, object]] | None = None,
) -> Dict[str, object]:
    candidate_labels = [str(label) for label in candidate_labels if str(label).strip()]
    if not candidate_labels:
        raise ValueError("candidate_labels must be non-empty.")

    if state_json is not None and action_effects is not None:
        effect_by_code = {str(effect.get("action_code", "")): dict(effect) for effect in action_effects}
        candidate_display_lines = []
        candidate_labels_in_order = []
        for label in candidate_labels:
            effect = effect_by_code.get(str(label))
            if effect is None:
                raise ValueError(f"Missing action effect for candidate label {label!r}")
            candidate_labels_in_order.append(str(label))
            candidate_display_lines.append(
                _build_candidate_display_line(
                    candidate_label=str(label),
                    effect=effect,
                    state_json=state_json,
                )
            )

        prefix_lines = _build_environment_lines(
            state_json=state_json,
            feasible_count=len(candidate_labels_in_order),
        )
        state_prefix_text = "\n".join(prefix_lines).strip()
        candidate_section_text = "\n".join(candidate_display_lines).strip()
        candidate_scoring_state_text = (
            f"{state_prefix_text}\n{candidate_section_text}"
            if state_prefix_text
            else candidate_section_text
        )
        return {
            "state_prefix_text": state_prefix_text,
            "candidate_display_lines": list(candidate_display_lines),
            "candidate_labels_in_order": list(candidate_labels_in_order),
            "candidate_scoring_state_text": candidate_scoring_state_text,
        }

    prompt_text = str(prompt_text or "")

    label_match_order = sorted(candidate_labels, key=len, reverse=True)
    prefix_lines: List[str] = []
    candidate_display_lines: List[str] = []
    candidate_labels_in_order: List[str] = []

    for raw_line in prompt_text.splitlines():
        stripped = raw_line.strip()
        matched_label = next(
            (
                label
                for label in label_match_order
                if stripped.startswith(f"{label} |") or stripped == label
            ),
            None,
        )
        if matched_label is None:
            prefix_lines.append(raw_line)
            continue
        candidate_display_lines.append(raw_line)
        candidate_labels_in_order.append(str(matched_label))

    if not candidate_display_lines:
        raise ValueError("No candidate display lines were found in prompt_text.")

    state_prefix_text = "\n".join(prefix_lines).strip()
    candidate_section_text = "\n".join(candidate_display_lines).strip()
    if state_prefix_text:
        candidate_scoring_state_text = f"{state_prefix_text}\n{candidate_section_text}"
    else:
        candidate_scoring_state_text = candidate_section_text

    return {
        "state_prefix_text": state_prefix_text,
        "candidate_display_lines": list(candidate_display_lines),
        "candidate_labels_in_order": list(candidate_labels_in_order),
        "candidate_scoring_state_text": candidate_scoring_state_text,
    }


def find_target_candidate_index(
    candidate_labels_in_order: Sequence[str],
    target_candidate_label: str,
) -> int:
    normalized = [str(label) for label in candidate_labels_in_order]
    target = str(target_candidate_label)
    if target not in normalized:
        raise ValueError(
            f"target_candidate_label={target!r} not found in candidate_labels_in_order={normalized}"
        )
    return int(normalized.index(target))


def strip_candidate_labels(candidate_display_lines: Iterable[str]) -> List[str]:
    stripped_lines = []
    for line in candidate_display_lines:
        text = str(line)
        if " | " in text:
            stripped_lines.append(text.split(" | ", 1)[1].strip())
        else:
            stripped_lines.append(text.strip())
    return stripped_lines
