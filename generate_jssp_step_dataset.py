"""Generate step-by-step JSSP training data from one-shot train.json."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from llm_jssp.utils.jssp_step_env import (
    ParsedTeacherAction,
    StaticJSSPStepEnv,
    parse_prompt_jobs_first,
    parse_solution_actions,
)
from llm_jssp.utils.jssp_dispatch_env import DispatchJSSPStepEnv
from llm_jssp.utils.jssp_step_stack import resolve_step_stack

# Safe import for Korean path in project structure.
try:
    from llm_jssp.utils.step_reasoning import (
        _sorted_reason_alternatives,
        build_reason_input_text,
        build_teacher_step_rationale,
    )
except ImportError:  # pragma: no cover
    import sys

    CURRENT_DIR = Path(__file__).resolve().parent
    LEGACY_UTILS = CURRENT_DIR / "llm_jssp" / "utils"
    if str(LEGACY_UTILS) not in sys.path:
        sys.path.insert(0, str(LEGACY_UTILS))
    from step_reasoning import (  # type: ignore
        _sorted_reason_alternatives,
        build_reason_input_text,
        build_teacher_step_rationale,
    )


def _serial_teacher_job_sequence(
    teacher_actions: Sequence[ParsedTeacherAction],
) -> List[int]:
    return [int(action.job_id) for action in teacher_actions]


def build_dispatch_teacher_actions(
    inst_for_ortools: Sequence[Sequence[Sequence[int]]],
    teacher_actions: Sequence[ParsedTeacherAction],
) -> Tuple[List[ParsedTeacherAction], Dict[str, int]]:
    """
    Convert serial teacher order into a dispatch-valid decision order.

    The raw one-shot teacher order can choose jobs that are not dispatchable at the
    current event time. To create strict dispatch supervision, we first reconstruct
    the target schedule with the serial environment, then replay the same operations
    in an event-driven dispatch environment.

    At each decision epoch, at least one feasible job must match the target
    schedule start_time exactly. Otherwise, dispatch supervision generation fails
    immediately.
    """
    serial_env = StaticJSSPStepEnv(inst_for_ortools)
    serial_env.rollout_teacher(_serial_teacher_job_sequence(teacher_actions))
    target_event_by_key: Dict[Tuple[int, int], Dict[str, int]] = {
        (int(event["job_id"]), int(event["op_idx"])): dict(event)
        for event in serial_env.get_event_log()
    }

    dispatch_env = DispatchJSSPStepEnv(inst_for_ortools)
    dispatch_env.reset()

    dispatch_actions: List[ParsedTeacherAction] = []
    exact_decisions = 0
    projected_decisions = 0

    while not dispatch_env.is_done():
        state_json = dispatch_env.get_state_json()
        current_time = int(state_json["current_time"])
        feasible_jobs = list(state_json["feasible_jobs"])
        if not feasible_jobs:
            raise RuntimeError(
                "Dispatch environment reached a non-decision epoch while rebuilding teacher."
            )

        candidates: List[Tuple[int, int, int, int]] = []
        for job_id in feasible_jobs:
            op_idx = int(dispatch_env.job_next_op[job_id])
            target_event = target_event_by_key.get((int(job_id), int(op_idx)))
            if target_event is None:
                raise KeyError(
                    f"Missing target event for dispatch teacher rebuild: job={job_id}, op={op_idx}"
                )
            candidates.append(
                (
                    int(target_event["start_time"]),
                    int(target_event["machine_id"]),
                    int(job_id),
                    int(op_idx),
                )
            )

        exact_candidates = [item for item in candidates if int(item[0]) == current_time]
        if not exact_candidates:
            raise ValueError(
                "Dispatch teacher rebuild requires an exact feasible match at the current event time. "
                f"current_time={current_time}, feasible_jobs={feasible_jobs}, candidates={candidates}"
            )
        chosen_start, chosen_machine, chosen_job, chosen_op_idx = min(exact_candidates)
        exact_decisions += 1

        dispatch_actions.append(
            ParsedTeacherAction(
                job_id=int(chosen_job),
                op_idx=int(chosen_op_idx),
                machine_id=int(chosen_machine),
            )
        )
        dispatch_env.step(int(chosen_job))

    meta = {
        "dispatch_exact_decisions": int(exact_decisions),
        "dispatch_projected_decisions": int(projected_decisions),
    }
    return dispatch_actions, meta


def convert_example_to_step_rows(
    example: Dict[str, object],
    source_index: int,
    instance_id: Optional[str] = None,
    strict: bool = True,
    strict_makespan: bool = False,
    action_code_seed: int = 42,
    action_code_width: int = 4,
    action_code_cap: int = 9999,
    dataset_role: str = "both",
    reason_topk: int = 3,
    env_mode: str = "serial",
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """
    Convert one one-shot JSSP sample into `J*M` step rows.
    """
    prompt_jobs_first = str(example.get("prompt_jobs_first", ""))
    solution_text = str(example.get("output", ""))
    num_jobs = int(example.get("num_jobs", 0))
    num_machines = int(example.get("num_machines", 0))

    inst_for_ortools = parse_prompt_jobs_first(prompt_jobs_first, strict=strict)
    teacher_actions, declared_makespan = parse_solution_actions(solution_text, strict=strict)
    step_stack = resolve_step_stack(env_mode)
    env = step_stack.env_cls(inst_for_ortools)

    if num_jobs and env.num_jobs != num_jobs:
        raise ValueError(
            f"num_jobs mismatch at source_index={source_index}: header {num_jobs}, parsed {env.num_jobs}"
        )
    if num_machines and max(env.operations_per_job) != num_machines:
        raise ValueError(
            f"num_machines mismatch at source_index={source_index}: "
            f"header {num_machines}, parsed max ops/job {max(env.operations_per_job)}"
        )

    expected_steps = env.total_ops
    if len(teacher_actions) != expected_steps:
        raise ValueError(
            f"teacher action count mismatch at source_index={source_index}: "
            f"{len(teacher_actions)} vs expected {expected_steps}"
        )

    dispatch_teacher_meta = {
        "dispatch_exact_decisions": 0,
        "dispatch_projected_decisions": 0,
    }
    if str(env_mode) == "dispatch":
        rollout_actions, dispatch_teacher_meta = build_dispatch_teacher_actions(
            inst_for_ortools=inst_for_ortools,
            teacher_actions=teacher_actions,
        )
    else:
        rollout_actions = list(teacher_actions)

    resolved_instance_id = instance_id or f"train_{source_index:06d}"
    rows: List[Dict[str, object]] = []
    problem_context_text = step_stack.build_problem_context_text(inst_for_ortools)
    env.reset()
    if dataset_role not in {"policy", "reason", "both"}:
        raise ValueError(
            f"Unsupported dataset_role={dataset_role}. Use one of: policy, reason, both."
        )

    for step_idx, action in enumerate(rollout_actions):
        state_json = env.get_state_json()
        feasible_jobs = list(state_json["feasible_jobs"])
        target_job = int(action.job_id)
        target_op_idx = int(action.op_idx)
        target_machine = int(action.machine_id)

        if target_job not in feasible_jobs:
            raise ValueError(
                f"infeasible teacher job at source_index={source_index}, step={step_idx}: "
                f"job={target_job}, feasible={feasible_jobs}"
            )

        expected_op_idx = state_json["job_next_op"][target_job]
        expected_machine = state_json["next_machine"][target_job]
        if strict and target_op_idx != expected_op_idx:
            raise ValueError(
                f"teacher op_idx mismatch at source_index={source_index}, step={step_idx}: "
                f"expected {expected_op_idx}, got {target_op_idx}"
            )
        if strict and target_machine != expected_machine:
            raise ValueError(
                f"teacher machine mismatch at source_index={source_index}, step={step_idx}: "
                f"expected M{expected_machine}, got M{target_machine}"
            )

        step_rng = random.Random(
            int(action_code_seed) + int(source_index) * 1_000_003 + int(step_idx)
        )
        action_code_to_job = step_stack.build_randomized_action_code_map(
            feasible_jobs=feasible_jobs,
            rng=step_rng,
            code_width=action_code_width,
            code_start=1,
            code_cap=action_code_cap,
        )
        job_to_action_code = step_stack.invert_action_code_map(action_code_to_job)
        if target_job not in job_to_action_code:
            raise ValueError(
                f"target_job={target_job} not found in action_code_to_job at source_index={source_index}, "
                f"step={step_idx}, mapping={action_code_to_job}"
            )
        target_action_code = job_to_action_code[target_job]
        feasible_action_codes = list(action_code_to_job.keys())
        _, action_effects = step_stack.compute_action_transition_features(
            state_json=state_json,
            action_code_to_job=action_code_to_job,
        )
        effect_by_code = {
            str(effect["action_code"]): dict(effect)
            for effect in action_effects
        }
        chosen_effect = effect_by_code[str(target_action_code)]
        contrast_effects = _sorted_reason_alternatives(
            action_effects=action_effects,
            chosen_action_code=str(target_action_code),
            top_k=int(reason_topk),
        )

        state_text = step_stack.build_step_prompt(
            state_json=state_json,
            feasible_jobs=feasible_jobs,
            step_idx=step_idx,
            total_steps=expected_steps,
            problem_context_text=problem_context_text,
            action_code_to_job=action_code_to_job,
        )
        target_reason_text = build_teacher_step_rationale(
            state_json=state_json,
            feasible_jobs=feasible_jobs,
            chosen_job=target_job,
            action_code_to_job=action_code_to_job,
            compute_action_effects_fn=step_stack.compute_action_transition_features,
        )
        reason_input_text = build_reason_input_text(
            state_text=state_text,
            chosen_action_code=str(target_action_code),
            chosen_effect=chosen_effect,
            action_effects=action_effects,
            top_k=int(reason_topk),
            state_json=state_json,
        )
        target_action_reason_text = f"{target_action_code}\n{target_reason_text}"
        common_row = {
            "instance_id": resolved_instance_id,
            "source_index": source_index,
            "num_jobs": env.num_jobs,
            "num_machines": env.num_machines,
            "total_steps": expected_steps,
            "step_idx": step_idx,
            "state_json": state_json,
            "state_text": state_text,
            "problem_context_text": problem_context_text,
            "feasible_jobs": feasible_jobs,
            "feasible_action_codes": feasible_action_codes,
            "action_code_to_job": action_code_to_job,
            "teacher_operation_idx": target_op_idx,
            "teacher_machine": target_machine,
            "env_mode": str(env_mode),
            "dispatch_teacher_exact": (
                bool(dispatch_teacher_meta["dispatch_projected_decisions"] == 0)
                if str(env_mode) == "dispatch"
                else None
            ),
        }
        policy_feature_schema = (
            "jssp_step_policy_v2_action_token"
            if str(env_mode) == "serial"
            else "jssp_step_policy_dispatch_v1_action_token"
        )
        reason_feature_schema = (
            "jssp_step_reason_v2_action_token"
            if str(env_mode) == "serial"
            else "jssp_step_reason_dispatch_v1_action_token"
        )
        mixed_feature_schema = (
            "jssp_step_v3_transition_action_token"
            if str(env_mode) == "serial"
            else "jssp_step_dispatch_v1_transition_action_token"
        )
        policy_row = {
            **common_row,
            "feature_schema_version": policy_feature_schema,
            "target_job": target_job,
            "target_action_code": target_action_code,
            "target_text": target_action_code,
        }
        reason_row = {
            **common_row,
            "feature_schema_version": reason_feature_schema,
            "selected_job": target_job,
            "selected_action_code": target_action_code,
            "reason_input_text": reason_input_text,
            "reason_target_text": target_reason_text,
            "chosen_transition_features": chosen_effect,
            "contrast_action_codes": [
                str(effect["action_code"]) for effect in contrast_effects
            ],
            "contrast_transition_features": contrast_effects,
            "reason_source": "deterministic_teacher_v1",
        }
        both_row = {
            **policy_row,
            "feature_schema_version": mixed_feature_schema,
            "target_reason_text": target_reason_text,
            "target_action_reason_text": target_action_reason_text,
            "reason_input_text": reason_input_text,
            "reason_target_text": target_reason_text,
            "selected_job": target_job,
            "selected_action_code": target_action_code,
            "chosen_transition_features": chosen_effect,
            "contrast_action_codes": [
                str(effect["action_code"]) for effect in contrast_effects
            ],
            "contrast_transition_features": contrast_effects,
            "reason_source": "deterministic_teacher_v1",
        }
        if dataset_role == "policy":
            rows.append(policy_row)
        elif dataset_role == "reason":
            rows.append(reason_row)
        else:
            rows.append(both_row)

        env.step(target_job)

    if not env.is_done():
        raise ValueError(
            f"rollout did not finish at source_index={source_index}: "
            f"{env.scheduled_ops}/{env.total_ops}"
        )

    computed_makespan = env.get_makespan()
    if strict_makespan and declared_makespan is not None and computed_makespan != declared_makespan:
        raise ValueError(
            f"makespan mismatch at source_index={source_index}: "
            f"declared={declared_makespan}, computed={computed_makespan}"
        )

    meta = {
        "feature_schema_version": (
            (
                "jssp_step_policy_v2_action_token"
                if str(env_mode) == "serial"
                else "jssp_step_policy_dispatch_v1_action_token"
            )
            if dataset_role == "policy"
            else (
                "jssp_step_reason_v2_action_token"
                if str(env_mode) == "serial"
                else "jssp_step_reason_dispatch_v1_action_token"
            )
            if dataset_role == "reason"
            else (
                "jssp_step_v3_transition_action_token"
                if str(env_mode) == "serial"
                else "jssp_step_dispatch_v1_transition_action_token"
            )
        ),
        "dataset_role": dataset_role,
        "env_mode": str(env_mode),
        "instance_id": resolved_instance_id,
        "source_index": source_index,
        "num_jobs": env.num_jobs,
        "num_machines": env.num_machines,
        "steps": expected_steps,
        "declared_makespan": declared_makespan,
        "computed_makespan": computed_makespan,
        "makespan_match": (
            declared_makespan == computed_makespan if declared_makespan is not None else None
        ),
        "dispatch_exact_decisions": int(dispatch_teacher_meta["dispatch_exact_decisions"]),
        "dispatch_projected_decisions": int(dispatch_teacher_meta["dispatch_projected_decisions"]),
    }
    return rows, meta


def iter_slice(
    data: List[Dict[str, object]],
    start_idx: int,
    end_idx: Optional[int],
    max_instances: Optional[int],
) -> Iterable[Tuple[int, Dict[str, object]]]:
    start = max(0, start_idx)
    end = len(data) if end_idx is None else min(end_idx, len(data))
    selected = range(start, end)
    if max_instances is not None:
        selected = range(start, min(end, start + max_instances))
    for idx in selected:
        yield idx, data[idx]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one-shot JSSP train.json into step-by-step action dataset (JSONL)."
    )
    parser.add_argument("--input", type=str, default="llm_jssp/train.json")
    parser.add_argument("--output", type=str, default="train_data/jssp_step_train.jsonl")
    parser.add_argument(
        "--dataset_role",
        type=str,
        default="both",
        choices=["policy", "reason", "both"],
        help="Output step dataset role: policy(action only), reason(explanation only), or both(legacy mixed).",
    )
    parser.add_argument(
        "--env_mode",
        type=str,
        default="serial",
        choices=["serial", "dispatch"],
        help="Step environment mode used during rollout and state construction.",
    )
    parser.add_argument(
        "--reason_topk",
        type=int,
        default=3,
        help="Number of strongest alternative actions to include in reason dataset rows.",
    )
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--max_instances", type=int, default=None)
    parser.add_argument(
        "--no_strict",
        action="store_true",
        default=False,
        help="Disable strict checks for operation index/machine consistency.",
    )
    parser.add_argument(
        "--strict_makespan",
        action="store_true",
        default=False,
        help="Raise error when declared makespan and computed makespan mismatch.",
    )
    parser.add_argument(
        "--action_code_seed",
        type=int,
        default=42,
        help="Base seed for step-wise randomized <Axxxx> action token mapping.",
    )
    parser.add_argument(
        "--action_code_width",
        type=int,
        default=4,
        help="Fixed digit width in action token (e.g., 4 => <A0001>).",
    )
    parser.add_argument(
        "--action_code_cap",
        type=int,
        default=9999,
        help="Upper bound of action code index pool (sampled sparsely per step from [1, cap]).",
    )
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        default=False,
        help="Stop immediately on first invalid sample.",
    )
    parser.add_argument("--progress_every", type=int, default=100)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list of examples.")

    total_instances = 0
    ok_instances = 0
    skipped_instances = 0
    total_rows = 0
    makespan_mismatch = 0
    dispatch_projected_instances = 0
    dispatch_projected_steps = 0
    failures: List[Dict[str, object]] = []

    with output_path.open("w", encoding="utf-8") as out_f:
        for idx, example in iter_slice(data, args.start_idx, args.end_idx, args.max_instances):
            total_instances += 1
            try:
                rows, meta = convert_example_to_step_rows(
                    example=example,
                    source_index=idx,
                    strict=not args.no_strict,
                    strict_makespan=args.strict_makespan,
                    action_code_seed=args.action_code_seed,
                    action_code_width=args.action_code_width,
                    action_code_cap=args.action_code_cap,
                    dataset_role=args.dataset_role,
                    reason_topk=args.reason_topk,
                    env_mode=args.env_mode,
                )
                for row in rows:
                    out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                total_rows += len(rows)
                ok_instances += 1
                if meta["makespan_match"] is False:
                    makespan_mismatch += 1
                dispatch_projected_steps += int(meta.get("dispatch_projected_decisions", 0))
                if int(meta.get("dispatch_projected_decisions", 0)) > 0:
                    dispatch_projected_instances += 1
            except Exception as exc:  # pragma: no cover - exercised in real generation runs
                skipped_instances += 1
                failures.append({"source_index": idx, "error": str(exc)})
                if args.fail_fast:
                    raise

            if total_instances % args.progress_every == 0:
                print(
                    f"[progress] processed={total_instances} ok={ok_instances} "
                    f"skipped={skipped_instances} rows={total_rows}"
                )

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dataset_role": args.dataset_role,
        "env_mode": args.env_mode,
        "reason_topk": int(args.reason_topk),
        "processed_instances": total_instances,
        "ok_instances": ok_instances,
        "skipped_instances": skipped_instances,
        "total_rows": total_rows,
        "makespan_mismatch_instances": makespan_mismatch,
        "dispatch_projected_instances": dispatch_projected_instances,
        "dispatch_projected_steps": dispatch_projected_steps,
    }

    summary_path = output_path.with_suffix(".summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "failures": failures[:1000]}, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
