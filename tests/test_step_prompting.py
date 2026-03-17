import random

from llm_jssp.utils.step_prompting import (
    build_randomized_action_code_map,
    build_problem_context_text,
    build_step_improvement_prompt,
    build_step_rationale_prompt,
    build_step_prompt,
)


def test_step_prompt_with_problem_context():
    inst_for_ortools = [
        [[0, 3], [1, 2]],
        [[1, 4], [0, 1]],
    ]
    context = build_problem_context_text(inst_for_ortools)
    state_json = {
        "step_idx": 0,
        "total_steps": 4,
        "scheduled_ratio": 0.0,
        "current_cmax": 0,
        "job_next_op": [0, 0],
        "job_total_ops": [2, 2],
        "job_total_work": [5, 5],
        "job_ready_time": [0, 0],
        "machine_ready_time": [0, 0],
        "next_machine": [0, 1],
        "next_proc_time": [3, 4],
        "next2_machine": [1, 0],
        "next2_proc_time": [2, 1],
        "remaining_ops": [2, 2],
        "remaining_work": [5, 5],
        "total_remaining_work": 10,
        "unfinished_jobs_count": 2,
        "unfinished_jobs_ratio": 1.0,
        "machine_ready_min": 0,
        "machine_ready_mean": 0.0,
        "machine_ready_max": 0,
        "machine_ready_std": 0.0,
        "machine_remaining_load": [4, 6],
        "machine_remaining_ops": [2, 2],
        "bottleneck_machine_id": 1,
        "bottleneck_machine_load": 6,
        "bottleneck_machine_ops_left": 2,
    }
    prompt = build_step_prompt(
        state_json=state_json,
        feasible_jobs=[0, 1],
        step_idx=0,
        total_steps=4,
        problem_context_text=context,
        action_code_to_job={"<A0001>": 1, "<A0002>": 0},
    )
    assert "Static problem context:" in prompt
    assert "Job 0: ops=2, total_work=5, route=(M0,t3) -> (M1,t2)" in prompt
    assert "Feasible action tokens: [<A0001>, <A0002>]" in prompt
    assert "Return exactly one token from the feasible action set" in prompt


def test_step_improvement_prompt_format():
    p = build_step_improvement_prompt(
        state_text="Step: 1/4\nFeasible action tokens: [<A0001>, <A0002>]",
        candidate_action_text="<A0002>",
        feasible_jobs=["<A0001>", "<A0002>"],
    )
    assert "Candidate action:" in p
    assert "<A0002>" in p
    assert "<Axxxx>" in p


def test_step_rationale_prompt_format():
    p = build_step_rationale_prompt(
        state_text="Step: 2/4\nFeasible action tokens: [<A0001>, <A0002>, <A0003>]",
        chosen_action_code="<A0002>",
        feasible_action_codes=["<A0001>", "<A0002>", "<A0003>"],
    )
    assert "Selected action (fixed, do not change): <A0002>" in p
    assert "Other feasible options: <A0001>, <A0003>" in p
    assert "Reason:" in p


def test_action_code_map_sparse_sampling_and_capacity():
    rng = random.Random(0)
    mapping = build_randomized_action_code_map(
        feasible_jobs=[0, 1, 2],
        rng=rng,
        code_width=4,
        code_start=100,
        code_cap=105,
    )
    assert len(mapping) == 3
    assert len(set(mapping.keys())) == 3
    for code in mapping.keys():
        assert code.startswith("<A")
        idx = int(code[2:-1])
        assert 100 <= idx <= 105

    try:
        build_randomized_action_code_map(
            feasible_jobs=[0, 1, 2],
            rng=random.Random(0),
            code_width=4,
            code_start=1,
            code_cap=2,
        )
    except ValueError as exc:
        assert "Not enough action-code slots" in str(exc)
    else:
        raise AssertionError("Expected ValueError for insufficient action-code capacity")
