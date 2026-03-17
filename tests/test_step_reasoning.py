from llm_jssp.utils.step_reasoning import build_teacher_step_rationale


def test_build_teacher_step_rationale_format():
    state_json = {
        "job_ready_time": [0, 5, 2],
        "machine_ready_time": [3, 1],
        "next_machine": [0, 1, 0],
        "next_proc_time": [7, 4, 3],
        "remaining_work": [20, 10, 13],
        "remaining_ops": [3, 1, 2],
    }
    text = build_teacher_step_rationale(
        state_json=state_json,
        feasible_jobs=[0, 1, 2],
        chosen_job=0,
        action_code_to_job={"<A0001>": 1, "<A0002>": 0, "<A0003>": 2},
        max_not_chosen=3,
    )
    lines = text.splitlines()
    assert lines[0].startswith("Reason:")
    assert "Not chosen:" in text
    assert "- <A0001>:" in text or "- <A0003>:" in text
