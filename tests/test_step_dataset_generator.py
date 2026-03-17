from generate_jssp_step_dataset import convert_example_to_step_rows
from llm_jssp.utils.jssp_step_env import StaticJSSPStepEnv
import re


def _build_output_from_job_sequence(inst_for_ortools, job_sequence):
    env = StaticJSSPStepEnv(inst_for_ortools)
    env.reset()
    lines = ["Solution:", ""]
    for job_id in job_sequence:
        op_idx = env.job_next_op[job_id]
        machine_id = env.inst_for_ortools[job_id][op_idx][0]
        lines.append(f"Job {job_id} Operation {op_idx}, M{machine_id}")
        env.step(job_id)
    lines.append("")
    lines.append(f"Makespan: {env.get_makespan()}")
    return "\n".join(lines)


def test_convert_example_to_step_rows_small():
    prompt = (
        "JSSP with 2 Jobs, 2 Machines. Makespan 최소화.\n\n"
        "Job 0 consists of Operations:\n"
        "Operation 0: M0,3\n"
        "Operation 1: M1,2\n\n"
        "Job 1 consists of Operations:\n"
        "Operation 0: M1,2\n"
        "Operation 1: M0,4\n"
    )
    inst_for_ortools = [
        [[0, 3], [1, 2]],
        [[1, 2], [0, 4]],
    ]
    teacher_jobs = [0, 1, 0, 1]
    output = _build_output_from_job_sequence(inst_for_ortools, teacher_jobs)

    example = {
        "num_jobs": 2,
        "num_machines": 2,
        "prompt_jobs_first": prompt,
        "output": output,
    }
    rows, meta = convert_example_to_step_rows(example, source_index=0, strict=True)

    assert len(rows) == 4
    assert re.fullmatch(r"Action:\s*<S\d{4}>", rows[0]["target_text"])
    assert rows[0]["target_reason_text"].startswith("Reason:")
    assert rows[0]["target_action_reason_text"].startswith(
        f"{rows[0]['target_text']}\nReason:"
    )
    assert re.fullmatch(r"Action:\s*<S\d{4}>", rows[1]["target_text"])
    assert rows[0]["step_idx"] == 0
    assert rows[-1]["step_idx"] == 3
    assert meta["steps"] == 4
    assert meta["computed_makespan"] == 7

    for row in rows:
        assert row["target_job"] in row["feasible_jobs"]
        assert row["target_action_code"] in row["feasible_action_codes"]
        assert int(row["action_code_to_job"][row["target_action_code"]]) == int(row["target_job"])


def test_10x10_generates_100_rows():
    # Deterministic 10x10 synthetic instance where op machine cycles with job offset.
    num_jobs = 10
    num_machines = 10
    inst_for_ortools = []
    for j in range(num_jobs):
        job_ops = []
        for op in range(num_machines):
            machine = (j + op) % num_machines
            duration = 1 + ((j * 7 + op * 3) % 9)
            job_ops.append([machine, duration])
        inst_for_ortools.append(job_ops)

    prompt_lines = [f"JSSP with {num_jobs} Jobs, {num_machines} Machines. Makespan 최소화.", ""]
    for j in range(num_jobs):
        prompt_lines.append(f"Job {j} consists of Operations:")
        for op, (m, p) in enumerate(inst_for_ortools[j]):
            prompt_lines.append(f"Operation {op}: M{m},{p}")
        prompt_lines.append("")
    prompt = "\n".join(prompt_lines).strip()

    teacher_jobs = []
    for op in range(num_machines):
        for j in range(num_jobs):
            teacher_jobs.append(j)
    output = _build_output_from_job_sequence(inst_for_ortools, teacher_jobs)

    example = {
        "num_jobs": num_jobs,
        "num_machines": num_machines,
        "prompt_jobs_first": prompt,
        "output": output,
    }
    rows, meta = convert_example_to_step_rows(example, source_index=1, strict=True)

    assert len(rows) == 100
    assert meta["steps"] == 100
    assert rows[0]["step_idx"] == 0
    assert rows[-1]["step_idx"] == 99
    for row in rows:
        assert row["target_job"] in row["feasible_jobs"]
        assert "target_reason_text" in row
        assert "target_action_reason_text" in row
        assert "target_action_code" in row
        assert "action_code_to_job" in row
