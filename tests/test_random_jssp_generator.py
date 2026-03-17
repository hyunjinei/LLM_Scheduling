import numpy as np

from llm_jssp.utils.random_jssp import generate_random_instance
from llm_jssp.utils.jssp_step_env import parse_prompt_jobs_first, StaticJSSPStepEnv


def main():
    rng = np.random.default_rng(123)
    instance = generate_random_instance(
        num_jobs=3,
        num_machines=3,
        process_time_range=(1, 10),
        rng=rng,
    )

    matrix = instance["matrix"]
    prompt = instance["prompt"]
    inst = instance["inst_for_ortools"]

    lines = matrix.strip().splitlines()
    n, m = map(int, lines[0].split())
    parsed_inst = []
    for row in lines[1:]:
        vals = list(map(int, row.split()))
        job_ops = []
        for i in range(0, len(vals), 2):
            job_ops.append([vals[i], vals[i + 1]])
        parsed_inst.append(job_ops)
    assert n == 3 and m == 3
    assert parsed_inst == inst
    assert parse_prompt_jobs_first(prompt, strict=True) == inst

    # ensure prompt contains expected structure
    assert prompt.startswith("JSSP with 3 Jobs, 3 Machines")
    assert "Job 0 consists of Operations:" in prompt
    assert "Operation 0: M" in prompt
    print("\nGenerated prompt_jobs_first style text:\n")
    print(prompt)
    print("\n" + "-" * 60 + "\n")

    env = StaticJSSPStepEnv(inst)
    env.reset()
    teacher_jobs = [job for job in range(n) for _ in range(m)]
    for job_id in teacher_jobs:
        env.step(job_id)
    assert env.is_done()
    assert len(env.get_event_log()) == n * m
    assert env.get_makespan() > 0
    print("Random JSSP generator test passed.")


if __name__ == "__main__":
    main()
