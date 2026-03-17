from llm_jssp.utils.jssp_step_env import StaticJSSPStepEnv


def test_static_env_step_and_done():
    inst_for_ortools = [
        [[0, 3], [1, 2]],
        [[1, 2], [0, 4]],
    ]
    env = StaticJSSPStepEnv(inst_for_ortools)
    state = env.reset()

    assert state["total_steps"] == 4
    assert state["step_idx"] == 0
    assert state["feasible_jobs"] == [0, 1]

    # step 0: Job 0 -> Op0 on M0, [0,3]
    _, _, done, info0 = env.step(0)
    assert done is False
    assert info0["job_id"] == 0
    assert info0["op_idx"] == 0
    assert info0["machine_id"] == 0
    assert info0["start_time"] == 0
    assert info0["end_time"] == 3

    # step 1: Job 1 -> Op0 on M1, [0,2]
    _, _, done, info1 = env.step(1)
    assert done is False
    assert info1["job_id"] == 1
    assert info1["start_time"] == 0
    assert info1["end_time"] == 2

    # step 2: Job 0 -> Op1 on M1, start=max(job0=3, M1=2)=3
    _, _, done, info2 = env.step(0)
    assert done is False
    assert info2["job_id"] == 0
    assert info2["op_idx"] == 1
    assert info2["machine_id"] == 1
    assert info2["start_time"] == 3
    assert info2["end_time"] == 5

    # step 3: Job 1 -> Op1 on M0, start=max(job1=2, M0=3)=3, end=7
    _, _, done, info3 = env.step(1)
    assert done is True
    assert info3["start_time"] == 3
    assert info3["end_time"] == 7
    assert env.get_makespan() == 7
    assert env.is_done() is True


def test_invalid_action_raises():
    inst_for_ortools = [
        [[0, 1]],
        [[1, 1]],
    ]
    env = StaticJSSPStepEnv(inst_for_ortools)
    env.reset()

    env.step(0)  # Job 0 completed

    try:
        env.step(0)
        raised = False
    except ValueError:
        raised = True

    assert raised is True
