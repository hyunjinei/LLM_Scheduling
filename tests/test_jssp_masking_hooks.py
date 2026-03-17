import types

try:
    import pytest  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback decorator
    def _identity_parametrize(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    pytest = types.SimpleNamespace(mark=types.SimpleNamespace(parametrize=_identity_parametrize))  # type: ignore

from tests.jssp_masking_test import (
    JSSPInstanceFSM,
    MockTokenizer,
    build_prefix_allowed_tokens_fn_from_instance,
)


def _tensor(ids):
    class _Wrapper(list):
        def tolist(self):
            return list(self)

    return _Wrapper(ids)


def build_fsm(num_jobs=4, ops_per_job=3):
    tokenizer = MockTokenizer()
    inst = []
    for job_idx in range(num_jobs):
        job_ops = []
        for op_idx in range(ops_per_job):
            machine = (job_idx + op_idx) % ops_per_job
            duration = 1
            job_ops.append([machine, duration])
        inst.append(job_ops)
    fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst)
    return tokenizer, inst, fsm


def test_factory_returns_fsm():
    tokenizer, inst, fsm = build_fsm()
    assert isinstance(fsm, JSSPInstanceFSM)
    assert fsm.num_jobs == len(inst)
    assert fsm.total_operations == len(inst) * len(inst[0])


def test_machine_mismatch_invalidates_prefix():
    tokenizer, inst, fsm = build_fsm(num_jobs=2, ops_per_job=2)
    text = "Solution:\nJob 0 Operation 0, M5"
    parse = fsm.update_from_input(0, _tensor(tokenizer.encode(text)))
    assert parse["valid"] is False


def test_duplicate_operation_rejected():
    tokenizer, inst, fsm = build_fsm(num_jobs=1, ops_per_job=2)
    text = (
        "Solution:\n"
        "Job 0 Operation 0, M0\n"
        "Job 0 Operation 0, M0\n"
    )
    parse = fsm.update_from_input(0, _tensor(tokenizer.encode(text)))
    assert parse["valid"] is False


def test_multi_digit_job_ids_supported():
    tokenizer, inst, fsm = build_fsm(num_jobs=12, ops_per_job=1)
    prefix = "Job "
    token_ids = tokenizer.encode(prefix)
    allowed_tokens = {tokenizer.id_to_token[idx] for idx in fsm(0, _tensor(token_ids))}
    for digit in map(str, range(10)):
        assert digit in allowed_tokens


def test_state_marks_completion_without_solution_prefix():
    tokenizer = MockTokenizer()
    inst = [
        [[0, 1], [1, 1]],
        [[1, 1], [0, 1]],
    ]
    fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst)
    text = (
        "Job 0 Operation 0, M0\n"
        "Job 1 Operation 0, M1\n"
        "Job 0 Operation 1, M1\n"
        "Job 1 Operation 1, M0\n"
        "Makespan: 10"
    )
    parse = fsm.update_from_input(0, _tensor(tokenizer.encode(text)))
    assert parse["valid"] and parse["all_done"]
    assert fsm._all_done() is True


@pytest.mark.parametrize(
    "partial,next_allowed",
    [
        ("Solution:\nJob ", {"0", "1"}),
        ("Solution:\nJob 0 Operation ", {"0"}),
        ("Solution:\nJob 0 Operation 0, M", {"0"}),
    ],
)
def test_slot_restrictions(partial, next_allowed):
    tokenizer, inst, fsm = build_fsm(num_jobs=2, ops_per_job=1)
    token_ids = tokenizer.encode(partial)
    allowed_tokens = {tokenizer.id_to_token[idx] for idx in fsm(0, _tensor(token_ids))}
    assert allowed_tokens == next_allowed


if __name__ == "__main__":
    test_factory_returns_fsm()
    test_machine_mismatch_invalidates_prefix()
    test_duplicate_operation_rejected()
    test_multi_digit_job_ids_supported()
    test_state_marks_completion_without_solution_prefix()
    for args in [
        ("Solution:\nJob ", {"0", "1"}),
        ("Solution:\nJob 0 Operation ", {"0"}),
        ("Solution:\nJob 0 Operation 0, M", {"0"}),
    ]:
        test_slot_restrictions(*args)
    print("All jssp_masking_hooks unit tests passed.")
