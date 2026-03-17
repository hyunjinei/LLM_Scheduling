import numpy as np
from typing import Dict, List, Tuple

__all__ = [
    "gen_operations_jsp",
    "generate_random_instance",
    "inst_to_matrix",
    "build_prompt_jobs_first",
]


def _sample_process_time(time_range: Tuple[int, int], rng: np.random.Generator) -> int:
    low, high = time_range
    if low > high:
        raise ValueError("time_range lower bound must not exceed upper bound.")
    return int(rng.integers(low, high + 1))


def gen_operations_jsp(
    op_num: int,
    machine_num: int,
    op_process_time_range: Tuple[int, int],
    rng: np.random.Generator,
) -> List[Dict]:
    """
    Generate a list of operations for a single job in JSSP format.

    Each operation uses a unique machine sampled without replacement.
    """
    if op_num > machine_num:
        raise ValueError("op_num cannot exceed machine_num when using unique machines.")

    operations = []
    machine_sequence = rng.permutation(machine_num)[:op_num]
    for op_id in range(op_num):
        process_time = _sample_process_time(op_process_time_range, rng)
        machine_id = int(machine_sequence[op_id])
        operations.append(
            {
                "id": op_id,
                "machine_and_processtime": [(machine_id, process_time)],
            }
        )
    return operations


def inst_to_matrix(inst_for_ortools: List[List[List[int]]]) -> str:
    """Convert inst structure to OR-Tools style matrix text."""
    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0]) if inst_for_ortools else 0
    lines = [f"{num_jobs} {num_machines}"]
    for job_ops in inst_for_ortools:
        tokens = []
        for machine, duration in job_ops:
            tokens.append(str(machine))
            tokens.append(str(duration))
        lines.append(" ".join(tokens))
    return "\n".join(lines)


def build_prompt_jobs_first(
    inst_for_ortools: List[List[List[int]]],
) -> str:
    """Create a 'prompt_jobs_first' formatted string."""
    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0]) if inst_for_ortools else 0
    lines = [
        f"JSSP with {num_jobs} Jobs, {num_machines} Machines. Makespan minimize.",
        "",
    ]
    for job_id, job_ops in enumerate(inst_for_ortools):
        lines.append(f"Job {job_id} consists of Operations:")
        for op_id, (machine, duration) in enumerate(job_ops):
            lines.append(f"Operation {op_id}: M{machine},{duration}")
        lines.append("")  # blank line between jobs
    return "\n".join(lines).strip()


def generate_random_instance(
    num_jobs: int,
    num_machines: int,
    process_time_range: Tuple[int, int] = (1, 100),
    rng: np.random.Generator = None,
) -> Dict:
    """
    Generate a random JSSP instance with the given parameters.

    Returns a dict containing inst_for_ortools, matrix text, and prompt string.
    """
    if rng is None:
        rng = np.random.default_rng()

    inst_for_ortools: List[List[List[int]]] = []
    for _ in range(num_jobs):
        operations = gen_operations_jsp(
            op_num=num_machines,
            machine_num=num_machines,
            op_process_time_range=process_time_range,
            rng=rng,
        )
        job_ops = [
            [machine, duration]
            for op in operations
            for machine, duration in op["machine_and_processtime"]
        ]
        inst_for_ortools.append(job_ops)

    matrix_text = inst_to_matrix(inst_for_ortools)
    prompt = build_prompt_jobs_first(inst_for_ortools)

    return {
        "inst_for_ortools": inst_for_ortools,
        "matrix": matrix_text,
        "prompt": prompt,
    }
