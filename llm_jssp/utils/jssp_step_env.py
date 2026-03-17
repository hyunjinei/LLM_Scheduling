"""Step-by-step environment and parsers for static JSSP."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


HEADER_PATTERN = re.compile(
    r"JSSP\s+with\s+(\d+)\s+Jobs,\s*(\d+)\s+Machines",
    re.IGNORECASE,
)
JOB_HEADER_PATTERN = re.compile(
    r"^\s*Job\s+(\d+)\s+consists\s+of\s+Operations:\s*$",
    re.IGNORECASE,
)
OPERATION_PATTERN = re.compile(
    r"^\s*Operation\s+(\d+):\s*M(\d+),\s*(\d+)\s*$",
    re.IGNORECASE,
)
SOLUTION_OPERATION_PATTERN = re.compile(
    r"Job\s*(\d+)\s*Operation\s*(\d+),\s*M(\d+)",
    re.IGNORECASE,
)
MAKESPAN_PATTERN = re.compile(r"Makespan\s*:\s*(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedTeacherAction:
    """Parsed action from one-shot output text."""

    job_id: int
    op_idx: int
    machine_id: int


def parse_prompt_jobs_first(prompt_text: str, strict: bool = True) -> List[List[List[int]]]:
    """
    Parse `prompt_jobs_first` text into `inst_for_ortools` format.

    Returns:
        inst_for_ortools[job][op] = [machine, duration]
    """
    if not isinstance(prompt_text, str) or not prompt_text.strip():
        raise ValueError("prompt_text must be a non-empty string.")

    header_match = HEADER_PATTERN.search(prompt_text)
    expected_jobs = int(header_match.group(1)) if header_match else None
    expected_machines = int(header_match.group(2)) if header_match else None

    jobs: Dict[int, List[List[int]]] = {}
    current_job: Optional[int] = None

    for raw_line in prompt_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        job_match = JOB_HEADER_PATTERN.match(line)
        if job_match:
            current_job = int(job_match.group(1))
            jobs.setdefault(current_job, [])
            continue

        op_match = OPERATION_PATTERN.match(line)
        if op_match:
            if current_job is None:
                raise ValueError(f"Operation line appears before job header: {raw_line!r}")

            op_idx = int(op_match.group(1))
            machine = int(op_match.group(2))
            duration = int(op_match.group(3))
            expected_op_idx = len(jobs[current_job])
            if strict and op_idx != expected_op_idx:
                raise ValueError(
                    f"Operation index mismatch in Job {current_job}: "
                    f"expected {expected_op_idx}, got {op_idx}."
                )
            jobs[current_job].append([machine, duration])

    if not jobs:
        raise ValueError("Failed to parse any jobs from prompt_text.")

    max_job_id = max(jobs.keys())
    if strict and set(jobs.keys()) != set(range(max_job_id + 1)):
        raise ValueError("Job IDs must be contiguous from 0 when strict=True.")

    inst_for_ortools = [jobs[job_id] for job_id in range(max_job_id + 1)]

    if strict:
        if expected_jobs is not None and expected_jobs != len(inst_for_ortools):
            raise ValueError(
                f"Header jobs ({expected_jobs}) != parsed jobs ({len(inst_for_ortools)})."
            )
        if expected_machines is not None:
            for job_id, job_ops in enumerate(inst_for_ortools):
                if len(job_ops) != expected_machines:
                    raise ValueError(
                        f"Job {job_id} has {len(job_ops)} ops, expected {expected_machines}."
                    )

    return inst_for_ortools


def parse_solution_actions(
    solution_text: str, strict: bool = True
) -> Tuple[List[ParsedTeacherAction], Optional[int]]:
    """
    Parse one-shot output into per-step teacher actions.

    Returns:
        actions in decode order, declared makespan from text if present.
    """
    if not isinstance(solution_text, str) or not solution_text.strip():
        raise ValueError("solution_text must be a non-empty string.")

    matches = SOLUTION_OPERATION_PATTERN.findall(solution_text)
    if not matches:
        raise ValueError("No 'Job X Operation Y, MZ' actions found in solution_text.")

    next_expected_op: Dict[int, int] = {}
    actions: List[ParsedTeacherAction] = []
    for job_str, op_str, machine_str in matches:
        job_id = int(job_str)
        op_idx = int(op_str)
        machine_id = int(machine_str)

        expected_op_idx = next_expected_op.get(job_id, 0)
        if strict and op_idx != expected_op_idx:
            raise ValueError(
                f"Teacher action order violation for job {job_id}: "
                f"expected op {expected_op_idx}, got {op_idx}."
            )

        next_expected_op[job_id] = op_idx + 1
        actions.append(ParsedTeacherAction(job_id=job_id, op_idx=op_idx, machine_id=machine_id))

    makespan_match = MAKESPAN_PATTERN.search(solution_text)
    declared_makespan = int(makespan_match.group(1)) if makespan_match else None
    return actions, declared_makespan


class StaticJSSPStepEnv:
    """
    Deterministic static JSSP environment for step-by-step action selection.

    Action:
        choose one feasible job id at each step.
    """

    def __init__(self, inst_for_ortools: Sequence[Sequence[Sequence[int]]]):
        if not inst_for_ortools:
            raise ValueError("inst_for_ortools must not be empty.")

        self.inst_for_ortools: List[List[Tuple[int, int]]] = []
        for job_ops in inst_for_ortools:
            parsed_job_ops: List[Tuple[int, int]] = []
            for op in job_ops:
                if len(op) != 2:
                    raise ValueError(f"Each operation must be [machine, duration], got {op}.")
                machine, duration = int(op[0]), int(op[1])
                parsed_job_ops.append((machine, duration))
            self.inst_for_ortools.append(parsed_job_ops)

        self.num_jobs = len(self.inst_for_ortools)
        self.operations_per_job = [len(job_ops) for job_ops in self.inst_for_ortools]
        self.job_total_ops = list(self.operations_per_job)
        self.job_total_work = [
            sum(duration for _, duration in job_ops) for job_ops in self.inst_for_ortools
        ]
        self.total_ops = sum(self.operations_per_job)

        if self.total_ops <= 0:
            raise ValueError("total_ops must be positive.")

        max_machine_id = max(
            machine for job_ops in self.inst_for_ortools for machine, _ in job_ops
        )
        self.num_machines = max_machine_id + 1

        self.job_next_op: List[int] = []
        self.job_ready_time: List[int] = []
        self.machine_ready_time: List[int] = []
        self.scheduled_ops = 0
        self.event_log: List[Dict[str, int]] = []
        self.reset()

    @classmethod
    def from_prompt_jobs_first(cls, prompt_text: str, strict: bool = True) -> "StaticJSSPStepEnv":
        inst_for_ortools = parse_prompt_jobs_first(prompt_text, strict=strict)
        return cls(inst_for_ortools)

    def reset(self) -> Dict[str, object]:
        self.job_next_op = [0] * self.num_jobs
        self.job_ready_time = [0] * self.num_jobs
        self.machine_ready_time = [0] * self.num_machines
        self.scheduled_ops = 0
        self.event_log = []
        return self.get_state_json()

    def is_done(self) -> bool:
        return self.scheduled_ops == self.total_ops

    def get_makespan(self) -> int:
        return max(self.machine_ready_time) if self.machine_ready_time else 0

    def get_feasible_jobs(self) -> List[int]:
        return [
            job_id
            for job_id in range(self.num_jobs)
            if self.job_next_op[job_id] < self.operations_per_job[job_id]
        ]

    def _remaining_work(self, job_id: int) -> int:
        next_op = self.job_next_op[job_id]
        return sum(duration for _, duration in self.inst_for_ortools[job_id][next_op:])

    def _next_operation(self, job_id: int, offset: int = 0) -> Tuple[int, int]:
        op_idx = self.job_next_op[job_id] + int(offset)
        if op_idx >= self.operations_per_job[job_id]:
            return -1, 0
        machine_id, duration = self.inst_for_ortools[job_id][op_idx]
        return int(machine_id), int(duration)

    def _remaining_machine_loads_and_ops(self) -> Tuple[List[int], List[int]]:
        machine_remaining_load = [0] * self.num_machines
        machine_remaining_ops = [0] * self.num_machines

        for job_id in range(self.num_jobs):
            next_op = self.job_next_op[job_id]
            for machine_id, duration in self.inst_for_ortools[job_id][next_op:]:
                machine_remaining_load[int(machine_id)] += int(duration)
                machine_remaining_ops[int(machine_id)] += 1

        return machine_remaining_load, machine_remaining_ops

    def get_state_json(self) -> Dict[str, object]:
        next_machine: List[int] = []
        next_proc_time: List[int] = []
        next2_machine: List[int] = []
        next2_proc_time: List[int] = []
        remaining_ops: List[int] = []
        remaining_work: List[int] = []
        remaining_work_ratio: List[float] = []
        job_progress_ratio: List[float] = []

        for job_id in range(self.num_jobs):
            op_idx = self.job_next_op[job_id]
            machine, duration = self._next_operation(job_id, offset=0)
            machine2, duration2 = self._next_operation(job_id, offset=1)
            next_machine.append(machine)
            next_proc_time.append(duration)
            next2_machine.append(machine2)
            next2_proc_time.append(duration2)

            rem_ops = self.operations_per_job[job_id] - op_idx
            rem_work = self._remaining_work(job_id)
            total_work = max(int(self.job_total_work[job_id]), 1)
            total_ops = max(int(self.job_total_ops[job_id]), 1)

            remaining_ops.append(int(rem_ops))
            remaining_work.append(int(rem_work))
            remaining_work_ratio.append(float(rem_work) / float(total_work))
            job_progress_ratio.append(
                float(total_ops - rem_ops) / float(total_ops)
            )

        current_cmax = self.get_makespan()
        total_remaining_work = int(sum(remaining_work))
        unfinished_jobs_count = sum(1 for x in remaining_ops if int(x) > 0)
        unfinished_jobs_ratio = (
            float(unfinished_jobs_count) / float(self.num_jobs) if self.num_jobs else 0.0
        )
        machine_ready_min = min(self.machine_ready_time) if self.machine_ready_time else 0
        machine_ready_max = max(self.machine_ready_time) if self.machine_ready_time else 0
        machine_ready_mean = (
            float(sum(self.machine_ready_time)) / float(len(self.machine_ready_time))
            if self.machine_ready_time
            else 0.0
        )
        machine_ready_var = (
            sum((float(x) - machine_ready_mean) ** 2 for x in self.machine_ready_time)
            / float(len(self.machine_ready_time))
            if self.machine_ready_time
            else 0.0
        )
        machine_ready_std = float(machine_ready_var ** 0.5)
        machine_remaining_load, machine_remaining_ops = self._remaining_machine_loads_and_ops()
        bottleneck_machine_load = max(machine_remaining_load) if machine_remaining_load else 0
        bottleneck_machine_id = (
            machine_remaining_load.index(bottleneck_machine_load)
            if machine_remaining_load
            else -1
        )
        bottleneck_machine_ops_left = (
            int(machine_remaining_ops[bottleneck_machine_id])
            if bottleneck_machine_id >= 0
            else 0
        )

        state = {
            "step_idx": self.scheduled_ops,
            "total_steps": self.total_ops,
            "scheduled_ratio": (
                float(self.scheduled_ops) / float(self.total_ops) if self.total_ops else 0.0
            ),
            "current_cmax": int(current_cmax),
            "job_next_op": list(self.job_next_op),
            "job_total_ops": list(self.job_total_ops),
            "job_total_work": list(self.job_total_work),
            "job_ready_time": list(self.job_ready_time),
            "machine_ready_time": list(self.machine_ready_time),
            "next_machine": next_machine,
            "next_proc_time": next_proc_time,
            "next2_machine": next2_machine,
            "next2_proc_time": next2_proc_time,
            "remaining_ops": remaining_ops,
            "remaining_work": remaining_work,
            "remaining_work_ratio": remaining_work_ratio,
            "job_progress_ratio": job_progress_ratio,
            "total_remaining_work": int(total_remaining_work),
            "unfinished_jobs_count": int(unfinished_jobs_count),
            "unfinished_jobs_ratio": float(unfinished_jobs_ratio),
            "machine_ready_min": int(machine_ready_min),
            "machine_ready_mean": float(machine_ready_mean),
            "machine_ready_max": int(machine_ready_max),
            "machine_ready_std": float(machine_ready_std),
            "machine_remaining_load": machine_remaining_load,
            "machine_remaining_ops": machine_remaining_ops,
            "bottleneck_machine_id": int(bottleneck_machine_id),
            "bottleneck_machine_load": int(bottleneck_machine_load),
            "bottleneck_machine_ops_left": int(bottleneck_machine_ops_left),
            "feasible_jobs": self.get_feasible_jobs(),
        }
        return state

    def step(self, job_id: int) -> Tuple[Dict[str, object], float, bool, Dict[str, int]]:
        if self.is_done():
            raise ValueError("Cannot step: environment is already done.")
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}.")

        op_idx = self.job_next_op[job_id]
        if op_idx >= self.operations_per_job[job_id]:
            raise ValueError(f"Job {job_id} has no remaining operation.")

        machine_id, duration = self.inst_for_ortools[job_id][op_idx]
        start_time = max(self.job_ready_time[job_id], self.machine_ready_time[machine_id])
        end_time = start_time + duration

        self.job_ready_time[job_id] = end_time
        self.machine_ready_time[machine_id] = end_time
        self.job_next_op[job_id] += 1
        self.scheduled_ops += 1

        event = {
            "step_idx": self.scheduled_ops - 1,
            "job_id": job_id,
            "op_idx": op_idx,
            "machine_id": machine_id,
            "duration": duration,
            "start_time": start_time,
            "end_time": end_time,
            "makespan_so_far": self.get_makespan(),
        }
        self.event_log.append(event)

        done = self.is_done()
        next_state = self.get_state_json()
        reward = 0.0
        return next_state, reward, done, event

    def rollout_teacher(self, job_sequence: Iterable[int]) -> List[Dict[str, object]]:
        """
        Roll out a full teacher sequence and collect step-level records.
        """
        self.reset()
        records: List[Dict[str, object]] = []
        for step_idx, job_id in enumerate(job_sequence):
            state_before = self.get_state_json()
            feasible_jobs = state_before["feasible_jobs"]
            if job_id not in feasible_jobs:
                raise ValueError(
                    f"Infeasible teacher action at step {step_idx}: "
                    f"job {job_id}, feasible={feasible_jobs}."
                )
            _, _, done, info = self.step(job_id)
            records.append(
                {
                    "step_idx": step_idx,
                    "state_json": state_before,
                    "feasible_jobs": list(feasible_jobs),
                    "target_job": job_id,
                    "info": info,
                    "done": done,
                }
            )
        if not self.is_done():
            raise ValueError(
                f"Teacher sequence ended early: scheduled {self.scheduled_ops}/{self.total_ops}."
            )
        return records

    def get_event_log(self) -> List[Dict[str, int]]:
        return list(self.event_log)
