"""Event-driven dispatching environment for JSSP step decisions."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .jssp_step_env import (
    ParsedTeacherAction,
    parse_prompt_jobs_first,
    parse_solution_actions,
)


class DispatchJSSPStepEnv:
    """
    Event-driven JSSP environment.

    At each decision epoch, the agent chooses one dispatchable job.
    If more dispatchable jobs remain at the same current_time, the next decision
    happens without advancing time. If no dispatchable job remains, current_time
    jumps to the next operation completion event.
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
        self.current_time = 0
        self.scheduled_ops = 0
        self.running_ops: List[Dict[str, int]] = []
        self.event_log: List[Dict[str, int]] = []
        self.reset()

    @classmethod
    def from_prompt_jobs_first(
        cls, prompt_text: str, strict: bool = True
    ) -> "DispatchJSSPStepEnv":
        inst_for_ortools = parse_prompt_jobs_first(prompt_text, strict=strict)
        return cls(inst_for_ortools)

    def reset(self) -> Dict[str, object]:
        self.job_next_op = [0] * self.num_jobs
        self.job_ready_time = [0] * self.num_jobs
        self.machine_ready_time = [0] * self.num_machines
        self.current_time = 0
        self.scheduled_ops = 0
        self.running_ops = []
        self.event_log = []
        return self.get_state_json()

    def is_done(self) -> bool:
        return self.scheduled_ops == self.total_ops

    def get_makespan(self) -> int:
        return max(self.machine_ready_time) if self.machine_ready_time else 0

    def _next_operation(self, job_id: int, offset: int = 0) -> Tuple[int, int]:
        op_idx = self.job_next_op[job_id] + int(offset)
        if op_idx >= self.operations_per_job[job_id]:
            return -1, 0
        machine_id, duration = self.inst_for_ortools[job_id][op_idx]
        return int(machine_id), int(duration)

    def _remaining_work(self, job_id: int) -> int:
        next_op = self.job_next_op[job_id]
        return sum(duration for _, duration in self.inst_for_ortools[job_id][next_op:])

    def _remaining_machine_loads_and_ops(self) -> Tuple[List[int], List[int]]:
        machine_remaining_load = [0] * self.num_machines
        machine_remaining_ops = [0] * self.num_machines
        for job_id in range(self.num_jobs):
            next_op = self.job_next_op[job_id]
            for machine_id, duration in self.inst_for_ortools[job_id][next_op:]:
                machine_remaining_load[int(machine_id)] += int(duration)
                machine_remaining_ops[int(machine_id)] += 1
        return machine_remaining_load, machine_remaining_ops

    def _dispatchable(self, job_id: int) -> bool:
        if self.job_next_op[job_id] >= self.operations_per_job[job_id]:
            return False
        machine_id, duration = self._next_operation(job_id, offset=0)
        if machine_id < 0 or duration <= 0:
            return False
        return (
            int(self.job_ready_time[job_id]) <= int(self.current_time)
            and int(self.machine_ready_time[machine_id]) <= int(self.current_time)
        )

    def get_feasible_jobs(self) -> List[int]:
        return [job_id for job_id in range(self.num_jobs) if self._dispatchable(job_id)]

    def _advance_to_next_event(self) -> None:
        if not self.running_ops:
            return
        next_time = min(int(op["end_time"]) for op in self.running_ops)
        self.current_time = int(next_time)
        self.running_ops = [
            dict(op) for op in self.running_ops if int(op["end_time"]) > int(self.current_time)
        ]

    def _advance_until_decision_epoch(self) -> None:
        while not self.is_done() and not self.get_feasible_jobs():
            if not self.running_ops:
                raise RuntimeError(
                    "Dispatch environment deadlocked: no dispatchable jobs and no running ops."
                )
            self._advance_to_next_event()

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
            job_progress_ratio.append(float(total_ops - rem_ops) / float(total_ops))

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
            "current_time": int(self.current_time),
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
            "dispatchable_jobs": self.get_feasible_jobs(),
            "idle_machines": [
                int(machine_id)
                for machine_id, ready in enumerate(self.machine_ready_time)
                if int(ready) <= int(self.current_time)
            ],
            "running_operations": [dict(op) for op in sorted(
                self.running_ops,
                key=lambda x: (int(x["end_time"]), int(x["machine_id"]), int(x["job_id"])),
            )],
        }
        return state

    def step(self, job_id: int) -> Tuple[Dict[str, object], float, bool, Dict[str, int]]:
        if self.is_done():
            raise ValueError("Cannot step: environment is already done.")
        if job_id < 0 or job_id >= self.num_jobs:
            raise ValueError(f"Invalid job_id: {job_id}.")
        feasible_jobs = self.get_feasible_jobs()
        if job_id not in feasible_jobs:
            raise ValueError(
                f"Job {job_id} is not dispatchable at t={self.current_time}. feasible={feasible_jobs}"
            )

        op_idx = self.job_next_op[job_id]
        machine_id, duration = self.inst_for_ortools[job_id][op_idx]
        start_time = int(self.current_time)
        end_time = start_time + int(duration)

        self.job_ready_time[job_id] = end_time
        self.machine_ready_time[machine_id] = end_time
        self.job_next_op[job_id] += 1
        self.scheduled_ops += 1
        self.running_ops.append(
            {
                "job_id": int(job_id),
                "op_idx": int(op_idx),
                "machine_id": int(machine_id),
                "start_time": int(start_time),
                "end_time": int(end_time),
                "duration": int(duration),
            }
        )

        event = {
            "step_idx": self.scheduled_ops - 1,
            "job_id": int(job_id),
            "op_idx": int(op_idx),
            "machine_id": int(machine_id),
            "duration": int(duration),
            "start_time": int(start_time),
            "end_time": int(end_time),
            "decision_time": int(start_time),
            "makespan_so_far": self.get_makespan(),
        }
        self.event_log.append(event)

        done = self.is_done()
        if not done:
            self._advance_until_decision_epoch()
        next_state = self.get_state_json()
        reward = 0.0
        return next_state, reward, done, event

    def rollout_teacher(self, job_sequence: Iterable[int]) -> List[Dict[str, object]]:
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
        return [dict(event) for event in self.event_log]


__all__ = [
    "DispatchJSSPStepEnv",
    "ParsedTeacherAction",
    "parse_prompt_jobs_first",
    "parse_solution_actions",
]
