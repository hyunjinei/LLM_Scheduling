from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parent
LA_TXT = ROOT / "la.txt"
LA_JSON = ROOT / "la.json"
LA_JSON_06_15 = ROOT / "la_06_15.json"
LA_JSON_BACKUP = ROOT / "la_01_05_backup.json"


INSTANCE_HEADER_RE = re.compile(r"^\s*instance\s+(la\d+)\s*$", re.IGNORECASE)
BEST_KNOWN_RE = re.compile(r"best known solution:\s*(\d+)", re.IGNORECASE)


def build_prompt_jobs_first(inst_for_ortools: List[List[List[int]]]) -> str:
    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0]) if inst_for_ortools else 0
    lines = [f"JSSP with {num_jobs} Jobs, {num_machines} Machines. Makespan 최소화.", ""]
    for job_id, job_ops in enumerate(inst_for_ortools):
        lines.append(f"J{job_id} consists of Operations:")
        for op_id, (machine, duration) in enumerate(job_ops):
            lines.append(f"Operation {op_id}: M{machine},{duration}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_prompt_machines_first(inst_for_ortools: List[List[List[int]]]) -> str:
    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0]) if inst_for_ortools else 0
    machine_to_ops: List[List[Tuple[int, int, int]]] = [[] for _ in range(num_machines)]
    for job_id, job_ops in enumerate(inst_for_ortools):
        for op_id, (machine, duration) in enumerate(job_ops):
            machine_to_ops[int(machine)].append((job_id, op_id, int(duration)))

    lines = [f"JSSP with {num_jobs} Jobs, {num_machines} Machines. Makespan 최소화.", ""]
    for machine_id, ops in enumerate(machine_to_ops):
        lines.append(f"M{machine_id} is used for:")
        for job_id, op_id, duration in ops:
            lines.append(f"J{job_id}-Op{op_id}: {duration}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def build_matrix_text(inst_for_ortools: List[List[List[int]]], benchmark_makespan: int) -> str:
    num_jobs = len(inst_for_ortools)
    num_machines = len(inst_for_ortools[0]) if inst_for_ortools else 0
    lines = [f"{num_jobs} {num_machines}"]
    for job_ops in inst_for_ortools:
        row: List[str] = []
        for machine, duration in job_ops:
            row.append(str(int(machine)))
            row.append(str(int(duration)))
        lines.append(" ".join(row))
    lines.append(str(int(benchmark_makespan)))
    return "\n".join(lines) + "\n"


def build_output_stub(benchmark_makespan: int) -> str:
    return f"Solution:\n\n# Best known makespan: {int(benchmark_makespan)}\n"


def parse_la_txt(path: Path) -> List[Dict[str, object]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    items: List[Dict[str, object]] = []
    i = 0
    while i < len(lines):
        header_match = INSTANCE_HEADER_RE.match(lines[i] or "")
        if not header_match:
            i += 1
            continue

        instance_name = header_match.group(1).lower()
        i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1
        if i < len(lines) and lines[i].strip().startswith("++++++++++++++++"):
            i += 1
        while i < len(lines) and not lines[i].strip():
            i += 1

        description = lines[i].strip()
        best_match = BEST_KNOWN_RE.search(description)
        if not best_match:
            raise ValueError(f"best known solution not found for {instance_name}: {description!r}")
        benchmark_makespan = int(best_match.group(1))
        i += 1

        while i < len(lines) and not lines[i].strip():
            i += 1
        size_tokens = lines[i].strip().split()
        if len(size_tokens) != 2:
            raise ValueError(f"size line malformed for {instance_name}: {lines[i]!r}")
        num_jobs, num_machines = map(int, size_tokens)
        i += 1

        inst_for_ortools: List[List[List[int]]] = []
        for _ in range(num_jobs):
            row_tokens = [int(tok) for tok in lines[i].strip().split()]
            if len(row_tokens) != 2 * num_machines:
                raise ValueError(
                    f"row length mismatch for {instance_name}: got {len(row_tokens)}, expected {2 * num_machines}"
                )
            job_ops = [[row_tokens[2 * j], row_tokens[2 * j + 1]] for j in range(num_machines)]
            inst_for_ortools.append(job_ops)
            i += 1

        items.append(
            {
                "instance_name": instance_name,
                "path": instance_name,
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "prompt_jobs_first": build_prompt_jobs_first(inst_for_ortools),
                "prompt_machines_first": build_prompt_machines_first(inst_for_ortools),
                "matrix": build_matrix_text(inst_for_ortools, benchmark_makespan),
                "output": build_output_stub(benchmark_makespan),
                "benchmark_makespan": benchmark_makespan,
            }
        )
    return items


def load_existing_la_01_05(path: Path) -> List[Dict[str, object]]:
    existing = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(existing, list):
        raise TypeError("Existing la.json is not a list.")
    out: List[Dict[str, object]] = []
    for idx, item in enumerate(existing, start=1):
        if not isinstance(item, dict):
            raise TypeError(f"Existing la.json item {idx} is not a dict.")
        cloned = dict(item)
        cloned.setdefault("instance_name", f"la{idx:02d}")
        cloned.setdefault("path", f"la{idx:02d}")
        out.append(cloned)
    return out


def sort_key(item: Dict[str, object]) -> int:
    name = str(item.get("instance_name", item.get("path", ""))).lower()
    m = re.search(r"la(\d+)", name)
    return int(m.group(1)) if m else 10**9


def main() -> None:
    parsed_06_15 = parse_la_txt(LA_TXT)
    existing_01_05 = load_existing_la_01_05(LA_JSON)

    LA_JSON_BACKUP.write_text(
        json.dumps(existing_01_05, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LA_JSON_06_15.write_text(
        json.dumps(parsed_06_15, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    merged = sorted(existing_01_05 + parsed_06_15, key=sort_key)
    LA_JSON.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"backup saved: {LA_JSON_BACKUP}")
    print(f"la06~la15 saved: {LA_JSON_06_15}")
    print(f"merged la01~la15 saved: {LA_JSON}")
    print(f"counts -> backup:{len(existing_01_05)} parsed:{len(parsed_06_15)} merged:{len(merged)}")


if __name__ == "__main__":
    main()
