import re
from typing import Dict, Iterable, List, Optional, Set, Tuple

import torch
from transformers import LogitsProcessor, LogitsProcessorList


class JSSPInstanceFSM:
    """Stateful prefix constraint helper for JSSP decoding."""

    _JOB_PATTERN = re.compile(r"\s*Job\s*(\d+)\s*Operation\s*(\d+),\s*M(\d+)\s*", re.IGNORECASE)
    _JOB_WORD_PATTERN = re.compile(r"\bjob\b", re.IGNORECASE)

    def __init__(self, tokenizer, inst_for_ortools, numeric_cap: int = 999):
        self.tokenizer = tokenizer
        self.inst_for_ortools = inst_for_ortools
        self.num_jobs = len(inst_for_ortools)
        self.operations_per_job = [len(job_ops) for job_ops in inst_for_ortools]
        self.machine_of: Dict[Tuple[int, int], int] = {}
        for job_idx, job_ops in enumerate(inst_for_ortools):
            for op_idx, op_data in enumerate(job_ops):
                machine_id = int(op_data[0])
                self.machine_of[(job_idx, op_idx)] = machine_id
        self.total_operations = sum(self.operations_per_job)

        self.numeric_cap = max(
            numeric_cap,
            max(self.num_jobs - 1, 0),
            max((machine for machine in self.machine_of.values()), default=0),
            max((ops_cnt - 1) for ops_cnt in self.operations_per_job) if self.operations_per_job else 0,
        )

        self.batch_states: Dict[int, Dict[str, object]] = {}
        self.allowed_token_ids = self._build_allowed_token_ids()
        self.fallback_tokens = self._build_fallback_tokens()
        self.eos_token_id = self.tokenizer.eos_token_id

    # ----- Public API -----------------------------------------------------

    def reset(self):
        self.batch_states.clear()

    def __call__(self, batch_id: int, input_ids) -> List[int]:
        state = self._update_state(batch_id, input_ids)
        parse_result = state["parse"]

        if not parse_result["valid"]:
            return self.fallback_tokens

        allowed: List[int] = []
        base_text: str = state["text"]

        for token_id in self.allowed_token_ids:
            if token_id == self.eos_token_id and not parse_result["all_done"]:
                continue
            token_str = self._decode_token(token_id)
            if token_str is None and token_id != self.eos_token_id:
                continue
            candidate_text = base_text + (token_str or "")
            candidate_parse = self._parse_text(candidate_text)
            if candidate_parse["valid"]:
                allowed.append(token_id)

        if not allowed:
            return self.fallback_tokens
        return allowed

    def update_from_input(self, batch_id: int, input_ids) -> Dict[str, object]:
        """Synchronise internal parse state with the provided sequence."""
        state = self._update_state(batch_id, input_ids)
        return state["parse"]

    def _all_done(self) -> bool:
        """Return True when every tracked batch has emitted all operations."""
        if not self.batch_states:
            return False
        for state in self.batch_states.values():
            parse = state.get("parse")
            if not parse or not (parse.get("valid") and parse.get("all_done")):
                return False
        return True

    # ----- Internal helpers ----------------------------------------------

    def _update_state(self, batch_id: int, input_ids) -> Dict[str, object]:
        ids_tuple = tuple(int(x) for x in input_ids.tolist())
        state = self.batch_states.setdefault(batch_id, {"last_ids": None, "parse": None, "text": ""})
        if state["last_ids"] != ids_tuple:
            text = self.tokenizer.decode(list(ids_tuple), skip_special_tokens=True)
            parse_result = self._parse_text(text)
            state["last_ids"] = ids_tuple
            state["parse"] = parse_result
            state["text"] = text
        return state

    def _build_allowed_token_ids(self) -> List[int]:
        tokens: Set[int] = set()
        base_strings = [
            "Solution",
            " Solution",
            "Solution:",
            "Solution:\n",
            "Job",
            " Job",
            "job",
            " job",
            "Job ",
            "job ",
            "Operation",
            " Operation",
            "operation",
            " operation",
            "Operation ",
            ",",
            ", ",
            "M",
            " M",
            "Makespan",
            " Makespan",
            "Makespan:",
            " Makespan:",
            ":",
            ": ",
            "\n",
            "\n\n",
            " ",
            "  ",
        ]

        digits = [str(i) for i in range(10)]
        base_strings.extend(digits)
        base_strings.extend(" " + d for d in digits)

        max_number = max(self.numeric_cap, 9)
        for number in range(max_number + 1):
            s = str(number)
            base_strings.append(s)
            base_strings.append(" " + s)
            base_strings.append("M" + s)
            base_strings.append(" M" + s)

        for snippet in base_strings:
            ids = self.tokenizer.encode(snippet, add_special_tokens=False)
            tokens.update(tid for tid in ids if tid >= 0)

        if self.tokenizer.pad_token_id is not None:
            tokens.add(self.tokenizer.pad_token_id)
        if self.tokenizer.eos_token_id is not None:
            tokens.add(self.tokenizer.eos_token_id)

        return list(tokens)

    def _build_fallback_tokens(self) -> List[int]:
        fallback: List[int] = []
        try:
            space_tokens = self.tokenizer.encode(" ", add_special_tokens=False)
        except Exception:
            space_tokens = []
        fallback.extend(space_tokens)
        if self.tokenizer.eos_token_id is not None:
            fallback.append(self.tokenizer.eos_token_id)
        return list(dict.fromkeys(fallback)) or [0]

    def _decode_token(self, token_id: int) -> Optional[str]:
        try:
            return self.tokenizer.decode([token_id], skip_special_tokens=True)
        except Exception:
            return None

    # ----- Parsing --------------------------------------------------------

    def _parse_text(self, text: str) -> Dict[str, object]:
        result = {
            "valid": True,
            "solution_started": False,
            "all_done": False,
            "remaining_ops": self.total_operations,
            "available_jobs": set(range(self.num_jobs)),
            "job_next": [0] * self.num_jobs,
            "tail": "",
            "makespan_started": False,
            "scheduled": set(),
        }

        lowered = text.lower()
        solution_pos = lowered.find("solution:")
        body_start: Optional[int] = None
        if solution_pos != -1:
            prefix_before_solution = text[:solution_pos]
            if prefix_before_solution.strip():
                result["valid"] = False
                return result
            result["solution_started"] = True
            body_start = solution_pos + len("solution:")
        else:
            for job_match in re.finditer(r"(?:^|\n)\s*Job\b", text, re.IGNORECASE):
                after = text[job_match.end():]
                after_stripped = after.lstrip()
                if not after_stripped:
                    result["solution_started"] = True
                    body_start = job_match.start()
                    break
                first = after_stripped[0]
                if first.isdigit():
                    result["solution_started"] = True
                    body_start = job_match.start()
                    break
            if body_start is None:
                return result

        body = text[body_start:]
        lowered_body = body.lower()
        makespan_pos = lowered_body.find("makespan")
        if makespan_pos == -1:
            before_makespan = body
            after_makespan = ""
        else:
            before_makespan = body[:makespan_pos]
            after_makespan = body[makespan_pos:]
            result["makespan_started"] = True

        job_next = [0] * self.num_jobs
        emitted_ops: Set[Tuple[int, int]] = set()
        cursor = 0

        for match in self._JOB_PATTERN.finditer(before_makespan):
            if match.start() != cursor:
                intermediate = before_makespan[cursor:match.start()]
                if intermediate.strip():
                    result["valid"] = False
                    return result
            job_id = int(match.group(1))
            op_idx = int(match.group(2))
            machine_id = int(match.group(3))

            if job_id < 0 or job_id >= self.num_jobs:
                result["valid"] = False
                return result

            if op_idx < 0 or op_idx >= self.operations_per_job[job_id]:
                result["valid"] = False
                return result

            expected_machine = self.machine_of[(job_id, op_idx)]
            if machine_id != expected_machine:
                result["valid"] = False
                return result

            if job_next[job_id] != op_idx:
                result["valid"] = False
                return result

            job_next[job_id] += 1
            emitted_ops.add((job_id, op_idx))
            cursor = match.end()

        tail = before_makespan[cursor:]
        remaining_ops = self.total_operations - len(emitted_ops)
        available_jobs = {
            job_id for job_id in range(self.num_jobs) if job_next[job_id] < self.operations_per_job[job_id]
        }

        if remaining_ops < 0:
            result["valid"] = False
            return result

        if result["makespan_started"] and remaining_ops > 0:
            result["valid"] = False
            return result

        if tail.strip():
            if not self._tail_matches_option(tail, available_jobs, job_next):
                result["valid"] = False
                return result

        if after_makespan:
            after_trim = after_makespan.lstrip()
            if remaining_ops > 0:
                result["valid"] = False
                return result
            if not self._makespan_prefix_valid(after_trim):
                result["valid"] = False
                return result

        result.update(
            {
                "job_next": job_next,
                "remaining_ops": remaining_ops,
                "available_jobs": available_jobs,
                "tail": tail,
                "scheduled": emitted_ops,
                "all_done": remaining_ops == 0,
            }
        )
        return result

    def _tail_matches_option(
        self, tail: str, available_jobs: Set[int], job_next: List[int]
    ) -> bool:
        tail_stripped = tail.lstrip()
        if not tail_stripped:
            return True

        options: Iterable[str]
        options_list: List[str] = []
        for job_id in available_jobs:
            op_idx = job_next[job_id]
            machine_id = self.machine_of[(job_id, op_idx)]
            options_list.append(f"Job {job_id} Operation {op_idx}, M{machine_id}")
        options = options_list

        lower_tail = tail_stripped.lower()
        for option in options:
            if option.lower().startswith(lower_tail):
                return True
        return False

    @staticmethod
    def _makespan_prefix_valid(text: str) -> bool:
        if not text:
            return True

        lower = text.lower()
        target = "makespan"

        if len(lower) <= len(target):
            return target.startswith(lower)

        if not lower.startswith(target):
            return False

        remainder = text[len("Makespan") :]
        if not remainder:
            return True

        allowed_chars = set(" :0123456789.\n\r\t")
        return all(char in allowed_chars for char in remainder)


class MakespanMaskProcessor(LogitsProcessor):
    """Logits processor that blocks 'Makespan' before all operations are emitted."""

    def __init__(self, tokenizer, fsm: JSSPInstanceFSM):
        self.tokenizer = tokenizer
        self.fsm = fsm
        self.block_token_ids = self._collect_makespan_tokens()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.size(0)
        for batch_idx in range(batch_size):
            parse = self.fsm.update_from_input(batch_idx, input_ids[batch_idx])
            if not (parse["valid"] and parse["all_done"]):
                for token_id in self.block_token_ids:
                    if 0 <= token_id < scores.size(1):
                        scores[batch_idx, token_id] = float("-inf")
        return scores

    def _collect_makespan_tokens(self) -> List[int]:
        tokens: Set[int] = set()
        vocab = self.tokenizer.get_vocab()
        for token_str, token_id in vocab.items():
            normalized = token_str.lower()
            stripped = normalized.lstrip("▁Ġ ")
            if stripped.startswith("makespan") or stripped.startswith("makes"):
                tokens.add(token_id)
        return list(tokens)


def build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst_for_ortools) -> JSSPInstanceFSM:
    """Factory for prefix_allowed_tokens_fn compatible callable."""
    return JSSPInstanceFSM(tokenizer, inst_for_ortools)


def build_logits_processors(tokenizer, fsm: JSSPInstanceFSM) -> LogitsProcessorList:
    """Build a logits processor list that masks premature 'Makespan' tokens."""
    processor = MakespanMaskProcessor(tokenizer, fsm)
    return LogitsProcessorList([processor])
