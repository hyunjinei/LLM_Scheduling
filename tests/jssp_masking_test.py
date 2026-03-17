"""Sanity tests for the decoding-time masking hooks."""

import copy
import importlib.util
import math
import sys
import types
from pathlib import Path

try:
    import torch  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    class _FakeTensor:
        def __init__(self, data):
            self._data = copy.deepcopy(data)

        def size(self, dim):
            if dim == 0:
                return len(self._data)
            if dim == 1:
                return len(self._data[0])
            raise IndexError("Unsupported dim")

        def __getitem__(self, idx):
            if isinstance(idx, tuple):
                row, col = idx
                return self._data[row][col]
            value = self._data[idx]
            if isinstance(value, list):
                return _FakeTensor(value)
            return value

        def __setitem__(self, idx, value):
            if isinstance(idx, tuple):
                row, col = idx
                self._data[row][col] = value
            else:
                self._data[idx] = value

        def clone(self):
            return _FakeTensor(self._data)

        def to(self, device):
            return self

        def tolist(self):
            return copy.deepcopy(self._data)

    torch = types.ModuleType("torch")

    def _tensor(data, dtype=None):
        return _FakeTensor(data)

    def _zeros(rows, cols):
        return _FakeTensor([[0.0 for _ in range(cols)] for _ in range(rows)])

    def _isinf(value):
        return math.isinf(value)

    torch.tensor = _tensor  # type: ignore[attr-defined]
    torch.zeros = _zeros  # type: ignore[attr-defined]
    torch.isinf = _isinf  # type: ignore[attr-defined]
    torch.long = "long"
    torch.float = "float"
    torch.LongTensor = _FakeTensor  # type: ignore[attr-defined]
    torch.FloatTensor = _FakeTensor  # type: ignore[attr-defined]
    sys.modules["torch"] = torch

try:
    from transformers import LogitsProcessor, LogitsProcessorList  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal envs
    transformers_stub = types.ModuleType("transformers")

    class _LogitsProcessor:
        pass

    class _LogitsProcessorList(list):
        pass

    transformers_stub.LogitsProcessor = _LogitsProcessor
    transformers_stub.LogitsProcessorList = _LogitsProcessorList
    sys.modules["transformers"] = transformers_stub
    from transformers import LogitsProcessor, LogitsProcessorList  # type: ignore


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_jssp" / "utils" / "jssp_masking_hooks.py"
spec = importlib.util.spec_from_file_location("jssp_masking_hooks", MODULE_PATH)
jssp_masking_hooks = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(jssp_masking_hooks)  # type: ignore[attr-defined]

JSSPInstanceFSM = jssp_masking_hooks.JSSPInstanceFSM
build_logits_processors = jssp_masking_hooks.build_logits_processors
build_prefix_allowed_tokens_fn_from_instance = jssp_masking_hooks.build_prefix_allowed_tokens_fn_from_instance


class MockTokenizer:
    """Minimal tokenizer compatible with masking hooks."""

    def __init__(self):
        base_tokens = [
            "Solution",
            "Solution:",
            "Job",
            "job",
            "Operation",
            "operation",
            "Makespan",
            "Makespan:",
            "M",
            ",",
            ":",
            " ",
            "\n",
        ]
        digit_tokens = [str(i) for i in range(10)]
        variants = []
        for tok in base_tokens:
            variants.append(tok)
            variants.append(" " + tok)
        variants.extend(digit_tokens)
        variants.extend(" " + d for d in digit_tokens)
        variants.extend("M" + d for d in digit_tokens)
        variants.extend(" M" + d for d in digit_tokens)

        deduped = []
        seen = set()
        for token in variants:
            if token not in seen:
                deduped.append(token)
                seen.add(token)

        self.tokens = deduped
        self.id_to_token = {idx: tok for idx, tok in enumerate(self.tokens)}
        self.token_to_id = {tok: idx for idx, tok in self.id_to_token.items()}
        self.sorted_tokens = sorted(self.tokens, key=len, reverse=True)
        self.pad_token_id = None
        self.eos_token_id = len(self.tokens)
        self.token_to_id["<eos>"] = self.eos_token_id
        self.id_to_token[self.eos_token_id] = "<eos>"

    def get_vocab(self):
        return dict(self.token_to_id)

    def decode(self, token_ids, skip_special_tokens=True):
        pieces = []
        for tid in token_ids:
            tid = int(tid)
            if skip_special_tokens and tid == self.eos_token_id:
                continue
            pieces.append(self.id_to_token[tid])
        return "".join(pieces)

    def encode(self, text, add_special_tokens=False):
        ids = []
        index = 0
        while index < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text.startswith(token, index):
                    ids.append(self.token_to_id[token])
                    index += len(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Unable to encode segment starting at {text[index: index + 10]!r}")
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, text, return_tensors=None):
        ids = self.encode(text, add_special_tokens=False)
        tensor = torch.tensor([ids], dtype=torch.long)
        return {"input_ids": tensor}


class TensorWrapper(list):
    """Lightweight tensor substitute exposing tolist()."""

    def tolist(self):
        return list(self)


def as_tensor(token_ids):
    return TensorWrapper(token_ids)


def build_test_fsm():
    tokenizer = MockTokenizer()
    inst_for_ortools = [
        [[0, 1], [1, 1], [2, 1]],
        [[2, 1], [0, 1], [1, 1]],
        [[1, 1], [2, 1], [0, 1]],
    ]
    fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst_for_ortools)
    logits_processors = build_logits_processors(tokenizer, fsm)
    return tokenizer, fsm, logits_processors


def test_full_schedule_passes():
    tokenizer, fsm, logits_processors = build_test_fsm()
    prompt = "Solution:\n"
    prompt_ids = tokenizer.encode(prompt)
    current_ids = list(prompt_ids)

    sequence = [
        "Job 0 Operation 0, M0\n",
        "Job 0 Operation 1, M1\n",
        "Job 0 Operation 2, M2\n",
        "Job 1 Operation 0, M2\n",
        "Job 1 Operation 1, M0\n",
        "Job 1 Operation 2, M1\n",
        "Job 2 Operation 0, M1\n",
        "Job 2 Operation 1, M2\n",
        "Job 2 Operation 2, M0\n",
        "Makespan: 9\n",
    ]

    zero_scores = torch.zeros(1, len(tokenizer.get_vocab()))

    for chunk in sequence:
        for tid in tokenizer.encode(chunk):
            allowed = fsm(0, as_tensor(current_ids))
            assert tid in allowed, f"Token {tokenizer.decode([tid])!r} unexpectedly masked."
            current_ids.append(tid)
            logits_processors[0](torch.tensor([current_ids], dtype=torch.long), zero_scores.clone())

    parse = fsm.update_from_input(0, as_tensor(current_ids))
    assert parse["valid"] and parse["all_done"], "FSM did not detect completion."
    assert fsm._all_done() is True, "_all_done() expected to be True."


def test_job_digit_restricted():
    tokenizer, fsm, _ = build_test_fsm()
    prefix = "Solution:\nJob "
    token_ids = tokenizer.encode(prefix)
    allowed_tokens = {tokenizer.id_to_token[idx] for idx in fsm(0, as_tensor(token_ids))}
    assert allowed_tokens == {"0", "1", "2"}


def test_operation_sequence_enforced():
    tokenizer, fsm, _ = build_test_fsm()
    prefix = "Solution:\nJob 0 Operation "
    token_ids = tokenizer.encode(prefix)
    allowed_tokens = {tokenizer.id_to_token[idx] for idx in fsm(0, as_tensor(token_ids))}
    assert allowed_tokens == {"0"}


def test_makespan_blocked_until_complete():
    tokenizer, fsm, logits_processors = build_test_fsm()
    partial = "Solution:\nJob 0 Operation 0, M0\n"
    current_ids = tokenizer.encode(partial)
    makespan_token = tokenizer.token_to_id["Makespan"]

    allowed = fsm(0, as_tensor(current_ids))
    assert makespan_token not in allowed, "Makespan token should be masked in prefix."

    scores = torch.zeros(1, len(tokenizer.get_vocab()))
    masked_scores = logits_processors[0](torch.tensor([current_ids], dtype=torch.long), scores)
    assert torch.isinf(masked_scores[0, makespan_token]), "Makespan score should be -inf before completion."


if __name__ == "__main__":
    test_full_schedule_passes()
    test_job_digit_restricted()
    test_operation_sequence_enforced()
    test_makespan_blocked_until_complete()
    print("All masking sanity checks passed.")
