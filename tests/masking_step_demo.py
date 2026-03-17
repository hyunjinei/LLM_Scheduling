"""Verbose step-by-step demo of the masking FSM.

This script exists purely for reporting/teaching purposes: it prints every
decision that the prefix-constrained decoder makes so that we can copy the
output into documentation without additional commentary.
"""

import torch

from llm_jssp.utils.jssp_masking_hooks import (
    build_logits_processors,
    build_prefix_allowed_tokens_fn_from_instance,
)

PREFIX_NOTES = {
    "Solution": "프롬프트가 막 시작된 상태. FSM은 아직 Solution: 키워드를 찾고 있습니다.",
    "Solution:": "콜론까지 적은 직후. 이제 줄바꿈이나 Job으로 바로 넘어가야 합니다.",
    "Solution:\n": "Solution 라인을 마치고 첫 Job을 기다리는 상태입니다.",
    "Job ": "Solution 없이 바로 Job으로 시작하는 엣지 케이스를 검사합니다.",
    "Solution:\nJob ": "Job 번호를 바로 적어야 하는 순간입니다.",
    "Solution:\nJob 0 ": "Job을 지정한 뒤 Operation 키워드를 강제합니다.",
    "Solution:\nJob 0 Operation ": "Operation 번호 선택 단계입니다.",
    "Solution:\nJob 0 Operation 0, ": "해당 작업에 맞는 기계 ID만 허용되는지 확인합니다.",
    "Solution:\nJob 0 Operation 0, M": "기계 ID 중 첫 자리만 열린 상태입니다.",
    "Solution:\nJob 0 Operation 0, M0\nJob ": "다음 Job 번호를 다시 선택해야 하는 지점입니다.",
    (
        "Solution:\nJob 0 Operation 0, M0\nJob 1 Operation 0, M1\n"
        "Job 0 Operation 1, M1\nJob 1 Operation 1, M0\n"
    ): "모든 작업을 다 적은 직후. 이제 Makespan만 허용되어야 합니다.",
}


class DemoTokenizer:
    """간단한 문자열 기반 토크나이저 (마스킹 데모용)."""

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
            ",",
            "M",
            ":",
            " ",
            "\n",
        ]
        digits = [str(i) for i in range(10)]

        token_list = []
        seen = set()

        def _register(token):
            if token not in seen:
                token_list.append(token)
                seen.add(token)

        for token in base_tokens:
            _register(token)
            _register(" " + token)
            if token.isalpha():
                _register(token + " ")

        for digit in digits:
            _register(digit)
            _register(" " + digit)
            _register("M" + digit)
            _register(" M" + digit)

        self.TOKEN_MAP = {tok: idx + 1 for idx, tok in enumerate(token_list)}
        self.id_to_token = {v: k for k, v in self.TOKEN_MAP.items()}
        self.pad_token_id = None
        self.eos_token_id = None

    def encode(self, text, add_special_tokens=False):
        tokens = []
        idx = 0
        keys = sorted(self.TOKEN_MAP.keys(), key=len, reverse=True)
        while idx < len(text):
            for token in keys:
                if text.startswith(token, idx):
                    tokens.append(self.TOKEN_MAP[token])
                    idx += len(token)
                    break
            else:
                raise ValueError(f"Unsupported substring near: {text[idx:idx+10]!r}")
        return tokens

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(self.id_to_token[token_id] for token_id in token_ids)

    def get_vocab(self):
        return self.TOKEN_MAP


def pretty_allowed(tokenizer, allowed_ids):
    return [tokenizer.id_to_token[token_id] for token_id in sorted(allowed_ids)]


def describe_numeric_options(tokens):
    stripped = [tok.strip() for tok in tokens if tok.strip()]
    digits = sorted({tok for tok in stripped if tok.isdigit()})
    machines = sorted({tok[1:] for tok in stripped if tok.startswith("M") and tok[1:].isdigit()})
    return digits, machines


def narrate_state_lines(state):
    """Return human-readable sentences for the FSM state."""

    lines = []
    remaining_ops = state["remaining_ops"]
    if remaining_ops:
        lines.append(f"남은 작업 수 {remaining_ops}개 → 아직 해당 Job/Operation을 더 써야 합니다.")
    else:
        lines.append("남은 작업이 0개 → 모든 Operation을 나열 완료.")

    jobs = sorted(state.get("available_jobs", []))
    if jobs:
        lines.append(f"다음에 시작할 수 있는 Job 후보: {jobs} (이외 Job은 아직 차례가 아님).")
    else:
        lines.append("현재 새로운 Job을 시작할 수 없음 (모두 처리됨).")

    makespan_started = state.get("makespan_started", False)
    if makespan_started:
        lines.append("Makespan 단어가 이미 시작되었으므로 추가 작업은 차단됩니다.")
    else:
        lines.append("Makespan 키워드는 아직 차단된 상태입니다.")
    return lines


def describe_blocked_tokens(tokenizer, allowed_ids, max_examples=8):
    vocab_ids = set(tokenizer.get_vocab().values())
    blocked_ids = sorted(vocab_ids - set(allowed_ids))
    examples = [tokenizer.id_to_token[idx] for idx in blocked_ids[:max_examples]]
    return blocked_ids, examples


def explain_prefix(prefix):
    return PREFIX_NOTES.get(prefix, "이 prefix는 FSM이 구조를 제대로 추적하는지 확인하는 임의 단계입니다.")


class _TensorWrapper(list):
    def tolist(self):
        return list(self)


def _as_tensor(token_ids):
    return _TensorWrapper(token_ids)


def run_demo(inst_for_ortools, description):
    tokenizer = DemoTokenizer()
    fsm = build_prefix_allowed_tokens_fn_from_instance(tokenizer, inst_for_ortools)
    logits_processors = build_logits_processors(tokenizer, fsm)
    vocab_size = len(tokenizer.get_vocab())

    print("=" * 72)
    print(description)
    print("=" * 72)
    print("이 데모는 두 계층의 마스킹(prefix_allowed_tokens_fn + LogitsProcessor)을")
    print("어떻게 적용하는지 그대로 출력하도록 만들어졌습니다.")
    print("  1) FSM이 prefix별로 허용/차단하는 토큰 목록을 보여 줌")
    print("  2) 숫자·기계 옵션을 요약하여 사람이 바로 읽을 수 있게 함")
    print("  3) Makespan logits를 -inf로 덮어쓰는 구간을 명확히 시각화")
    print()
    print("[1/3] prefix_allowed_tokens_fn 상태 추적")
    print("-" * 72)

    prefixes = [
        "Solution",
        "Solution:",
        "Solution:\n",
        "Job ",
        "Solution:\nJob ",
        "Solution:\nJob 0 ",
        "Solution:\nJob 0 Operation ",
        "Solution:\nJob 0 Operation 0, ",
        "Solution:\nJob 0 Operation 0, M",
        "Solution:\nJob 0 Operation 0, M0\nJob ",
        "Solution:\nJob 0 Operation 0, M0\nJob 1 Operation 0, M1\n"
        "Job 0 Operation 1, M1\nJob 1 Operation 1, M0\n",
    ]

    for prefix in prefixes:
        try:
            token_ids = tokenizer.encode(prefix)
        except ValueError as err:
            print(f"[SKIP] '{prefix}': {err}")
            continue
        state = fsm.update_from_input(0, _as_tensor(token_ids))
        allowed = fsm(0, _as_tensor(token_ids))
        state_desc = (
            f"remaining_ops={state['remaining_ops']}, "
            f"allowed_jobs={sorted(state.get('available_jobs', []))}, "
            f"makespan_started={state['makespan_started']}"
        )
        print(f"Prefix: {repr(prefix)}")
        print(f"  설명: {explain_prefix(prefix)}")
        print(f"  State -> {state_desc}")
        for line in narrate_state_lines(state):
            print(f"    - {line}")
        allowed_tokens = pretty_allowed(tokenizer, allowed)
        digits, machines = describe_numeric_options(allowed_tokens)
        print(f"  허용 토큰 ({len(allowed_tokens)}개): {allowed_tokens}")
        if digits:
            print(f"    • 숫자 옵션: {digits} (Job/Operation 인덱스 후보)")
        if machines:
            print(f"    • 기계 옵션: {machines} (FSM이 허용한 Machine ID)")
        blocked_ids, blocked_examples = describe_blocked_tokens(tokenizer, allowed)
        print(f"  차단된 토큰 수: {len(blocked_ids)} / vocab={vocab_size}")
        if blocked_examples:
            print(f"    • 대표 차단 토큰: {blocked_examples}")
        print(f"  (허용되지 않은 토큰은 FSM이 곧바로 구조 위반으로 판단해 버립니다.)")
        print()

    print("[2/3] 모든 Job/Operation을 마친 뒤 Makespan 해제 여부")
    print("-" * 72)
    full_solution = (
        "Solution:\n"
        "Job 0 Operation 0, M0\n"
        "Job 1 Operation 0, M1\n"
        "Job 0 Operation 1, M1\n"
        "Job 1 Operation 1, M0\n"
    )
    full_ids = tokenizer.encode(full_solution)
    print("모든 (Job, Operation) 출력 후 'Makespan' 허용 여부:")
    allowed_after = fsm(0, _as_tensor(full_ids))
    allowed_after_tokens = pretty_allowed(tokenizer, allowed_after)
    makespan_unlocked = any(tok.strip().lower().startswith("makespan") for tok in allowed_after_tokens)
    print(f"  허용 토큰: {allowed_after_tokens}")
    if makespan_unlocked:
        print("  → Makespan 계열 토큰이 '마지막 단계'에서만 열린다는 뜻입니다.")
    else:
        print("  → [경고] Makespan이 열리지 않았습니다. FSM 설정을 확인하세요.")

    print()
    print("[3/3] LogitsProcessor(-inf 마스킹) 데모")
    print("-" * 72)

    def inspect_logit(prefix_text):
        ids = tokenizer.encode(prefix_text)
        input_tensor = torch.tensor([ids], dtype=torch.long)
        zero_scores = torch.zeros(1, vocab_size)
        masked_scores = logits_processors(input_tensor, zero_scores.clone())
        makespan_id = tokenizer.TOKEN_MAP["Makespan"]
        return masked_scores[0, makespan_id].item()

    partial_prefix = (
        "Solution:\n"
        "Job 0 Operation 0, M0\n"
        "Job 1 Operation 0, M1\n"
    )
    partial_logit = inspect_logit(partial_prefix)
    complete_logit = inspect_logit(full_solution)

    print("  • 미완료 스케줄에서 Makespan logit:", partial_logit)
    print("    (FSM이 all_done=False 상태라 -inf가 된 것을 확인합니다.)")
    print("  • 완료 스케줄에서 Makespan logit:", complete_logit)
    print("    (모든 작업을 마치면 LogitsProcessor가 더 이상 막지 않습니다.)")

    if makespan_unlocked and partial_logit == float("-inf") and complete_logit != float("-inf"):
        print("All masking sanity checks passed.")
    else:
        print("[경고] 마스킹 체크가 기대와 다릅니다. 출력 로그를 확인하세요.")

    print()
    print()


if __name__ == "__main__":
    small_inst = [
        [[0, 21], [1, 28]],
        [[1, 26], [0, 27]],
    ]
    run_demo(small_inst, "2 Jobs × 2 Machines 예제 (Makespan: 54)")
