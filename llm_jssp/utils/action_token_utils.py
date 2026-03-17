"""Utilities for single-token action-code policies."""

from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence


ACTION_TOKEN_PREFIX = "A"
LEGACY_ACTION_TOKEN_PREFIX = "S"

ACTION_CODE_PATTERN = re.compile(
    r"(?:Action\s*:\s*)?<\s*[aAsS]\s*(\d+)\s*>",
    re.IGNORECASE,
)


def format_action_code(
    code_index: int,
    code_width: int = 4,
    prefix: str = ACTION_TOKEN_PREFIX,
) -> str:
    if int(code_index) < 0:
        raise ValueError(f"code_index must be non-negative, got {code_index}.")
    if int(code_width) < 1:
        raise ValueError(f"code_width must be >= 1, got {code_width}.")
    return f"<{str(prefix)}{int(code_index):0{int(code_width)}d}>"


def parse_action_code(text: str, code_width: int = 4) -> Optional[str]:
    if not isinstance(text, str):
        return None
    match = ACTION_CODE_PATTERN.search(text)
    if not match:
        return None
    return format_action_code(int(match.group(1)), code_width=code_width)


def build_action_special_tokens(
    code_width: int = 4,
    code_start: int = 1,
    code_cap: int = 9999,
    prefix: str = ACTION_TOKEN_PREFIX,
) -> List[str]:
    return [
        format_action_code(code_idx, code_width=code_width, prefix=prefix)
        for code_idx in range(int(code_start), int(code_cap) + 1)
    ]


def ensure_action_special_tokens(
    tokenizer,
    model=None,
    code_width: int = 4,
    code_start: int = 1,
    code_cap: int = 9999,
    prefix: str = ACTION_TOKEN_PREFIX,
) -> Dict[str, int]:
    action_tokens = build_action_special_tokens(
        code_width=code_width,
        code_start=code_start,
        code_cap=code_cap,
        prefix=prefix,
    )
    existing = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    missing = [tok for tok in action_tokens if tok not in existing]
    added = 0
    if missing:
        added = int(tokenizer.add_special_tokens({"additional_special_tokens": missing}))

    if model is not None:
        target_size = len(tokenizer)
        input_size = int(model.get_input_embeddings().weight.shape[0])
        if input_size != target_size:
            model.resize_token_embeddings(target_size)

    return {
        "num_added_tokens": int(added),
        "vocab_size": int(len(tokenizer)),
        "action_token_count": int(len(action_tokens)),
    }


def action_code_to_token_id(tokenizer, action_code: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(str(action_code))
    if token_id is None:
        raise ValueError(f"Tokenizer returned None token id for action_code={action_code!r}")
    token_id = int(token_id)
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is not None and token_id == int(unk_id):
        encoded = tokenizer.encode(str(action_code), add_special_tokens=False)
        if len(encoded) == 1:
            return int(encoded[0])
        raise ValueError(
            f"Action code {action_code!r} is not a single tokenizer token. "
            "Special tokens were likely not installed correctly."
        )
    return token_id


def action_codes_to_token_ids(tokenizer, action_codes: Iterable[str]) -> List[int]:
    return [action_code_to_token_id(tokenizer, str(code)) for code in action_codes]


def token_id_to_action_code(tokenizer, token_id: int, code_width: int = 4) -> Optional[str]:
    token = tokenizer.convert_ids_to_tokens(int(token_id))
    if isinstance(token, bytes):
        token = token.decode("utf-8", errors="ignore")
    token_text = str(token) if token is not None else ""
    parsed = parse_action_code(token_text, code_width=code_width)
    if parsed is not None:
        return parsed

    decoded = tokenizer.decode([int(token_id)], skip_special_tokens=False)
    return parse_action_code(decoded, code_width=code_width)


def validate_action_tokenizer_installation(
    tokenizer,
    code_width: int = 4,
    code_start: int = 1,
    code_cap: int = 9999,
) -> None:
    for action_code in (
        format_action_code(code_start, code_width=code_width),
        format_action_code(min(code_start + 7, code_cap), code_width=code_width),
        format_action_code(code_cap, code_width=code_width),
    ):
        token_id = action_code_to_token_id(tokenizer, action_code)
        roundtrip = token_id_to_action_code(tokenizer, token_id, code_width=code_width)
        if roundtrip != action_code:
            raise ValueError(
                f"Action token round-trip failed: action_code={action_code}, "
                f"token_id={token_id}, roundtrip={roundtrip}"
            )
