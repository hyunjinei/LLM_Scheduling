import math
from functools import lru_cache
from typing import Iterable, List, Sequence


class StepActionParseError(ValueError):
    """Raised when a generated step action cannot be parsed."""


def _normalize_text(text: str) -> str:
    return text.replace("\r", "")


def _is_prefix_of_any(candidate_text: str, feasible_action_codes: Sequence[str]) -> bool:
    candidate_text = _normalize_text(candidate_text)
    for code in feasible_action_codes:
        if code.startswith(candidate_text):
            return True
    return False


def build_step_prefix_allowed_tokens_fn(
    tokenizer,
    feasible_action_codes_provider: Iterable[str],
    prompt_len: int = 0,
    code_width: int = 4,
    code_cap: int = 9999,
):
    feasible_action_codes = tuple(str(code) for code in feasible_action_codes_provider)
    if not feasible_action_codes:
        raise RuntimeError("No feasible action codes available at this step.")

    vocab_size = int(getattr(tokenizer, "vocab_size", None) or len(tokenizer))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    @lru_cache(maxsize=65536)
    def decode_suffix(token_id: int) -> str:
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        return _normalize_text(text)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        generated_ids = input_ids[int(prompt_len) :]
        generated_text = _normalize_text(
            tokenizer.decode(generated_ids, skip_special_tokens=False)
        )
        allowed: List[int] = []
        exact_match = generated_text in feasible_action_codes

        if exact_match and eos_token_id is not None:
            allowed.append(int(eos_token_id))

        for token_id in range(vocab_size):
            suffix = decode_suffix(token_id)
            if not suffix:
                continue
            candidate_text = generated_text + suffix
            if _is_prefix_of_any(candidate_text, feasible_action_codes):
                allowed.append(int(token_id))

        if not allowed:
            raise RuntimeError(
                "No valid next tokens for step action generation. "
                f"generated_text={generated_text!r}, feasible_action_codes={list(feasible_action_codes)}"
            )
        return allowed

    return prefix_allowed_tokens_fn
