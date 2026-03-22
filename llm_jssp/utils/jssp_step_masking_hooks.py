from typing import Iterable, List, Sequence

from .action_token_utils import action_codes_to_token_ids, token_id_to_action_code


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

    feasible_token_ids = tuple(
        int(token_id)
        for token_id in action_codes_to_token_ids(tokenizer, feasible_action_codes)
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids) -> List[int]:
        if hasattr(input_ids, "tolist"):
            input_ids = input_ids.tolist()
        generated_ids = input_ids[int(prompt_len) :]
        if len(generated_ids) == 0:
            return list(feasible_token_ids)

        chosen_action_code = token_id_to_action_code(
            tokenizer,
            int(generated_ids[0]),
            code_width=code_width,
        )
        if (
            len(generated_ids) == 1
            and chosen_action_code is not None
            and str(chosen_action_code) in feasible_action_codes
            and eos_token_id is not None
        ):
            return [int(eos_token_id)]

        if len(generated_ids) == 1 and chosen_action_code is not None and str(chosen_action_code) in feasible_action_codes:
            return list(feasible_token_ids)

        generated_text = _normalize_text(
            tokenizer.decode(generated_ids, skip_special_tokens=False)
        )
        if not feasible_token_ids:
            raise RuntimeError(
                "No valid next tokens for step action generation. "
                f"generated_text={generated_text!r}, feasible_action_codes={list(feasible_action_codes)}"
            )
        return list(feasible_token_ids)

    return prefix_allowed_tokens_fn
