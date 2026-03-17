from llm_jssp.utils.data_preprocessing_english import create_step_prompt_formats


class _DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}]{m['content']}")
        return "\n".join(parts)


def test_create_step_prompt_formats_action_reason():
    tokenizer = _DummyTokenizer()
    example = {
        "state_text": "Step: 1/4\nFeasible action tokens: [<A0001>, <A0002>]",
        "target_text": "<A0001>",
        "target_action_reason_text": (
            "<A0001>\n"
            "Reason: <A0001> is feasible and earliest.\n"
            "Not chosen:\n"
            "- <A0002>: later estimated start."
        ),
    }
    out = create_step_prompt_formats(
        example=example,
        tokenizer=tokenizer,
        step_supervision_mode="action_reason",
    )
    assert "<A0001>" in out["text"]
    assert "Reason:" in out["text"]
    assert "Not chosen:" in out["text"]


def test_create_step_prompt_formats_action_only():
    tokenizer = _DummyTokenizer()
    example = {
        "state_text": "Step: 1/4\nFeasible action tokens: [<A0001>, <A0002>]",
        "target_text": "<A0002>",
    }
    out = create_step_prompt_formats(
        example=example,
        tokenizer=tokenizer,
        step_supervision_mode="action_only",
    )
    assert "<A0002>" in out["text"]
