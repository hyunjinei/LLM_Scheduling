from llm_jssp.utils.data_preprocessing_english import (
    build_step_supervision_example,
    create_step_prompt_formats,
)


class _DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.additional_special_tokens = ["<A0001>", "<A0002>", "<A0003>"]
        self._vocab = {
            "<pad>": 0,
            "<eos>": 1,
            "[system]": 2,
            "[user]": 3,
            "[assistant]": 4,
            "<A0001>": 1001,
            "<A0002>": 1002,
            "<A0003>": 1003,
        }
        self._next_id = 2000

    def _encode_text(self, text):
        ids = []
        for raw_token in str(text).replace("\n", " \n ").split():
            token = raw_token
            if token not in self._vocab:
                self._vocab[token] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[token])
        return ids

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            parts.append(f"[{m['role']}] {m['content']}")
        if add_generation_prompt:
            parts.append("[assistant]")
        rendered = "\n".join(parts)
        if not tokenize:
            return rendered
        return self._encode_text(rendered)

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self._encode_text(text)}

    def convert_tokens_to_ids(self, token):
        return self._vocab.get(str(token))


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


def test_build_step_supervision_example_action_only_masks_to_action_token():
    tokenizer = _DummyTokenizer()
    example = {
        "state_text": "Step: 1/4\nFeasible action tokens: [<A0001>, <A0002>]",
        "target_text": "<A0002>",
    }
    out = build_step_supervision_example(
        example=example,
        tokenizer=tokenizer,
        step_supervision_mode="action_only",
        max_length=512,
        action_loss_weight=4.0,
    )
    supervised_positions = [idx for idx, label in enumerate(out["labels"]) if label != -100]
    assert len(supervised_positions) == 1
    supervised_idx = supervised_positions[0]
    assert out["input_ids"][supervised_idx] == tokenizer.convert_tokens_to_ids("<A0002>")
    assert out["loss_weights"][supervised_idx] == 4.0


def test_build_step_supervision_example_action_reason_supervises_assistant_and_weights_action():
    tokenizer = _DummyTokenizer()
    example = {
        "state_text": "Step: 1/4\nFeasible action tokens: [<A0001>, <A0002>]",
        "target_text": "<A0001>",
        "target_action_reason_text": (
            "<A0001>\n"
            "Reason: <A0001> is feasible.\n"
            "Not chosen:\n"
            "- <A0002>: later."
        ),
    }
    out = build_step_supervision_example(
        example=example,
        tokenizer=tokenizer,
        step_supervision_mode="action_reason",
        max_length=512,
        action_loss_weight=3.0,
    )
    supervised_positions = [idx for idx, label in enumerate(out["labels"]) if label != -100]
    assert len(supervised_positions) > 1
    action_idx = next(
        idx for idx in supervised_positions
        if out["input_ids"][idx] == tokenizer.convert_tokens_to_ids("<A0001>")
    )
    assert out["loss_weights"][action_idx] == 3.0
    assert out["supervised_token_count"] == len(supervised_positions)
