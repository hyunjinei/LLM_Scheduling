import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "llm_jssp"
    / "utils"
    / "jssp_step_masking_hooks.py"
)
spec = importlib.util.spec_from_file_location("jssp_step_masking_hooks", MODULE_PATH)
jssp_step_masking_hooks = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(jssp_step_masking_hooks)  # type: ignore[attr-defined]

build_step_prefix_allowed_tokens_fn = jssp_step_masking_hooks.build_step_prefix_allowed_tokens_fn
parse_action_code = jssp_step_masking_hooks.parse_action_code


class TensorWrapper(list):
    def tolist(self):
        return list(self)


def as_tensor(token_ids):
    return TensorWrapper(token_ids)


class MockTokenizer:
    def __init__(self):
        base_tokens = [
            "<A0001>",
            "<A0002>",
            "<A0003>",
        ]
        tokens = list(dict.fromkeys(base_tokens))
        self.tokens = tokens
        self.id_to_token = {idx: tok for idx, tok in enumerate(tokens)}
        self.token_to_id = {tok: idx for idx, tok in self.id_to_token.items()}
        self.sorted_tokens = sorted(tokens, key=len, reverse=True)
        self.eos_token_id = len(tokens)
        self.id_to_token[self.eos_token_id] = "<eos>"
        self.token_to_id["<eos>"] = self.eos_token_id
        self.unk_token_id = self.eos_token_id + 1
        self.pad_token_id = None

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
        idx = 0
        while idx < len(text):
            matched = False
            for token in self.sorted_tokens:
                if text.startswith(token, idx):
                    ids.append(self.token_to_id[token])
                    idx += len(token)
                    matched = True
                    break
            if not matched:
                raise ValueError(f"Cannot encode text from position {idx}: {text[idx:idx+20]!r}")
        if add_special_tokens:
            ids.append(self.eos_token_id)
        return ids

    def convert_tokens_to_ids(self, token):
        return self.token_to_id.get(token, self.unk_token_id)

    def convert_ids_to_tokens(self, token_id):
        return self.id_to_token.get(int(token_id))

    def get_vocab(self):
        return dict(self.token_to_id)


def test_code_slot_restricted_to_feasible_actions():
    tokenizer = MockTokenizer()
    feasible_codes = ["<A0001>", "<A0003>"]
    fsm = build_step_prefix_allowed_tokens_fn(tokenizer, feasible_codes, code_width=4)
    fsm(0, as_tensor([]))  # initialize prompt length to 0 for this unit test

    token_ids = []
    allowed = fsm(0, as_tensor(token_ids))
    allowed_tokens = {tokenizer.id_to_token[tid] for tid in allowed if tid != tokenizer.eos_token_id}

    assert "<A0001>" in allowed_tokens
    assert "<A0003>" in allowed_tokens
    assert "<A0002>" not in allowed_tokens


def test_eos_unblocked_only_after_complete_action_code():
    tokenizer = MockTokenizer()
    feasible_codes = ["<A0003>"]
    fsm = build_step_prefix_allowed_tokens_fn(tokenizer, feasible_codes, code_width=4)
    fsm(0, as_tensor([]))  # initialize prompt length to 0 for this unit test

    partial_ids = []
    partial_allowed = set(fsm(0, as_tensor(partial_ids)))
    assert tokenizer.eos_token_id not in partial_allowed

    complete_ids = tokenizer.encode("<A0003>")
    complete_allowed = set(fsm(0, as_tensor(complete_ids)))
    assert tokenizer.eos_token_id in complete_allowed


def test_parse_action_code():
    assert parse_action_code("<A0007>", code_width=4) == "<A0007>"
    assert parse_action_code("  Action : <a12> \n", code_width=4) == "<A0012>"
    assert parse_action_code("No action here", code_width=4) is None


if __name__ == "__main__":
    test_code_slot_restricted_to_feasible_actions()
    test_eos_unblocked_only_after_complete_action_code()
    test_parse_action_code()
    print("All step masking hook tests passed.")
