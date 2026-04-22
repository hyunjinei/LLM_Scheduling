"""Shared action-code candidate-scoring helpers for JSSP step policies."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .action_token_utils import action_code_to_token_id, parse_action_code


CANDIDATE_SCORE_TOKEN_DEFAULT = "<CAND_SCORE>"


def _try_forward_last_hidden_state(module, *, input_ids, attention_mask):
    outputs = module(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    last_hidden = getattr(outputs, "last_hidden_state", None)
    if last_hidden is not None:
        return last_hidden
    if isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
        return outputs[0]
    raise RuntimeError("last_hidden_state is unavailable from candidate-scoring backbone forward.")


def _forward_final_hidden_state(model, *, input_ids, attention_mask):
    candidate_modules = []

    def _append_module(module):
        if isinstance(module, nn.Module):
            candidate_modules.append(module)

    _append_module(getattr(model, "model", None))
    base_model = getattr(model, "base_model", None)
    _append_module(getattr(base_model, "model", None))
    _append_module(base_model)
    _append_module(getattr(model, "backbone", None))
    _append_module(model)

    seen = set()
    for module in candidate_modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        try:
            return _try_forward_last_hidden_state(
                module,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        except Exception:
            continue

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    hidden_states = getattr(outputs, "hidden_states", None)
    if hidden_states is None:
        raise RuntimeError("Unable to obtain final hidden state for candidate scoring.")
    return hidden_states[-1]


def ensure_candidate_score_token(
    tokenizer,
    model=None,
    token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
) -> Dict[str, int]:
    token = str(token)
    current_vocab = tokenizer.get_vocab()
    existing_specials = set(getattr(tokenizer, "additional_special_tokens", []) or [])
    added = 0
    if token not in current_vocab and token not in existing_specials:
        added = int(tokenizer.add_special_tokens({"additional_special_tokens": [token]}))
    if model is not None:
        target_size = len(tokenizer)
        input_size = int(model.get_input_embeddings().weight.shape[0])
        if input_size != target_size:
            model.resize_token_embeddings(target_size)
    token_id = tokenizer.convert_tokens_to_ids(token)
    return {
        "token": token,
        "token_id": int(token_id),
        "num_added_tokens": int(added),
        "vocab_size": int(len(tokenizer)),
    }


def maybe_reinitialize_action_token_rows(
    tokenizer,
    model,
    train_lm_head: bool,
    train_embed_tokens: bool,
    code_width: int = 4,
    code_start: int = 1,
    code_cap: int = 9999,
    enabled: bool = True,
    scale: float = 0.02,
    seed: int | None = None,
    share_input_output_rows: bool = True,
) -> Dict[str, object]:
    if not bool(enabled):
        return {"applied": False, "reason": "disabled"}
    if bool(train_lm_head) or bool(train_embed_tokens):
        return {"applied": False, "reason": "trainable_rows"}

    from .action_token_utils import format_action_code

    start_token = format_action_code(code_start, code_width=code_width)
    end_token = format_action_code(code_cap, code_width=code_width)
    start_id = action_code_to_token_id(tokenizer, start_token)
    end_id = action_code_to_token_id(tokenizer, end_token)
    if int(end_id) < int(start_id):
        raise ValueError(
            f"Invalid action token range: start_id={start_id}, end_id={end_id}"
        )

    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    if input_embeddings is None or getattr(input_embeddings, "weight", None) is None:
        raise RuntimeError("Input embedding weights are unavailable for action-token reinitialization.")
    if output_embeddings is None or getattr(output_embeddings, "weight", None) is None:
        raise RuntimeError("Output embedding weights are unavailable for action-token reinitialization.")

    input_weight = input_embeddings.weight
    output_weight = output_embeddings.weight
    if int(start_id) <= 0:
        raise ValueError("Action token ids unexpectedly start at 0; cannot build reference distribution.")

    row_count = int(end_id) - int(start_id) + 1
    if row_count <= 0:
        raise ValueError(f"Invalid action row count: {row_count}")

    device_list = []
    if input_weight.is_cuda:
        device_list.append(input_weight.device)

    with torch.random.fork_rng(devices=device_list):
        if seed is not None:
            torch.manual_seed(int(seed))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(seed))

        with torch.no_grad():
            ref_input = input_weight[: int(start_id)].detach().float()
            input_mean = ref_input.mean(dim=0, keepdim=True)
            input_std = ref_input.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
            input_noise = torch.randn(
                (row_count, input_weight.shape[1]),
                device=input_weight.device,
                dtype=torch.float32,
            )
            new_input = input_mean + (input_std * input_noise * float(scale))
            input_weight[int(start_id): int(end_id) + 1] = new_input.to(dtype=input_weight.dtype)

            if bool(share_input_output_rows):
                if int(output_weight.shape[1]) != int(new_input.shape[1]):
                    raise RuntimeError(
                        "Cannot share action input/output rows because dimensions differ: "
                        f"input_dim={int(new_input.shape[1])}, output_dim={int(output_weight.shape[1])}"
                    )
                new_output = new_input.to(device=output_weight.device, dtype=torch.float32)
            else:
                ref_output = output_weight[: int(start_id)].detach().float()
                output_mean = ref_output.mean(dim=0, keepdim=True)
                output_std = ref_output.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
                output_noise = torch.randn(
                    (row_count, output_weight.shape[1]),
                    device=output_weight.device,
                    dtype=torch.float32,
                )
                new_output = output_mean + (output_std * output_noise * float(scale))
            output_weight[int(start_id): int(end_id) + 1] = new_output.to(dtype=output_weight.dtype)

    return {
        "applied": True,
        "reason": "frozen_action_rows_reinitialized",
        "start_id": int(start_id),
        "end_id": int(end_id),
        "row_count": int(row_count),
        "scale": float(scale),
        "seed": None if seed is None else int(seed),
        "share_input_output_rows": bool(share_input_output_rows),
    }


def summarize_action_token_row_geometry(
    tokenizer,
    model,
    sample_tokens=None,
    code_width: int = 4,
) -> Dict[str, object]:
    from .action_token_utils import format_action_code

    if sample_tokens is None:
        sample_tokens = [
            format_action_code(1, code_width=code_width),
            format_action_code(2, code_width=code_width),
            format_action_code(3, code_width=code_width),
            format_action_code(100, code_width=code_width),
            format_action_code(1000, code_width=code_width),
            format_action_code(9999, code_width=code_width),
        ]

    action_ids = [int(action_code_to_token_id(tokenizer, tok)) for tok in sample_tokens]
    input_weight = model.get_input_embeddings().weight.detach().float().cpu()
    output_weight = model.get_output_embeddings().weight.detach().float().cpu()

    input_pairwise = []
    output_pairwise = []
    for i in range(len(action_ids)):
        for j in range(i + 1, len(action_ids)):
            input_pairwise.append(float(torch.norm(input_weight[action_ids[i]] - input_weight[action_ids[j]], p=2).item()))
            output_pairwise.append(float(torch.norm(output_weight[action_ids[i]] - output_weight[action_ids[j]], p=2).item()))

    input_norms = [float(torch.norm(input_weight[token_id], p=2).item()) for token_id in action_ids]
    output_norms = [float(torch.norm(output_weight[token_id], p=2).item()) for token_id in action_ids]

    return {
        "sample_tokens": list(sample_tokens),
        "action_ids": list(action_ids),
        "input_pairwise_min": min(input_pairwise) if input_pairwise else 0.0,
        "input_pairwise_max": max(input_pairwise) if input_pairwise else 0.0,
        "input_pairwise_mean": (sum(input_pairwise) / len(input_pairwise)) if input_pairwise else 0.0,
        "output_pairwise_min": min(output_pairwise) if output_pairwise else 0.0,
        "output_pairwise_max": max(output_pairwise) if output_pairwise else 0.0,
        "output_pairwise_mean": (sum(output_pairwise) / len(output_pairwise)) if output_pairwise else 0.0,
        "input_norm_min": min(input_norms) if input_norms else 0.0,
        "input_norm_max": max(input_norms) if input_norms else 0.0,
        "output_norm_min": min(output_norms) if output_norms else 0.0,
        "output_norm_max": max(output_norms) if output_norms else 0.0,
    }


def extract_candidate_transition_entries_for_scoring(
    state_text: str,
    feasible_action_codes: Sequence[str],
    code_width: int = 4,
) -> Dict[str, List[str] | str]:
    if not isinstance(state_text, str) or not state_text.strip():
        raise ValueError("state_text must be a non-empty string for candidate scoring.")
    feasible_set = {str(code).strip() for code in feasible_action_codes if str(code).strip()}
    if not feasible_set:
        raise ValueError("feasible_action_codes must be non-empty for candidate scoring.")

    rebuilt_lines: List[str] = []
    candidate_action_codes_in_order: List[str] = []
    candidate_display_lines_in_order: List[str] = []
    candidate_original_lines_in_order: List[str] = []

    candidate_idx = 0
    for line in state_text.splitlines():
        stripped = line.strip()
        parsed_code = parse_action_code(stripped, code_width=code_width)
        if parsed_code is None or parsed_code not in feasible_set or " | " not in line:
            rebuilt_lines.append(line)
            continue

        feature_suffix = str(line.split(" | ", 1)[1]).strip()
        candidate_idx += 1
        display_line = f"Candidate {candidate_idx} | {feature_suffix}"
        rebuilt_lines.append(display_line)
        candidate_action_codes_in_order.append(str(parsed_code))
        candidate_display_lines_in_order.append(display_line)
        candidate_original_lines_in_order.append(str(line))

    if not candidate_action_codes_in_order:
        raise ValueError(
            "No candidate transition lines were found for candidate scoring. "
            f"feasible_action_codes={list(feasible_set)[:8]}"
        )

    return {
        "ordinalized_state_text": "\n".join(rebuilt_lines),
        "candidate_action_codes_in_order": list(candidate_action_codes_in_order),
        "candidate_display_lines_in_order": list(candidate_display_lines_in_order),
        "candidate_original_lines_in_order": list(candidate_original_lines_in_order),
    }


def build_candidate_query_prompt_text(
    tokenizer,
    state_text: str,
    eval_candidate_line: str,
    score_token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert JSSP scheduler. "
                "Read the full current step state and the entire feasible candidate set. "
                "The queried candidate is one member of that feasible set. "
                "Represent how strong the queried candidate is relative to the other feasible actions."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{state_text}\n\n"
                "Candidate under evaluation:\n"
                f"{str(eval_candidate_line)}\n"
                "Task: Compare the queried candidate against all feasible candidates shown above. "
                "Represent the queried candidate score at the single marker shown below.\n"
                f"{str(score_token)}"
            ),
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def _prepare_candidate_query_batch(
    tokenizer,
    prompt_text: str,
    feasible_action_codes: Sequence[str],
    *,
    code_width: int = 4,
    score_token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
    max_length: int = 16384,
):
    prepared = extract_candidate_transition_entries_for_scoring(
        state_text=prompt_text,
        feasible_action_codes=feasible_action_codes,
        code_width=int(code_width),
    )
    ordinalized_state_text = str(prepared["ordinalized_state_text"])
    candidate_codes_in_order = list(prepared["candidate_action_codes_in_order"])
    candidate_display_lines_in_order = list(prepared["candidate_display_lines_in_order"])

    query_prompts = [
        build_candidate_query_prompt_text(
            tokenizer=tokenizer,
            state_text=ordinalized_state_text,
            eval_candidate_line=candidate_line,
            score_token=score_token,
        )
        for candidate_line in candidate_display_lines_in_order
    ]
    tokenized = tokenizer(
        query_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )

    score_token_id = int(tokenizer.convert_tokens_to_ids(score_token))
    score_marker_positions = []
    for row_ids in tokenized["input_ids"].tolist():
        positions = [idx for idx, token_id in enumerate(row_ids) if int(token_id) == int(score_token_id)]
        if len(positions) != 1:
            raise ValueError(
                "Expected exactly one score marker token per candidate query. "
                f"count={len(positions)}, score_token={score_token}"
            )
        score_marker_positions.append(int(positions[0]))

    return {
        "prepared": prepared,
        "ordinalized_state_text": ordinalized_state_text,
        "candidate_codes_in_order": candidate_codes_in_order,
        "candidate_display_lines_in_order": candidate_display_lines_in_order,
        "query_prompts": query_prompts,
        "tokenized": tokenized,
        "score_marker_positions": score_marker_positions,
    }


def _compute_candidate_scores(
    model,
    candidate_score_head,
    *,
    tokenized,
    score_marker_positions,
    query_forward_batch_size: int = 16,
    require_grad: bool = False,
    detach_backbone: bool = False,
):
    device = infer_module_device(model)
    score_marker_positions_cpu = torch.tensor(
        score_marker_positions,
        dtype=torch.long,
    )

    all_scores = []
    was_training = bool(model.training)
    model.eval()
    try:
        total_queries = int(tokenized["input_ids"].size(0))
        for start in range(0, total_queries, int(query_forward_batch_size)):
            end = min(total_queries, start + int(query_forward_batch_size))
            chunk_input_ids = tokenized["input_ids"][start:end].to(device)
            chunk_attention_mask = tokenized["attention_mask"][start:end].to(device)

            if bool(require_grad) and bool(detach_backbone):
                with torch.no_grad():
                    hidden_states = _forward_final_hidden_state(
                        model,
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                    )
                    row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                    chunk_score_positions = score_marker_positions_cpu[start:end].to(hidden_states.device)
                    query_hidden = hidden_states[row_indices, chunk_score_positions].detach()
                query_hidden = query_hidden.to(dtype=candidate_score_head.weight.dtype)
                chunk_scores = candidate_score_head(query_hidden).squeeze(-1).float()
            else:
                grad_context = torch.enable_grad() if bool(require_grad) else torch.no_grad()
                with grad_context:
                    hidden_states = _forward_final_hidden_state(
                        model,
                        input_ids=chunk_input_ids,
                        attention_mask=chunk_attention_mask,
                    )
                    row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
                    chunk_score_positions = score_marker_positions_cpu[start:end].to(hidden_states.device)
                    query_hidden = hidden_states[row_indices, chunk_score_positions]
                    query_hidden = query_hidden.to(dtype=candidate_score_head.weight.dtype)
                    chunk_scores = candidate_score_head(query_hidden).squeeze(-1).float()

            all_scores.append(chunk_scores)
            del chunk_input_ids, chunk_attention_mask, hidden_states, row_indices, chunk_score_positions, query_hidden, chunk_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        if was_training:
            model.train()

    return torch.cat(all_scores, dim=0)


def build_candidate_scoring_example(
    example,
    tokenizer,
    max_length: int,
    score_token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
    code_width: int = 4,
):
    state_text = str(example.get("state_text", ""))
    feasible_action_codes = [str(x) for x in example.get("feasible_action_codes", [])]
    target_action_code = str(example.get("target_text", "")).strip()
    if not target_action_code:
        raise ValueError("Missing target_text for candidate scoring example.")

    prepared = extract_candidate_transition_entries_for_scoring(
        state_text=state_text,
        feasible_action_codes=feasible_action_codes,
        code_width=int(code_width),
    )
    ordinalized_state_text = str(prepared["ordinalized_state_text"])
    candidate_codes_in_order = list(prepared["candidate_action_codes_in_order"])
    candidate_display_lines_in_order = list(prepared["candidate_display_lines_in_order"])
    candidate_original_lines_in_order = list(prepared["candidate_original_lines_in_order"])

    if target_action_code not in candidate_codes_in_order:
        raise ValueError(
            "target_action_code was not found in candidate order derived from state_text. "
            f"target={target_action_code}, candidate_codes={candidate_codes_in_order[:16]}"
        )

    score_token_id = int(tokenizer.convert_tokens_to_ids(score_token))
    if score_token_id < 0:
        raise ValueError(f"score token is not in tokenizer vocab: {score_token}")

    candidate_query_texts: List[str] = []
    candidate_query_input_ids: List[List[int]] = []
    candidate_query_attention_masks: List[List[int]] = []
    candidate_score_marker_positions: List[int] = []

    for candidate_line in candidate_display_lines_in_order:
        prompt_text = build_candidate_query_prompt_text(
            tokenizer=tokenizer,
            state_text=ordinalized_state_text,
            eval_candidate_line=candidate_line,
            score_token=score_token,
        )
        tokenized = tokenizer(
            prompt_text,
            add_special_tokens=False,
            truncation=True,
            max_length=int(max_length),
        )
        input_ids = [int(x) for x in tokenized.get("input_ids", [])]
        attention_mask = [int(x) for x in tokenized.get("attention_mask", [1] * len(input_ids))]
        marker_positions = [
            int(idx) for idx, token_id in enumerate(input_ids)
            if int(token_id) == int(score_token_id)
        ]
        if len(marker_positions) != 1:
            raise ValueError(
                "Expected exactly one candidate score marker per query prompt. "
                f"count={len(marker_positions)}, candidate_line={candidate_line}, max_length={int(max_length)}"
            )
        candidate_query_texts.append(prompt_text)
        candidate_query_input_ids.append(input_ids)
        candidate_query_attention_masks.append(attention_mask)
        candidate_score_marker_positions.append(int(marker_positions[0]))

    out = dict(example)
    out["ordinalized_state_text"] = str(ordinalized_state_text)
    out["candidate_query_texts"] = list(candidate_query_texts)
    out["candidate_query_input_ids"] = list(candidate_query_input_ids)
    out["candidate_query_attention_masks"] = list(candidate_query_attention_masks)
    out["candidate_score_marker_positions"] = list(candidate_score_marker_positions)
    out["candidate_action_codes_in_order"] = list(candidate_codes_in_order)
    out["candidate_display_lines_in_order"] = list(candidate_display_lines_in_order)
    out["candidate_original_lines_in_order"] = list(candidate_original_lines_in_order)
    out["target_candidate_index"] = int(candidate_codes_in_order.index(target_action_code))
    out["prompt_token_count"] = int(sum(len(ids) for ids in candidate_query_input_ids))
    out["assistant_token_count"] = 0
    out["supervised_token_count"] = int(len(candidate_codes_in_order))
    return out


class CandidateScoringCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    def __call__(self, features):
        flat_input_ids = []
        flat_attention_mask = []
        flat_score_marker_positions = []
        candidate_counts = []
        target_candidate_index = []
        candidate_action_codes = []
        for feature in features:
            query_ids = [list(x) for x in feature.get("candidate_query_input_ids", [])]
            query_masks = [list(x) for x in feature.get("candidate_query_attention_masks", [])]
            query_marker_positions = [int(x) for x in feature.get("candidate_score_marker_positions", [])]
            query_codes = list(feature.get("candidate_action_codes_in_order", []))
            count = int(len(query_ids))
            if count <= 0:
                raise ValueError("candidate_query_input_ids must be non-empty for candidate scoring.")
            if not (len(query_masks) == count == len(query_marker_positions) == len(query_codes)):
                raise ValueError(
                    "Candidate query field lengths mismatch: "
                    f"ids={len(query_ids)}, masks={len(query_masks)}, markers={len(query_marker_positions)}, codes={len(query_codes)}"
                )
            candidate_counts.append(count)
            target_candidate_index.append(int(feature["target_candidate_index"]))
            candidate_action_codes.append(query_codes)
            flat_input_ids.extend(query_ids)
            flat_attention_mask.extend(query_masks)
            flat_score_marker_positions.extend(query_marker_positions)

        max_len = max(len(ids) for ids in flat_input_ids)
        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(flat_input_ids, flat_attention_mask):
            pad_len = max_len - len(ids)
            padded_input_ids.append(list(ids) + [int(self.pad_token_id)] * pad_len)
            padded_attention_mask.append(list(mask) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "score_marker_positions": torch.tensor(flat_score_marker_positions, dtype=torch.long),
            "candidate_counts": torch.tensor(candidate_counts, dtype=torch.long),
            "target_candidate_index": torch.tensor(target_candidate_index, dtype=torch.long),
            "candidate_action_codes": candidate_action_codes,
        }


class CandidateScoringModel(nn.Module):
    def __init__(
        self,
        backbone_model,
        score_token: str,
        head_init_std: float = 0.02,
        query_forward_batch_size: int = 16,
    ):
        super().__init__()
        self.backbone = backbone_model
        hidden_size = int(getattr(backbone_model.config, "hidden_size"))
        backbone_param = next(backbone_model.parameters())
        head_device = backbone_param.device
        head_dtype = backbone_param.dtype
        self.candidate_score_head = nn.Linear(
            hidden_size,
            1,
            bias=True,
            device=head_device,
            dtype=head_dtype,
        )
        nn.init.normal_(self.candidate_score_head.weight, mean=0.0, std=float(head_init_std))
        nn.init.zeros_(self.candidate_score_head.bias)
        self.score_token = str(score_token)
        self.policy_head_type = "candidate_scoring"
        self.query_forward_batch_size = max(1, int(query_forward_batch_size))
        self.config = getattr(backbone_model, "config", None)

    def _score_flat_queries(self, input_ids, attention_mask, score_marker_positions):
        flat_scores = []
        total_queries = int(input_ids.size(0))
        for start in range(0, total_queries, int(self.query_forward_batch_size)):
            end = min(total_queries, start + int(self.query_forward_batch_size))
            chunk_input_ids = input_ids[start:end]
            chunk_attention_mask = attention_mask[start:end]
            chunk_positions = score_marker_positions[start:end]
            hidden_states = _forward_final_hidden_state(
                self.backbone,
                input_ids=chunk_input_ids,
                attention_mask=chunk_attention_mask,
            )
            row_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            gathered = hidden_states[row_indices, chunk_positions.to(hidden_states.device).long()]
            gathered = gathered.to(dtype=self.candidate_score_head.weight.dtype)
            chunk_scores = self.candidate_score_head(gathered).squeeze(-1)
            flat_scores.append(chunk_scores)
            del hidden_states, row_indices, gathered, chunk_scores
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return torch.cat(flat_scores, dim=0)

    def forward(
        self,
        input_ids,
        attention_mask,
        score_marker_positions,
        candidate_counts,
        target_candidate_index=None,
        **kwargs,
    ):
        flat_scores = self._score_flat_queries(
            input_ids=input_ids,
            attention_mask=attention_mask,
            score_marker_positions=score_marker_positions,
        )
        counts = [int(x) for x in candidate_counts.tolist()]
        if sum(counts) != int(flat_scores.shape[0]):
            raise ValueError(
                f"candidate_counts sum mismatch: counts_sum={sum(counts)}, flat_scores={int(flat_scores.shape[0])}"
            )
        max_candidates = max(counts)
        pad_value = torch.finfo(flat_scores.dtype).min
        candidate_scores = flat_scores.new_full((len(counts), max_candidates), pad_value)
        offset = 0
        for row_idx, count in enumerate(counts):
            candidate_scores[row_idx, :count] = flat_scores[offset:offset + count]
            offset += count
        loss = None
        if target_candidate_index is not None:
            loss = F.cross_entropy(candidate_scores.float(), target_candidate_index.long())
        return {
            "loss": loss,
            "candidate_scores": candidate_scores,
            "flat_scores": flat_scores,
        }

    def save_checkpoint_bundle(self, save_dir):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(str(save_dir))
        torch.save(
            {
                "state_dict": self.candidate_score_head.state_dict(),
                "score_token": self.score_token,
                "policy_head_type": self.policy_head_type,
                "query_forward_batch_size": int(self.query_forward_batch_size),
            },
            save_dir / "candidate_scorer.pt",
        )


def load_candidate_score_head(
    scorer_path: str | Path,
    hidden_size: int,
    device,
    dtype,
):
    scorer_payload = torch.load(str(scorer_path), map_location="cpu")
    candidate_score_head = nn.Linear(
        int(hidden_size),
        1,
        bias=True,
        device=device,
        dtype=dtype,
    )
    candidate_score_head.load_state_dict(scorer_payload["state_dict"])
    candidate_score_head = candidate_score_head.to(device=device, dtype=dtype)
    candidate_score_head.eval()
    return candidate_score_head, scorer_payload


def infer_module_device(model_or_module) -> torch.device:
    try:
        emb = model_or_module.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    try:
        return next(model_or_module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def score_candidate_actions(
    model,
    tokenizer,
    candidate_score_head,
    prompt_text: str,
    feasible_action_codes: Sequence[str],
    *,
    do_sample: bool = True,
    temperature: float = 1.0,
    code_width: int = 4,
    score_token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
    max_length: int = 16384,
    query_forward_batch_size: int = 16,
    topk: int = 5,
) -> Dict[str, object]:
    batch = _prepare_candidate_query_batch(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        feasible_action_codes=feasible_action_codes,
        code_width=int(code_width),
        score_token=score_token,
        max_length=int(max_length),
    )
    ordinalized_state_text = str(batch["ordinalized_state_text"])
    candidate_codes_in_order = list(batch["candidate_codes_in_order"])
    candidate_display_lines_in_order = list(batch["candidate_display_lines_in_order"])
    candidate_scores = _compute_candidate_scores(
        model=model,
        candidate_score_head=candidate_score_head,
        tokenized=batch["tokenized"],
        score_marker_positions=batch["score_marker_positions"],
        query_forward_batch_size=int(query_forward_batch_size),
        require_grad=False,
    )
    temp = max(float(temperature), 1e-6)
    probs = torch.softmax(candidate_scores / temp, dim=-1)
    if bool(do_sample) and int(probs.numel()) > 1:
        chosen_idx = int(torch.multinomial(probs, num_samples=1).item())
    else:
        chosen_idx = int(torch.argmax(probs).item())

    chosen_action_code = str(candidate_codes_in_order[chosen_idx])
    topk = min(int(topk), len(candidate_codes_in_order))
    top_vals, top_idx = torch.topk(probs, k=topk)
    entropy = float((-(probs * torch.log(probs.clamp_min(1e-12))).sum()).item())
    debug_payload = {
        "feasible_count": int(len(candidate_codes_in_order)),
        "chosen_action_code": chosen_action_code,
        "chosen_candidate_line": str(candidate_display_lines_in_order[chosen_idx]),
        "chosen_prob": float(probs[chosen_idx].item()),
        "score_min": float(candidate_scores.min().item()),
        "score_max": float(candidate_scores.max().item()),
        "score_std": float(candidate_scores.std(unbiased=False).item()) if int(candidate_scores.numel()) > 1 else 0.0,
        "entropy": entropy,
        "top_probs": [
            (str(candidate_codes_in_order[int(j)]), float(p), str(candidate_display_lines_in_order[int(j)]))
            for p, j in zip(top_vals.tolist(), top_idx.tolist())
        ],
    }
    return {
        "chosen_action_code": chosen_action_code,
        "debug_payload": debug_payload,
        "candidate_scores": candidate_scores,
        "candidate_probs": probs,
        "candidate_codes_in_order": candidate_codes_in_order,
        "candidate_display_lines_in_order": candidate_display_lines_in_order,
        "ordinalized_state_text": ordinalized_state_text,
    }


def compute_candidate_action_log_prob(
    model,
    tokenizer,
    candidate_score_head,
    prompt_text: str,
    feasible_action_codes: Sequence[str],
    chosen_action_code: str,
    *,
    temperature: float = 1.0,
    code_width: int = 4,
    score_token: str = CANDIDATE_SCORE_TOKEN_DEFAULT,
    max_length: int = 16384,
    query_forward_batch_size: int = 16,
    require_grad: bool = True,
    detach_backbone: bool = False,
):
    batch = _prepare_candidate_query_batch(
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        feasible_action_codes=feasible_action_codes,
        code_width=int(code_width),
        score_token=score_token,
        max_length=int(max_length),
    )
    candidate_codes_in_order = list(batch["candidate_codes_in_order"])
    chosen_action_code = str(chosen_action_code)
    if chosen_action_code not in candidate_codes_in_order:
        raise ValueError(
            "chosen_action_code not found in feasible candidate order. "
            f"chosen={chosen_action_code}, feasible_head={candidate_codes_in_order[:8]}"
        )
    candidate_scores = _compute_candidate_scores(
        model=model,
        candidate_score_head=candidate_score_head,
        tokenized=batch["tokenized"],
        score_marker_positions=batch["score_marker_positions"],
        query_forward_batch_size=int(query_forward_batch_size),
        require_grad=bool(require_grad),
        detach_backbone=bool(detach_backbone),
    )
    temp = max(float(temperature), 1e-6)
    log_probs = torch.log_softmax(candidate_scores / temp, dim=-1)
    chosen_idx = int(candidate_codes_in_order.index(chosen_action_code))
    return log_probs[chosen_idx]
