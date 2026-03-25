from functools import partial
import random
import re
from typing import Any, Dict, List, Sequence

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = object  # type: ignore[misc,assignment]

from llm_jssp.utils.action_token_utils import action_code_to_token_id, parse_action_code


def create_prompt_formats(example, tokenizer):
    """
    Creates JSSP prompt formats for model training.

    Args:
        example (dict): A single example from the dataset.
        tokenizer (AutoTokenizer): Tokenizer that handles chat template formatting.

    Returns:
        dict: Example with formatted text added.
    """
    # print(f"[DEBUG] create_prompt_formats started")  # 디버그 메시지 비활성화
    
    # Standard user prompt variations
    user_variations = [
        "Instruction: Solve the following JSSP problem and provide the optimal job sequence. Also display the makespan.",
        "Task: Provide the solution for the following Job Shop Scheduling Problem as machine-wise job sequences.",
        "Command: Find a solution for the JSSP problem and present the machine-wise work order that minimizes makespan.",
        "Guide: Find the optimal schedule that can minimize the makespan for the following JSSP problem.",
    ]
    
    # Basic constraints for JSSP problems
    constraints = "Rules:\n" + "\n".join([
        "1. Each machine can process only one job at a time. (Operations on a machine cannot overlap)",
        "2. Jobs must follow the specified operation sequence. (e.g., Job 0's Operation 0 must complete before Job 0's Operation 1 starts)",
        "3. The goal is to minimize the total completion time (makespan).",
        "4. The next operation of the same job cannot start until the previous operation is completed.",
        "5. A machine cannot perform more than one operation simultaneously.",
        "6. Makespan is determined by the time when the last operation completes on all machines.",
        "7. Each job must be processed on the designated machine and cannot be changed to another machine.",
        "8. Jobs must follow the specified operation sequence. (Job 0: Operation 0 → Operation 1 → Operation 2...)",
        "9. Machines can have idle time between operations. Minimizing idle time can be an optimization goal.",
        "10. Each machine's operation must complete before starting the next operation.",
        "11. Even if multiple machines are available, one job cannot be processed in parallel on multiple machines.",
        "12. The goal is to minimize the total time (makespan) to complete all operations.",
        "13. A job is complete only after its last operation is processed.",
        "14. Each machine has a fixed processing capacity that cannot be exceeded.",
        "15. Jobs are processed in the order assigned to machines."
    ])
    
    # Add operation format explanation
    operation_format_explanation = "\n\nOperation Format Explanation:\n" + "\n".join([
        "In the 'Operation X: MY,Z' format:",
        "- X is the operation number (starting from 0)",
        "- Y is the machine number where this operation should be performed",
        "- Z is the processing time for this operation",
        "For example, 'Operation 0: M6,69' means the first operation is performed on machine 6 and takes 69 time units."
    ])


    # Updated work sequence selection strategy with strict constraints
    selection_strategy = "\n\nWork Sequence Selection Strategy:\n" + "\n".join([
        "- Ensure that operations of each job are executed in sequence, while finding an order that minimizes overall makespan.",
        "- Output in 'Job X Operation Y, MZ' format, listing each operation in execution order. For example: 'Job 0 Operation 0, M0' means Job 0's first operation is executed on Machine 0.",
        "- All operations from all jobs must be fully sequenced with no duplication or omission.",
        "- The total number of operations must equal (number of jobs) × (number of machines).",
        "- Double-check that each job’s operations appear exactly once and in correct order."
    ])

    # Extract prompt and output directly from example
    prompt = example["prompt_jobs_first"]
    output = example["output"]

    # Randomly select standard message
    user_standard = random.choice(user_variations)

    # Create message array with user and assistant roles
    messages = [
        {
            "role": "system",
            "content": f"You are an expert in Job Shop Scheduling Problem (JSSP). Follow these rules:\n\n{constraints}\n{operation_format_explanation}\n{selection_strategy}"
        },
        {
            "role": "user",
            "content": (
                f"{user_standard}\n{prompt}\n\n"
                "Check your answer against the following rules:\n"
                "- Each operation must appear exactly once.\n"
                "- No operation should be omitted.\n"
                "- The total number of operations must match (number of jobs × number of machines).\n"
                "- Maintain the correct order of operations for each job.\n"
                "- Ensure no machine is assigned overlapping operations.\n"
                "- Clearly state the makespan at the end."
            )
        },
        {
            "role": "assistant",
            "content": f"{output}"
        }
    ]

    # Apply chat template with settings from tokenizer
    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Check final prompt
    if "text" in example:
        # print(f"[DEBUG] Final text length: {len(example['text'])}") # 디버그 메시지 비활성화
        pass

    return example


def create_step_prompt_formats(example, tokenizer, step_supervision_mode="action_only"):
    """
    Creates step-by-step JSSP prompt formats for model training.

    Expected input keys:
        - state_text
        - target_text
    """
    messages = build_step_messages(
        example=example,
        step_supervision_mode=step_supervision_mode,
    )

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


def build_step_messages(example, step_supervision_mode="action_only"):
    state_text = str(example.get("state_text", ""))
    reason_input_text = str(example.get("reason_input_text", ""))
    target_text = str(example.get("target_text", ""))
    reason_target_text = str(example.get("reason_target_text", "")).strip()

    if step_supervision_mode not in {"action_only", "action_reason", "reason_only"}:
        raise ValueError(
            f"Unsupported step_supervision_mode={step_supervision_mode}. "
            "Use one of: action_only, action_reason, reason_only."
        )

    if step_supervision_mode == "reason_only":
        if not reason_input_text:
            raise ValueError("Missing 'reason_input_text' for reason dataset sample.")
        if not reason_target_text:
            raise ValueError("Missing 'reason_target_text' for reason dataset sample.")
        user_content = reason_input_text
        assistant_content = reason_target_text
        system_content = (
            "You are an expert JSSP scheduling analyst. "
            "Explain a fixed already-selected action. "
            "Primary objective context is final makespan (Cmax). "
            "Do not output a new action. "
            "Output format:\n"
            "Reason: ...\n"
            "Not chosen:\n"
            "- <Axxxx>: ..."
        )
    elif step_supervision_mode == "action_reason":
        if not state_text:
            raise ValueError("Missing 'state_text' for step dataset sample.")
        target_action_reason_text = str(example.get("target_action_reason_text", "")).strip()
        if not target_action_reason_text:
            raise ValueError("Missing 'target_action_reason_text' for mixed step dataset sample.")
        assistant_content = target_action_reason_text
        system_content = (
            "You are an expert JSSP scheduler. "
            "Primary objective: minimize final makespan (Cmax). "
            "At each step, choose exactly one feasible action code and explain briefly. "
            "Output format:\n"
            "<Axxxx>\n"
            "Reason: ...\n"
            "Not chosen:\n"
            "- <Axxxx>: ..."
        )
        user_content = state_text
    else:
        if not state_text:
            raise ValueError("Missing 'state_text' for step dataset sample.")
        if not target_text:
            raise ValueError("Missing 'target_text' for step dataset sample.")
        assistant_content = target_text
        system_content = (
            "You are an expert JSSP scheduler. "
            "Primary objective: minimize final makespan (Cmax). "
            "At each step, choose exactly one feasible action code. "
            "Always output exactly one code in this format: <Axxxx>."
        )
        user_content = state_text

    return [
        {
            "role": "system",
            "content": system_content,
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]


def _normalize_token_ids(tokenized_output: Any) -> List[int]:
    if tokenized_output is None:
        return []
    if hasattr(tokenized_output, "tolist"):
        tokenized_output = tokenized_output.tolist()
    if isinstance(tokenized_output, tuple):
        tokenized_output = list(tokenized_output)
    if isinstance(tokenized_output, list):
        if tokenized_output and isinstance(tokenized_output[0], list):
            if len(tokenized_output) != 1:
                raise ValueError("Expected a single tokenized sequence, got a batch.")
            tokenized_output = tokenized_output[0]
        return [int(x) for x in tokenized_output]
    raise TypeError(f"Unsupported tokenized output type: {type(tokenized_output)!r}")


def _collect_action_token_ids(tokenizer) -> List[int]:
    cached = getattr(tokenizer, "_cached_action_token_ids", None)
    if cached is not None:
        return list(cached)

    token_ids = []
    seen = set()
    for token in getattr(tokenizer, "additional_special_tokens", []) or []:
        parsed = parse_action_code(str(token))
        if parsed is None:
            continue
        try:
            token_id = action_code_to_token_id(tokenizer, str(token))
        except Exception:
            continue
        token_id = int(token_id)
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
    setattr(tokenizer, "_cached_action_token_ids", tuple(token_ids))
    return token_ids


def _find_prompt_token_count(tokenizer, prompt_messages, full_ids: Sequence[int]) -> int:
    prompt_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
        )
    )
    if list(full_ids[: len(prompt_ids)]) == prompt_ids:
        return len(prompt_ids)

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if callable(getattr(tokenizer, "__call__", None)):
        prompt_text_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_text_ids = [int(x) for x in prompt_text_ids]
        if list(full_ids[: len(prompt_text_ids)]) == prompt_text_ids:
            return len(prompt_text_ids)

    raise ValueError(
        "Could not align prompt and assistant token boundary for step supervision."
    )


def _extract_action_codes(example: Dict[str, Any]) -> List[str]:
    codes = []
    raw_codes = example.get("action_codes")
    if isinstance(raw_codes, list):
        codes.extend(str(code) for code in raw_codes if str(code).strip())
    action_code_to_job = example.get("action_code_to_job")
    if isinstance(action_code_to_job, dict):
        codes.extend(str(code) for code in action_code_to_job.keys() if str(code).strip())
    if not codes:
        raise ValueError(
            "Missing structured action code metadata in step example. "
            "Expected `action_codes` or `action_code_to_job`."
        )
    deduped = []
    seen = set()
    for code in codes:
        parsed = parse_action_code(str(code))
        if parsed is None or parsed in seen:
            continue
        seen.add(parsed)
        deduped.append(parsed)
    return deduped


def build_step_supervision_example(
    example: Dict[str, Any],
    tokenizer,
    step_supervision_mode: str = "action_only",
    max_length: int | None = None,
    action_loss_weight: float = 1.0,
):
    messages = build_step_messages(
        example=example,
        step_supervision_mode=step_supervision_mode,
    )
    full_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )
    prompt_len = _find_prompt_token_count(tokenizer, messages[:-1], full_ids)
    if prompt_len >= len(full_ids):
        raise ValueError("Assistant span is empty after chat template tokenization.")

    labels = [-100] * len(full_ids)
    loss_weights = [0.0] * len(full_ids)
    attention_mask = [1] * len(full_ids)
    action_target_mask = [0] * len(full_ids)
    assistant_positions = list(range(prompt_len, len(full_ids)))

    action_token_ids = set(_collect_action_token_ids(tokenizer))
    feasible_action_ids = []
    for code in _extract_action_codes(example):
        try:
            token_id = action_code_to_token_id(tokenizer, str(code))
        except Exception:
            continue
        feasible_action_ids.append(int(token_id))
    feasible_action_ids = sorted(set(feasible_action_ids))
    action_position = None
    if step_supervision_mode in {"action_only", "action_reason"}:
        for pos in assistant_positions:
            if int(full_ids[pos]) in action_token_ids:
                action_position = pos
                break
        if action_position is None:
            raise ValueError(
                "Could not find action token inside assistant span. "
                "Check action special-token installation and step targets."
            )

    if step_supervision_mode == "action_only":
        labels[action_position] = int(full_ids[action_position])
        loss_weights[action_position] = float(max(1.0, action_loss_weight))
        action_target_mask[action_position] = 1
    elif step_supervision_mode == "action_reason":
        for pos in assistant_positions:
            labels[pos] = int(full_ids[pos])
            loss_weights[pos] = 1.0
        loss_weights[action_position] = float(max(1.0, action_loss_weight))
        action_target_mask[action_position] = 1
    else:
        for pos in assistant_positions:
            labels[pos] = int(full_ids[pos])
            loss_weights[pos] = 1.0

    if max_length is not None and int(max_length) > 0 and len(full_ids) > int(max_length):
        full_ids = full_ids[: int(max_length)]
        attention_mask = attention_mask[: int(max_length)]
        labels = labels[: int(max_length)]
        loss_weights = loss_weights[: int(max_length)]
        action_target_mask = action_target_mask[: int(max_length)]

    supervised_token_count = sum(1 for label in labels if int(label) != -100)
    if supervised_token_count <= 0:
        raise ValueError(
            "No supervised assistant tokens remain after truncation. "
            "Increase max_length or shorten prompts."
        )

    out = dict(example)
    out["text"] = ""
    out["input_ids"] = [int(x) for x in full_ids]
    out["attention_mask"] = [int(x) for x in attention_mask]
    out["labels"] = [int(x) for x in labels]
    out["loss_weights"] = [float(x) for x in loss_weights]
    out["action_target_mask"] = [int(x) for x in action_target_mask]
    out["feasible_action_ids"] = [int(x) for x in feasible_action_ids]
    out["prompt_token_count"] = int(min(prompt_len, len(full_ids)))
    out["assistant_token_count"] = int(max(0, len(full_ids) - min(prompt_len, len(full_ids))))
    out["supervised_token_count"] = int(supervised_token_count)
    return out


def preprocess_batch(batch, tokenizer, max_length):
    """
    Tokenizes example batches.

    Args:
        batch (dict): Example batch.
        tokenizer (AutoTokenizer): Tokenizer to use.
        max_length (int): Maximum length of tokenized sequences.

    Returns:
        dict: Tokenized batch.
    """
    return tokenizer(
        batch["text"],
        padding=False,  # 동적 패딩을 위해 False로 변경 (SFTTrainer가 배치별로 패딩)
        truncation=True,
        max_length=max_length,
        # return_tensors="pt",
    )


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed: int, dataset, num_proc: int = 1):
    """
    Formats and tokenizes dataset for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer to use.
        max_length (int): Maximum number of tokens to generate from tokenizer.
        seed (int): Seed for dataset shuffling.
        dataset: Dataset to preprocess.

    Returns:
        Dataset: Preprocessed dataset.
    """

    # Add prompts to each sample
    # print("Preprocessing dataset...")  # 디버그 메시지 비활성화
    _create_prompt_formats = partial(create_prompt_formats, tokenizer=tokenizer)
    map_num_proc = max(1, int(num_proc))
    dataset = dataset.map(_create_prompt_formats, num_proc=map_num_proc)

    # Tokenize dataset
    _preprocessing_function = partial(preprocess_batch, tokenizer=tokenizer, max_length=max_length)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["prompt_jobs_first", "output"],
        num_proc=map_num_proc,
    )

    # Shuffle dataset
    dataset = dataset.shuffle(seed=seed)

    return dataset 
