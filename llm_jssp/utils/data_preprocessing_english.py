from functools import partial
import random

try:
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = object  # type: ignore[misc,assignment]

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
        user_content = reason_input_text or state_text
        if not user_content:
            raise ValueError("Missing 'reason_input_text'/'state_text' for reason dataset sample.")
        if not reason_target_text:
            legacy_reason_text = str(example.get("target_reason_text", "")).strip()
            if legacy_reason_text:
                reason_target_text = legacy_reason_text
        if not reason_target_text:
            raise ValueError("Missing 'reason_target_text' for reason dataset sample.")
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
        if not target_text:
            raise ValueError("Missing 'target_text' for step dataset sample.")
        target_action_reason_text = str(example.get("target_action_reason_text", "")).strip()
        if target_action_reason_text:
            assistant_content = target_action_reason_text
        else:
            # Backward compatible fallback for older datasets.
            assistant_content = (
                f"{target_text}\n{reason_target_text}" if reason_target_text else target_text
            )
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

    messages = [
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

    example["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return example


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
