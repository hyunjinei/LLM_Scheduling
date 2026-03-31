RL Canonicalization Checklist

1. Model loading must follow:
   - resolve source path/repo
   - if adapter source: load base family first
   - install action tokens
   - load policy adapter
   - enter training/inference mode

2. Never call `model.to(device)` on 4bit bitsandbytes / Unsloth models.

3. RL notebooks and `RL_jssp_fssp.py` must agree on:
   - `ensure_action_special_tokens(...)` call
   - adapter-root vs base-model loading
   - optional checkpoint subfolder handling via `model_checkpoint_tag`
   - `adapter_only` trainable-parameter validation

4. `BOPO` in this repo means:
   - collect rollout groups
   - sort feasible rollouts by makespan
   - construct winner/loser step pairs
   - update by pairwise preference, not pseudo-label teacher forcing

5. If you want self-label RL instead:
   - generate multiple rollouts
   - choose best trajectory by final makespan
   - convert its step actions into pseudo-labels
   - run a teacher-forcing update on that trajectory

6. Before serious RL runs, verify:
   - model source path
   - action token install log
   - policy adapter loaded log
   - RL trainable params log
   - no `model.to(device)` calls remain in the active path
