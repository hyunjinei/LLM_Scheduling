"""Canonical SFT entrypoint.

Uses the action-code + `<CAND_SCORE>` candidate-scoring training path from
the `[action_code_add][fix]` notebooks.
"""

from train_jssp_action_code_candidate_scoring import main


if __name__ == "__main__":
    main()
