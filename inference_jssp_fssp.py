"""Canonical inference entrypoint.

Uses the action-code + candidate-scoring reranker policy from the
`[action_code_add][fix]` notebooks.
"""

from inference_jssp_action_code_candidate_scoring import main


if __name__ == "__main__":
    main()
