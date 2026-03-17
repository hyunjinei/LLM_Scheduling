"""Helpers to resolve serial vs dispatch JSSP step stack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Type


@dataclass(frozen=True)
class StepStack:
    env_cls: Type[object]
    build_problem_context_text: Callable[..., str]
    build_randomized_action_code_map: Callable[..., dict]
    build_step_prompt: Callable[..., str]
    build_step_improvement_prompt: Callable[..., str]
    compute_action_transition_features: Callable[..., tuple]
    invert_action_code_map: Callable[..., dict]


def resolve_step_stack(env_mode: str = "serial") -> StepStack:
    normalized = str(env_mode or "serial").strip().lower()
    if normalized == "serial":
        from .jssp_step_env import StaticJSSPStepEnv
        from .step_prompting import (
            build_problem_context_text,
            build_randomized_action_code_map,
            build_step_improvement_prompt,
            build_step_prompt,
            compute_action_transition_features,
            invert_action_code_map,
        )

        return StepStack(
            env_cls=StaticJSSPStepEnv,
            build_problem_context_text=build_problem_context_text,
            build_randomized_action_code_map=build_randomized_action_code_map,
            build_step_prompt=build_step_prompt,
            build_step_improvement_prompt=build_step_improvement_prompt,
            compute_action_transition_features=compute_action_transition_features,
            invert_action_code_map=invert_action_code_map,
        )
    if normalized == "dispatch":
        from .jssp_dispatch_env import DispatchJSSPStepEnv
        from .step_prompting_dispatch import (
            build_problem_context_text,
            build_randomized_action_code_map,
            build_step_improvement_prompt,
            build_step_prompt,
            compute_action_transition_features,
            invert_action_code_map,
        )

        return StepStack(
            env_cls=DispatchJSSPStepEnv,
            build_problem_context_text=build_problem_context_text,
            build_randomized_action_code_map=build_randomized_action_code_map,
            build_step_prompt=build_step_prompt,
            build_step_improvement_prompt=build_step_improvement_prompt,
            compute_action_transition_features=compute_action_transition_features,
            invert_action_code_map=invert_action_code_map,
        )
    raise ValueError(f"Unsupported env_mode={env_mode}. Use 'serial' or 'dispatch'.")


__all__ = ["StepStack", "resolve_step_stack"]
