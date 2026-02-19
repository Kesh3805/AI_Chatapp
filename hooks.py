"""Extension hooks — customize pipeline behavior without touching core.

Register hooks using decorators:

    from hooks import Hooks

    @Hooks.before_generation
    def log_intent(pipeline_result):
        print(f"Intent: {pipeline_result.intent}")
        return pipeline_result

    @Hooks.after_generation
    def filter_response(response, pipeline_result):
        return response.replace("bad_word", "***")

    @Hooks.policy_override
    def always_inject_profile(features, decision):
        decision.inject_profile = True
        return decision

Hooks run in registration order.  Return the (possibly modified) value
to pass it to the next hook.  Return None to keep the original.

Load your hooks file at startup (e.g. import user_hooks in main.py).
"""

from __future__ import annotations

from typing import Any, Callable


class Hooks:
    """Registry for pipeline extension points."""

    _before_generation: list[Callable] = []
    _after_generation: list[Callable] = []
    _policy_override: list[Callable] = []
    _before_persist: list[Callable] = []

    # ── Decorators ────────────────────────────────────────────────

    @classmethod
    def before_generation(cls, fn: Callable) -> Callable:
        """Called with PipelineResult before LLM generation.

        Signature: fn(pipeline_result) -> pipeline_result
        """
        cls._before_generation.append(fn)
        return fn

    @classmethod
    def after_generation(cls, fn: Callable) -> Callable:
        """Called with (response_text, pipeline_result) after generation.

        Signature: fn(response: str, pipeline_result) -> str
        """
        cls._after_generation.append(fn)
        return fn

    @classmethod
    def policy_override(cls, fn: Callable) -> Callable:
        """Called with (features, decision) after BehaviorPolicy.resolve().

        Signature: fn(features, decision) -> decision
        """
        cls._policy_override.append(fn)
        return fn

    @classmethod
    def before_persist(cls, fn: Callable) -> Callable:
        """Called with (pipeline_result, response_text) before DB persistence.

        Signature: fn(pipeline_result, response_text) -> None
        """
        cls._before_persist.append(fn)
        return fn

    # ── Runners ───────────────────────────────────────────────────

    @classmethod
    def run_before_generation(cls, pipeline_result: Any) -> Any:
        for fn in cls._before_generation:
            result = fn(pipeline_result)
            if result is not None:
                pipeline_result = result
        return pipeline_result

    @classmethod
    def run_after_generation(cls, response: str, pipeline_result: Any) -> str:
        for fn in cls._after_generation:
            result = fn(response, pipeline_result)
            if result is not None:
                response = result
        return response

    @classmethod
    def run_policy_override(cls, features: Any, decision: Any) -> Any:
        for fn in cls._policy_override:
            result = fn(features, decision)
            if result is not None:
                decision = result
        return decision

    @classmethod
    def run_before_persist(cls, pipeline_result: Any, response: str) -> None:
        for fn in cls._before_persist:
            try:
                fn(pipeline_result, response)
            except Exception:
                pass

    # ── Utilities ─────────────────────────────────────────────────

    @classmethod
    def clear(cls) -> None:
        """Remove all registered hooks (useful for testing)."""
        cls._before_generation.clear()
        cls._after_generation.clear()
        cls._policy_override.clear()
        cls._before_persist.clear()
