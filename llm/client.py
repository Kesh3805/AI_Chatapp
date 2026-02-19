"""LLM client — thin wrapper that delegates to the configured provider.

Every module in llm/ calls these functions.  The actual provider
(Cerebras, OpenAI, Anthropic) is determined by LLM_PROVIDER in settings.

Usage:
    from llm.client import completion, stream_text_deltas, MAX_RESPONSE_TOKENS
"""

from __future__ import annotations

import logging
from typing import Generator

from settings import settings

logger = logging.getLogger(__name__)

# ── Token budgets (re-exported from settings for backward compat) ─────────
MAX_RESPONSE_TOKENS = settings.MAX_RESPONSE_TOKENS
MAX_CLASSIFIER_TOKENS = settings.MAX_CLASSIFIER_TOKENS
MAX_PROFILE_DETECT_TOKENS = settings.MAX_PROFILE_DETECT_TOKENS
MAX_TITLE_TOKENS = settings.MAX_TITLE_TOKENS


def completion(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> str:
    """Send a chat-completion request and return the response text."""
    from .providers import provider

    return provider().complete(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def stream_text_deltas(
    messages: list[dict],
    *,
    temperature: float = 0.3,
    max_tokens: int = MAX_RESPONSE_TOKENS,
) -> Generator[str, None, None]:
    """Send a streaming request and yield text deltas."""
    from .providers import provider

    return provider().stream_text_deltas(
        messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
