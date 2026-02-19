"""Dynamic LLM provider loader.

Reads LLM_PROVIDER from settings and returns the matching provider instance.
Provider SDKs are imported lazily — only the selected provider's SDK
needs to be installed.

Usage:
    from llm.providers import provider
    text = provider().complete(messages)
"""

from __future__ import annotations

import logging

from .base import LLMProvider

logger = logging.getLogger(__name__)

_provider: LLMProvider | None = None


def _load_provider() -> LLMProvider:
    """Instantiate the configured provider."""
    from settings import settings

    name = settings.LLM_PROVIDER.lower()

    if name == "cerebras":
        from .cerebras import CerebrasProvider

        return CerebrasProvider(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
        )
    elif name == "openai":
        from .openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
            base_url=settings.LLM_BASE_URL,
        )
    elif name == "anthropic":
        from .anthropic import AnthropicProvider

        return AnthropicProvider(
            api_key=settings.LLM_API_KEY,
            model=settings.LLM_MODEL,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{name}'.  "
            f"Supported: cerebras, openai, anthropic"
        )


def provider() -> LLMProvider:
    """Get the singleton LLM provider instance."""
    global _provider
    if _provider is None:
        _provider = _load_provider()
    return _provider


def reset() -> None:
    """Reset the provider (forces re-initialization on next call)."""
    global _provider
    _provider = None
