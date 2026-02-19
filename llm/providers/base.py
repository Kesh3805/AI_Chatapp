"""LLM provider base class.

Every provider implements two methods:
  - complete(messages, ...) -> str       (returns response text)
  - stream_text_deltas(messages, ...)    (yields text delta strings)

To add a new provider:
  1. Create llm/providers/your_provider.py
  2. Subclass LLMProvider
  3. Register it in llm/providers/__init__.py
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generator


class LLMProvider(ABC):
    """Abstract base for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g. 'cerebras', 'openai')."""
        ...

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Send messages and return the full response text."""
        ...

    @abstractmethod
    def stream_text_deltas(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Send messages and yield text deltas as they arrive."""
        ...
