"""Anthropic LLM provider.

Handles Anthropic-specific API differences:
  - System prompt is a separate parameter (not a message)
  - Messages must strictly alternate user/assistant
  - Response format differs from OpenAI
"""

from __future__ import annotations

import logging
from typing import Generator

from .base import LLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API."""

    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str, model: str = ""):
        from anthropic import Anthropic  # type: ignore[import-untyped]

        self._client = Anthropic(api_key=api_key)
        self._model = model or self.DEFAULT_MODEL
        logger.info(f"Anthropic provider ready (model={self._model})")

    @property
    def name(self) -> str:
        return "anthropic"

    # ── Internal helpers ──────────────────────────────────────────

    @staticmethod
    def _split_messages(
        messages: list[dict],
    ) -> tuple[str | None, list[dict]]:
        """Separate system messages from chat messages.

        Anthropic API takes system as a top-level param, not a message.
        """
        system_parts: list[str] = []
        chat: list[dict] = []
        for msg in messages:
            if msg["role"] == "system":
                system_parts.append(msg["content"])
            else:
                chat.append({"role": msg["role"], "content": msg["content"]})

        # Merge consecutive same-role messages (Anthropic requires alternation)
        merged: list[dict] = []
        for msg in chat:
            if merged and msg["role"] == merged[-1]["role"]:
                merged[-1]["content"] += "\n\n" + msg["content"]
            else:
                merged.append(msg)

        # Anthropic requires first message to be user
        if merged and merged[0]["role"] != "user":
            merged.insert(0, {"role": "user", "content": "(continued)"})

        system = "\n\n".join(system_parts) if system_parts else None
        return system, merged

    # ── Public API ────────────────────────────────────────────────

    def complete(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        system, chat_msgs = self._split_messages(messages)
        kwargs: dict = dict(
            model=self._model,
            messages=chat_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def stream_text_deltas(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        system, chat_msgs = self._split_messages(messages)
        kwargs: dict = dict(
            model=self._model,
            messages=chat_msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if system:
            kwargs["system"] = system

        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text
