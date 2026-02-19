"""Prompt orchestrator — builds LLM message lists from policy decisions.

This replaces the old ``_build_messages()`` helper.  It is policy-aware:
flags like ``privacy_mode`` and ``greeting_name`` control which frames
are injected, keeping that logic out of the generators.

Token budgeting
---------------
Before appending conversation history, the history block is fitted to
``settings.MAX_HISTORY_TOKENS`` via ``context_manager.fit_messages_to_budget``.
When ``settings.ENABLE_HISTORY_SUMMARIZATION`` is True, overflowing turns
are compressed into an LLM-generated summary instead of being silently
dropped.
"""

import logging

import context_manager
from settings import settings
from .prompts import (
    SYSTEM_PROMPT,
    PROFILE_CONTEXT_FRAME,
    RAG_CONTEXT_FRAME,
    QA_CONTEXT_FRAME,
    PRIVACY_QA_FRAME,
    GREETING_PERSONALIZATION_FRAME,
)

logger = logging.getLogger(__name__)


def build_messages(
    user_query: str,
    *,
    chat_history: list | None = None,
    curated_history: list | None = None,
    rag_context: str = "",
    profile_context: str = "",
    similar_qa_context: str = "",
    privacy_mode: bool = False,
    greeting_name: str | None = None,
) -> list[dict]:
    """Assemble the OpenAI-format message list for the LLM.

    Parameters
    ----------
    user_query : str
        The current user message.
    chat_history : list | None
        Raw recent messages (fallback when curated_history is None).
    curated_history : list | None
        Policy-selected (recency + semantic) history slice.
    rag_context : str
        Retrieved knowledge-base documents.
    profile_context : str
        Formatted user profile text.
    similar_qa_context : str
        Prior Q&A for continuity.
    privacy_mode : bool
        Whether the privacy response frame should be injected.
    greeting_name : str | None
        If set, inject the greeting personalization frame.
    """
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    # ── Greeting personalization (before profile, so LLM sees name first) ─
    if greeting_name:
        messages.append({
            "role": "system",
            "content": GREETING_PERSONALIZATION_FRAME.format(name=greeting_name),
        })

    # ── Profile context ───────────────────────────────────────────────────
    if profile_context:
        messages.append({
            "role": "system",
            "content": PROFILE_CONTEXT_FRAME.format(profile=profile_context),
        })

    # ── Knowledge-base documents ──────────────────────────────────────────
    if rag_context:
        messages.append({
            "role": "system",
            "content": RAG_CONTEXT_FRAME.format(context=rag_context),
        })

    # ── Privacy frame or prior Q&A ────────────────────────────────────────
    if privacy_mode:
        messages.append({
            "role": "system",
            "content": PRIVACY_QA_FRAME,
        })
    elif similar_qa_context:
        messages.append({
            "role": "system",
            "content": QA_CONTEXT_FRAME.format(qa=similar_qa_context),
        })

    # ── Conversation history (token-budget enforced) ──────────────────────
    history = curated_history if curated_history is not None else chat_history
    if history:
        if settings.ENABLE_HISTORY_SUMMARIZATION:
            from .client import completion
            history = context_manager.summarize_old_turns(
                history,
                max_history_tokens=settings.MAX_HISTORY_TOKENS,
                completion_fn=completion,
            )
        else:
            history = context_manager.fit_messages_to_budget(
                history,
                budget_tokens=settings.MAX_HISTORY_TOKENS,
            )
        messages.extend(history)

    # ── Current user message ──────────────────────────────────────────────
    messages.append({"role": "user", "content": user_query})

    logger.info(
        "Messages: %d total (rag=%s, profile=%s, qa=%s, privacy=%s, greeting=%s)",
        len(messages),
        "yes" if rag_context else "no",
        "yes" if profile_context else "no",
        "yes" if similar_qa_context else "no",
        "yes" if privacy_mode else "no",
        greeting_name or "no",
    )
    return messages
