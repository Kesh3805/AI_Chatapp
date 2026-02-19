"""Token-budget context management — history fitting and conversation summarization.

Why this module exists
----------------------
LLMs have hard context window limits.  Without token budgeting,
a long conversation silently overflows the model's window, causing
truncation artifacts or outright errors.

The RECENCY_WINDOW heuristic (last N messages) is a fast guard, but it
is *count-based*, not *token-based*.  A single long message can consume
as many tokens as 20 short ones.  This module enforces precision.

Token estimation
----------------
We estimate tokens as ``len(text) // 4``.  This is a well-known
approximation (~97 % accurate for English prose, ~90 % for code).
For exact counting, replace ``estimate_tokens()`` with
``tiktoken.encoding_for_model("gpt-4o").encode(text)``.

Public API
----------
    estimate_tokens(text)             → int
    message_tokens(msg)               → int
    history_tokens(messages)          → int
    fit_messages_to_budget(messages, budget_tokens, min_recent=4) → list[dict]
    summarize_old_turns(messages, max_history_tokens, completion_fn) → list[dict]
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
#  Constants
# --------------------------------------------------------------------------

# Characters-per-token approximation.  4 chars ≈ 1 GPT token for English.
_CHARS_PER_TOKEN: int = 4

# Fixed per-message overhead: role string + JSON framing (~10 tokens).
_MSG_OVERHEAD: int = 10


# --------------------------------------------------------------------------
#  Token estimation
# --------------------------------------------------------------------------

def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (chars / 4).

    Fast, zero-dependency approximation.  Replace the body with a tiktoken
    call for exact counting when precision matters.
    """
    return max(1, len(text) // _CHARS_PER_TOKEN)


def message_tokens(msg: dict) -> int:
    """Estimated token cost for a single OpenAI-format message dict."""
    return estimate_tokens(msg.get("content", "")) + _MSG_OVERHEAD


def history_tokens(messages: list[dict]) -> int:
    """Total estimated token cost for a list of messages."""
    return sum(message_tokens(m) for m in messages)


# --------------------------------------------------------------------------
#  Budget fitting
# --------------------------------------------------------------------------

def fit_messages_to_budget(
    messages: list[dict],
    budget_tokens: int,
    min_recent: int = 4,
) -> list[dict]:
    """Trim oldest messages until the list fits within *budget_tokens*.

    Always preserves the last ``min_recent`` messages regardless of budget,
    so the model always has some immediate conversational context.

    Args:
        messages:      Chronological ``[{"role": ..., "content": ...}]`` list.
        budget_tokens: Maximum allowed token count for the history block.
        min_recent:    Minimum tail messages to keep unconditionally.

    Returns:
        Trimmed list (same dict objects, not copies).  Empty list is returned
        unchanged.
    """
    if not messages:
        return messages

    # Fast path — already fits.
    total = history_tokens(messages)
    if total <= budget_tokens:
        return messages

    trimmed = list(messages)
    while len(trimmed) > min_recent and history_tokens(trimmed) > budget_tokens:
        trimmed.pop(0)

    dropped = len(messages) - len(trimmed)
    logger.info(
        "Context budget: dropped %d oldest messages to fit %d-token budget "
        "(kept %d, ~%d tokens estimated)",
        dropped,
        budget_tokens,
        len(trimmed),
        history_tokens(trimmed),
    )
    return trimmed


# --------------------------------------------------------------------------
#  Summarization
# --------------------------------------------------------------------------

_SUMMARIZE_SYSTEM = (
    "You are a concise conversation summarizer. "
    "Summarize the following conversation turns into 2-5 sentences. "
    "Capture: the main topics discussed, any facts the user stated about themselves, "
    "unresolved questions, and the user's apparent goal. "
    "Output ONLY the summary — no preamble, no labels, no bullet points."
)


def summarize_old_turns(
    messages: list[dict],
    max_history_tokens: int,
    completion_fn: Callable[[list[dict]], str],
    min_recent: int = 6,
) -> list[dict]:
    """Replace overflow turns with an LLM-generated summary.

    If the history fits within *max_history_tokens*, returns it unchanged.
    Otherwise:

    1. Splits messages into ``[overflow | tail(min_recent)]``.
    2. Posts the overflow transcript to *completion_fn* for summarization.
    3. Returns ``[summary_system_msg] + tail``.

    On summarization failure, falls back to ``fit_messages_to_budget``
    (silent drop) so the pipeline always gets a valid message list.

    Args:
        messages:           Chronological conversation messages.
        max_history_tokens: Token budget for the full history block.
        completion_fn:      A callable with signature ``(messages) -> str``,
                            typically ``llm.client.completion``.
        min_recent:         Recent turns to always keep verbatim.

    Returns:
        History list, possibly prefixed with a compacted summary message.
    """
    if not messages or history_tokens(messages) <= max_history_tokens:
        return messages

    # Keep the freshest `min_recent` turns verbatim.
    recent = messages[-min_recent:]
    to_summarize = messages[:-min_recent]

    if not to_summarize:
        # Not enough messages to split — just trim.
        return fit_messages_to_budget(messages, max_history_tokens, min_recent)

    transcript = "\n".join(
        f"{m['role'].title()}: {m['content'][:400]}"
        for m in to_summarize
    )

    try:
        summary_text = completion_fn(
            [
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": transcript},
            ]
        )
        summary_msg: dict = {
            "role": "system",
            "content": f"[Summary of earlier conversation]: {summary_text.strip()}",
        }
        logger.info(
            "Context: compressed %d turns into ~%d-token summary",
            len(to_summarize),
            estimate_tokens(summary_text),
        )
        return [summary_msg] + list(recent)

    except Exception as exc:
        logger.warning("Summarization failed (%s) — falling back to recency trim", exc)
        return fit_messages_to_budget(messages, max_history_tokens, min_recent)
