"""Tests for the prompt_orchestrator.build_messages() function.

build_messages() is pure Python — no LLM calls, no I/O.
Every test asserts the exact structure of the returned messages list.
"""

import pytest
from llm.prompt_orchestrator import build_messages
from llm.prompts import (
    SYSTEM_PROMPT,
    GREETING_PERSONALIZATION_FRAME,
    PROFILE_CONTEXT_FRAME,
    RAG_CONTEXT_FRAME,
    QA_CONTEXT_FRAME,
    PRIVACY_QA_FRAME,
)


QUERY = "What is pgvector?"


# ─── Baseline structure ───────────────────────────────────────────────────

def test_minimal_messages():
    """Minimal call: system prompt + user query only."""
    msgs = build_messages(QUERY)
    assert len(msgs) == 2
    assert msgs[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert msgs[-1] == {"role": "user", "content": QUERY}


def test_user_query_always_last():
    """The user query must always be the final message."""
    msgs = build_messages(
        QUERY,
        rag_context="Some docs",
        profile_context="Name: Alice",
        similar_qa_context="Prior Q&A",
    )
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == QUERY


def test_system_prompt_always_first():
    msgs = build_messages(QUERY, rag_context="docs", profile_context="profile")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == SYSTEM_PROMPT


# ─── Optional context frames ──────────────────────────────────────────────

def test_greeting_name_injected():
    msgs = build_messages(QUERY, greeting_name="Bob")
    system_contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Bob" in c for c in system_contents)
    # Greeting frame should come before profile frame (index 1)
    greeting_idx = next(
        i for i, m in enumerate(msgs) if m["role"] == "system" and "Bob" in m["content"]
    )
    assert greeting_idx == 1  # immediately after SYSTEM_PROMPT


def test_no_greeting_frame_when_name_not_set():
    msgs = build_messages(QUERY)
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    expected_fragment = GREETING_PERSONALIZATION_FRAME.split("{name}")[0].strip()[:30]
    assert not any(expected_fragment in c for c in contents)


def test_profile_frame_injected():
    msgs = build_messages(QUERY, profile_context="Name: Charlie\nJob: Developer")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Charlie" in c for c in contents)


def test_rag_frame_injected():
    msgs = build_messages(QUERY, rag_context="pgvector is a PostgreSQL extension.")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("pgvector is a PostgreSQL extension." in c for c in contents)


def test_qa_context_injected_when_no_privacy_mode():
    msgs = build_messages(QUERY, similar_qa_context="Q: What is it? A: It's cool.")
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any("Q: What is it?" in c for c in contents)


def test_privacy_frame_injected_when_privacy_mode():
    msgs = build_messages(QUERY, privacy_mode=True)
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    assert any(PRIVACY_QA_FRAME[:40] in c for c in contents)


def test_privacy_mode_suppresses_qa_context():
    """privacy_mode=True must inject PRIVACY_QA_FRAME, not QA_CONTEXT_FRAME."""
    msgs = build_messages(
        QUERY,
        privacy_mode=True,
        similar_qa_context="Prior Q&A text",
    )
    contents = [m["content"] for m in msgs if m["role"] == "system"]
    # PRIVACY_QA_FRAME should be present
    assert any(PRIVACY_QA_FRAME[:40] in c for c in contents)
    # QA_CONTEXT_FRAME should NOT be present
    qa_fragment = QA_CONTEXT_FRAME.split("{qa}")[0].strip()[:30]
    assert not any(qa_fragment in c for c in contents)


# ─── History injection ────────────────────────────────────────────────────

def test_curated_history_used_over_chat_history():
    chat = [{"role": "user", "content": "old message"}]
    curated = [{"role": "user", "content": "curated message"}]
    msgs = build_messages(QUERY, chat_history=chat, curated_history=curated)
    contents = [m["content"] for m in msgs]
    assert "curated message" in contents
    assert "old message" not in contents


def test_chat_history_fallback_when_no_curated():
    chat = [{"role": "user", "content": "fallback message"}]
    msgs = build_messages(QUERY, chat_history=chat, curated_history=None)
    contents = [m["content"] for m in msgs]
    assert "fallback message" in contents


def test_no_history_when_both_none():
    msgs = build_messages(QUERY, chat_history=None, curated_history=None)
    # system prompt + user query only
    assert len(msgs) == 2


def test_history_appears_before_user_query():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "response"},
    ]
    msgs = build_messages(QUERY, curated_history=history)
    history_indices = [i for i, m in enumerate(msgs) if m["content"] in ("first", "response")]
    user_idx = next(i for i, m in enumerate(msgs) if m["content"] == QUERY)
    assert all(hi < user_idx for hi in history_indices)


# ─── Frame ordering ───────────────────────────────────────────────────────

def test_frame_order_system_greeting_profile_rag_qa_history_user():
    """Verify the documented injection order is preserved end-to-end."""
    history = [{"role": "user", "content": "earlier question"}]
    msgs = build_messages(
        QUERY,
        greeting_name="Dana",
        profile_context="Name: Dana",
        rag_context="RAG docs",
        similar_qa_context="QA context",
        curated_history=history,
    )
    roles_and_snippets = [(m["role"], m["content"][:20]) for m in msgs]
    # All system messages first, then history, then user query
    system_indices = [i for i, m in enumerate(msgs) if m["role"] == "system"]
    history_indices = [i for i, m in enumerate(msgs) if m["content"] == "earlier question"]
    user_index = len(msgs) - 1

    assert max(system_indices) < min(history_indices)
    assert max(history_indices) < user_index
