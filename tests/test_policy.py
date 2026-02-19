"""Tests for the BehaviorPolicy engine.

All tests are pure — no LLM calls, no DB, no Redis.
We test the deterministic rule matrix that policy.resolve() produces.
"""

import pytest
from policy import (
    BehaviorPolicy,
    ContextFeatures,
    extract_context_features,
    PolicyDecision,
)


POLICY = BehaviorPolicy()


# ─── Helpers ──────────────────────────────────────────────────────────────

def features(**kwargs) -> ContextFeatures:
    """Return a ContextFeatures with sensible defaults, overridden by kwargs."""
    defaults = dict(
        is_greeting=False,
        references_profile=False,
        privacy_signal=False,
        is_followup=False,
        is_profile_statement=False,
        is_profile_question=False,
        topic_similarity=None,
        has_profile_data=False,
        profile_name=None,
        conversation_length=0,
    )
    defaults.update(kwargs)
    return ContextFeatures(**defaults)


# ─── Intent: general ──────────────────────────────────────────────────────

class TestGeneralIntent:
    def test_no_injections(self):
        d = POLICY.resolve(features(), "general")
        assert not d.inject_profile
        assert not d.inject_rag
        assert not d.inject_qa_history
        assert d.retrieval_route == "llm_only"
        assert not d.use_curated_history

    def test_name_overlay_without_profile_data(self):
        """Name in profile but inject_profile stays False → lightweight greeting_name used."""
        d = POLICY.resolve(features(has_profile_data=True, profile_name="Alice"), "general")
        assert not d.inject_profile
        assert d.greeting_name == "Alice"

    def test_personal_ref_triggers_profile_injection(self):
        d = POLICY.resolve(
            features(references_profile=True, has_profile_data=True, profile_name="Bob"),
            "general",
        )
        assert d.inject_profile
        # Full profile injected → no need for separate greeting_name frame
        assert d.greeting_name is None


# ─── Intent: knowledge_base ───────────────────────────────────────────────

class TestKnowledgeBaseIntent:
    def test_rag_and_qa_injected(self):
        d = POLICY.resolve(features(), "knowledge_base")
        assert d.inject_rag
        assert d.inject_qa_history
        assert d.retrieval_route == "rag"

    def test_greeting_name_overlay(self):
        d = POLICY.resolve(
            features(has_profile_data=True, profile_name="Carol"),
            "knowledge_base",
        )
        assert d.greeting_name == "Carol"
        assert d.inject_rag
        assert not d.inject_profile


# ─── Intent: continuation ─────────────────────────────────────────────────

class TestContinuationIntent:
    def test_qa_injected(self):
        d = POLICY.resolve(features(is_followup=True), "continuation")
        assert d.inject_qa_history
        assert d.retrieval_route == "conversation"
        assert not d.inject_rag

    def test_no_inject_rag(self):
        d = POLICY.resolve(features(), "continuation")
        assert not d.inject_rag


# ─── Intent: profile ──────────────────────────────────────────────────────

class TestProfileIntent:
    def test_profile_statement_no_inject(self):
        d = POLICY.resolve(features(is_profile_statement=True), "profile")
        assert d.retrieval_route == "profile_update"
        assert not d.inject_profile
        assert not d.use_curated_history

    def test_profile_question_injects_profile(self):
        d = POLICY.resolve(
            features(is_profile_question=True, has_profile_data=True),
            "profile",
        )
        assert d.inject_profile
        assert d.retrieval_route == "profile"


# ─── Intent: privacy ──────────────────────────────────────────────────────

class TestPrivacyIntent:
    def test_privacy_mode_set(self):
        d = POLICY.resolve(features(privacy_signal=True), "privacy")
        assert d.privacy_mode
        assert d.retrieval_route == "privacy"
        assert not d.use_curated_history

    def test_profile_injected_when_data_exists(self):
        d = POLICY.resolve(
            features(privacy_signal=True, has_profile_data=True),
            "privacy",
        )
        assert d.inject_profile

    def test_personal_ref_overlay_skipped_for_privacy(self):
        """Privacy intent already injects profile; personal-ref overlay must not re-set greeting_name."""
        d = POLICY.resolve(
            features(privacy_signal=True, references_profile=True,
                     has_profile_data=True, profile_name="Dave"),
            "privacy",
        )
        # inject_profile should be True, but not through the overlay logic
        assert d.inject_profile
        # greeting_name should remain None (privacy path doesn't set it)
        assert d.greeting_name is None


# ─── extract_context_features ─────────────────────────────────────────────

class TestExtractContextFeatures:
    def test_greeting_detection(self):
        cf = extract_context_features("hello", "general")
        assert cf.is_greeting

    def test_not_greeting_when_long(self):
        cf = extract_context_features(
            "hello how are you doing this fine afternoon today", "general"
        )
        assert not cf.is_greeting

    def test_profile_name_extracted(self):
        profile = [{"key": "name", "value": "Eve", "category": "personal"}]
        cf = extract_context_features("hi", "general", profile_entries=profile)
        assert cf.profile_name == "Eve"

    def test_no_profile_name_key(self):
        profile = [{"key": "occupation", "value": "engineer", "category": "professional"}]
        cf = extract_context_features("hi", "general", profile_entries=profile)
        assert cf.profile_name is None

    def test_has_profile_data_false_when_empty(self):
        cf = extract_context_features("hello", "general", profile_entries=[])
        assert not cf.has_profile_data

    def test_privacy_signal_from_intent(self):
        cf = extract_context_features("do you store my data", "privacy")
        assert cf.privacy_signal

    def test_personal_ref_signal(self):
        cf = extract_context_features("what is my job", "general")
        assert cf.references_profile
