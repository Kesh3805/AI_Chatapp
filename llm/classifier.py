"""Intent classification — pre-heuristics + LLM call.

Pre-heuristic fast-paths (no LLM round-trip):
  greeting          → general   (saves ~1-2s per greeting message)
  profile statement → profile   (saves ~1-2s for "my name is …" etc.)
  privacy phrase    → privacy
  short pronoun q.  → continuation  (needs context)
"""

import json
import logging
from typing import Optional

import cache
from .client import completion, MAX_CLASSIFIER_TOKENS
from .prompts import INTENT_PROMPT

logger = logging.getLogger(__name__)

# ── Pre-heuristic signal lists ────────────────────────────────────────────

# Greetings — map immediately to "general" without hitting the LLM
_GREETING_PATTERNS = [
    "hello", "hi", "hey", "good morning", "good afternoon",
    "good evening", "good night", "howdy", "sup", "what's up",
    "whats up", "yo", "hola", "greetings", "hi there",
    "hey there", "hello there",
]

# Profile statement openers — map to "profile" without hitting the LLM
_PROFILE_OPENERS = (
    "my name is ", "i am ", "i'm ", "i have ", "i like ", "i prefer ",
    "i use ", "i work ", "i live ", "call me ", "remember that ",
    "i am a ", "i'm a ", "i speak ", "i study ", "i graduated ",
    "i code ", "i built ", "i am from ",
)

PRIVACY_SIGNALS = [
    "invasion of privacy", "privacy concern", "what do you know about me",
    "what data do you", "what information do you", "do you store",
    "do you track", "are you tracking", "delete my data", "erase my",
    "forget about me", "what have you stored", "my personal data",
    "data privacy", "how is my data", "who has access", "is my data safe",
    "stop storing", "do you collect", "what do you remember about me",
    "stored about me", "data about me", "what info do you have",
]

CONTINUATION_PRONOUNS = {
    "that", "it", "this", "those", "these", "they", "them",
    "its", "their", "the same",
}

VALID_INTENTS = {"general", "continuation", "knowledge_base", "profile", "privacy"}


def classify_intent(
    user_query: str,
    conversation_context: Optional[list] = None,
) -> dict:
    """Classify user intent using pre-heuristics then LLM fallback.

    Returns ``{"intent": str, "confidence": float}``.
    """
    query_lower = user_query.strip().lower()

    # ── Cache check ─────────────────────────────────────────────────────
    cached = cache.get_classification(user_query)
    if cached:
        logger.info(f"Cache hit: intent={cached['intent']}")
        return cached

    # ── Greeting fast-path (saves a full LLM round-trip) ─────────────────
    _words = query_lower.split()
    if len(_words) <= 8:
        for _pat in _GREETING_PATTERNS:
            _rest = query_lower[len(_pat):]
            if query_lower == _pat or (
                query_lower.startswith(_pat) and (_rest == "" or _rest[0] in " ,!?")
            ):
                logger.info("Pre-heuristic: greeting → general (no LLM)")
                return {"intent": "general", "confidence": 0.97}

    # ── Profile statement fast-path ───────────────────────────────────────
    if "?" not in user_query and any(query_lower.startswith(p) for p in _PROFILE_OPENERS):
        logger.info("Pre-heuristic: profile statement → profile (no LLM)")
        return {"intent": "profile", "confidence": 0.92}

    # ── Fast privacy check ────────────────────────────────────────────────
    if any(sig in query_lower for sig in PRIVACY_SIGNALS):
        logger.info("Pre-heuristic: privacy signal detected")
        return {"intent": "privacy", "confidence": 0.95}

    # ── Continuation check (needs context) ────────────────────────────────
    if conversation_context and len(conversation_context) >= 2:
        words = set(query_lower.split())
        if (
            len(words) <= 8
            and words & CONTINUATION_PRONOUNS
            and "?" in user_query
        ):
            logger.info("Pre-heuristic: continuation (pronoun + short question)")
            return {"intent": "continuation", "confidence": 0.85}

    # ── LLM classification ────────────────────────────────────────────────
    try:
        context_text = ""
        if conversation_context:
            recent = conversation_context[-6:]
            lines = [f"{m['role'].title()}: {m['content'][:120]}" for m in recent]
            context_text = "\nRecent conversation:\n" + "\n".join(lines)

        messages = [
            {"role": "system", "content": INTENT_PROMPT},
            {"role": "user", "content": f"Message: {user_query}{context_text}"},
        ]

        raw = completion(
            messages,
            temperature=0.0,
            max_tokens=MAX_CLASSIFIER_TOKENS,
        ).strip()

        # Robust JSON extraction
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start >= 0 and end > start:
            raw = raw[start:end]

        result = json.loads(raw)
        intent = result.get("intent", "general")
        confidence = float(result.get("confidence", 0.5))

        if intent not in VALID_INTENTS:
            logger.warning(f"Unknown intent '{intent}' → general")
            intent = "general"

        result = {"intent": intent, "confidence": confidence}
        cache.set_classification(user_query, result)
        return result

    except Exception as e:
        logger.error(f"Classification error: {e}")
        return {"intent": "general", "confidence": 0.5}
