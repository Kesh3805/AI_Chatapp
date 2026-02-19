"""Profile-update detection — extract personal info from user messages."""

import json
import logging

from .client import completion, MAX_PROFILE_DETECT_TOKENS
from .prompts import PROFILE_DETECT_PROMPT

logger = logging.getLogger(__name__)

# Pre-check signals — skip LLM call if message is unlikely to contain
# personal information.
PERSONAL_SIGNALS = [
    "my name", "i am", "i'm", "i prefer", "i like", "i use",
    "i work", "i live", "call me", "remember that i", "i'm a",
    "my job", "my role", "my favorite", "my preferred",
    "i weigh", "my weight", "i'm ", "i am ", "i stand",
    "my height", "i measure", "i speak", "my age", "i'm from",
    "i studied", "i graduated", "i have", "my degree",
    "i code", "i built", "i've been", "years of experience",
]


def detect_profile_updates(user_message: str, assistant_response: str) -> list:
    """Detect explicit personal info worth storing in the user profile.

    Returns a list of ``{"key": str, "value": str, "category": str}``
    dicts, or an empty list if nothing is found.
    """
    try:
        lower = user_message.lower()
        if not any(sig in lower for sig in PERSONAL_SIGNALS):
            return []

        raw = completion(
            messages=[
                {"role": "system", "content": PROFILE_DETECT_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"User said: {user_message}\n\n"
                        f"Assistant responded: {assistant_response[:300]}"
                    ),
                },
            ],
            temperature=0.0,
            max_tokens=MAX_PROFILE_DETECT_TOKENS,
        ).strip()

        # Strip markdown fences if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        entries = json.loads(raw)
        if not isinstance(entries, list):
            return []

        valid = []
        for e in entries:
            if isinstance(e, dict) and "key" in e and "value" in e:
                valid.append({
                    "key": str(e["key"]).strip().lower().replace(" ", "_"),
                    "value": str(e["value"]).strip(),
                    "category": str(e.get("category", "general")).strip(),
                })

        if valid:
            logger.info("Profile updates: %s", [e["key"] for e in valid])
        return valid

    except (json.JSONDecodeError, Exception) as e:
        logger.debug("No profile updates: %s", e)
        return []
