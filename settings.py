"""Centralized configuration — every tunable in one place.

Environment variables override defaults. Import anywhere:

    from settings import settings

All values are frozen at startup. To change, update .env and restart.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────

def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int = 0) -> int:
    return int(os.getenv(key, str(default)))


def _env_float(key: str, default: float = 0.0) -> float:
    return float(os.getenv(key, str(default)))


def _env_bool(key: str, default: bool = False) -> bool:
    return os.getenv(key, str(default)).lower() in ("true", "1", "yes")


# ── Settings ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Settings:
    """Application settings.  Immutable after creation."""

    # ── LLM Provider ──────────────────────────────────────────────
    # Supported: cerebras, openai, anthropic
    LLM_PROVIDER: str = _env("LLM_PROVIDER", "cerebras")
    LLM_API_KEY: str = _env("LLM_API_KEY", _env("CEREBRAS_API_KEY"))
    LLM_MODEL: str = _env("LLM_MODEL", _env("CEREBRAS_MODEL"))
    # Empty LLM_MODEL → each provider picks its own default.
    LLM_BASE_URL: str = _env("LLM_BASE_URL")
    # Optional: override API endpoint (useful for Azure OpenAI, vLLM, etc.)

    # ── Token Budgets ─────────────────────────────────────────────
    MAX_RESPONSE_TOKENS: int = _env_int("MAX_RESPONSE_TOKENS", 2048)
    MAX_CLASSIFIER_TOKENS: int = 50
    MAX_PROFILE_DETECT_TOKENS: int = 300
    MAX_TITLE_TOKENS: int = 20
    MAX_CONTEXT_WINDOW: int = _env_int("MAX_CONTEXT_WINDOW", 65536)

    # ── Embeddings ────────────────────────────────────────────────
    # BAAI/bge-base-en-v1.5: 768-dim, top MTEB ranking, still runs locally.
    # Faster/lighter: BAAI/bge-small-en-v1.5 (384-dim, ~133 MB)
    # Highest quality: BAAI/bge-large-en-v1.5 (1024-dim, ~1.3 GB)
    # Symmetric (no prefix needed): sentence-transformers/all-mpnet-base-v2 (768-dim)
    EMBEDDING_MODEL: str = _env("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_DIMENSION: int = _env_int("EMBEDDING_DIMENSION", 768)
    # Optional query prefix for asymmetric retrieval models (bge, e5, nomic).
    # Leave empty for symmetric models (all-mpnet-base-v2, all-MiniLM*).
    QUERY_INSTRUCTION: str = _env("QUERY_INSTRUCTION", "")

    # ── Retrieval ─────────────────────────────────────────────────
    RETRIEVAL_K: int = _env_int("RETRIEVAL_K", 4)
    QA_K: int = _env_int("QA_K", 4)
    QA_MIN_SIMILARITY: float = _env_float("QA_MIN_SIMILARITY", 0.65)

    # ── Context Window ─────────────────────────────────────────────
    # Token budget reserved exclusively for conversation history.
    # System prompts + RAG docs + profile use the remaining context window.
    MAX_HISTORY_TOKENS: int = _env_int("MAX_HISTORY_TOKENS", 8000)
    # When True, old turns beyond MAX_HISTORY_TOKENS are summarized by the LLM
    # instead of silently dropped.  Costs one extra LLM call.
    ENABLE_HISTORY_SUMMARIZATION: bool = _env_bool("ENABLE_HISTORY_SUMMARIZATION", True)
    # How many recent chat messages to load from DB per request.
    # Higher values give the summarizer more material to work from;
    # the token budget (MAX_HISTORY_TOKENS) keeps the LLM context bounded.
    HISTORY_FETCH_LIMIT: int = _env_int("HISTORY_FETCH_LIMIT", 100)

    # ── Pipeline ──────────────────────────────────────────────────
    TOPIC_CONTINUATION_THRESHOLD: float = _env_float(
        "TOPIC_CONTINUATION_THRESHOLD", 0.35
    )
    TOPIC_DECAY_ALPHA: float = _env_float("TOPIC_DECAY_ALPHA", 0.2)
    RECENCY_WINDOW: int = _env_int("RECENCY_WINDOW", 6)
    SEMANTIC_K: int = _env_int("SEMANTIC_K", 3)
    SIMILARITY_THRESHOLD: float = _env_float("SIMILARITY_THRESHOLD", 0.65)

    # ── Knowledge Base ────────────────────────────────────────────
    KNOWLEDGE_DIR: str = _env("KNOWLEDGE_DIR", "knowledge")
    CHUNK_SIZE: int = _env_int("CHUNK_SIZE", 500)
    CHUNK_OVERLAP: int = _env_int("CHUNK_OVERLAP", 50)
    FORCE_REINDEX: bool = _env_bool("FORCE_REINDEX", False)

    # ── Database (PostgreSQL + pgvector) ──────────────────────────
    DATABASE_URL: str = _env("DATABASE_URL")
    POSTGRES_HOST: str = _env("POSTGRES_HOST", "localhost")
    POSTGRES_PORT: int = _env_int("POSTGRES_PORT", 55432)
    POSTGRES_DB: str = _env("POSTGRES_DB", "chatapp")
    POSTGRES_USER: str = _env("POSTGRES_USER", "root")
    POSTGRES_PASSWORD: str = _env("POSTGRES_PASSWORD", "password")
    DB_POOL_MIN: int = _env_int("DB_POOL_MIN", 1)
    DB_POOL_MAX: int = _env_int("DB_POOL_MAX", 10)

    # ── Cache (Optional Redis) ────────────────────────────────────
    ENABLE_CACHE: bool = _env_bool("ENABLE_CACHE", False)
    REDIS_URL: str = _env("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL: int = _env_int("CACHE_TTL", 3600)
    # ── Security ────────────────────────────────────────────────
    # Comma-separated origins allowed by CORS middleware.
    # Use "*" for local dev only — always restrict in production.
    # Example: ALLOWED_ORIGINS=https://app.example.com,https://admin.example.com
    ALLOWED_ORIGINS: str = _env("ALLOWED_ORIGINS", "*")
    # Default user identity when no user_id is included in a ChatRequest.
    DEFAULT_USER_ID: str = _env("DEFAULT_USER_ID", "public")
    # ── Server ────────────────────────────────────────────────────
    HOST: str = _env("HOST", "0.0.0.0")
    PORT: int = _env_int("PORT", 8000)
    DEBUG_MODE: bool = _env_bool("DEBUG_MODE", False)
    STAGE_STREAMING: bool = _env_bool("STAGE_STREAMING", True)


settings = Settings()
