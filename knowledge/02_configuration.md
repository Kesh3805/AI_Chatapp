# Configuration Reference — settings.py

## Overview

All configuration lives in settings.py as a frozen dataclass. Import it from anywhere:
`from settings import settings`

Settings are read from environment variables at startup via python-dotenv. All values are immutable after startup. To change a setting, update .env and restart. No defaults are scattered across files — one file, one source of truth.

## LLM Provider Settings

**LLM_PROVIDER** (default: "cerebras") — Which LLM provider to use. Supported values: "cerebras", "openai", "anthropic". The provider module is lazily imported, so only the selected provider's SDK needs to be installed.

**LLM_API_KEY** (default: "") — API key for the selected provider. Also checks CEREBRAS_API_KEY as a legacy fallback for backward compatibility with older .env files.

**LLM_MODEL** (default: "") — Model name override. When empty, each provider uses its own default model. Cerebras default: gpt-oss-120b. OpenAI default: gpt-4o. Anthropic default: claude-sonnet-4-20250514.

**LLM_BASE_URL** (default: "") — Optional API endpoint override. Useful for Azure OpenAI, vLLM, Ollama, or any OpenAI-compatible server. Only used by the OpenAI provider.

## Token Budget Settings

**MAX_RESPONSE_TOKENS** (default: 2048) — Maximum tokens in a generated response. Passed to the LLM as max_tokens.

**MAX_CLASSIFIER_TOKENS** (hardcoded: 50) — Maximum tokens for the intent classification LLM call. Classification returns a short JSON object so 50 is always sufficient.

**MAX_PROFILE_DETECT_TOKENS** (hardcoded: 300) — Maximum tokens for the profile detection LLM call. Returns a JSON array so 300 is sufficient.

**MAX_TITLE_TOKENS** (hardcoded: 20) — Maximum tokens for auto-generating conversation titles. Titles are 3-6 words so 20 is always sufficient.

**MAX_CONTEXT_WINDOW** (default: 65536) — Total context window size for the model. Used by the frontend AITokenMeter component to display context usage percentage.

**MAX_HISTORY_TOKENS** (default: 8000) — Token budget reserved for conversation history. History exceeding this limit is trimmed from the oldest end (or summarized when ENABLE_HISTORY_SUMMARIZATION=True) before being passed to the LLM. Enforced in prompt_orchestrator.py via context_manager.fit_messages_to_budget().

**ENABLE_HISTORY_SUMMARIZATION** (default: False) — When True, overflow turns are compressed into an LLM-generated system message instead of being silently dropped. Costs one extra LLM call per overflowing request. Set to True for long-session applications where conversation history is important.

## Embedding Settings

**EMBEDDING_MODEL** (default: "BAAI/bge-base-en-v1.5") — The sentence-transformers model for generating embeddings. Runs entirely locally; no API key required. Downloaded on first startup (~440 MB) and cached. Alternatives: BAAI/bge-small-en-v1.5 (384-dim, 133 MB, faster), BAAI/bge-large-en-v1.5 (1024-dim, 1.3 GB, highest quality), sentence-transformers/all-mpnet-base-v2 (768-dim, 420 MB, no prefix needed).

**EMBEDDING_DIMENSION** (default: 768) — Vector dimension. Must match the model. All pgvector columns use vector(EMBEDDING_DIMENSION). Changing this on an existing database requires dropping and recreating the vector columns (CREATE TABLE IF NOT EXISTS won't change existing column types).

**QUERY_INSTRUCTION** (default: "") — Optional prefix applied to queries at retrieval time via get_query_embedding(). Leave empty for bge-v1.5 (works without prefix). For maximum recall, set to "Represent this sentence for searching relevant passages: ". Documents are always encoded without any prefix.

## Retrieval Settings

**RETRIEVAL_K** (default: 4) — Number of document chunks to retrieve from pgvector for knowledge_base queries. Higher values inject more context but increase token usage.

**QA_K** (default: 4) — Number of similar past Q&A pairs to retrieve from user_queries table. Used for cross-conversation continuity on knowledge_base and continuation intents.

**QA_MIN_SIMILARITY** (default: 0.65) — Minimum cosine similarity score for a Q&A pair to be included in QA context. Pairs below this threshold are excluded. Range: 0.0-1.0.

## Pipeline Thresholds

**TOPIC_CONTINUATION_THRESHOLD** (default: 0.35) — Minimum cosine similarity required to keep the "continuation" intent. If the query embedding is less similar than this to the conversation's rolling topic vector, the intent is downgraded to "general". Lower values are more permissive (allow more topic drift while still treating as continuation).

**TOPIC_DECAY_ALPHA** (default: 0.2) — Exponential decay rate for the rolling topic vector update. At 0.2, new messages have 20% influence on the topic direction, older messages have 80% influence. Higher alpha makes the topic vector respond faster to topic changes.

**RECENCY_WINDOW** (default: 6) — Number of most recent messages to always include in curated history. Combined with semantic history to form the full context window for continuation and knowledge_base intents.

**SEMANTIC_K** (default: 3) — Maximum number of semantically similar older messages to add to curated history beyond the recency window. Only used when conversation has more messages than RECENCY_WINDOW.

**SIMILARITY_THRESHOLD** (default: 0.65) — Minimum similarity for semantic history retrieval. Messages below this threshold are excluded from semantic history expansion.

## Knowledge Base Settings

**KNOWLEDGE_DIR** (default: "knowledge") — Path to the directory containing knowledge base files. Supports .txt and .md files. All files in this directory are ingested on startup if the document_chunks table is empty, or when FORCE_REINDEX=true.

**CHUNK_SIZE** (default: 500) — Maximum character length of each document chunk. Chunks are created by sliding window over the text.

**CHUNK_OVERLAP** (default: 50) — Number of characters to overlap between consecutive chunks. A chunk at position i has characters [i, i+CHUNK_SIZE], and the next chunk starts at [i+CHUNK_SIZE-CHUNK_OVERLAP].

**FORCE_REINDEX** (default: false) — When true, clears and rebuilds the document_chunks table on every startup. Useful during development when knowledge base content changes frequently. Set to false in production.

## Database Settings

**DATABASE_URL** (default: "") — Full PostgreSQL connection string in format postgresql://user:password@host:port/database. When set, overrides all individual POSTGRES_* settings below.

**POSTGRES_HOST** (default: "localhost") — PostgreSQL host. Only used if DATABASE_URL is not set.

**POSTGRES_PORT** (default: 55432) — PostgreSQL port. Default is 55432 (not standard 5432) to avoid conflicts with system PostgreSQL installations.

**POSTGRES_DB** (default: "chatapp") — Database name.

**POSTGRES_USER** (default: "root") — Database user.

**POSTGRES_PASSWORD** (default: "password") — Database password.

**DB_POOL_MIN** (default: 1) — Minimum connections in the psycopg2 SimpleConnectionPool.

**DB_POOL_MAX** (default: 10) — Maximum connections in the SimpleConnectionPool.

## Cache Settings

**ENABLE_CACHE** (default: false) — Set to true to enable Redis caching for intent classifications and embeddings. When false, all cache operations are no-ops. Redis is completely optional.

**REDIS_URL** (default: "redis://localhost:6379/0") — Redis connection URL. Only used when ENABLE_CACHE=true.

**CACHE_TTL** (default: 3600) — Default cache TTL in seconds (1 hour). Intent classifications use a shorter 1800-second TTL (30 minutes) hardcoded in cache.py.

## Server Settings

**HOST** (default: "0.0.0.0") — Server bind address. Used by cli.py dev command.

**PORT** (default: 8000) — Server port. Used by cli.py dev command.

**DEBUG_MODE** (default: false) — Enable verbose logging. Does not affect the frontend's Debug Mode toggle (that is client-side state).

**STAGE_STREAMING** (default: true) — Whether to emit pipeline stage events (classified, retrieved, generating) before text tokens in the stream. Set to false to skip stage events and get pure token streaming.

## How Settings Are Loaded

The Settings dataclass is instantiated once as a module-level singleton: `settings = Settings()`. The dataclass is frozen (immutable), so no code can accidentally mutate settings at runtime. The dotenv load_dotenv() call at the top of settings.py reads .env before the dataclass is constructed, so environment variables in .env are available to the default value expressions.
