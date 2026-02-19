# RAG Chat — Framework Architecture

> **Stop building naive RAG.  Start building policy-aware AI systems.**

An opinionated, extensible conversational AI starter framework.  Intent-driven, policy-gated, observable by default.

## What Makes This Different

Most RAG templates do this: dump docs into a vector DB, call the LLM with context.

This framework does this: **classify intent first, then selectively retrieve only what's needed, governed by deterministic policy rules, with full observability in the UI.**

Three differentiators:

1. **Intent → Policy → Orchestrator separation** — Rare, powerful, your signature
2. **Stage streaming protocol** — Real-time pipeline observability in the browser
3. **AI-native UI primitives** — Debug mode, intent badges, retrieval panels out of the box

---

## Quick Start

```bash
# 1. Clone and init
git clone <repo> && cd rag-chat
python cli.py init            # Creates knowledge/, copies .env

# 2. Add your API key
# Edit .env → LLM_API_KEY=your-key-here

# 3. Start PostgreSQL
docker compose up postgres -d

# 4. Add knowledge (optional — ships with example)
# Drop .txt/.md files into knowledge/
python cli.py ingest

# 5. Run
python cli.py dev             # → http://localhost:8000
```

Or with Docker: `docker compose up --build`

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   React Frontend (frontend/)                     │
│  AI-native components · Debug mode · Command palette · Streaming │
│  Vite + Tailwind + Vercel AI SDK (useChat) + Zustand             │
└──────────────────────┬───────────────────────────────────────────┘
                       │  HTTP / SSE (Vercel AI SDK data stream)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                    FastAPI  (main.py)  v4.0.0                    │
│                                                                  │
│  POST /chat/stream   ──► run_pipeline() ──► hooks ──► generate   │
│  POST /chat          │                                           │
│  GET /health         │   Conversations + Profile CRUD            │
│  GET /               │   Serve React or fallback HTML            │
└─────────┬───────────┴────────┬────────┬──────────────────────────┘
          │                    │        │
  ┌───────▼───────┐    ┌──────▼──┐  ┌──▼──────────┐
  │  llm/ package │    │query_db │  │vector_store  │
  │  ├ providers/ │    │PostgreSQL│  │  pgvector    │
  │  │ ├ base    │    │ pgvector │  │ (persistent) │
  │  │ ├ cerebras│    └──────────┘  └──────────────┘
  │  │ ├ openai  │
  │  │ └ anthrop.│    ┌──────────┐  ┌──────────────┐
  │  ├ client    │    │policy.py │  │ settings.py  │
  │  ├ classifier│    │ Behavior │  │  All config   │
  │  ├ orchestr. │    │  Policy  │  └──────────────┘
  │  ├ generators│    └──────────┘
  │  └ profiler  │                  ┌──────────────┐
  └──────────────┘                  │  hooks.py    │
                                    │  Extensions  │
                                    └──────────────┘
```

---

## File Map

```
settings.py              ← Every tunable in one place
hooks.py                 ← Extension points (before/after generation, policy override)
cache.py                 ← Optional Redis (no-op when disabled)
worker.py                ← Background task runner (thread-based)
cli.py                   ← CLI: init, ingest, dev

main.py                  ← FastAPI app + pipeline + endpoints
policy.py                ← BehaviorPolicy engine (deterministic rules)
query_db.py              ← PostgreSQL + pgvector (all persistence)
vector_store.py          ← Document search (pgvector-backed, in-memory fallback)
embeddings.py            ← Local sentence-transformer model

llm/
  providers/
    base.py              ← LLMProvider ABC (2 methods: complete, stream_text_deltas)
    cerebras.py          ← Cerebras Cloud SDK
    openai.py            ← OpenAI (also Azure, vLLM, Ollama via base_url)
    anthropic.py         ← Anthropic Messages API
    __init__.py          ← Dynamic loader (reads LLM_PROVIDER from settings)
  client.py              ← Thin wrapper: delegates to active provider
  classifier.py          ← Intent classification (pre-heuristics + LLM)
  prompts.py             ← All prompt templates (single source of truth)
  prompt_orchestrator.py ← Builds message lists from PolicyDecision
  generators.py          ← Response generation (stream + batch + titles)
  profile_detector.py    ← Extract personal info from messages

knowledge/               ← Drop .txt/.md files here → auto-indexed
  example.txt            ← Shipped example knowledge base

frontend/                ← React 18 + Vite + Tailwind + AI SDK + Zustand
  src/components/ai/     ← AI-native observability primitives

.env.example             ← Template with all settings documented
docker-compose.yml       ← PostgreSQL + optional Redis + app
```

---

## Core Concepts

### 1. Intent Classification

Every message is classified into one of five intents **before** any retrieval:

| Intent | What Happens | LLM Receives |
|---|---|---|
| `general` | No retrieval | System prompt + query only |
| `knowledge_base` | pgvector doc search + cross-conv Q&A | Docs + Q&A + history + query |
| `continuation` | Same-conversation Q&A | Curated history + Q&A + query |
| `profile` (statement) | Nothing (saved in background) | System prompt + query |
| `profile` (question) | User profile data | Profile + query |
| `privacy` | Profile + transparency rules | Profile + privacy frame + query |

**Pre-heuristics** skip the LLM call entirely for common patterns:
- Greetings ("hi", "hello") → `general` at 0.97 confidence
- Profile statements ("My name is…") → `profile` at 0.92
- Privacy phrases ("do you store my data") → `privacy` at 0.95
- Short pronoun follow-ups ("what about that?") → `continuation` at 0.85

### 2. BehaviorPolicy Engine

`policy.py` contains **deterministic rules** that decide what context to inject:

```python
features = extract_context_features(query, intent, profile_entries, ...)
decision = BehaviorPolicy().resolve(features, intent)
decision = Hooks.run_policy_override(features, decision)  # ← your code here
```

The `PolicyDecision` dataclass controls everything:
- `inject_rag` — fetch documents from pgvector
- `inject_profile` — include user profile
- `inject_qa_history` — include prior Q&A
- `privacy_mode` — activate transparency rules
- `greeting_name` — personalize with user's name
- `retrieval_route` — label for observability

**Why this matters:** Rules are editable without touching prompts or LLM code.

### 3. Pluggable LLM Providers

```env
LLM_PROVIDER=cerebras      # or: openai, anthropic
LLM_API_KEY=your-key
LLM_MODEL=                  # empty = provider default
LLM_BASE_URL=               # optional: vLLM, Azure, Ollama
```

Every provider implements `LLMProvider` (2 methods):
- `complete(messages) → str`
- `stream_text_deltas(messages) → Generator[str]`

Add a new provider: subclass `llm/providers/base.py`, register in `__init__.py`.

### 4. Stage Streaming Protocol

Vercel AI SDK data stream format over SSE:

```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4}}]
8:[{"stage":"generating"}]
0:"The"                    ← text delta
0:" answer"
0:" is..."
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...}}]
e:{"finishReason":"stop"}
d:{"finishReason":"stop"}
```

The frontend renders these as a real-time pipeline timeline.

### 5. Extension Hooks

Register hooks with decorators — no core code changes needed:

```python
from hooks import Hooks

@Hooks.before_generation
def add_custom_context(pipeline_result):
    pipeline_result.rag_context += "\nCustom: extra context"
    return pipeline_result

@Hooks.after_generation
def filter_response(response, pipeline_result):
    return response.replace("sensitive_word", "***")

@Hooks.policy_override
def force_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision
```

### 6. Centralized Configuration

Everything tunable lives in `settings.py`, driven by env vars:

```python
from settings import settings

settings.RETRIEVAL_K              # 4  (documents per search)
settings.TOPIC_CONTINUATION_THRESHOLD  # 0.35
settings.CHUNK_SIZE               # 500
settings.ENABLE_CACHE             # False (set ENABLE_CACHE=true for Redis)
settings.LLM_PROVIDER             # "cerebras"
settings.MAX_CONTEXT_WINDOW       # 65536
```

No hunting through scattered constants across files.

### 7. Single Database Layer

PostgreSQL + pgvector handles **everything**:
- Conversations & messages
- User profile (key-value)
- Query embeddings (semantic Q&A search)
- Document chunks (knowledge base vectors)
- Topic vectors (conversation continuity)

No FAISS. No separate vector DB. One database, one backup, one deployment.

Tables: `conversations`, `chat_messages`, `user_queries`, `user_profile`, `document_chunks`

### 8. Optional Redis Cache

Set `ENABLE_CACHE=true` + `REDIS_URL` → caches intent classifications and embeddings.
When Redis is unavailable: **everything still works** — graceful no-op degradation.

---

## Pipeline Flow

```
User message arrives
       │
       ▼
  ┌─ PARALLEL ─────────────────────────────────┐
  │  1. Embed query (sentence-transformers)     │
  │  2. Load history (PostgreSQL)               │
  │  3. Load profile (PostgreSQL)               │
  └─────────────────────────────────────────────┘
       │
       ▼
  4. Classify intent (pre-heuristics → LLM fallback → cache)
       │
       ▼
  5. Topic gate (continuation: cosine sim < threshold → general)
       │
       ▼
  6. Extract features + BehaviorPolicy.resolve() + Hooks.policy_override()
       │
       ▼
  7. History pruning (recency window + semantic retrieval)
       │
       ▼
  8. Selective retrieval (only what PolicyDecision says)
       │
       ▼
  9. Hooks.before_generation()
       │
       ▼
  10. Generate response (stream or batch via current LLM provider)
       │
       ▼
  11. Hooks.after_generation() → Hooks.before_persist()
       │
       ▼
  12. Background persist (worker.submit → DB writes, profile detection, auto-title)
```

---

## Swapping Components

| Want to... | Do this |
|---|---|
| Change LLM | Set `LLM_PROVIDER` + `LLM_API_KEY` in `.env` |
| Add LLM provider | Subclass `llm/providers/base.py`, register in `__init__.py` |
| Change knowledge base | Drop files in `knowledge/`, run `python cli.py ingest` |
| Modify behavior rules | Edit `policy.py` → `BehaviorPolicy.resolve()` |
| Add custom logic | Use decorators in `hooks.py` |
| Change retrieval depth | Set `RETRIEVAL_K`, `QA_K` in `.env` |
| Change embedding model | Set `EMBEDDING_MODEL` in `.env` |
| Add caching | Set `ENABLE_CACHE=true`, `REDIS_URL` in `.env` |
| Remove features | Delete the component — nothing is tightly coupled |

---

## Frontend: AI-Native UI

```
frontend/src/components/ai/
  AIIntentBadge.tsx    → Color-coded intent + confidence dot
  AIStatusBar.tsx      → Horizontal pipeline stage timeline
  AIRetrievalPanel.tsx → Expandable retrieval breakdown
  AITokenMeter.tsx     → Context window usage bar
  AIDebugPanel.tsx     → Raw PolicyDecision JSON (Debug Mode only)
```

Key features:
- **Debug Mode** — Toggle from sidebar or header to see raw system decisions
- **Command Palette** (Ctrl+K) — Quick actions with fuzzy search
- **Streaming phases** — "Classifying..." → "Retrieving..." → "Generating..."
- **Sidebar intelligence** — Category icons from conversation titles

**Development:** `cd frontend && npm install && npm run dev`
**Production:** `npm run build` → served by FastAPI

---

## Stack

| Layer | Technology |
|---|---|
| LLM | Pluggable: Cerebras, OpenAI, Anthropic (or any via `base_url`) |
| Embeddings | `BAAI/bge-base-en-v1.5` (768-dim, local, asymmetric retrieval, no API key) |
| Vector search | PostgreSQL + pgvector (persistent, HNSW index) |
| Database | PostgreSQL 16 + pgvector (single DB for everything) |
| Backend | FastAPI + Uvicorn (Python 3.12) |
| Frontend | React 18 + Vite + Tailwind CSS + Vercel AI SDK + Zustand |
| AI UI | AIMessage, AIStatusBar, AIIntentBadge, AIRetrievalPanel, AIDebugPanel |
| Streaming | Vercel AI SDK data stream protocol over SSE (with stage events) |
| Cache | Optional Redis (graceful degradation) |
| Background | Thread-based worker (zero-dependency) |
| Context Mgmt | `context_manager.py` (token budgeting, history trimming, LLM summarization) |
| Config | `settings.py` (env-driven, single source of truth) |
| Extensions | `hooks.py` (4 decorator-based extension points) |
| CLI | `python cli.py init/ingest/dev` |
| Deploy | Docker Compose (single command startup) |

---

## Design Principles

1. **Opinionated defaults** — Works out of the box. Smart choices so users don't have to.
2. **Swappable components** — LLM, knowledge base, behavior rules — all replaceable.
3. **Minimal cognitive load** — Config in one file, hooks in one file, prompts in one file.
4. **Production-safe** — Connection pooling, graceful degradation, background persistence.
5. **Extensible without forking** — Hooks let you customize without touching core.
6. **Observable** — Every pipeline decision is visible in the UI's Debug Mode.
7. **One database** — PostgreSQL + pgvector for vectors, messages, profiles, documents.
