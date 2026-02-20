# RAG Chat

> **Policy-driven, intent-gated retrieval-augmented generation.**  
> A self-hosted chat platform that decides *what* to retrieve — and *whether* to retrieve at all — before spending a single token on context.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![pgvector](https://img.shields.io/badge/pgvector-PostgreSQL-336791.svg)](https://github.com/pgvector/pgvector)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Is This?

RAG Chat is a **production-ready, fully self-hosted conversational AI backend** built on FastAPI, PostgreSQL/pgvector, and pluggable LLM providers (Cerebras, OpenAI, Anthropic, Ollama).

Unlike naive RAG systems that blindly embed every query and dump retrieved chunks into the prompt, this system first **classifies the user’s intent**, then **resolves a policy decision** that determines exactly what — if anything — to retrieve. The result is a pipeline that is cheaper to run, less likely to hallucinate from irrelevant context, and fully observable.

### Core idea in one sentence

> *Classify first, retrieve only what the intent demands, generate with the minimum context needed.*

---

## How It Differs From Naive RAG

| Layer | Naive RAG | This System |
|---|---|---|
| **Retrieval trigger** | Every query | Only when policy says so |
| **Intent awareness** | None | 5 classified intents |
| **Context assembly** | Dump all chunks | Policy-resolved, per-intent |
| **History management** | Last N messages | Semantic pruning + token budget |
| **User isolation** | None / hardcoded | Per-request `user_id` |
| **Observability** | Black box | Pipeline stages exposed in UI |
| **Vector DB** | Separate service | PostgreSQL + pgvector (single DB) |

---

## Intent Gate — The Core Mechanic

Every message passes through a classifier before any retrieval occurs:

| Intent | What Gets Retrieved | Approx. Cost |
|---|---|---|
| `general` | Nothing — LLM only | Minimal |
| `continuation` | Curated history + semantic pruning | Low |
| `knowledge_base` | pgvector docs + cross-conversation Q&A | Full |
| `profile` | User profile entries (when a question) | Low |
| `privacy` | Profile + transparency rules | Low |

A **topic similarity gate** additionally prevents false `continuation` signals when the user switches domains mid-conversation — a silent failure mode in most RAG systems.

---

## Quick Start

```bash
git clone <repo> && cd rag-chat

# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — set LLM_API_KEY and POSTGRES_* vars

# 4. Start PostgreSQL
docker compose up postgres -d

# 5. Run
python cli.py dev               # http://localhost:8000
```

**One-command Docker start:**

```bash
docker compose up --build
```

---

## Project Structure

```
settings.py              # Every tunable in one place (env-driven)
main.py                  # FastAPI app + 12-step pipeline + all endpoints
policy.py                # BehaviorPolicy engine — deterministic rules
context_manager.py       # Token-budget history trimming + LLM summarization
query_db.py              # PostgreSQL + pgvector (conversations, messages,
                         #   profiles, query embeddings, document vectors)
vector_store.py          # Document search (pgvector + numpy fallback)
embeddings.py            # BAAI/bge-base-en-v1.5, 768-dim, asymmetric retrieval
hooks.py                 # Extension points (decorator-based)
cache.py                 # Optional Redis cache (no-op when disabled)
worker.py                # Background task runner
cli.py                   # Developer CLI: init / ingest / dev

llm/
  providers/
    base.py              # LLMProvider ABC
    cerebras.py          # Cerebras Cloud SDK
    openai.py            # OpenAI / Azure / vLLM / Ollama
    anthropic.py         # Anthropic Messages API
    __init__.py          # Dynamic provider loader
  client.py              # Active-provider wrapper
  classifier.py          # Intent classification (heuristics + LLM)
  prompts.py             # All prompt templates (single source of truth)
  prompt_orchestrator.py # Policy-aware message builder
  generators.py          # Response generation (streaming + batch)
  profile_detector.py    # Extract personal facts from conversation

knowledge/               # Drop .txt / .md files here — auto-indexed
frontend/                # React 18 + Vite + Tailwind + Vercel AI SDK
tests/                   # 117 tests across classifier, policy,
                         #   context manager, prompt orchestrator, chunker
```

---

## Configuration

All settings live in [`settings.py`](settings.py) and are overridden by environment variables.  
See [`.env.example`](.env.example) for the full reference.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `cerebras` | `cerebras` · `openai` · `anthropic` |
| `LLM_API_KEY` | — | **Required** — your provider API key |
| `LLM_MODEL` | *(provider default)* | Model name override |
| `LLM_BASE_URL` | — | Custom endpoint (Ollama, Azure, vLLM) |
| `DATABASE_URL` | — | Full PostgreSQL connection string |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | Local sentence-transformer model |
| `EMBEDDING_DIMENSION` | `768` | Must match model output size |
| `RETRIEVAL_K` | `4` | Document chunks per knowledge query |
| `CHUNK_SIZE` | `500` | Knowledge base chunk size (chars) |
| `MAX_HISTORY_TOKENS` | `8000` | Token budget for conversation history |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `DEFAULT_USER_ID` | `public` | Fallback user when none provided |
| `ENABLE_CACHE` | `false` | Redis caching for embeddings + classifications |
| `FORCE_REINDEX` | `false` | Re-index knowledge base on every startup |

---

## CLI

```bash
python cli.py init          # Scaffold project — creates knowledge/, copies .env
python cli.py ingest        # Index knowledge base into PostgreSQL
python cli.py ingest docs/  # Index from a custom directory
python cli.py dev           # Start dev server (uvicorn --reload)
```

---

## Swapping the LLM

```env
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Anthropic Claude
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...

# Ollama (fully local, no API key)
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

Add a new provider: subclass `LLMProvider` in [`llm/providers/base.py`](llm/providers/base.py).

---

## Extension Hooks

Customize behavior without touching core files:

```python
from hooks import Hooks

@Hooks.before_generation
def inject_date(pipeline_result):
    from datetime import date
    pipeline_result.rag_context += f"\nToday: {date.today()}"
    return pipeline_result

@Hooks.policy_override
def always_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision
```

Four hook points: `before_generation` · `after_generation` · `policy_override` · `before_persist`.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/stream` | Streaming chat (SSE — Vercel AI SDK compatible) |
| `POST` | `/chat` | Non-streaming chat |
| `GET` | `/health` | Status, DB connection, doc count, provider name |
| `GET/POST/PUT/DELETE` | `/conversations` | Conversation CRUD |
| `GET/POST/PUT/DELETE` | `/profile` | Per-user profile management |

---

## Frontend

React 18 + Vite + Tailwind UI with observable pipeline stages:

- **Intent badge** — classification + confidence score on every response
- **Pipeline timeline** — real-time stage chips: `Classified → Retrieved → Generating`
- **Retrieval panel** — expandable breakdown of documents and Q&A used
- **Debug mode** — raw `PolicyDecision` JSON per message
- **Command palette** — `Ctrl+K` quick navigation
- **Token meter** — context window usage visualization

```bash
cd frontend && npm install && npm run dev    # port 5173
npm run build                                # output to frontend/dist/
```

---

## Gaps vs ChatGPT

An honest comparison. The goal of this project is **privacy, control, and extensibility** — not feature parity with a $10B product.

### What ChatGPT Has That This Doesn’t (yet)

| Feature | Notes |
|---|---|
| **Multi-modal input** | Images, audio, video — not yet supported |
| **File upload + analysis** | PDF / CSV / spreadsheet parsing, in-context analysis |
| **Code interpreter** | Sandboxed Python execution with output rendering |
| **Web browsing / search** | Real-time retrieval from the live web |
| **Image generation** | DALL-E / image model integration |
| **Voice mode** | Real-time speech-to-speech |
| **Structured tool / function calling** | Schema-validated tool use and result handling |
| **Rich automatic memory** | ChatGPT Memory synthesizes facts at scale automatically |
| **Mobile native apps** | iOS / Android with push notifications |
| **Collaborative canvas** | Shared document co-editing with the model |
| **GPT Store / plugins** | Third-party capability marketplace |

### What This Has That ChatGPT Doesn’t

| Feature | Notes |
|---|---|
| **Full self-hosting** | Your data never leaves your infrastructure |
| **Custom knowledge base** | RAG over your own private documents |
| **Policy engine** | Deterministic, auditable retrieval rules — no prompt hacks |
| **Intent gating** | Skips retrieval entirely for non-KB queries — saves cost |
| **Provider portability** | Swap LLM in two env vars; run fully local via Ollama |
| **Per-user data isolation** | `user_id` threaded through every DB read and write |
| **Token budget management** | Automatic history trimming prevents silent context overflow |
| **Extension hooks** | Inject logic at 4 pipeline stages without forking |
| **Transparent pipeline** | UI exposes every routing decision the system makes |
| **Single-DB architecture** | No separate vector store, cache, or memory service needed |
| **Open source** | Fully auditable, forkable, and self-improvable |

### Realistic Roadmap to Close the Gaps

1. **File upload** — parse PDF/DOCX on upload with `pymupdf` or `unstructured`, chunk and index in-place  
2. **Tool calling** — add a `tools` field to `ChatRequest`, pass schema to LLM, handle tool-result turn in pipeline  
3. **Web search** — add a `web_search` retrieval route in the policy engine (Brave/Tavily API)  
4. **Richer memory** — periodic background job that LLM-summarizes profile entries into higher-level facts  
5. **Voice** — `whisper` for STT on the frontend; `openai.audio.speech` or `kokoro` for TTS  

---

## License

MIT
