# RAG Chat

**Policy-driven, intent-gated RAG framework.** Stop building naive RAG. Start building systems that decide *what* to retrieve before they retrieve it.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What Makes This Different

Every message is **classified** before **any** retrieval:

| Intent | What Gets Retrieved | Cost |
|---|---|---|
| `general` | Nothing  LLM only | Minimal |
| `continuation` | Curated history + semantic pruning | Low |
| `knowledge_base` | pgvector docs + cross-conversation Q&A | Full |
| `profile` | User profile data (if a question) | Low |
| `privacy` | Profile + transparency rules | Low |

**No FAISS. No separate vector DB.** PostgreSQL + pgvector is the single database for conversations, messages, profiles, query embeddings, and document vectors.

**Pluggable LLM providers.** Switch Cerebras  OpenAI  Anthropic  Ollama by changing two env vars.

---

## Quick Start

`ash
git clone <repo> && cd rag-chat

# 1. Install dependencies
python -m venv .venv && .venv\Scripts\activate       # Windows
# source .venv/bin/activate                           # macOS/Linux
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: set LLM_API_KEY=your-key-here

# 3. Start PostgreSQL
docker compose up postgres -d

# 4. Run
python cli.py dev    # http://localhost:8000
`

**Or with Docker (one command):**
`ash
docker compose up --build
`

---

## Project Structure

`
settings.py          # Every tunable in one place (env-driven)
hooks.py             # Extension points (decorator-based)
cache.py             # Optional Redis cache (no-op when disabled)
worker.py            # Background task runner
cli.py               # Developer CLI: init, ingest, dev

main.py              # FastAPI app + 12-step pipeline + endpoints
policy.py            # BehaviorPolicy engine (deterministic rules)
context_manager.py   # Token budgeting, history trimming, LLM summarization
query_db.py          # PostgreSQL + pgvector (all persistence)
vector_store.py      # Document search (pgvector + numpy fallback)
embeddings.py        # BAAI/bge-base-en-v1.5 768-dim, asymmetric retrieval

llm/
  providers/
    base.py          # LLMProvider ABC
    cerebras.py      # Cerebras Cloud SDK
    openai.py        # OpenAI / Azure / vLLM / Ollama
    anthropic.py     # Anthropic Messages API
    __init__.py      # Dynamic provider loader
  client.py          # Thin wrapper  active provider
  classifier.py      # Intent classification (heuristics + LLM)
  prompts.py         # All prompt templates (single source)
  prompt_orchestrator.py  # Policy-aware message builder
  generators.py      # Response generation (stream + batch)
  profile_detector.py     # Extract personal facts from messages

knowledge/           # Drop .txt/.md files here  auto-indexed
frontend/            # React 18 + Vite + Tailwind + Vercel AI SDK
`

---

## Configuration

All settings live in settings.py and are driven by environment variables. See .env.example for the full reference.

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `cerebras` | Provider: `cerebras`, `openai`, `anthropic` |
| `LLM_API_KEY` |  | **Required**  your API key |
| `LLM_MODEL` | *(provider default)* | Model name override |
| `LLM_BASE_URL` |  | Custom endpoint (Ollama, Azure, vLLM) |
| `DATABASE_URL` |  | Full PostgreSQL connection string |
| `RETRIEVAL_K` | `4` | Document chunks per knowledge query |
| `TOPIC_CONTINUATION_THRESHOLD` | `0.35` | Continuation sensitivity |
| `CHUNK_SIZE` | `500` | Knowledge base chunk size (chars) |
| `ENABLE_CACHE` | `false` | Redis caching for classifications + embeddings |
| `FORCE_REINDEX` | `false` | Re-index knowledge base on every startup |

---

## CLI

`ash
python cli.py init          # Scaffold project  creates knowledge/, copies .env
python cli.py ingest        # Index knowledge base into PostgreSQL
python cli.py ingest docs/  # Index from a custom directory
python cli.py dev           # Start dev server (uvicorn --reload)
`

---

## Swapping the LLM

`env
# OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-...

# Anthropic
LLM_PROVIDER=anthropic
LLM_API_KEY=sk-ant-...

# Ollama (local)
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
`

Add a new provider: subclass `LLMProvider` from `llm/providers/base.py`.

---

## Extension Hooks

Customize behavior without editing core files:

`python
from hooks import Hooks

@Hooks.before_generation
def inject_date(pipeline_result):
    from datetime import date
    pipeline_result.rag_context += f"\nDate: {date.today()}"
    return pipeline_result

@Hooks.policy_override
def always_rag_for_questions(features, decision):
    if "?" in features.query:
        decision.inject_rag = True
    return decision
`

Four hook points: `before_generation`, `after_generation`, `policy_override`, `before_persist`.

---

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat/stream` | Streaming chat (Vercel AI SDK SSE) |
| `POST` | `/chat` | Non-streaming chat |
| `GET` | `/health` | Status, DB, document count, provider |
| `GET/POST/PUT/DELETE` | `/conversations` | Conversation management |
| `GET/POST/PUT/DELETE` | `/profile` | User profile management |

---

## Frontend

AI-native React UI with observable pipeline decisions:

- **Intent badge**  shows classification + confidence on every response
- **Pipeline timeline**  real-time stage chips: Classified  Retrieved  Generating
- **Retrieval panel**  expandable breakdown of what documents/Q&A were used
- **Debug Mode**  raw PolicyDecision JSON on every message
- **Command palette**  Ctrl+K quick navigation
- **Token meter**  context window usage visualization

`ash
cd frontend && npm install && npm run dev    # port 5173
npm run build                                # output to frontend/dist/
`

---

## License

MIT
