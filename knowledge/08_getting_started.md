# Getting Started, Deployment & Customization

## Quick Start (5 Steps)

Step 1 — Clone and init:
```
git clone <repo> && cd rag-chat
python cli.py init        # Creates knowledge/, copies .env.example to .env
```

Step 2 — Configure:
Edit .env and set LLM_API_KEY=your-key-here. Set LLM_PROVIDER if not using Cerebras.

Step 3 — Start PostgreSQL:
```
docker compose up postgres -d
```
Or manually: docker run -d --name chatapp-postgres -e POSTGRES_USER=root -e POSTGRES_PASSWORD=password -e POSTGRES_DB=chatapp -p 55432:5432 pgvector/pgvector:pg16

Step 4 — Add your knowledge base (optional — ships with example content):
Drop .txt or .md files into the knowledge/ directory, then:
```
python cli.py ingest
```

Step 5 — Run:
```
python cli.py dev         # starts on http://localhost:8000
```

## Docker Compose Deployment

Start everything with one command:
```
docker compose up --build
```

Services defined in docker-compose.yml:
- postgres: pgvector/pgvector:pg16 image, port 55432:5432, named volume "pgdata" for persistence
- app: Python 3.12, builds from Dockerfile, port 8000:8000, mounts model_cache volume for sentence-transformers model caching, depends_on postgres with healthcheck

Optional Redis (commented out by default, uncomment in docker-compose.yml):
- redis: redis:7-alpine, port 6379:6379
- Add REDIS_URL and ENABLE_CACHE=true to app environment

## Environment Variables (.env)

Required:
- LLM_API_KEY — your API key for the selected provider

Common:
- LLM_PROVIDER — "cerebras" (default), "openai", or "anthropic"
- LLM_MODEL — override the provider's default model
- DATABASE_URL — full PostgreSQL connection string (overrides POSTGRES_*)
- FORCE_REINDEX — set to "true" during development to re-index on every restart

Optional tuning:
- RETRIEVAL_K — document chunks per query (default 4)
- QA_K — Q&A pairs per query (default 4)
- TOPIC_CONTINUATION_THRESHOLD — continuation sensitivity (default 0.35)
- RECENCY_WINDOW — history messages to keep (default 6)
- CHUNK_SIZE — document chunk character size (default 500)
- MAX_RESPONSE_TOKENS — max tokens per response (default 2048)
- ENABLE_CACHE — set to "true" for Redis caching
- REDIS_URL — Redis connection URL if ENABLE_CACHE=true

## Swapping the LLM Provider

Change two lines in .env:
```
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-key
```

For Ollama (local LLM):
```
LLM_PROVIDER=openai
LLM_API_KEY=ollama
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.2
```

To add a new provider, create llm/providers/myprovider.py subclassing LLMProvider from llm/providers/base.py, then add it to the registry in llm/providers/__init__.py.

## Adding Knowledge Base Content

Drop .txt or .md files into the knowledge/ directory. Run:
```
python cli.py ingest
```

The cli.py ingest command: reads each file, chunks it into CHUNK_SIZE-character overlapping segments, generates embeddings using sentence-transformers (locally, no API key), stores all chunks in the document_chunks table with the filename as source. On re-ingest, existing chunks for that source are cleared first (clean re-index per file).

To force re-indexing on startup: set FORCE_REINDEX=true in .env.

## Customizing Pipeline Behavior

To change when profile data is injected: edit the BehaviorPolicy.resolve() rules in policy.py.

To change what gets retrieved for knowledge queries: modify the "knowledge_base" branch in BehaviorPolicy.resolve().

To add a new intent category: add it to VALID_INTENTS in llm/classifier.py, add it to the INTENT_PROMPT in llm/prompts.py, and add a branch in BehaviorPolicy.resolve().

To add custom logic without editing core files: use Hooks in hooks.py (see the Policy Engine & Extension Hooks document for details).

## Frontend Development

Start React dev server (with hot reload and proxy to backend):
```
cd frontend && npm install && npm run dev
```

Build for production:
```
npm run build
```
FastAPI automatically serves frontend/dist/ when it exists.

To remove the React frontend entirely and use the vanilla HTML fallback: just don't build the frontend. FastAPI will serve index.html from the project root instead.

## Profile System

The profile system automatically extracts personal facts from conversation. When a user says anything starting with "My name is", "I am", "I work as", "I like", etc., the profile detector LLM call fires in the background after the response is sent.

Profile data is injected into LLM context when: the user asks a personal question ("what's my name?", "where do I work?"), the intent is "profile", or when the query contains personal reference signals ("my job", "my role"). For privacy queries, all stored profile data is always shown to the LLM for transparency.

The profile is a flat key-value store. Keys are snake_case strings like "name", "job_title", "preferred_language", "employer". Values are text strings. Categories are: personal, professional, preferences, health, education, other.

## Health Monitoring

GET /health returns: status ("ok"), database ("connected" or "unavailable"), documents (chunk count), llm_provider (active provider name), version ("4.0.0"). Use this endpoint for:
- Load balancer health checks
- Docker Compose healthcheck
- Monitoring dashboards
- Verifying which provider is active after config changes

## Graceful Degradation

When PostgreSQL is not available (init_db() returns False), the application runs in in-memory mode:
- Conversations are not persisted (lost on restart)
- Profile learning is disabled
- Semantic Q&A search is disabled
- Knowledge base uses the in-memory numpy fallback in vector_store.py
- Knowledge base content IS still loaded from knowledge/ files into memory
- All chat endpoints still work (generate responses using LLM)
- The React frontend works normally

When Redis is not available (or ENABLE_CACHE=false), caching is silently disabled. All operations proceed without caching. No errors are raised.

## Performance Notes

The parallel step 1+2 (embed + load history + load profile simultaneously) reduces pipeline preamble from ~90ms serial to ~50ms using ThreadPoolExecutor(max_workers=3). For even faster responses at the cost of stale cache: enable Redis with ENABLE_CACHE=true and tune CACHE_TTL.

The HNSW indexes on user_queries.embedding and document_chunks.embedding provide sub-millisecond approximate nearest-neighbor search at scale. For very large knowledge bases (millions of chunks), HNSW is highly recommended over the default flat index.

Sentence-transformers model loading takes 1-2 seconds on first request (model is not loaded until first embedding call). Model stays in memory for the process lifetime. The Docker compose configuration mounts a model_cache volume to avoid re-downloading on container restart.

## Known Limitations

The vanilla index.html fallback UI only supports basic chat — no streaming UI, no Debug Mode, no AI-native components. Build the React frontend for the full experience.

Profile detection fires on every conversation turn where the user message contains personal-statement signals. This is a background LLM call that may increase per-message cost slightly. Disable by setting a policy hook to skip profile detection if needed.

The default model is BAAI/bge-base-en-v1.5 generating 768-dimensional embeddings. Changing models requires matching EMBEDDING_DIMENSION and recreating the database tables (vector column size is fixed at table creation). Safer alternatives that preserve dimension: BAAI/bge-small-en-v1.5 (384-dim, matches old MiniLM column size), BAAI/bge-large-en-v1.5 (1024-dim).
