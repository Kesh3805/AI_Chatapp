# Policy Engine & Extension Hooks

## BehaviorPolicy — policy.py

The BehaviorPolicy engine translates intent + context features into a concrete set of retrieval and framing instructions (PolicyDecision). It is entirely deterministic — no LLM calls. All decisions are human-readable if/elif rules.

### ContextFeatures Dataclass

extract_context_features(query, intent, profile_entries=None, conversation_length=0, topic_similarity=None) → ContextFeatures

Computed fields:
- is_greeting: True if query ≤8 words and starts with a greeting pattern
- references_profile: True if query contains any PERSONAL_REF_SIGNALS phrase (e.g., "my job", "my name", "who am i")
- privacy_signal: True if intent == "privacy"
- is_followup: True if intent == "continuation"
- is_profile_statement: True if query starts with a PROFILE_STATEMENT_PREFIX and has no "?"
- is_profile_question: True if intent == "profile" and not is_profile_statement
- has_profile_data: True if profile_entries is non-empty
- profile_name: The value from the first entry with key in ("name", "first_name", "full_name", "username")
- conversation_length: Number of messages in recent history
- topic_similarity: The cosine similarity score from the topic gate step (or None)

### PolicyDecision Dataclass

The output of BehaviorPolicy.resolve(). Controls the entire pipeline behavior:

- inject_profile (bool, default False) — Include formatted user profile in LLM context
- inject_rag (bool, default False) — Fetch documents from document_chunks + cross-conversation Q&A
- inject_qa_history (bool, default False) — Fetch same-conversation Q&A history
- use_curated_history (bool, default True) — Apply semantic + recency history pruning
- privacy_mode (bool, default False) — Inject privacy transparency frame
- greeting_name (str|None, default None) — If set, inject greeting personalization frame with this name
- retrieval_route (str, default "llm_only") — Label for metadata/debugging (llm_only, rag, conversation, profile, profile_update, privacy)
- rag_k (int, default 4) — How many document chunks to retrieve (overrides RETRIEVAL_K)
- qa_k (int, default 4) — How many Q&A pairs to retrieve
- qa_min_similarity (float, default 0.65) — Minimum similarity for Q&A retrieval

### BehaviorPolicy.resolve() Rules

Base intent rules (evaluated in if/elif/else — NOT early returns, so cross-intent overlays always run):

**privacy intent**: privacy_mode=True, inject_profile=has_profile_data, use_curated_history=False, retrieval_route="privacy".

**profile intent (statement — no question mark, starts with profile prefix)**: retrieval_route="profile_update", use_curated_history=False. (No injection — profile is being saved, not queried.)

**profile intent (question)**: inject_profile=has_profile_data, retrieval_route="profile".

**knowledge_base intent**: inject_rag=True, inject_qa_history=True, retrieval_route="rag".

**continuation intent**: inject_qa_history=True, retrieval_route="conversation".

**general intent (else)**: use_curated_history=False, retrieval_route="llm_only". (Fastest path — LLM only, no retrieval, no history.)

Cross-intent overlays (always run after base rules):

1. Name injection: If profile_name is known AND inject_profile is still False, set greeting_name=profile_name. This gives a lightweight personalization for general/continuation/knowledge_base without injecting the full profile dump.

2. Personal reference overlay: If references_profile=True AND has_profile_data=True AND inject_profile is still False AND intent is not "privacy": set inject_profile=True and clear greeting_name (name is now in full profile, no need for separate frame).

### Signal Lists in policy.py

GREETING_PATTERNS: 17 patterns including "hello", "hi", "hey", "good morning", "howdy", "sup", "what's up", "yo", "hola".

PERSONAL_REF_SIGNALS: 34 phrases including "my job", "my role", "my name", "my work", "what do i do", "who am i", "i work as", "my skills", "my experience", "whats my", "my weight", "my height", "my age", "my location", "my degree".

PROFILE_STATEMENT_PREFIXES: Tuple of 18 openers used to detect when a user is sharing personal info rather than asking a question.

## Extension Hooks — hooks.py

The Hooks class provides four pipeline extension points as class-level decorator registries. Hooks run in registration order. Each hook receives and returns a value (modified or original).

### Registering Hooks

Import and use decorators:
```python
from hooks import Hooks
```

### @Hooks.before_generation
Called with PipelineResult before the LLM generation step.
Signature: fn(pipeline_result) → pipeline_result

Use cases: add custom context to rag_context, modify the intent, log pipeline state, add metadata to retrieval_info.

Example:
```python
@Hooks.before_generation
def inject_current_date(p):
    from datetime import date
    p.rag_context += f"\nToday's date: {date.today()}"
    return p
```

### @Hooks.after_generation
Called with (response_text, pipeline_result) after text is generated.
Signature: fn(response: str, pipeline_result) → str

Use cases: content filtering, response logging, post-processing text, analytics.

Example:
```python
@Hooks.after_generation
def log_response(response, p):
    logger.info(f"Generated {len(response)} chars for intent={p.intent}")
    return response  # unchanged, just logged
```

### @Hooks.policy_override
Called with (features, decision) after BehaviorPolicy.resolve().
Signature: fn(features: ContextFeatures, decision: PolicyDecision) → decision

Use cases: custom business rules that override the default policy, A/B testing, tenant-specific behavior.

Example:
```python
@Hooks.policy_override
def always_rag_for_questions(features, decision):
    if features.is_profile_question:
        decision.inject_rag = True  # always fetch docs for profile questions
    return decision
```

### @Hooks.before_persist
Called with (pipeline_result, response_text) before the background DB writes.
Signature: fn(pipeline_result, response_text) → None

Use cases: send to analytics pipeline, webhook notifications, custom logging.

Example:
```python
@Hooks.before_persist
def notify_analytics(p, response):
    analytics.track("chat_message", {
        "intent": p.intent,
        "confidence": p.confidence,
        "response_length": len(response),
    })
```

### Hooks.clear()
Unregisters all hooks from all slots. Useful in tests to reset state between test cases.

### Loading Custom Hooks
Create a file like my_hooks.py in the project root and import it in main.py before the app starts. Since hooks are class-level lists, importing the file that registers hooks is sufficient — no explicit registration call needed.

## Optional Redis Cache — cache.py

when ENABLE_CACHE=false (default), all cache functions are no-ops that immediately return None. When true, cache.py lazily connects to Redis on first use.

cache.get_classification(query) → dict|None: Check cache for {"intent","confidence"} result for this exact query string. Key is MD5 of "intent:"+query[:200].

cache.set_classification(query, result): Store classification with 1800-second TTL (30 minutes).

cache.get_embedding(text) → list|None: Check cache for embedding vector as list[float].

cache.set_embedding(text, vector): Store embedding (converts numpy array to list[float]).

cache.get(key) / cache.put(key, value, ttl): Generic get/set (uses CACHE_TTL from settings, default 3600s).

If Redis becomes unavailable after startup, all operations silently return None/no-op — zero exceptions are raised to calling code.

## Background Worker — worker.py

worker.submit(fn, *args) submits a callable to a daemon thread. The thread runs independently — the main request handler does not wait for it. Used for all DB persistence after response generation.

Current implementation uses threading.Thread(target=fn, daemon=True). Daemon threads die when the main process exits. Future versions can swap to celery, rq, or arq by replacing this single file.

## CLI — cli.py

python cli.py init: Scaffold a new project. Creates the knowledge/ directory, copies .env.example to .env (if .env doesn't exist), and prints a getting-started message.

python cli.py ingest [DIR]: Index knowledge base files into PostgreSQL. Optional DIR overrides the KNOWLEDGE_DIR setting. Reads all .txt and .md files. For each file: calls vector_store.clear(source) to remove old chunks for that file, then chunks the text and calls vector_store.add_documents(). Prints progress.

python cli.py dev: Start the development server. Runs uvicorn main:app --host HOST --port PORT --reload using the settings.HOST and settings.PORT values. Equivalent to running uvicorn directly.

## Vector Store — vector_store.py

The vector store is backed by PostgreSQL pgvector with an in-memory numpy fallback for when db_enabled=False.

vector_store.init(db_enabled): Called at startup. If db_enabled, uses pgvector exclusively via query_db functions. If not db_enabled, initializes an in-memory list (_fallback_docs) for basic functionality.

vector_store.add_documents(chunks, source): Takes a list of text strings. Generates embeddings for each chunk using get_embeddings() (batch). Stores in document_chunks table (pgvector mode) or appends to _fallback_docs (in-memory mode).

vector_store.search(query, k=4): Generates embedding for query. Queries document_chunks via query_db.search_document_chunks() (pgvector mode) or computes cosine similarity over _fallback_docs with numpy (in-memory mode).

vector_store.has_documents(): Returns True if document count > 0.

vector_store.count(): Returns total document chunk count.

vector_store.clear(source=None): Removes all chunks or chunks for a specific source file.

## Embeddings — embeddings.py

get_embedding(text: str) → np.ndarray: Encodes a document/passage string (no prefix). Used for document indexing.

get_query_embedding(text: str) → np.ndarray: Encodes a search query, applying settings.QUERY_INSTRUCTION prefix if set. Used for all user query embeddings at retrieval time. This is the correct function to call when computing similarity to indexed documents.

get_embeddings(texts: list[str]) → np.ndarray: Batch encoding for multiple document texts. More efficient than calling get_embedding() in a loop. Used by vector_store.add_documents() for bulk document indexing.

get_dim() → int: Returns the actual embedding dimension of the loaded model (calls model.get_sentence_embedding_dimension()).

The model is loaded lazily on first call as a module-level singleton (not at import time). Default model: BAAI/bge-base-en-v1.5, 768-dim, ~440 MB download on first use.

## Context Management — context_manager.py

This module enforces token-budget discipline on conversation history before messages are assembled for the LLM. It is called inside prompt_orchestrator.build_messages() on every request.

estimate_tokens(text: str) → int: Estimates token count as len(text) // 4. This is a 97%-accurate approximation for English prose without requiring tiktoken. To use exact counting, replace this function with tiktoken.encoding_for_model("gpt-4o").encode(text).

message_tokens(msg: dict) → int: Estimates token cost for a single {"role": ..., "content": ...} dict. Adds 10 tokens of overhead to account for role string and JSON framing.

history_tokens(messages: list[dict]) → int: Total estimated token cost for a list of messages. Sum of message_tokens() over all messages.

fit_messages_to_budget(messages, budget_tokens, min_recent=4) → list[dict]: Trims oldest messages until the list fits within budget_tokens. Always keeps at least min_recent messages unconditionally. Returns a tail slice of the original list (same dict objects). The dropped messages are logged at INFO level. Used by default (when ENABLE_HISTORY_SUMMARIZATION=False).

summarize_old_turns(messages, max_history_tokens, completion_fn, min_recent=6) → list[dict]: When history overflows max_history_tokens, splits into [overflow | recent] parts. Sends the overflow transcript to completion_fn (llm.client.completion) with a summarization system prompt. Prepends the summary as a system message: "[Summary of earlier conversation]: ...". On LLM failure, falls back to fit_messages_to_budget silently. Used when ENABLE_HISTORY_SUMMARIZATION=True.

Integration point: prompt_orchestrator.build_messages() calls context_manager after selecting history (curated_history or chat_history) and before extending the messages list. This means token budgeting is enforced for ALL generation paths (streaming and non-streaming, all intents).
