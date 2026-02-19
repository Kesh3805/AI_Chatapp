# RAG Chat Framework — Architecture & Core Concepts

## What This Is

RAG Chat is a policy-driven, intent-gated Retrieval-Augmented Generation framework. It is NOT a naive RAG system that dumps all context into every LLM call. Instead, it classifies user intent first, then selectively retrieves only the context that is actually needed. This reduces noise, cuts latency, and produces better responses.

The framework is designed to be adopted as a boilerplate: opinionated defaults, swappable components, minimal configuration, production-safe behavior, and extensible without forking.

Version: 4.0.0. Backend: FastAPI. Frontend: React 18 + Vite + Tailwind + Vercel AI SDK. Database: PostgreSQL 16 + pgvector (single DB, no FAISS). LLM: Pluggable (Cerebras, OpenAI, Anthropic).

## The Five Intent Categories

Every message is classified into one of five intents BEFORE any retrieval happens:

- **general**: Greetings, opinions, open questions, small talk. No retrieval. LLM gets system prompt + query only.
- **knowledge_base**: Factual or technical questions that benefit from document retrieval. LLM gets knowledge base excerpts + prior Q&A + history + query.
- **continuation**: Follow-up messages referencing earlier conversation (uses pronouns like "it", "that", "those"). LLM gets curated history + same-conversation Q&A + query.
- **profile**: Personal information sharing ("My name is Alex") or personal queries ("What's my name?"). Profile updates are saved in background. Profile questions inject the stored profile.
- **privacy**: Questions about what data is stored, deletion requests, tracking concerns. LLM gets full profile data + privacy transparency rules.

## The Pipeline (12 Steps)

Every chat message goes through this exact sequence in run_pipeline() in main.py:

Step 1 — EMBED: Convert user query string to a 768-dimensional float32 vector using sentence-transformers (BAAI/bge-base-en-v1.5, runs locally, no API key needed). Uses get_query_embedding() which applies QUERY_INSTRUCTION prefix if set (enables asymmetric retrieval).

Step 2 — PARALLEL LOAD: Simultaneously load conversation history (last 20 messages from chat_messages table) and user profile (all entries from user_profile table) from PostgreSQL. These run in parallel using ThreadPoolExecutor to cut latency from ~90ms serial to ~50ms.

Step 3 — CLASSIFY INTENT: Determine the intent using fast pre-heuristics first. If no heuristic matches, fall back to an LLM API call. Returns {"intent": str, "confidence": float}. If Redis cache is enabled, the classification result is looked up in cache before any heuristic work — cache TTL is 30 minutes.

Step 4 — TOPIC GATE: Only applies to "continuation" intent. Computes cosine similarity between the current query embedding and the conversation's rolling topic vector stored in the conversations table. If similarity is below TOPIC_CONTINUATION_THRESHOLD (default 0.35), the intent is downgraded to "general". This prevents the system from treating a topic change as a follow-up.

Step 5 — CONTEXT FEATURES: Call extract_context_features() to compute deterministic boolean features: is_greeting, references_profile, privacy_signal, is_followup, is_profile_statement, is_profile_question, has_profile_data, profile_name, conversation_length.

Step 6 — POLICY RESOLVE: Pass features + intent to BehaviorPolicy().resolve() to get a PolicyDecision dataclass. This decision controls what gets retrieved and how the response is framed. Policy is purely deterministic rules — no LLM involved. After BehaviorPolicy, Hooks.run_policy_override() runs any registered policy hook functions.

Step 7 — HISTORY PRUNING: If the policy decision says use_curated_history=True, take the last RECENCY_WINDOW (default 6) messages plus up to SEMANTIC_K (default 3) semantically similar older messages from the same conversation that exceed SIMILARITY_THRESHOLD (default 0.65). This is smarter than raw history injection.

Step 8 — SELECTIVE RETRIEVAL: Based on PolicyDecision flags, fetch only what is needed:
- inject_rag=True: search document_chunks table via pgvector for top RETRIEVAL_K (default 4) chunks, then also search user_queries for up to QA_K (default 4) similar past Q&A pairs above QA_MIN_SIMILARITY (default 0.65).
- inject_qa_history=True + continuation intent: search user_queries for same-conversation Q&A.
- inject_profile=True: format user_profile entries as key: value text.
- privacy_mode=True: force profile injection regardless of inject_profile flag.

Step 9 — BEFORE_GENERATION HOOKS: Run all registered @Hooks.before_generation functions on the PipelineResult before sending it to the LLM.

Step 10 — GENERATE: Build the LLM message list via build_messages() in prompt_orchestrator.py, then call the configured LLM provider. For streaming (/chat/stream), the response is streamed using the Vercel AI SDK data stream protocol with stage event annotations. For batch (/chat), a complete string is returned.

Step 11 — AFTER_GENERATION HOOKS: Run all registered @Hooks.after_generation functions on the response text and PipelineResult.

Step 12 — BACKGROUND PERSIST: Use worker.submit() to run _work() in a daemon thread: store user message (chat_messages), store assistant response (chat_messages), store query embedding (user_queries), update topic vector (conversations.topic_embedding with exponential decay alpha TOPIC_DECAY_ALPHA=0.2), increment message count, generate title on first message, detect and save profile updates. All DB writes are non-blocking.

## Why Policy-Driven Architecture

The BehaviorPolicy engine separates behavior from model calls. When behavior is wrong (e.g., greeting doesn't use the user's name), you fix a rule in policy.py. You never edit prompt strings or generator functions. Similarly, when retrieval is wrong, you adjust a policy rule. This architectural separation makes the system debuggable and extensible.

## Streaming Stage Protocol

Before any text tokens are emitted, the backend sends three stage annotation events so the frontend can show real-time pipeline progress:

1. `8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]`
2. `8:[{"stage":"retrieved","retrieval_info":{"num_docs":4,"similar_queries":2}}]` (only if something was retrieved)
3. `8:[{"stage":"generating"}]`

After all text tokens, a final annotation is sent: `8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...},"query_tags":[...]}]`

Then: `e:{"finishReason":"stop"}` and `d:{"finishReason":"stop"}`.

Text tokens are emitted as: `0:"token text here"`.

## Topic Vector Rolling Average

Each conversation has a rolling topic embedding vector stored in conversations.topic_embedding (vector(EMBEDDING_DIMENSION), default 768). On every message, this vector is updated using exponential moving average: new_vector = (1 - alpha) * old_vector + alpha * query_embedding, where alpha = TOPIC_DECAY_ALPHA (default 0.2). This means recent messages influence the topic more than older ones. The topic vector is used in Step 4 to measure whether the current message is a continuation of the active topic.

## Single Database Architecture

PostgreSQL 16 + pgvector is the single database for EVERYTHING: conversations metadata, chat messages, user profile key-value pairs, query embeddings for semantic Q&A search, document chunk vectors for knowledge retrieval, and the rolling topic embedding per conversation. No FAISS. No separate vector database. One connection pool, one backup strategy, one deployment concern.
