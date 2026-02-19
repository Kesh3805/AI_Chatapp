# Database Schema & Query Functions — query_db.py

## Connection Management

The database layer uses psycopg2.pool.SimpleConnectionPool. Configuration is read from settings.py — DATABASE_URL takes priority over individual POSTGRES_* settings. The pool is lazily initialized on first use via _get_pool(). Min connections: DB_POOL_MIN (default 1). Max connections: DB_POOL_MAX (default 10).

The function init_db() creates all tables using CREATE TABLE IF NOT EXISTS plus necessary indexes. It also runs migrations (ALTER TABLE ADD COLUMN IF NOT EXISTS) for columns added after initial schema creation. Returns True on success, False if PostgreSQL is unreachable. If init_db() returns False, the application runs in in-memory mode (no persistence).

## Schema — Five Tables

### conversations
Primary key: id (TEXT, UUID).
Columns: title (TEXT, default "New Chat"), created_at (TIMESTAMP), updated_at (TIMESTAMP), message_count (INTEGER default 0), is_archived (BOOLEAN default false), metadata (JSONB), topic_embedding (vector(EMBEDDING_DIMENSION), default 768-dim).

The topic_embedding column stores the rolling conversation topic vector, updated on every message using exponential decay. Used for the topic similarity gate in the pipeline.

Indexes: idx_convs_upd on updated_at DESC for efficient listing by recency.

### chat_messages
Primary key: id (SERIAL).
Columns: user_id (TEXT default "public"), conversation_id (TEXT FK to conversations), role (TEXT — "user" or "assistant"), content (TEXT), tags (TEXT[] default '{}'), timestamp (TIMESTAMP), metadata (JSONB).

Indexes: idx_msgs_conv on (conversation_id, timestamp) for efficient message loading.

### user_queries
Primary key: id (SERIAL).
Columns: query_text (TEXT), embedding (vector(EMBEDDING_DIMENSION), default 768-dim), user_id (TEXT default "public"), conversation_id (TEXT), response_text (TEXT), tags (TEXT[] array), timestamp (TIMESTAMP), metadata (JSONB).

This table enables semantic search over past conversations. Every user message with its response and embedding is stored here. The pipeline searches this table when inject_rag=True (cross-conversation QA) and inject_qa_history=True (same-conversation QA).

Indexes: idx_queries_conv on conversation_id, idx_queries_ts on timestamp DESC, idx_queries_emb HNSW index on embedding using vector_cosine_ops for fast approximate nearest-neighbor search.

### user_profile
Primary key: id (SERIAL).
Columns: key (TEXT, UNIQUE constraint), value (TEXT), category (TEXT default "general"), created_at (TIMESTAMP), updated_at (TIMESTAMP).

Categories used in practice: personal (name, age, location), professional (job_title, employer, skills), preferences (preferred_language, framework), health, education, other.

The UNIQUE constraint on key means updating an existing key uses upsert logic in update_profile_entry().

### document_chunks
Primary key: id (SERIAL).
Columns: content (TEXT), embedding (vector(EMBEDDING_DIMENSION), default 768-dim), source (TEXT default "default"), chunk_index (INTEGER default 0), metadata (JSONB), created_at (TIMESTAMP).

This table IS the knowledge base vector store. Documents are chunked, embedded, and stored here via cli.py ingest or auto-ingestion at startup. The source column identifies which file the chunk came from.

Indexes: idx_doc_chunks_emb HNSW index on embedding for fast vector search, idx_doc_chunks_src on source for efficient per-source operations (clear, count by source).

## Conversation Functions

**create_conversation(title="New Chat")** — Inserts a new conversation with a UUID id. Returns the full conversation dict.

**list_conversations(limit=50)** — Returns all conversations ordered by updated_at DESC. Each dict has: id, title, created_at, updated_at, message_count, is_archived, metadata.

**get_conversation(conversation_id)** — Returns single conversation dict or None.

**rename_conversation(conversation_id, title)** — Updates title, sets updated_at to NOW(). Returns updated conversation dict or None.

**delete_conversation(conversation_id)** — Deletes conversation and cascades: also deletes all chat_messages and user_queries for that conversation_id. Returns True on success.

**touch_conversation(conversation_id)** — Updates updated_at to NOW(). Called on every message persist to keep conversations sorted by recency.

**increment_message_count(conversation_id, n)** — Increments message_count by n (typically 2 per turn: one user + one assistant message).

## Message Functions

**store_chat_message(role, content, conversation_id, tags=None, metadata=None)** — Inserts a message. Creates the conversation record if it doesn't exist (via ensure_conversation_exists() helper).

**get_conversation_messages(conversation_id, limit=200)** — Returns all messages in chronological order as list of dicts with role and content.

**get_recent_chat_messages(conversation_id, k=20)** — Returns last k messages as list of dicts, used in pipeline for conversation history.

## Query (Semantic Search) Functions

**store_query(query_text, embedding, response_text, conversation_id, tags=None, metadata=None)** — Stores a Q&A pair with its embedding vector. Called in background during persist_after_response().

**retrieve_similar_queries(embedding, k=4, conversation_id=None, current_tags=None, min_similarity=0.65)** — Searches user_queries using pgvector cosine similarity. Returns top-k results above min_similarity threshold. Excludes results from the current conversation_id to avoid self-reference. Used for cross-conversation Q&A injection.

**retrieve_same_conversation_queries(embedding, conversation_id, k=3, min_similarity=0.65)** — Same as above but restricted to the current conversation_id. Used for same-conversation Q&A on continuation intents.

**get_similar_messages_in_conversation(embedding, conversation_id, k=3, min_similarity=0.65)** — Returns semantically similar older messages from the same conversation. Used during history pruning to augment the recency window.

## Profile Functions

**get_user_profile()** — Returns all entries from user_profile as list of dicts with id, key, value, category.

**get_profile_as_text()** — Returns profile as formatted string "key: value\n..." suitable for LLM injection.

**update_profile_entry(key, value, category="general")** — Upserts a profile entry. If the key exists, updates value and updated_at. If new, inserts. Returns the entry id.

**delete_profile_entry(entry_id)** — Deletes profile entry by integer id. Returns True on success.

## Topic Vector Functions

**get_topic_vector(conversation_id)** — Returns topic_embedding as numpy float32 array, or None if not yet set.

**update_topic_vector(conversation_id, new_embedding, alpha=0.2)** — Updates rolling topic vector. If no existing vector: sets to new_embedding directly. If existing: applies exponential moving average (1-alpha)*old + alpha*new, then normalizes to unit length. The alpha parameter is settings.TOPIC_DECAY_ALPHA.

## Document Chunk Functions

**store_document_chunks(chunks, source)** — Takes a list of (text, embedding) tuples and bulk-inserts them into document_chunks with the given source label. Used by vector_store.add_documents() and cli.py ingest.

**search_document_chunks(embedding, k=4)** — Searches document_chunks using pgvector ORDER BY embedding <=> $1 LIMIT k. Returns list of content strings.

**count_document_chunks()** — Returns total row count in document_chunks.

**clear_document_chunks(source=None)** — Deletes all document chunks (if source is None) or only chunks matching a specific source filename. Used by cli.py ingest before re-indexing.

## Tag Inference

**infer_tags(query_text)** — Lightweight function that scans the query for keywords and assigns tags. Examples: queries containing "python", "code", "function" → ["programming"]. Queries containing "database", "sql" → ["database"]. Returns a list of string labels used in metadata and filtering.
