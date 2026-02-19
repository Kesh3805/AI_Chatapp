# API Reference — Endpoints, Data Models, Streaming

## Base URL

Default: http://localhost:8000. All endpoints return JSON unless noted.

## Chat Endpoints

### POST /chat
Non-streaming chat. Runs the full pipeline and returns complete response as JSON.

Request body:
```json
{
  "user_query": "What is the BehaviorPolicy engine?",
  "conversation_id": "optional-uuid-string",
  "tags": ["optional", "tag", "list"]
}
```

If conversation_id is omitted, a new UUID is generated and returned.
If tags is omitted, tags are inferred via query_db.infer_tags().

Response body:
```json
{
  "response": "The full response text as a string.",
  "conversation_id": "uuid-string",
  "intent": "knowledge_base",
  "confidence": 0.92,
  "retrieval_info": {
    "intent": "knowledge_base",
    "confidence": 0.92,
    "topic_similarity": null,
    "route": "rag",
    "num_docs": 4,
    "similar_queries": 2,
    "profile_injected": false
  },
  "query_tags": ["technical"]
}
```

### POST /chat/stream
Streaming chat using Vercel AI SDK data stream protocol over Server-Sent Events.

Request body: same as POST /chat.

Response: text/event-stream with the following line format (each line ends with \n):

Stage events (before text tokens):
```
8:[{"stage":"classified","intent":"knowledge_base","confidence":0.92}]
8:[{"stage":"retrieved","retrieval_info":{"num_docs":4,"similar_queries":2}}]
8:[{"stage":"generating"}]
```

Text delta tokens (one per streamed token):
```
0:"Hello"
0:", here"
0:" is the"
0:" answer."
```

Final metadata annotation (after all tokens):
```
8:[{"intent":"knowledge_base","confidence":0.92,"retrieval_info":{...},"query_tags":[...]}]
```

Finish events:
```
e:{"finishReason":"stop"}
d:{"finishReason":"stop"}
```

The retrieved stage annotation is only emitted if any retrieval occurred (num_docs, similar_queries, same_conv_qa, or profile_injected).

## Conversation Endpoints

### POST /conversations
Create a new conversation.

Request: `{"title": "My Chat"}` (title defaults to "New Chat").
Response: `{"id":"uuid","title":"New Chat","created_at":"...","updated_at":"...","message_count":0}`.

### GET /conversations?limit=50
List all conversations ordered by most recently updated. Default limit: 50.
Response: `{"conversations":[...], "count": N}`.

### GET /conversations/{conversation_id}
Get a single conversation by ID.
Response: conversation dict or 404.

### GET /conversations/{conversation_id}/messages?limit=200
Get all messages in a conversation in chronological order.
Response: `{"conversation_id":"uuid","messages":[{"role":"user","content":"..."},...],"count":N}`.

### PUT /conversations/{conversation_id}
Rename a conversation.
Request: `{"title": "New Title"}`.
Response: updated conversation dict.

### DELETE /conversations/{conversation_id}
Delete a conversation and all its messages and query history.
Response: `{"deleted": true}` or 404.

## Profile Endpoints

### GET /profile
Get all stored user profile entries.
Response: `{"entries":[{"id":1,"key":"name","value":"Alex","category":"personal"},...], "count": N}`.

### POST /profile
Manually add a profile entry.
Request: `{"key": "preferred_language", "value": "Python", "category": "preferences"}`.
Response: `{"id": N, "key": "...", "value": "...", "category": "..."}`.

### PUT /profile/{entry_id}
Update a profile entry by id.
Request: same as POST.
Response: updated entry.

### DELETE /profile/{entry_id}
Delete a profile entry by id.
Response: `{"deleted": true}`.

## Health Endpoint

### GET /health
Returns system status for monitoring and load balancers.
Response:
```json
{
  "status": "ok",
  "database": "connected",
  "documents": 47,
  "llm_provider": "cerebras",
  "version": "4.0.0"
}
```

The "documents" field is the count of rows in document_chunks table. The "llm_provider" field is the .name property of the active provider instance.

## Static Asset Serving

### GET /
If frontend/dist/index.html exists, serves the React build. Otherwise falls back to the root index.html (vanilla HTML fallback). FastAPI mounts /assets from frontend/dist/assets when the React build exists.

## When Database Is Unavailable

All conversation and profile endpoints return graceful responses when DB_ENABLED is False:
- GET /conversations → {"conversations":[], "count":0}
- POST/PUT/DELETE with DB unavailable → 503 with "Database not available"
- GET /profile → {"entries":[], "count":0}
- /chat and /chat/stream still work (in-memory mode, no persistence)

## Request Headers

### Cache-Control and X-Accel-Buffering
The /chat/stream endpoint sets Cache-Control: no-cache and X-Accel-Buffering: no to prevent proxy servers (Nginx, Caddy) from buffering the SSE stream.

## CORS

The FastAPI app has CORSMiddleware configured with allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]. This allows the Vite dev server on port 5173 to call the API on port 8000 during development.
