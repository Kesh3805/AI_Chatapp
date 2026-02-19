# LLM Package — Providers, Classifier, Orchestrator, Generators

## Package Structure (llm/)

The llm/ directory is a Python package with these files:
- __init__.py — barrel re-exports for backward compatibility
- client.py — thin wrapper delegating to the active provider
- providers/ — pluggable LLM provider implementations
- classifier.py — intent classification with pre-heuristics + LLM fallback
- prompts.py — ALL prompt templates (single source of truth)
- prompt_orchestrator.py — builds LLM message lists from pipeline data
- generators.py — response generation (streaming and batch)
- profile_detector.py — extracts personal facts from messages

## Providers Package (llm/providers/)

### llm/providers/base.py — Abstract Base Class
LLMProvider is an abstract class with two required methods:

complete(messages, temperature=0.3, max_tokens=2048) → str: Send a list of {"role": str, "content": str} dicts and return the complete response as a string.

stream_text_deltas(messages, temperature=0.3, max_tokens=2048) → Generator[str]: Send a list of messages and yield plain text strings (NOT formatted SSE lines — just the token text). The generators.py module wraps these in the "0:..." SSE format.

Every provider also has a .name property returning the provider name string.

### llm/providers/cerebras.py — Cerebras
Uses cerebras-cloud-sdk. DEFAULT_MODEL = "gpt-oss-120b" (65,536 token context window). Model supports both completion and streaming. If LLM_MODEL setting is empty, uses gpt-oss-120b. API key from settings.LLM_API_KEY.

### llm/providers/openai.py — OpenAI (and compatible)
Uses openai>=1.0.0. DEFAULT_MODEL = "gpt-4o". Supports LLM_BASE_URL for Azure OpenAI, vLLM, Ollama, and any OpenAI-compatible endpoint. If LLM_BASE_URL is set, passes it as base_url to the OpenAI client. Useful for pointing at a local Ollama instance: LLM_BASE_URL=http://localhost:11434/v1.

### llm/providers/anthropic.py — Anthropic
Uses anthropic>=0.20.0. DEFAULT_MODEL = "claude-sonnet-4-20250514". The Anthropic Messages API requires special handling: the system message must be passed as a separate "system" parameter (not as a message in the array), and consecutive messages with the same role must be merged (Anthropic rejects alternating-role violations). The _split_messages() helper extracts system messages and enforces role alternation. The first message in the array must be "user" role.

### llm/providers/__init__.py — Dynamic Loader
The provider() function is a module-level singleton. On first call, it reads settings.LLM_PROVIDER ("cerebras", "openai", or "anthropic"), lazily imports ONLY that provider's module (avoiding ImportError for uninstalled SDKs), instantiates the provider class, and caches it. Subsequent calls return the cached instance. The reset() function clears the cache (used in tests). If an unknown provider is specified, raises ValueError with a helpful message listing valid options.

## llm/client.py — Provider Wrapper

This is the only module that knows about providers. All other modules (classifier.py, generators.py, profile_detector.py) call completion() or stream_text_deltas() from client.py without knowing which provider is active.

completion(messages, temperature=0.3, max_tokens=MAX_RESPONSE_TOKENS) → str: Calls provider().complete(messages, ...). Returns response text string.

stream_text_deltas(messages, temperature=0.3, max_tokens=MAX_RESPONSE_TOKENS) → Generator[str]: Calls provider().stream_text_deltas(messages, ...). Yields text strings.

Token budget constants are re-exported from settings: MAX_RESPONSE_TOKENS=2048, MAX_CLASSIFIER_TOKENS=50, MAX_PROFILE_DETECT_TOKENS=300, MAX_TITLE_TOKENS=20.

## llm/classifier.py — Intent Classification

classify_intent(user_query, conversation_context=None) → {"intent": str, "confidence": float}

This function first checks the Redis cache (if enabled). On cache miss, it runs pre-heuristics in order:

1. Greeting fast-path: If query ≤8 words and starts with a greeting pattern ("hello", "hi", "hey", etc.), return {"intent":"general","confidence":0.97} without calling the LLM.

2. Profile statement fast-path: If query has no "?" and starts with a profile opener ("my name is", "i am", "i work", "i like", etc.), return {"intent":"profile","confidence":0.92}.

3. Privacy fast-path: If query contains any of ~22 privacy signal phrases ("do you store", "what data do you", "delete my data", etc.), return {"intent":"privacy","confidence":0.95}.

4. Continuation fast-path: If there is conversation context (≥2 messages), query is ≤8 words, query contains a continuation pronoun ("that", "it", "this", "those", etc.), AND query contains "?", return {"intent":"continuation","confidence":0.85}.

5. LLM fallback: Build a messages array with INTENT_PROMPT as system message and the query (plus last 6 context messages) as user message. Call completion() with temperature=0.0 and MAX_CLASSIFIER_TOKENS=50. Parse the JSON response. Clean up markdown fences and extract the JSON object if surrounded by other text. Validate the intent against VALID_INTENTS set. Cache the result. Return the result.

On any exception: return {"intent":"general","confidence":0.5} as safe fallback.

## llm/prompts.py — All Prompt Templates

INTENT_PROMPT: Instructions for the intent classifier. Defines the 5 intent labels with decision rules in priority order, plus confidence guidelines (0.9-1.0 obvious, 0.7-0.89 likely, 0.5-0.69 uncertain).

SYSTEM_PROMPT: Core personality and behavior rules for the assistant. 8 rules: Accuracy (no guessing), Context (use injected material), Tone (match user register), Formatting (Markdown with code blocks), Continuity (reference earlier conversation), Profile (treat injected data as ground truth), Length (thorough but not verbose), Safety (decline harmful requests).

PROFILE_CONTEXT_FRAME: Template injected when inject_profile=True. Uses {profile} placeholder. Instructs the LLM to use ONLY relevant profile entries, to say "You mentioned that..." when referencing them, never deny having the data, and never make health/medical/financial assessments.

RAG_CONTEXT_FRAME: Template injected when inject_rag=True. Uses {context} placeholder. Instructs LLM to support answers with retrieved excerpts and distinguish between retrieved content and its own knowledge.

QA_CONTEXT_FRAME: Template injected for same-conversation or cross-conversation Q&A context. Uses {qa} placeholder. Instructs LLM to synthesize rather than repeat verbatim.

PRIVACY_QA_FRAME: Template injected when privacy_mode=True. Instructs LLM to be fully transparent about data storage: list what is stored (came from user's own messages), explain usage is personalization-only, offer to delete, never be defensive or evasive.

GREETING_PERSONALIZATION_FRAME: Template injected when greeting_name is set. Uses {name} placeholder. Instructs LLM to address user by name in greetings, not overuse the name, and never claim to not know the user's name.

PROFILE_DETECT_PROMPT: Instructions for extracting personal facts. Output: JSON array of {"key", "value", "category"}. Categories: personal, professional, preferences, health, education, other. Rules: only extract explicit statements, not inferences; return [] for no-fact messages.

TITLE_PROMPT: Instructions for generating a 3-6 word conversation title from the first message. Return only the title text, no quotes, no punctuation.

## llm/prompt_orchestrator.py — Message Builder

build_messages(user_query, *, chat_history=None, curated_history=None, rag_context="", profile_context="", similar_qa_context="", privacy_mode=False, greeting_name=None) → list[dict]

Assembles the OpenAI-format messages array in this specific order:
1. {"role":"system","content":SYSTEM_PROMPT} — always first
2. {"role":"system","content":GREETING_PERSONALIZATION_FRAME.format(name=greeting_name)} — only if greeting_name is set
3. {"role":"system","content":PROFILE_CONTEXT_FRAME.format(profile=profile_context)} — only if profile_context is non-empty
4. {"role":"system","content":RAG_CONTEXT_FRAME.format(context=rag_context)} — only if rag_context is non-empty
5. {"role":"system","content":PRIVACY_QA_FRAME} — only if privacy_mode=True
   OR {"role":"system","content":QA_CONTEXT_FRAME.format(qa=similar_qa_context)} — only if similar_qa_context is non-empty and not privacy_mode
6. All messages from curated_history (or chat_history if no curated) — each is {"role":"user/assistant","content":"..."}
7. {"role":"user","content":user_query} — the current message, always last

## llm/generators.py — Response Generation

generate_response(...) → str: Calls build_messages() then completion(). On exception, returns a user-friendly error string.

generate_response_stream(...) → Generator[str]: Calls build_messages() then stream_text_deltas(). Yields formatted SSE lines: "0:\"token\"" for each delta, then "e:{...}" and "d:{...}" finish events. On exception, yields an error delta and error finish events.

generate_title(user_message) → str: Builds a minimal messages array with TITLE_PROMPT + user message, calls completion() with temperature=0.5 and MAX_TITLE_TOKENS=20. Returns the cleaned title (max 50 chars, stripping quotes). On exception, returns the first 5 words of the message.

## llm/profile_detector.py — Profile Extraction

detect_profile_updates(user_message, assistant_response) → list[dict]

First does a pre-check: tests whether the user message starts with any of the PROFILE_STATEMENT_PREFIXES or contains "my name is", "i am", etc. from PERSONAL_SIGNALS. If no signals found, returns [] immediately without calling the LLM (saves API call for 99% of messages).

If signals found, calls completion() with PROFILE_DETECT_PROMPT + user_message + assistant_response, temperature=0.0, MAX_PROFILE_DETECT_TOKENS=300. Parses the JSON array response. Returns list of {"key", "value", "category"} dicts. On any error (JSON parse, API error), returns [].

Called asynchronously in persist_after_response() via worker.submit(). The profile detection result is passed to query_db.update_profile_entry() for each extracted fact.

## llm/__init__.py — Barrel Exports

Re-exports the most commonly used functions for backward compatibility and convenience:
- classify_intent from .classifier
- generate_response, generate_response_stream, generate_title from .generators
- detect_profile_updates from .profile_detector
