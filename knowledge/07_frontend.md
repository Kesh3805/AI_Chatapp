# Frontend Architecture — React + Vite + Tailwind + AI SDK

## Stack

React 18, Vite (build tool and dev server), Tailwind CSS v3 (dark theme), Zustand (state management), Vercel AI SDK (useChat hook for streaming), Lucide React (icons).

Dev server: port 5173, proxies all /chat, /conversations, /profile, /health requests to FastAPI on port 8000 (configured in vite.config.ts). Production: npm run build outputs to frontend/dist/, served by FastAPI's static file mount.

## Main Files

**frontend/src/main.tsx**: React entry point. Mounts App component to #root.

**frontend/src/App.tsx**: Root shell component. Renders Sidebar + ChatArea side by side. Manages command palette open state. Renders ProfileModal when profile button is clicked.

**frontend/src/api.ts**: Typed fetch wrappers for all backend endpoints. Functions: fetchConversations(), createConversation(), deleteConversation(), renameConversation(), getMessages(), getProfile(), addProfileEntry(), deleteProfileEntry(). All return typed response objects.

**frontend/src/store.ts**: Zustand store with these slices:
- conversations: list of conversation objects
- activeConversationId: currently selected conversation UUID
- profile: list of profile entries
- debugMode: boolean toggle for Debug Mode
- commandPaletteOpen: boolean
- Actions: setConversations, addConversation, removeConversation, setActiveConversation, setProfile, toggleDebug, openCommandPalette, closeCommandPalette

**frontend/src/hooks/use-chat-stream.ts**: Custom hook wrapping Vercel AI SDK useChat. Transforms the request body from the default {messages} format to our backend format {user_query, conversation_id}. Uses streamProtocol: "data" to parse the Vercel AI SDK data stream lines. Returns messages, input, handleInputChange, handleSubmit, isLoading, stop, setMessages.

## Component Tree

```
App
├── Sidebar
│   ├── Conversation list items (with category icons)
│   ├── Debug Mode toggle button
│   ├── Command Palette shortcut hint (Ctrl+K)
│   └── Profile button
├── ChatArea
│   ├── Header bar (with Debug Mode toggle)
│   ├── Messages container
│   │   └── AIMessage (for each assistant message)
│   │       ├── AIStatusBar (pipeline timeline chips)
│   │       ├── AIIntentBadge (intent + confidence)
│   │       ├── AIRetrievalPanel (expandable retrieval info)
│   │       ├── AITokenMeter (context usage bar)
│   │       └── AIDebugPanel (raw JSON, Debug Mode only)
│   ├── WelcomeScreen (when no messages)
│   └── InputArea (textarea + send/stop)
├── ProfileModal (when profile button clicked)
└── CommandPalette (when Ctrl+K pressed)
```

## AI-Native Components (frontend/src/components/ai/)

These components make invisible pipeline decisions visible in the UI.

**AIIntentBadge**: Displays the intent label and confidence score with a colored dot. Green dot for confidence ≥0.85, yellow for 0.6-0.84, gray for <0.6. Labels shown: general, knowledge_base, continuation, profile, privacy.

**AIStatusBar**: Horizontal chip timeline above assistant messages. Renders chips for each stage annotation received via message.annotations. Chips: "Classified: intent" (always shown), "Retrieved: N docs" (if num_docs), "Similar Q&A: N" (if similar_queries), "Same-conv Q&A: N" (if same_conv_qa), "Topic: X.XX" (if topic_similarity), "Profile" (if profile_injected), "Generating..." or "Complete". Clickable chips may expand the retrieval panel.

**AIRetrievalPanel**: Expandable drawer below the AIStatusBar. Shows detailed retrieval breakdown: document source count from num_docs, Q&A match count from similar_queries, whether profile was injected, topic similarity score with a color scale. Triggered by clicking "Details" or a retrieval chip.

**AITokenMeter**: Visual progress bar showing estimated token usage. Estimates: 1 token ≈ 4 characters. Context total from settings.MAX_CONTEXT_WINDOW (65536 for Cerebras). Bar color: green <50%, yellow 50-80%, red >80%.

**AIDebugPanel**: Raw JSON inspector. Only visible when debugMode=true in Zustand store. Shows expandable sections: PolicyDecision fields (inject_rag, inject_profile, privacy_mode, retrieval_route, greeting_name), retrieval_info object, query_tags array. Implemented as collapsible JSON viewer.

## Key Design Decisions

**useChat with custom fetch**: The Vercel AI SDK useChat hook handles streaming state, error handling, and message accumulation. We intercept the fetch call to transform the request body format and route to /chat/stream.

**streamProtocol: "data"**: Our backend emits the Vercel AI SDK data stream protocol (0:, 8:, e:, d: lines), so the AI SDK parses these correctly. Stage annotations arrive in message.annotations as the stream progresses.

**Stage annotation parsing**: In AIMessage, message.annotations is an array that grows as annotations are received during streaming. The component separates annotations with a "stage" field (for AIStatusBar live updates) from the final metadata annotation with an "intent" field (for AIIntentBadge).

**Zustand over Redux/Context**: Minimal boilerplate. Store is accessed directly in any component. debugMode toggle propagates to every AIMessage without prop drilling.

**Dark theme only**: Tailwind configured with dark mode class strategy. The root html element always has "dark" class. The entire UI uses slate/zinc color palette with custom CSS variables for accent colors.

## Sidebar Intelligence — Category Icons

The sidebar detects conversation topics from titles using regex patterns and renders category icons:
- Brain icon: ML, AI, model, neural, machine learning topics
- Shield icon: privacy, security, auth topics
- Database icon: SQL, database, query, postgres topics
- Code icon: Python, JavaScript, programming, function, code topics
- BookOpen icon: RAG, embedding, vector, retrieval topics
- Globe icon: web, deploy, Docker, nginx topics
- HelpCircle icon: fallback for all other topics

## Debug Mode

Toggled via the sidebar footer button or the chat header toggle. When active: a "ON" badge appears in the sidebar and header, every AIMessage renders an AIDebugPanel showing the raw PolicyDecision JSON and retrieval_info. Debug mode is purely client-side — no API calls change when it's enabled.

## Command Palette (Ctrl+K)

Triggered by keyboard shortcut or sidebar button. Opens a centered modal with a search input. Commands: New Chat (creates a new conversation and makes it active), View/Edit Profile (opens ProfileModal), Toggle Debug Mode, recent conversation entries (switch to a conversation). Features fuzzy text filtering as you type, arrow key navigation, Enter to execute, Escape to close.

## Development Setup

cd frontend && npm install && npm run dev — starts Vite on port 5173 with hot module replacement and proxy to backend on 8000.

npm run build — compiles TypeScript, bundles with Vite, outputs to frontend/dist/. FastAPI serves this automatically.

npx tsc --noEmit — TypeScript type check without building.

The vite.config.ts configures a server.proxy to forward all API paths to localhost:8000, so the frontend dev server can call the backend without CORS issues during development.
