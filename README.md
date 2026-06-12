# Multi-Agent Productivity OS

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)
![Groq](https://img.shields.io/badge/Groq-Llama--3-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A production-grade agentic system that manages your emails, calendar, documents, tasks, and web research through natural language — orchestrated with LangGraph and powered by Groq (Llama 3).

---

## Architecture

```
User Goal
   │
   ▼
Orchestrator (plan_node)          ← Llama-3.3-70b, decomposes goal → SubTasks
   │
   ├── Email Agent    ──► Gmail API
   ├── Calendar Agent ──► Google Calendar API       [parallel fan-out]
   ├── Doc Agent      ──► Google Drive + ChromaDB RAG
   ├── Task Agent     ──► Todoist API
   └── Web Agent      ──► Tavily search
   │
   ▼
Synthesise (synthesise_node)      ← merges all results → final response
   │
   ▼
User
```

**Sub-agent models**
- Orchestrator / Synthesiser / Email / Task / Web: `llama-3.3-70b-versatile` (Groq)
- Calendar / Doc: `llama-3.1-8b-instant` (Groq)

**Memory layers**
- Short-term: Redis (session state, conversation history, 24hr TTL)
- Long-term: ChromaDB (vector store, episodic memory, RAG retrieval)

---

## Project Structure

```
productivity_os/
├── graph.py              # LangGraph graph — nodes, router, fan-out/converge
├── state.py              # AgentState TypedDict, SubTask, AgentResult schemas
├── api.py                # FastAPI — POST /chat, POST /chat/stream (SSE), GET /history
├── streamlit_app.py      # Chat UI (requires API server running)
├── setup_env.py          # Environment checker — run before first use
├── requirements.txt
├── agents/
│   ├── orchestrator.py   # plan_node + synthesise_node (Groq)
│   ├── email_agent.py    # ReAct email agent — list, read, draft, send
│   ├── calendar_agent.py # ReAct calendar agent — list, find slots, create events
│   ├── doc_agent.py      # ReAct doc agent — Drive listing + ChromaDB RAG search
│   ├── task_agent.py     # ReAct task agent — Todoist get/create/update/complete
│   └── web_agent.py      # ReAct web agent — Tavily search
├── tools/
│   ├── gmail_tools.py    # Gmail API wrappers (OAuth, list/read/draft/send)
│   ├── calendar_tools.py # Google Calendar API wrappers (events, free/busy)
│   ├── drive_rag.py      # Drive ingestion + chunking + ChromaDB upsert + search
│   └── todoist_tools.py  # Todoist REST API v1 wrappers
├── memory/
│   └── memory.py         # ShortTermMemory (Redis) + LongTermMemory (Chroma)
├── config/               # OAuth tokens — gitignored
└── chroma_db/            # Persistent vector store — gitignored
```

---

## Setup

### 1. Clone and install

```bash
git clone <your-repo>
cd productivity_os
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your API keys (see table below)
```

| Variable | Required | Where to get it |
|---|---|---|
| `GROQ_API_KEY` | ✅ | [console.groq.com](https://console.groq.com) |
| `TAVILY_API_KEY` | ✅ | [app.tavily.com](https://app.tavily.com) |
| `TODOIST_API_KEY` | ✅ | Todoist → Settings → Integrations → API token |
| `ANTHROPIC_API_KEY` | Optional | Used only by `setup_env.py` connectivity test |
| `OPENAI_API_KEY` | Optional | Better Drive embeddings; falls back to local model |
| `REDIS_URL` | Optional | Defaults to `redis://localhost:6379` |
| `CHROMA_PERSIST_DIR` | Optional | Defaults to `./chroma_db` |
| `GOOGLE_CLIENT_SECRETS` | ✅ | `./config/client_secrets.json` (see step 3) |
| `GOOGLE_TOKEN_PATH` | Optional | Defaults to `./config/google_token.json` |

### 3. Google OAuth (Gmail + Calendar + Drive)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Gmail API**, **Google Calendar API**, and **Google Drive API**
3. Go to APIs & Services → Credentials → Create OAuth 2.0 Client ID (Desktop app)
4. Download the JSON → save as `config/client_secrets.json`
5. On first run a browser window opens for OAuth consent — this generates `config/google_token.json` automatically

### 4. Start Redis

```bash
docker run -d -p 6379:6379 redis:alpine
```

### 5. Verify setup

```bash
python setup_env.py
```

This checks Python version, all packages, API keys, Redis, ChromaDB, and Google OAuth in one pass.

### 6. Run

```bash
# CLI smoke test
python graph.py

# API server
uvicorn api:app --reload --port 8000

# Streamlit chat UI (separate terminal, requires API server)
streamlit run streamlit_app.py
```

---

## API

### POST `/chat` — blocking

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"goal": "Summarise my last 5 emails and flag any urgent ones"}'
```

Returns: `{"session_id": "...", "response": "..."}`

### POST `/chat/stream` — Server-Sent Events

```bash
curl -N http://localhost:8000/chat/stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"goal": "What meetings do I have next week and do I need to prepare anything?"}'
```

SSE event types: `node_start`, `plan_ready`, `final_response`, `[DONE]`

### GET `/history/{session_id}`

Fetches conversation history from Redis for a given session.

### GET `/health`

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

Pass `session_id` in the request body to continue a conversation across calls.

---

## Key Technical Decisions

**Why LangGraph over CrewAI?**
LangGraph gives explicit control over graph structure, conditional routing, and state management — essential for production systems where you need to inspect, replay, or modify execution paths. CrewAI abstracts this away.

**Why Groq instead of Claude for sub-agents?**
Sub-agents run on every tool call. Groq's inference speed (hundreds of tokens/second on Llama 3) keeps latency low while the parallel fan-out structure means all agents run concurrently. The orchestrator and synthesiser use the larger 70b model for reasoning quality; simpler calendar/doc tasks run on 8b.

**Why two memory layers?**
Redis for speed (sub-millisecond session state and conversation history) and Chroma for semantic retrieval (vector search over past goals and Drive documents). They serve different access patterns and neither replaces the other.

**Why a custom ReAct loop instead of LangChain tools?**
Each agent implements its own JSON-based tool-call loop (thought → tool → result → repeat). This keeps the agents self-contained, easy to debug (every message is a plain dict), and avoids framework-level abstractions that are hard to trace when something goes wrong.

---

## Resume Bullet (copy this)

> **Multi-Agent Productivity OS** — Built a LangGraph-based agentic system with an orchestrator that decomposes natural-language goals into parallel sub-tasks, dispatching to 5 specialised agents (email, calendar, document, task, web). Implemented a two-layer memory architecture (Redis for session state, ChromaDB for RAG-based episodic memory), a Google Drive ingestion pipeline with semantic search, and a FastAPI backend with Server-Sent Events streaming. Sub-agents run on Groq (Llama 3) for low-latency parallel execution.
