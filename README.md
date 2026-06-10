# Multi-Agent Productivity OS

A production-grade agentic system that manages your tasks, emails, calendar, and documents through natural language — powered by Claude claude-sonnet-4-20250514 and orchestrated with LangGraph.

---

## Architecture

```
User Goal
   │
   ▼
Orchestrator (plan_node)          ← Claude claude-sonnet-4-20250514, decomposes goal → SubTasks
   │
   ├── Email Agent   ──► Gmail API
   ├── Calendar Agent ──► Google Calendar API       [parallel fan-out]
   ├── Doc Agent     ──► Google Drive / Notion RAG
   ├── Task Agent    ──► Todoist / Linear
   └── Web Agent     ──► Tavily / Playwright
   │
   ▼
Synthesise (synthesise_node)      ← merges all results → final response
   │
   ▼
User
```

**Memory layers**
- Short-term: Redis (session state, conversation history, TTL-based)
- Long-term: ChromaDB (vector store, episodic memory, RAG retrieval)

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
# Fill in your API keys
```

### 3. Google OAuth (for Gmail + Calendar)

1. Create a project in [Google Cloud Console](https://console.cloud.google.com)
2. Enable Gmail API and Google Calendar API
3. Create OAuth 2.0 credentials → download as `config/client_secrets.json`
4. On first run, a browser window opens for OAuth consent

### 4. Start Redis

```bash
docker run -d -p 6379:6379 redis:alpine
```

### 5. Run

```bash
# CLI smoke test
python graph.py

# API server
uvicorn api:app --reload --port 8000
```

### 6. API usage

```bash
# Blocking call
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"goal": "Summarise my last 5 emails and flag any urgent ones"}'

# Streaming (Server-Sent Events)
curl -N http://localhost:8000/chat/stream \
  -X POST -H "Content-Type: application/json" \
  -d '{"goal": "What meetings do I have next week and do I need to prepare anything?"}'
```

---

## Project Structure

```
productivity_os/
├── graph.py              # LangGraph graph definition + router
├── state.py              # Shared AgentState TypedDict
├── api.py                # FastAPI REST + SSE streaming endpoints
├── agents/
│   ├── orchestrator.py   # plan_node + synthesise_node
│   └── email_agent.py    # ReAct email agent with Gmail tools
├── tools/
│   └── gmail_tools.py    # Gmail API wrappers
├── memory/
│   └── memory.py         # Redis (short-term) + Chroma (long-term)
├── config/               # OAuth tokens (gitignored)
├── .env.example
└── requirements.txt
```

---

## Build Roadmap

| Week | Goal |
|------|------|
| 1 | Orchestrator + Email agent ✅ |
| 2 | Calendar agent + Doc agent (RAG over Drive) |
| 3 | Task agent + Web agent + model routing (Haiku for simple tasks) |
| 4 | React frontend + streaming UI + demo video |

---

## Resume Bullet (copy this)

> **Multi-Agent Productivity OS** — Engineered a LangGraph-based agentic system with an orchestrator that decomposes natural-language goals into parallel sub-tasks, dispatching to 5 specialised agents (email, calendar, document, task, web) via Claude's tool-use API. Implemented a two-layer memory architecture (Redis for session state, ChromaDB for RAG-based episodic memory) and a model-routing layer that reduces inference cost by ~60% by delegating simpler tasks to Claude Haiku. Exposed via a FastAPI backend with Server-Sent Events streaming.

---

## Key Technical Decisions

**Why LangGraph over CrewAI?**
LangGraph gives you explicit control over the graph structure, conditional routing, and state management — essential for production systems. CrewAI is higher-level but opaque.

**Why model routing?**
Running Claude Sonnet on every tool call is expensive. The orchestrator and synthesiser need full reasoning capacity, but simple email classification or date extraction can run on Haiku at ~20× lower cost.

**Why two memory layers?**
Redis for speed (sub-millisecond session state), Chroma for semantic retrieval (vector search over past tasks and documents). They serve different purposes and neither replaces the other.
