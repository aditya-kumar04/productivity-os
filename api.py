"""
api.py — FastAPI backend for the Productivity OS.

Endpoints:
  POST /chat          — run a goal, return final response (blocking)
  POST /chat/stream   — stream tokens as the graph executes (SSE)
  GET  /history/{sid} — fetch session history from Redis
  GET  /health        — health check
"""

from __future__ import annotations

import uuid
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from graph import build_graph
from memory.memory import get_short_term
from state import AgentState

app = FastAPI(title="Productivity OS API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ChatRequest(BaseModel):
    goal: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    response: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """Run the full agent graph and return the synthesised response."""
    session_id = req.session_id or str(uuid.uuid4())
    graph = build_graph()

    initial_state: AgentState = {
        "messages": [],
        "user_goal": req.goal,
        "plan": [],
        "results": [],
        "final_response": "",
        "session_id": session_id,
        "memory_context": "",
    }

    # Save user message to session history
    get_short_term().append_history(session_id, "user", req.goal)

    final_state = graph.invoke(initial_state)
    return ChatResponse(session_id=session_id, response=final_state["final_response"])


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Stream intermediate agent events as Server-Sent Events.
    Each event is a JSON line: {"event": "...", "data": "..."}
    """
    session_id = req.session_id or str(uuid.uuid4())

    async def event_generator() -> AsyncGenerator[str, None]:
        graph = build_graph()
        initial_state: AgentState = {
            "messages": [],
            "user_goal": req.goal,
            "plan": [],
            "results": [],
            "final_response": "",
            "session_id": session_id,
            "memory_context": "",
        }

        import json
        async for event in graph.astream_events(initial_state, version="v2"):
            kind = event.get("event", "")
            name = event.get("name", "")

            if kind == "on_chain_start" and name in ("plan", "synthesise"):
                yield f'data: {json.dumps({"event": "node_start", "node": name})}\n\n'

            elif kind == "on_chain_end" and name == "plan":
                plan = event.get("data", {}).get("output", {}).get("plan", [])
                agents = [t.agent for t in plan] if plan else []
                yield f'data: {json.dumps({"event": "plan_ready", "agents": agents})}\n\n'

            elif kind == "on_chain_end" and name == "synthesise":
                response = event.get("data", {}).get("output", {}).get("final_response", "")
                yield f'data: {json.dumps({"event": "final_response", "data": response})}\n\n'

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return conversation history for a session."""
    history = get_short_term().get_history(session_id)
    return {"session_id": session_id, "history": history}
