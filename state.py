"""
state.py — Shared state schema for the Multi-Agent Productivity OS.

Every node in the LangGraph reads from and writes to this TypedDict.
LangGraph merges partial updates automatically via the `add_messages` reducer.
"""

from __future__ import annotations

from typing import Annotated, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


# ── Sub-task issued by orchestrator to a sub-agent ────────────────────────────

class SubTask(BaseModel):
    agent: str = Field(description="Target agent: email | calendar | doc | task | web")
    action: str = Field(description="What the agent should do")
    context: dict[str, Any] = Field(default_factory=dict, description="Extra params")


# ── Result returned by a sub-agent ────────────────────────────────────────────

class AgentResult(BaseModel):
    agent: str
    success: bool
    output: str
    data: dict[str, Any] = Field(default_factory=dict)


# ── Main graph state ───────────────────────────────────────────────────────────

class AgentState(dict):
    """
    LangGraph state container.  Fields:

    messages      — full conversation history (auto-appended via add_messages)
    user_goal     — raw user input
    plan          — orchestrator's decomposed list of sub-tasks
    results       — collected sub-agent results
    final_response— synthesised answer shown to user
    session_id    — Redis key prefix for this session
    memory_context— relevant snippets retrieved from long-term vector store
    """

    messages: Annotated[list[BaseMessage], add_messages]
    user_goal: str
    plan: list[SubTask]
    results: list[AgentResult]
    final_response: str
    session_id: str
    memory_context: str
