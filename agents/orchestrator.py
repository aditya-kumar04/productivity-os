"""
agents/orchestrator.py — Orchestrator Agent

The "brain" of the system.  Given a user goal it:
  1. Retrieves relevant long-term memories
  2. Calls Claude to decompose the goal into a list of SubTasks
  3. Writes the plan into state so the router can dispatch sub-agents
  4. After all sub-agents finish, calls Claude again to synthesise the final answer
"""

from __future__ import annotations

import json
import os

import anthropic

from state import AgentState, SubTask
from memory.memory import get_long_term, get_short_term


client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── 1. Plan node ──────────────────────────────────────────────────────────────

PLAN_SYSTEM = """You are the Orchestrator of a multi-agent productivity assistant.

Available agents:
- email     : read, search, draft, or send emails via Gmail
- calendar  : list events, create meetings, find free slots
- doc       : summarise or search documents in Google Drive / Notion
- task      : create or update tasks in Todoist / Linear
- web       : search the web for current information

Given the user's goal, output a JSON object with a single key "tasks" — a list of sub-tasks.
Each sub-task has:
  - agent   : one of the agent names above
  - action  : a clear, specific instruction for that agent (1–2 sentences)
  - context : optional dict with extra parameters (e.g. date ranges, keywords)

Only include agents that are actually needed.
Respond with valid JSON only — no markdown, no explanation."""

def plan_node(state: AgentState) -> dict:
    """Decompose the user goal into a plan of sub-tasks."""
    goal = state.get("user_goal", "")
    session_id = state.get("session_id", "default")

    # Pull relevant past context from vector store
    memory_context = get_long_term().retrieve(goal)

    messages_payload = [{"role": "user", "content": goal}]
    if memory_context:
        messages_payload[0]["content"] = f"{memory_context}\n\nUser goal: {goal}"

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=PLAN_SYSTEM,
        messages=messages_payload,
    )

    raw = response.content[0].text.strip()

    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)
    tasks = [SubTask(**t) for t in parsed.get("tasks", [])]

    # Persist plan to Redis for auditability
    get_short_term().save(session_id, "plan", [t.model_dump() for t in tasks])

    return {
        "plan": tasks,
        "memory_context": memory_context,
    }


# ── 2. Synthesise node ────────────────────────────────────────────────────────

SYNTH_SYSTEM = """You are the Orchestrator of a multi-agent productivity assistant.
Sub-agents have completed their tasks. Synthesise their outputs into a single,
clear, helpful response for the user.
- Lead with the most important information.
- Bullet lists are fine for action items or email summaries.
- End with suggested next steps if relevant.
- Be concise — aim for under 300 words unless detail is necessary."""

def synthesise_node(state: AgentState) -> dict:
    """Merge all sub-agent results into a final user-facing answer."""
    goal = state.get("user_goal", "")
    results = state.get("results", [])
    session_id = state.get("session_id", "default")

    # Format sub-agent outputs
    results_text = "\n\n".join(
        f"[{r.agent.upper()} AGENT]\n{r.output}" for r in results
    )

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=SYNTH_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": f"Original goal: {goal}\n\nAgent results:\n{results_text}",
            }
        ],
    )

    final = response.content[0].text.strip()

    # Store this interaction as a long-term memory
    memory_text = f"Goal: {goal}\nOutcome: {final[:300]}"
    get_long_term().store(
        memory_id=f"{session_id}_latest",
        text=memory_text,
        metadata={"session_id": session_id, "agents_used": [r.agent for r in results]},
    )

    # Log to session history
    get_short_term().append_history(session_id, "assistant", final)

    return {"final_response": final}
