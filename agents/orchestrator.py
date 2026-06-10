"""
agents/orchestrator.py — Orchestrator Agent (Groq backend)
"""

from __future__ import annotations

import json
import os

from groq import Groq

from state import AgentState, SubTask
from memory.memory import get_long_term, get_short_term


client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

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
    goal = state.get("user_goal", "")
    session_id = state.get("session_id", "default")

    memory_context = get_long_term().retrieve(goal)

    content = f"{memory_context}\n\nUser goal: {goal}" if memory_context else goal

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": content},
        ],
    )

    raw = response.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    parsed = json.loads(raw)
    tasks = [SubTask(**t) for t in parsed.get("tasks", [])]

    get_short_term().save(session_id, "plan", [t.model_dump() for t in tasks])

    return {"plan": tasks, "memory_context": memory_context}


SYNTH_SYSTEM = """You are the Orchestrator of a multi-agent productivity assistant.
Sub-agents have completed their tasks. Synthesise their outputs into a single,
clear, helpful response for the user.
- Lead with the most important information.
- Bullet lists are fine for action items or email summaries.
- End with suggested next steps if relevant.
- Be concise — aim for under 300 words unless detail is necessary."""

def synthesise_node(state: AgentState) -> dict:
    goal = state.get("user_goal", "")
    results = state.get("results", [])
    session_id = state.get("session_id", "default")

    results_text = "\n\n".join(
        f"[{r.agent.upper()} AGENT]\n{r.output}" for r in results
    )

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": SYNTH_SYSTEM},
            {"role": "user", "content": f"Original goal: {goal}\n\nAgent results:\n{results_text}"},
        ],
    )

    final = response.choices[0].message.content.strip()

    memory_text = f"Goal: {goal}\nOutcome: {final[:300]}"
    get_long_term().store(
        memory_id=f"{session_id}_latest",
        text=memory_text,
        metadata={"session_id": session_id, "agents_used": [r.agent for r in results]},
    )

    get_short_term().append_history(session_id, "assistant", final)

    return {"final_response": final}
