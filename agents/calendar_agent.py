"""
agents/calendar_agent.py — Calendar Sub-Agent (Groq backend)
"""

from __future__ import annotations

import json
import os

from groq import Groq

from state import AgentState, AgentResult
from tools.calendar_tools import list_upcoming_events, find_free_slots, create_event, get_event_details

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama3-8b-8192"

TOOLS_DESC = """You have access to these tools:
- list_upcoming_events(days_ahead: int, max_results: int)
- find_free_slots(date: str, duration_minutes: int)
- create_event(title: str, start: str, end: str, attendees: list, description: str)
- get_event_details(event_id: str)

To use a tool respond with JSON only:
{"thought": "...", "tool": "tool_name", "args": {...}}

When done respond with JSON only:
{"thought": "...", "answer": "your summary"}"""


def _dispatch(name: str, args: dict):
    return {
        "list_upcoming_events": lambda i: list_upcoming_events(**i),
        "find_free_slots": lambda i: find_free_slots(**i),
        "create_event": lambda i: create_event(**i),
        "get_event_details": lambda i: get_event_details(**i),
    }[name](args)


def calendar_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "calendar"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Calendar Agent inside a multi-agent productivity system.
Complete the given calendar task using the provided tools.
- When asked to schedule something, first check free slots before creating an event.
- Be specific about times and dates in your summary.

{TOOLS_DESC}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"},
    ]

    final_text = ""
    for _ in range(5):
        response = client.chat.completions.create(model=MODEL, max_tokens=1000, messages=messages)
        raw = response.choices[0].message.content.strip()

        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            final_text = raw
            break

        if "answer" in parsed:
            final_text = parsed["answer"]
            break

        if "tool" in parsed:
            tool_result = _dispatch(parsed["tool"], parsed.get("args", {}))
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result)}"})

    result = AgentResult(agent="calendar", success=True, output=final_text or "Calendar task completed.")
    return {"results": state.get("results", []) + [result]}
