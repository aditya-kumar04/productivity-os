"""
agents/calendar_agent.py — Calendar Sub-Agent

Same ReAct pattern as the email agent — Claude reasons, picks a
calendar tool, gets the result, reasons again until done.
"""

from __future__ import annotations

import json
import os

import anthropic

from state import AgentState, AgentResult
from tools.calendar_tools import (
    list_upcoming_events,
    find_free_slots,
    create_event,
    get_event_details,
)

TOOLS: list[dict] = [
    {
        "name": "list_upcoming_events",
        "description": "List upcoming calendar events. Specify how many days ahead to look.",
        "input_schema": {
            "type": "object",
            "properties": {
                "days_ahead": {"type": "integer", "description": "Number of days ahead to look (default 7)"},
                "max_results": {"type": "integer", "description": "Max events to return (default 20)"},
            },
        },
    },
    {
        "name": "find_free_slots",
        "description": "Find free time slots on a specific date during working hours (9am–6pm).",
        "input_schema": {
            "type": "object",
            "properties": {
                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                "duration_minutes": {"type": "integer", "description": "Minimum slot duration in minutes (default 60)"},
            },
            "required": ["date"],
        },
    },
    {
        "name": "create_event",
        "description": "Create a new calendar event.",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "start": {"type": "string", "description": "ISO 8601 datetime"},
                "end": {"type": "string", "description": "ISO 8601 datetime"},
                "attendees": {"type": "array", "items": {"type": "string"}, "description": "List of email addresses"},
                "description": {"type": "string"},
            },
            "required": ["title", "start", "end"],
        },
    },
    {
        "name": "get_event_details",
        "description": "Get full details of a specific event by its ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "event_id": {"type": "string"},
            },
            "required": ["event_id"],
        },
    },
]


def _dispatch(name: str, inputs: dict):
    return {
        "list_upcoming_events": lambda i: list_upcoming_events(**i),
        "find_free_slots": lambda i: find_free_slots(**i),
        "create_event": lambda i: create_event(**i),
        "get_event_details": lambda i: get_event_details(**i),
    }[name](inputs)


def calendar_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "calendar"), None)
    if not task:
        return {}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are the Calendar Agent inside a multi-agent productivity system.
Complete the given calendar task using the provided tools.
- When asked to schedule something, first check free slots before creating an event.
- Be specific about times and dates in your summary.
- Never create events without being asked — prefer listing and summarising."""

    messages = [{"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"}]
    final_text = ""

    for _ in range(5):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # ← model routing: simpler tasks use Haiku
            max_tokens=1000,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]
        if text_blocks:
            final_text = " ".join(text_blocks)
        if not tool_uses or response.stop_reason == "end_turn":
            break

        messages.append({"role": "assistant", "content": response.content})
        tool_results = [
            {"type": "tool_result", "tool_use_id": tu.id, "content": json.dumps(_dispatch(tu.name, tu.input))}
            for tu in tool_uses
        ]
        messages.append({"role": "user", "content": tool_results})

    result = AgentResult(agent="calendar", success=True, output=final_text or "Calendar task completed.")
    return {"results": state.get("results", []) + [result]}
