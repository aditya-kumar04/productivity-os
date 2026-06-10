"""
agents/email_agent.py — Email Sub-Agent

Receives a SubTask from the orchestrator, calls Gmail tools via
Claude's tool-use API, and returns an AgentResult.

Pattern: ReAct loop — Claude reasons → picks a tool → gets result →
         reasons again → until it has enough info to answer.
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic

from state import AgentState, AgentResult
from tools.gmail_tools import list_recent_emails, get_email_body, draft_email, send_email

# ── Tool definitions (Claude tool-use schema) ─────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "list_recent_emails",
        "description": "List recent emails from Gmail. Use Gmail search syntax in the query param.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Max emails to return (default 10)"},
                "query": {"type": "string", "description": "Gmail search query, e.g. 'from:boss subject:meeting'"},
            },
        },
    },
    {
        "name": "get_email_body",
        "description": "Get the full body of a specific email by its message ID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "message_id": {"type": "string", "description": "Gmail message ID"},
            },
            "required": ["message_id"],
        },
    },
    {
        "name": "draft_email",
        "description": "Create a Gmail draft for the user to review before sending.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "send_email",
        "description": "Send an email immediately. Only use when user explicitly asked to send.",
        "input_schema": {
            "type": "object",
            "properties": {
                "to": {"type": "string"},
                "subject": {"type": "string"},
                "body": {"type": "string"},
            },
            "required": ["to", "subject", "body"],
        },
    },
]

# ── Tool dispatcher ────────────────────────────────────────────────────────────

def _dispatch_tool(name: str, inputs: dict) -> Any:
    dispatch = {
        "list_recent_emails": lambda i: list_recent_emails(**i),
        "get_email_body": lambda i: get_email_body(**i),
        "draft_email": lambda i: draft_email(**i),
        "send_email": lambda i: send_email(**i),
    }
    fn = dispatch.get(name)
    if not fn:
        return {"error": f"Unknown tool: {name}"}
    return fn(inputs)


# ── Agent node ────────────────────────────────────────────────────────────────

def email_agent_node(state: AgentState) -> dict:
    """
    LangGraph node.  Finds the email sub-task in state.plan,
    runs a ReAct loop with Claude, and appends an AgentResult.
    """
    # Find our task
    task = next((t for t in state.get("plan", []) if t.agent == "email"), None)
    if not task:
        return {}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are the Email Agent inside a multi-agent productivity system.
Your job: complete the given email task using the provided tools.
- Always list emails first to understand context before drafting replies.
- Never send emails unless explicitly instructed; prefer drafting.
- Be concise in your final summary — the orchestrator will synthesise everything.
- When done, respond with a plain text summary of what you did and found."""

    messages = [
        {
            "role": "user",
            "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}",
        }
    ]

    # ── ReAct loop: max 5 rounds to prevent runaway ────────────────────────────
    final_text = ""
    for _ in range(5):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system_prompt,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text blocks
        text_blocks = [b.text for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if text_blocks:
            final_text = " ".join(text_blocks)

        # If no tool calls, Claude is done
        if not tool_uses or response.stop_reason == "end_turn":
            break

        # Append assistant message and all tool results
        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tu in tool_uses:
            result = _dispatch_tool(tu.name, tu.input)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": json.dumps(result),
            })

        messages.append({"role": "user", "content": tool_results})

    # Append result to state
    result = AgentResult(
        agent="email",
        success=True,
        output=final_text or "Email task completed.",
        data={"messages_processed": len(messages)},
    )
    return {"results": state.get("results", []) + [result]}
