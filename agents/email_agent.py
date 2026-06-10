"""
agents/email_agent.py — Email Sub-Agent (Groq backend)

Groq doesn't support native tool-use the same way as Claude, so we use
a prompted ReAct loop: model outputs JSON with {thought, tool, args} or
{thought, answer} and we parse + dispatch manually.
"""

from __future__ import annotations

import json
import os
from typing import Any

from groq import Groq

from state import AgentState, AgentResult
from tools.gmail_tools import list_recent_emails, get_email_body, draft_email, send_email

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

TOOLS_DESC = """You have access to these tools:
- list_recent_emails(max_results: int, query: str) — list emails from Gmail
- get_email_body(message_id: str) — get full body of an email
- draft_email(to: str, subject: str, body: str) — create a draft
- send_email(to: str, subject: str, body: str) — send immediately

To use a tool respond with JSON only:
{"thought": "...", "tool": "tool_name", "args": {...}}

When done respond with JSON only:
{"thought": "...", "answer": "your summary"}"""

def _dispatch(name: str, args: dict) -> Any:
    dispatch = {
        "list_recent_emails": lambda i: list_recent_emails(**i),
        "get_email_body": lambda i: get_email_body(**i),
        "draft_email": lambda i: draft_email(**i),
        "send_email": lambda i: send_email(**i),
    }
    fn = dispatch.get(name)
    return fn(args) if fn else {"error": f"Unknown tool: {name}"}


def email_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "email"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Email Agent inside a multi-agent productivity system.
Complete the given email task using the provided tools.
- Always list emails first to understand context before drafting replies.
- Never send emails unless explicitly instructed; prefer drafting.
- Be concise in your final answer.

{TOOLS_DESC}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"},
    ]

    final_text = ""
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences
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

    result = AgentResult(
        agent="email",
        success=True,
        output=final_text or "Email task completed.",
    )
    return {"results": state.get("results", []) + [result]}
