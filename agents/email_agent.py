"""
agents/email_agent.py — Email Sub-Agent (Groq backend)
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


def _trim(text: str, max_chars: int = 400) -> str:
    return text[:max_chars] + "…" if len(text) > max_chars else text


def _trim_tool_result(name: str, result: Any) -> Any:
    """Truncate tool results that can blow up token counts."""
    if name == "get_email_body":
        if isinstance(result, dict) and "body" in result:
            result = {**result, "body": _trim(result["body"], 400)}
        elif isinstance(result, str):
            result = _trim(result, 400)
    elif name == "list_recent_emails":
        # Trim snippet on each email in the list
        if isinstance(result, list):
            result = [
                {**e, "snippet": _trim(e.get("snippet", ""), 150)}
                if isinstance(e, dict) else e
                for e in result[:5]   # cap at 5 emails
            ]
    return result


def _dispatch(name: str, args: dict) -> Any:
    dispatch = {
        "list_recent_emails": lambda i: list_recent_emails(**i),
        "get_email_body":     lambda i: get_email_body(**i),
        "draft_email":        lambda i: draft_email(**i),
        "send_email":         lambda i: send_email(**i),
    }
    fn = dispatch.get(name)
    raw_result = fn(args) if fn else {"error": f"Unknown tool: {name}"}
    return _trim_tool_result(name, raw_result)


def email_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "email"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Email Agent inside a multi-agent productivity system.
Complete the given email task using the provided tools.
- Always list emails first to understand context before drafting replies.
- Never send emails unless explicitly instructed; prefer drafting.
- Be concise in your final answer.
- Do NOT call get_email_body more than twice per task.
{TOOLS_DESC}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"},
    ]

    final_text = ""
    for _ in range(5):
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=800,
            messages=messages,
        )
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
            # Serialise and hard-cap the appended result string
            result_str = json.dumps(tool_result)
            if len(result_str) > 1500:
                result_str = result_str[:1500] + "…"
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {result_str}"})

    result = AgentResult(
        agent="email",
        success=True,
        output=final_text or "Email task completed.",
    )
    return {"results": state.get("results", []) + [result]}