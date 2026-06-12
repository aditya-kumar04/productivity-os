"""
agents/doc_agent.py — Document Sub-Agent (Groq backend)
"""

from __future__ import annotations

import json
import os

from groq import Groq

from state import AgentState, AgentResult
from tools.drive_rag import list_drive_files, ingest_file, ingest_all_docs, search_docs

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.1-8b-instant"

TOOLS_DESC = """You have access to these tools:
- search_docs(query: str, n_results: int) — semantic search over Drive docs
- list_drive_files(max_results: int, query: str)
- ingest_file(file_id: str, file_name: str, mime_type: str)
- ingest_all_docs(max_files: int)

To use a tool respond with JSON only:
{"thought": "...", "tool": "tool_name", "args": {...}}

When done respond with JSON only:
{"thought": "...", "answer": "your summary"}"""


def _dispatch(name: str, args: dict):
    return {
        "search_docs": lambda i: search_docs(**i),
        "list_drive_files": lambda i: list_drive_files(**i),
        "ingest_file": lambda i: ingest_file(**i),
        "ingest_all_docs": lambda i: ingest_all_docs(**i),
    }[name](args)


def doc_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "doc"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Document Agent inside a multi-agent productivity system.
You have access to the user's Google Drive documents via semantic search (RAG).
- Always call search_docs first before answering any content question.
- If search returns no results, call ingest_all_docs then search again.
- Cite the document name when referencing specific content.

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
            try:
                tool_result = _dispatch(parsed["tool"], parsed.get("args", {}))
            except Exception as e:
                tool_result = {"error": str(e)}
                final_text = f"Could not complete task: {e}"
                break

            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result)}"})

    result = AgentResult(agent="doc", success=True, output=final_text or "Document task completed.")
    return {"results": state.get("results", []) + [result]}
