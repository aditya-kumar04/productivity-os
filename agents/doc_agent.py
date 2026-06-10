"""
agents/doc_agent.py — Document Sub-Agent

Uses RAG over Google Drive to answer questions about documents,
summarise content, and extract key information.

Model routing: uses Claude Haiku for simple retrieval + summarisation
to keep costs low — Sonnet is reserved for the orchestrator.
"""

from __future__ import annotations

import json
import os

import anthropic

from state import AgentState, AgentResult
from tools.drive_rag import list_drive_files, ingest_file, ingest_all_docs, search_docs

TOOLS: list[dict] = [
    {
        "name": "search_docs",
        "description": "Semantic search over all ingested Google Drive documents. Use this first for any document question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "n_results": {"type": "integer", "description": "Number of results to return (default 5)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_drive_files",
        "description": "List files available in Google Drive.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_results": {"type": "integer", "description": "Max files to list (default 20)"},
                "query": {"type": "string", "description": "Drive search query, e.g. name contains 'Q3'"},
            },
        },
    },
    {
        "name": "ingest_file",
        "description": "Ingest a specific file into the vector store by file ID and name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_id": {"type": "string"},
                "file_name": {"type": "string"},
                "mime_type": {"type": "string", "description": "e.g. application/vnd.google-apps.document"},
            },
            "required": ["file_id", "file_name", "mime_type"],
        },
    },
    {
        "name": "ingest_all_docs",
        "description": "Ingest all Google Docs from Drive into the vector store. Use when the user says 'index my docs' or search returns no results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "max_files": {"type": "integer", "description": "Max files to ingest (default 30)"},
            },
        },
    },
]


def _dispatch(name: str, inputs: dict):
    return {
        "search_docs": lambda i: search_docs(**i),
        "list_drive_files": lambda i: list_drive_files(**i),
        "ingest_file": lambda i: ingest_file(**i),
        "ingest_all_docs": lambda i: ingest_all_docs(**i),
    }[name](inputs)


def doc_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "doc"), None)
    if not task:
        return {}

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    system_prompt = """You are the Document Agent inside a multi-agent productivity system.
You have access to the user's Google Drive documents via semantic search (RAG).

Guidelines:
- Always call search_docs first before answering any content question.
- If search returns no results, call ingest_all_docs then search again.
- Cite the document name when referencing specific content.
- Summarise clearly — the orchestrator will combine your output with other agents."""

    messages = [{"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"}]
    final_text = ""

    for _ in range(5):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",   # model routing: Haiku for doc retrieval
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

    result = AgentResult(agent="doc", success=True, output=final_text or "Document task completed.")
    return {"results": state.get("results", []) + [result]}
