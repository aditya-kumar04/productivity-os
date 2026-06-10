"""
agents/web_agent.py — Web Search Agent (Tavily + Groq)
"""

from __future__ import annotations

import json
import os

from groq import Groq
from tavily import TavilyClient

from state import AgentState, AgentResult

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

TOOLS_DESC = """You have access to these tools:
- search_web(query: str, max_results: int) — search the web for current information

To use a tool respond with JSON only:
{"thought": "...", "tool": "search_web", "args": {"query": "...", "max_results": 5}}

When done respond with JSON only:
{"thought": "...", "answer": "your summary"}"""


def _search_web(query: str, max_results: int = 5) -> dict:
    try:
        results = tavily_client.search(query=query, max_results=max_results)
        return {
            "results": [
                {"title": r.get("title"), "url": r.get("url"), "content": r.get("content", "")[:500]}
                for r in results.get("results", [])
            ]
        }
    except Exception as e:
        return {"error": str(e)}


def web_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "web"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Web Search Agent inside a multi-agent productivity system.
Use the search tool to find current, accurate information from the web.
- Always search before answering — never rely on your training data for current events.
- Cite sources (title + URL) in your summary.
- Be concise and factual.

{TOOLS_DESC}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"},
    ]

    final_text = ""
    for _ in range(5):
        response = groq_client.chat.completions.create(
            model=MODEL,
            max_tokens=1500,
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

        if "tool" in parsed and parsed["tool"] == "search_web":
            args = parsed.get("args", {})
            tool_result = _search_web(**args)
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Search results: {json.dumps(tool_result)}"})

    result = AgentResult(agent="web", success=True, output=final_text or "Web search completed.")
    return {"results": state.get("results", []) + [result]}
