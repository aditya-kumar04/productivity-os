"""
agents/task_agent.py — Task Management Agent (Todoist + Groq)
"""

from __future__ import annotations

import json
import os

from groq import Groq

from state import AgentState, AgentResult
from tools.todoist_tools import get_tasks, create_task, complete_task, get_projects, update_task

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

TOOLS_DESC = """You have access to these tools:
- get_tasks(project_id: str, filter: str) — get tasks, filter examples: "today", "overdue", "priority 1"
- create_task(content: str, due_string: str, priority: int, project_id: str) — create a task, priority 1-4 (4=urgent)
- complete_task(task_id: str) — mark a task as done
- get_projects() — list all projects
- update_task(task_id: str, content: str, due_string: str, priority: int) — update a task

To use a tool respond with JSON only:
{"thought": "...", "tool": "tool_name", "args": {...}}

When done respond with JSON only:
{"thought": "...", "answer": "your summary"}"""


def _dispatch(name: str, args: dict):
    return {
        "get_tasks": lambda i: get_tasks(**i),
        "create_task": lambda i: create_task(**i),
        "complete_task": lambda i: complete_task(**i),
        "get_projects": lambda i: get_projects(**i),
        "update_task": lambda i: update_task(**i),
    }[name](args)


def task_agent_node(state: AgentState) -> dict:
    task = next((t for t in state.get("plan", []) if t.agent == "task"), None)
    if not task:
        return {}

    system_prompt = f"""You are the Task Management Agent inside a multi-agent productivity system.
You manage the user's tasks in Todoist.
- Always get_projects first if you need to create a task in a specific project.
- Use priority 4 for urgent tasks, 3 for high, 2 for medium, 1 for low.
- When listing tasks, summarise clearly with due dates and priorities.
- Be concise in your final answer.

{TOOLS_DESC}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task.action}\nContext: {json.dumps(task.context)}"},
    ]

    final_text = ""
    for _ in range(5):
        response = client.chat.completions.create(model=MODEL, max_tokens=1500, messages=messages)
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
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(tool_result)}"})

    result = AgentResult(agent="task", success=True, output=final_text or "Task completed.")
    return {"results": state.get("results", []) + [result]}
