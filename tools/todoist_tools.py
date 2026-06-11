"""
tools/todoist_tools.py — Todoist API tools (v1)
"""

from __future__ import annotations

import os
import requests

BASE_URL = "https://api.todoist.com/api/v1"

def _headers():
    return {"Authorization": f"Bearer {os.getenv('TODOIST_API_KEY')}"}


def get_tasks(project_id: str = None, filter: str = None) -> dict:
    params = {}
    if project_id:
        params["project_id"] = project_id
    if filter:
        params["filter"] = filter
    resp = requests.get(f"{BASE_URL}/tasks", headers=_headers(), params=params)
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    tasks = data.get("results", data) if isinstance(data, dict) else data
    return {"tasks": [{"id": t["id"], "content": t["content"], "due": t.get("due"), "priority": t.get("priority", 1)} for t in tasks]}


def create_task(content: str, due_string: str = None, priority: int = 1, project_id: str = None) -> dict:
    payload = {"content": content, "priority": priority}
    if due_string:
        payload["due_string"] = due_string
    if project_id:
        payload["project_id"] = project_id
    resp = requests.post(f"{BASE_URL}/tasks", headers=_headers(), json=payload)
    if resp.status_code not in (200, 201):
        return {"error": resp.text}
    t = resp.json()
    return {"id": t["id"], "content": t["content"], "due": t.get("due"), "url": t.get("url")}


def complete_task(task_id: str) -> dict:
    resp = requests.post(f"{BASE_URL}/tasks/{task_id}/close", headers=_headers())
    return {"success": resp.status_code in (200, 204)}


def get_projects() -> dict:
    resp = requests.get(f"{BASE_URL}/projects", headers=_headers())
    if resp.status_code != 200:
        return {"error": resp.text}
    data = resp.json()
    projects = data.get("results", data) if isinstance(data, dict) else data
    return {"projects": [{"id": p["id"], "name": p["name"]} for p in projects]}


def update_task(task_id: str, content: str = None, due_string: str = None, priority: int = None) -> dict:
    payload = {}
    if content:
        payload["content"] = content
    if due_string:
        payload["due_string"] = due_string
    if priority:
        payload["priority"] = priority
    resp = requests.post(f"{BASE_URL}/tasks/{task_id}", headers=_headers(), json=payload)
    if resp.status_code != 200:
        return {"error": resp.text}
    return {"success": True, "task": resp.json()}