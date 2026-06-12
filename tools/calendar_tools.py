"""
tools/calendar_tools.py — Google Calendar API wrappers for the Calendar Agent.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]


def _get_service():
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "./config/google_token.json")
    creds = None
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.getenv("GOOGLE_CLIENT_SECRETS", "./config/client_secrets.json"), SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as f:
            f.write(creds.to_json())
    return build("calendar", "v3", credentials=creds)


def list_upcoming_events(days_ahead: int = 7, max_results: int = 20) -> list[dict[str, Any]]:
    """List upcoming calendar events within the next N days."""
    service = _get_service()
    now = datetime.now(timezone.utc).isoformat()
    from datetime import timedelta
    end = (datetime.now(timezone.utc) + timedelta(days=days_ahead)).isoformat()

    result = service.events().list(
        calendarId="primary",
        timeMin=now,
        timeMax=end,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()

    events = []
    for e in result.get("items", []):
        start = e["start"].get("dateTime", e["start"].get("date", ""))
        end_t = e["end"].get("dateTime", e["end"].get("date", ""))
        events.append({
            "id": e["id"],
            "title": e.get("summary", "(no title)"),
            "start": start,
            "end": end_t,
            "attendees": [a["email"] for a in e.get("attendees", [])],
            "description": e.get("description", ""),
            "location": e.get("location", ""),
        })
    return events


def find_free_slots(date: str, duration_minutes: int = 60) -> list[dict[str, Any]]:
    """
    Find free time slots on a given date (YYYY-MM-DD).
    Returns a list of available windows during working hours (9am–6pm).
    """
    from datetime import timedelta
    service = _get_service()

    day_start = datetime.fromisoformat(f"{date}T09:00:00+00:00")
    day_end = datetime.fromisoformat(f"{date}T18:00:00+00:00")

    result = service.freebusy().query(body={
        "timeMin": day_start.isoformat(),
        "timeMax": day_end.isoformat(),
        "items": [{"id": "primary"}],
    }).execute()

    busy = result["calendars"]["primary"]["busy"]
    busy_intervals = [
        (datetime.fromisoformat(b["start"]), datetime.fromisoformat(b["end"]))
        for b in busy
    ]

    # Walk the day and collect free windows >= duration_minutes
    free_slots = []
    cursor = day_start
    for b_start, b_end in sorted(busy_intervals):
        gap = (b_start - cursor).total_seconds() / 60
        if gap >= duration_minutes:
            free_slots.append({
                "start": cursor.isoformat(),
                "end": b_start.isoformat(),
                "duration_minutes": int(gap),
            })
        cursor = max(cursor, b_end)

    # Check remaining time after last busy block
    gap = (day_end - cursor).total_seconds() / 60
    if gap >= duration_minutes:
        free_slots.append({
            "start": cursor.isoformat(),
            "end": day_end.isoformat(),
            "duration_minutes": int(gap),
        })

    return free_slots


def create_event(
    title: str,
    start: str,
    end: str,
    attendees: list[str] | None = None,
    description: str = "",
) -> dict[str, Any]:
    """
    Create a calendar event.
    start / end should be ISO 8601 strings e.g. '2025-06-15T10:00:00+05:30'
    """
    service = _get_service()
    body: dict[str, Any] = {
        "summary": title,
        "start": {"dateTime": start},
        "end": {"dateTime": end},
        "description": description,
    }
    if attendees:
        body["attendees"] = [{"email": e} for e in attendees]

    event = service.events().insert(calendarId="primary", body=body).execute()
    return {"event_id": event["id"], "title": title, "start": start, "link": event.get("htmlLink", "")}


def get_event_details(event_id: str) -> dict[str, Any]:
    """Fetch full details for a single event by ID."""
    service = _get_service()
    e = service.events().get(calendarId="primary", eventId=event_id).execute()
    return {
        "id": e["id"],
        "title": e.get("summary", ""),
        "start": e["start"].get("dateTime", e["start"].get("date")),
        "end": e["end"].get("dateTime", e["end"].get("date")),
        "attendees": [a["email"] for a in e.get("attendees", [])],
        "description": e.get("description", ""),
        "location": e.get("location", ""),
    }
