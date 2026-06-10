"""
tools/gmail_tools.py — Gmail API wrappers for the Email Agent.

Handles OAuth token refresh automatically.
Returns plain dicts so they're easy to pass to Claude as tool results.
"""

from __future__ import annotations

import base64
import os
from email.mime.text import MIMEText
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/gmail.compose",
]


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_service():
    """Return an authenticated Gmail service, refreshing token if needed."""
    token_path = os.getenv("GOOGLE_TOKEN_PATH", "./config/google_token.json")
    creds = None

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                os.getenv("GOOGLE_CLIENT_SECRETS", "./config/client_secrets.json"),
                SCOPES,
            )
            creds = flow.run_local_server(port=0)
        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, "w") as f:
            f.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ── Tool functions ─────────────────────────────────────────────────────────────

def list_recent_emails(max_results: int = 10, query: str = "") -> list[dict[str, Any]]:
    """
    List recent emails.  `query` accepts Gmail search syntax,
    e.g. 'from:boss@example.com after:2024/01/01 subject:meeting'.
    """
    service = _get_service()
    results = service.users().messages().list(
        userId="me",
        maxResults=max_results,
        q=query,
    ).execute()

    messages = results.get("messages", [])
    emails = []
    for msg in messages:
        detail = service.users().messages().get(
            userId="me", id=msg["id"], format="metadata",
            metadataHeaders=["Subject", "From", "Date"],
        ).execute()
        headers = {h["name"]: h["value"] for h in detail["payload"]["headers"]}
        snippet = detail.get("snippet", "")
        emails.append({
            "id": msg["id"],
            "subject": headers.get("Subject", "(no subject)"),
            "from": headers.get("From", ""),
            "date": headers.get("Date", ""),
            "snippet": snippet,
        })
    return emails


def get_email_body(message_id: str) -> dict[str, Any]:
    """Fetch and decode the full body of a single email."""
    service = _get_service()
    msg = service.users().messages().get(
        userId="me", id=message_id, format="full"
    ).execute()

    payload = msg["payload"]
    body = ""

    def _extract(parts):
        nonlocal body
        for part in parts:
            if part["mimeType"] == "text/plain" and "data" in part.get("body", {}):
                body += base64.urlsafe_b64decode(part["body"]["data"]).decode("utf-8")
            if "parts" in part:
                _extract(part["parts"])

    if "parts" in payload:
        _extract(payload["parts"])
    elif "body" in payload and "data" in payload["body"]:
        body = base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8")

    headers = {h["name"]: h["value"] for h in payload["headers"]}
    return {
        "id": message_id,
        "subject": headers.get("Subject", ""),
        "from": headers.get("From", ""),
        "date": headers.get("Date", ""),
        "body": body.strip(),
    }


def draft_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Create a Gmail draft (does NOT send — user reviews first)."""
    service = _get_service()
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    draft = service.users().drafts().create(
        userId="me", body={"message": {"raw": raw}}
    ).execute()
    return {"draft_id": draft["id"], "to": to, "subject": subject}


def send_email(to: str, subject: str, body: str) -> dict[str, Any]:
    """Send an email immediately."""
    service = _get_service()
    mime = MIMEText(body)
    mime["to"] = to
    mime["subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    sent = service.users().messages().send(
        userId="me", body={"raw": raw}
    ).execute()
    return {"message_id": sent["id"], "status": "sent"}
