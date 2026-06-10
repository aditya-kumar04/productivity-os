"""
memory/memory.py — Two-layer memory for the Productivity OS.

  ShortTermMemory  → Redis  (fast session K/V, TTL-based)
  LongTermMemory   → Chroma (persistent vector store, local embeddings)
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import redis
import chromadb


# ── Short-term: Redis ─────────────────────────────────────────────────────────

class ShortTermMemory:
    TTL_SECONDS = 86_400  # 24 hours

    def __init__(self) -> None:
        self._client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )

    def save(self, session_id: str, key: str, value: Any) -> None:
        self._client.setex(f"session:{session_id}:{key}", self.TTL_SECONDS, json.dumps(value))

    def load(self, session_id: str, key: str) -> Any | None:
        raw = self._client.get(f"session:{session_id}:{key}")
        return json.loads(raw) if raw else None

    def append_history(self, session_id: str, role: str, content: str) -> None:
        history = self.load(session_id, "history") or []
        history.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
        self.save(session_id, "history", history)

    def get_history(self, session_id: str) -> list[dict]:
        return self.load(session_id, "history") or []


# ── Long-term: Chroma vector store ────────────────────────────────────────────

class LongTermMemory:
    COLLECTION_NAME = "productivity_os_memories"
    TOP_K = 5

    def __init__(self) -> None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Use ChromaDB's built-in local embedding model — no API key needed
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
        )

    def store(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        self._collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def retrieve(self, query: str, n_results: int = TOP_K) -> str:
        count = self._collection.count()
        if count == 0:
            return ""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        return "Relevant past context:\n" + "\n".join(f"- {doc}" for doc in docs)


# ── Singletons ────────────────────────────────────────────────────────────────

_short: ShortTermMemory | None = None
_long: LongTermMemory | None = None


def get_short_term() -> ShortTermMemory:
    global _short
    if _short is None:
        _short = ShortTermMemory()
    return _short


def get_long_term() -> LongTermMemory:
    global _long
    if _long is None:
        _long = LongTermMemory()
    return _long