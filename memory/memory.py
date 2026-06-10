"""
memory/memory.py — Two-layer memory for the Productivity OS.

  ShortTermMemory  → Redis  (fast session K/V, TTL-based)
  LongTermMemory   → Chroma (persistent vector store for RAG)
"""

from __future__ import annotations

import json
import os
from datetime import datetime

import redis
import chromadb
from chromadb.utils import embedding_functions


# ── Short-term: Redis ─────────────────────────────────────────────────────────

class ShortTermMemory:
    """Stores per-session working state in Redis with a 24-hour TTL."""

    TTL_SECONDS = 86_400  # 24 hours

    def __init__(self) -> None:
        self._client = redis.from_url(
            os.getenv("REDIS_URL", "redis://localhost:6379"),
            decode_responses=True,
        )

    def save(self, session_id: str, key: str, value: Any) -> None:
        redis_key = f"session:{session_id}:{key}"
        self._client.setex(redis_key, self.TTL_SECONDS, json.dumps(value))

    def load(self, session_id: str, key: str) -> Any | None:
        redis_key = f"session:{session_id}:{key}"
        raw = self._client.get(redis_key)
        return json.loads(raw) if raw else None

    def append_history(self, session_id: str, role: str, content: str) -> None:
        """Append a message to the session's conversation log."""
        history = self.load(session_id, "history") or []
        history.append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
        self.save(session_id, "history", history)

    def get_history(self, session_id: str) -> list[dict]:
        return self.load(session_id, "history") or []


# ── Long-term: Chroma vector store ────────────────────────────────────────────

class LongTermMemory:
    """
    Stores and retrieves episodic memories (past tasks, email summaries,
    meeting notes) using semantic search via ChromaDB.
    """

    COLLECTION_NAME = "productivity_os_memories"
    TOP_K = 5

    def __init__(self) -> None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self._client = chromadb.PersistentClient(path=persist_dir)

        # Use OpenAI embeddings if key present, else fall back to a local model
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-3-small",
            )
        else:
            ef = embedding_functions.DefaultEmbeddingFunction()

        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=ef,
        )

    def store(self, memory_id: str, text: str, metadata: dict | None = None) -> None:
        """Embed and persist a memory (e.g. a completed task summary)."""
        self._collection.upsert(
            ids=[memory_id],
            documents=[text],
            metadatas=[metadata or {}],
        )

    def retrieve(self, query: str, n_results: int = TOP_K) -> str:
        """Return the top-K relevant memories as a formatted string."""
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count() or 1),
        )
        docs = results.get("documents", [[]])[0]
        if not docs:
            return ""
        lines = [f"- {doc}" for doc in docs]
        return "Relevant past context:\n" + "\n".join(lines)


# ── Convenience singleton accessors ───────────────────────────────────────────

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
