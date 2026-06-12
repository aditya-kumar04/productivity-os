"""
tools/drive_rag.py — Google Drive ingestion + RAG pipeline.

Flow:
  1. List files from Google Drive
  2. Export/download content
  3. Chunk text
  4. Embed + store in ChromaDB
  5. Semantic search (used at query time)
"""

from __future__ import annotations

import io
import os
import re
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
import chromadb
from chromadb.utils import embedding_functions

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]
COLLECTION_NAME = "drive_documents"
CHUNK_SIZE = 800       # characters per chunk
CHUNK_OVERLAP = 100    # overlap between consecutive chunks


# ── Auth ──────────────────────────────────────────────────────────────────────

def _get_drive_service():
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
    return build("drive", "v3", credentials=creds)


# ── ChromaDB collection ───────────────────────────────────────────────────────

def _get_collection():
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    client = chromadb.PersistentClient(path=persist_dir)
    api_key = os.getenv("OPENAI_API_KEY")
    ef = (
        embedding_functions.OpenAIEmbeddingFunction(api_key=api_key, model_name="text-embedding-3-small")
        if api_key
        else embedding_functions.DefaultEmbeddingFunction()
    )
    return client.get_or_create_collection(COLLECTION_NAME, embedding_function=ef)


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk_text(text: str, doc_id: str, doc_name: str) -> list[dict]:
    """Split text into overlapping chunks with metadata."""
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    chunks = []
    start = 0
    idx = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end]
        chunks.append({
            "id": f"{doc_id}_chunk_{idx}",
            "text": chunk,
            "metadata": {"doc_id": doc_id, "doc_name": doc_name, "chunk_idx": idx},
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP
        idx += 1
    return chunks


# ── Ingestion ─────────────────────────────────────────────────────────────────

def list_drive_files(max_results: int = 20, query: str = "") -> list[dict[str, Any]]:
    """List files in Google Drive. query uses Drive search syntax."""
    service = _get_drive_service()
    q = "trashed=false"
    if query:
        q += f" and {query}"
    results = service.files().list(
        q=q,
        pageSize=max_results,
        fields="files(id, name, mimeType, modifiedTime)",
    ).execute()
    return results.get("files", [])


def ingest_file(file_id: str, file_name: str, mime_type: str) -> dict[str, Any]:
    """
    Download/export a Drive file, chunk it, and upsert into ChromaDB.
    Supports Google Docs (exported as plain text) and plain text/PDF files.
    """
    service = _get_drive_service()
    collection = _get_collection()

    # Export Google Docs as plain text; download others
    if mime_type == "application/vnd.google-apps.document":
        content = service.files().export(fileId=file_id, mimeType="text/plain").execute()
        text = content.decode("utf-8") if isinstance(content, bytes) else content
    else:
        request = service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        text = buf.getvalue().decode("utf-8", errors="ignore")

    chunks = _chunk_text(text, file_id, file_name)
    if not chunks:
        return {"file_id": file_id, "chunks_ingested": 0}

    collection.upsert(
        ids=[c["id"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )
    return {"file_id": file_id, "file_name": file_name, "chunks_ingested": len(chunks)}


def ingest_all_docs(max_files: int = 30) -> dict[str, Any]:
    """Ingest all Google Docs from Drive into the vector store."""
    files = list_drive_files(
        max_results=max_files,
        query="mimeType='application/vnd.google-apps.document'",
    )
    results = [ingest_file(f["id"], f["name"], f["mimeType"]) for f in files]
    total_chunks = sum(r["chunks_ingested"] for r in results)
    return {"files_ingested": len(results), "total_chunks": total_chunks, "details": results}


# ── Retrieval ─────────────────────────────────────────────────────────────────

def search_docs(query: str, n_results: int = 5) -> list[dict[str, Any]]:
    """Semantic search over ingested Drive documents."""
    collection = _get_collection()
    count = collection.count()
    if count == 0:
        return [{"text": "No documents ingested yet. Run ingest_all_docs() first.", "doc_name": "", "score": 0}]

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, count),
    )
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    return [
        {
            "text": doc,
            "doc_name": meta.get("doc_name", ""),
            "chunk_idx": meta.get("chunk_idx", 0),
            "relevance_score": round(1 - dist, 3),
        }
        for doc, meta, dist in zip(docs, metas, distances)
    ]
