#!/usr/bin/env python3
"""
setup_env.py — Environment setup checker for Productivity OS.

Run this after filling in your .env to verify everything is connected.
Usage: python setup_env.py
"""

import os
import sys
import importlib
import subprocess

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BLUE   = "\033[94m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):   print(f"  {GREEN}✓{RESET}  {msg}")
def fail(msg): print(f"  {RED}✗{RESET}  {msg}")
def warn(msg): print(f"  {YELLOW}!{RESET}  {msg}")
def info(msg): print(f"  {BLUE}→{RESET}  {msg}")
def section(title): print(f"\n{BOLD}{title}{RESET}\n" + "─" * 40)


# ── 1. Python version ─────────────────────────────────────────────────────────
section("Python version")
major, minor = sys.version_info[:2]
if major == 3 and minor >= 11:
    ok(f"Python {major}.{minor} (>=3.11 required)")
else:
    fail(f"Python {major}.{minor} — upgrade to 3.11+")
    sys.exit(1)


# ── 2. .env file ──────────────────────────────────────────────────────────────
section(".env file")
if not os.path.exists(".env"):
    fail(".env not found — copy .env.example and fill it in")
    info("Run:  cp .env.example .env")
    sys.exit(1)
ok(".env file found")

# Load it
from dotenv import load_dotenv
load_dotenv()


# ── 3. Required packages ──────────────────────────────────────────────────────
section("Python packages")
REQUIRED = [
    ("anthropic",                 "anthropic"),
    ("langgraph",                 "langgraph"),
    ("langchain_anthropic",       "langchain-anthropic"),
    ("langchain_core",            "langchain-core"),
    ("redis",                     "redis"),
    ("chromadb",                  "chromadb"),
    ("google.auth",               "google-auth"),
    ("googleapiclient",           "google-api-python-client"),
    ("dotenv",                    "python-dotenv"),
    ("fastapi",                   "fastapi"),
    ("uvicorn",                   "uvicorn"),
    ("pydantic",                  "pydantic"),
    ("tavily",                    "tavily-python"),
]
missing = []
for import_name, pkg_name in REQUIRED:
    try:
        importlib.import_module(import_name)
        ok(pkg_name)
    except ImportError:
        fail(f"{pkg_name} — not installed")
        missing.append(pkg_name)

if missing:
    print()
    warn("Missing packages detected. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    ok("Packages installed")


# ── 4. API keys ───────────────────────────────────────────────────────────────
section("API keys")

anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")
if anthropic_key and anthropic_key != "your_anthropic_api_key":
    ok("ANTHROPIC_API_KEY is set")
else:
    fail("ANTHROPIC_API_KEY is missing")
    info("Get it from: https://console.anthropic.com/settings/keys")

tavily_key = os.getenv("TAVILY_API_KEY", "")
if tavily_key and tavily_key != "your_tavily_api_key":
    ok("TAVILY_API_KEY is set")
else:
    warn("TAVILY_API_KEY is missing (needed for web agent in Week 3)")
    info("Get it from: https://app.tavily.com")

openai_key = os.getenv("OPENAI_API_KEY", "")
if openai_key and openai_key != "your_openai_api_key":
    ok("OPENAI_API_KEY is set (used for embeddings)")
else:
    warn("OPENAI_API_KEY not set — will use default local embeddings (slower)")


# ── 5. Anthropic connectivity ─────────────────────────────────────────────────
section("Anthropic API connectivity")
if anthropic_key and anthropic_key != "your_anthropic_api_key":
    try:
        import anthropic as _anthropic
        client = _anthropic.Anthropic(api_key=anthropic_key)
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{"role": "user", "content": "ping"}],
        )
        ok(f"Connected — model: {resp.model}")
    except Exception as e:
        fail(f"Anthropic API error: {e}")
else:
    warn("Skipping connectivity test (no key)")


# ── 6. Redis ──────────────────────────────────────────────────────────────────
section("Redis")
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
try:
    import redis as _redis
    r = _redis.from_url(redis_url, socket_connect_timeout=2)
    r.ping()
    ok(f"Redis connected at {redis_url}")
except Exception as e:
    fail(f"Redis not reachable: {e}")
    info("Start Redis with:  docker run -d -p 6379:6379 redis:alpine")
    info("Or install locally: https://redis.io/docs/getting-started/installation/")


# ── 7. ChromaDB ───────────────────────────────────────────────────────────────
section("ChromaDB")
persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
try:
    import chromadb as _chromadb
    client = _chromadb.PersistentClient(path=persist_dir)
    col = client.get_or_create_collection("_setup_test")
    client.delete_collection("_setup_test")
    ok(f"ChromaDB ready — persisting to {persist_dir}/")
except Exception as e:
    fail(f"ChromaDB error: {e}")


# ── 8. Google OAuth ───────────────────────────────────────────────────────────
section("Google OAuth")
client_secrets = os.getenv("GOOGLE_CLIENT_SECRETS", "./config/client_secrets.json")
token_path     = os.getenv("GOOGLE_TOKEN_PATH",    "./config/google_token.json")

if os.path.exists(client_secrets):
    ok(f"client_secrets.json found at {client_secrets}")
else:
    fail("client_secrets.json not found")
    info("1. Go to https://console.cloud.google.com")
    info("2. APIs & Services → Credentials → Create OAuth 2.0 Client ID")
    info("3. Enable: Gmail API, Google Calendar API, Google Drive API")
    info(f"4. Download JSON → save as {client_secrets}")

if os.path.exists(token_path):
    ok(f"google_token.json found — already authenticated")
else:
    warn("google_token.json not found — first run will open a browser for OAuth consent")
    info("This is normal. Just run: python graph.py")


# ── 9. Summary ────────────────────────────────────────────────────────────────
section("Next steps")
info("1. Fix any ✗ items above")
info("2. Run a smoke test:  python graph.py")
info("3. Start the API:     uvicorn api:app --reload --port 8000")
info("4. Test the API:")
print()
print('     curl -X POST http://localhost:8000/chat \\')
print('       -H "Content-Type: application/json" \\')
print('       -d \'{"goal": "Summarise my last 3 emails"}\'')
print()
