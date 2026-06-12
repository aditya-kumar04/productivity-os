"""
streamlit_app.py — Chat UI for the Multi-Agent Productivity OS.

Run:
    streamlit run streamlit_app.py

Requires FastAPI backend running at http://localhost:8000
    uvicorn api:app --reload --port 8000
"""

import json
import time
import uuid
from datetime import datetime

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

AGENT_META = {
    "email":    {"icon": "✉️",  "color": "#4A90D9", "label": "Email"},
    "calendar": {"icon": "📅",  "color": "#7B61FF", "label": "Calendar"},
    "doc":      {"icon": "📄",  "color": "#00B4A2", "label": "Docs"},
    "task":     {"icon": "✅",  "color": "#F5A623", "label": "Tasks"},
    "web":      {"icon": "🌐",  "color": "#E86F3A", "label": "Web"},
}

EXAMPLE_GOALS = [
    "Summarise my last 5 emails and flag urgent ones",
    "What meetings do I have this week?",
    "Search for recent AI agent papers and save a summary",
    "Create a Todoist task: finish the demo video by Friday",
    "Find emails about the Q3 report and list action items",
]

# ── Page setup ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Productivity OS",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background: #0F1117;
}
[data-testid="stSidebar"] {
    background: #161B27;
    border-right: 1px solid #1E2536;
}

/* ── Header ── */
.pos-header {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 0 24px;
    border-bottom: 1px solid #1E2536;
    margin-bottom: 24px;
}
.pos-logo {
    font-size: 28px;
    font-weight: 800;
    letter-spacing: -1px;
    color: #E8EAF0;
    font-family: 'SF Pro Display', system-ui, sans-serif;
}
.pos-logo span { color: #7B61FF; }
.pos-badge {
    font-size: 11px;
    font-weight: 600;
    background: #1E2536;
    color: #7B61FF;
    padding: 3px 8px;
    border-radius: 99px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid #7B61FF33;
}

/* ── Chat messages ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 12px 0;
}
.msg-user-bubble {
    background: #7B61FF;
    color: #fff;
    padding: 12px 16px;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 15px;
    line-height: 1.5;
    font-family: system-ui, sans-serif;
}
.msg-assistant {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    margin: 12px 0;
}
.msg-avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: #1E2536;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
    border: 1px solid #2A3350;
}
.msg-assistant-bubble {
    background: #161B27;
    color: #C8D0E0;
    padding: 14px 18px;
    border-radius: 4px 18px 18px 18px;
    max-width: 80%;
    font-size: 15px;
    line-height: 1.6;
    border: 1px solid #1E2536;
    font-family: system-ui, sans-serif;
    white-space: pre-wrap;
}

/* ── Agent pipeline ── */
.pipeline-label {
    font-size: 11px;
    font-weight: 600;
    color: #4A5270;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 8px 0 6px 42px;
}
.pipeline-row {
    display: flex;
    gap: 6px;
    margin: 0 0 10px 42px;
    flex-wrap: wrap;
}
.agent-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 4px 10px;
    border-radius: 99px;
    font-size: 12px;
    font-weight: 600;
    background: #1E2536;
    border: 1px solid #2A3350;
    color: #8A96B4;
}
.agent-chip.active {
    background: #1A1F35;
    color: #fff;
    border-color: var(--chip-color);
    box-shadow: 0 0 0 1px var(--chip-color)22;
}

/* ── Thinking / streaming ── */
.thinking-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 12px 0;
    padding: 10px 16px;
    background: #161B27;
    border: 1px solid #1E2536;
    border-radius: 12px;
    color: #4A5270;
    font-size: 13px;
    font-family: system-ui, sans-serif;
}
.dot-pulse {
    display: flex; gap: 4px;
}
.dot-pulse span {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #7B61FF;
    animation: blink 1.2s infinite;
}
.dot-pulse span:nth-child(2) { animation-delay: 0.2s; }
.dot-pulse span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.8); }
    40% { opacity: 1; transform: scale(1); }
}

/* ── Sidebar widgets ── */
.sid-section {
    font-size: 11px;
    font-weight: 700;
    color: #4A5270;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 20px 0 8px;
}
.sid-stat {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #1E2536;
    font-size: 13px;
    color: #8A96B4;
}
.sid-stat-val {
    font-weight: 700;
    color: #E8EAF0;
}
.example-btn {
    width: 100%;
    text-align: left;
    background: #1E2536;
    border: 1px solid #2A3350;
    color: #8A96B4;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 4px 0;
    font-size: 13px;
    cursor: pointer;
    transition: all 0.15s;
}
.status-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.status-dot.green { background: #22C55E; box-shadow: 0 0 6px #22C55E66; }
.status-dot.red   { background: #EF4444; }
.status-dot.grey  { background: #4A5270; }

/* ── Input area ── */
[data-testid="stChatInput"] {
    background: #161B27 !important;
    border: 1px solid #2A3350 !important;
    border-radius: 12px !important;
    color: #E8EAF0 !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7B61FF !important;
    box-shadow: 0 0 0 2px #7B61FF22 !important;
}

/* ── Scrollable chat ── */
.chat-scroll {
    max-height: calc(100vh - 220px);
    overflow-y: auto;
    padding-right: 8px;
    scroll-behavior: smooth;
}

/* ── Misc ── */
.timestamp {
    font-size: 11px;
    color: #2A3350;
    margin-top: 4px;
    margin-left: 42px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────

def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "total_queries" not in st.session_state:
        st.session_state.total_queries = 0
    if "agents_used" not in st.session_state:
        st.session_state.agents_used = set()
    if "use_streaming" not in st.session_state:
        st.session_state.use_streaming = True
    if "api_status" not in st.session_state:
        st.session_state.api_status = "unknown"

init_state()


# ── API helpers ───────────────────────────────────────────────────────────────

def check_health():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return "online" if r.status_code == 200 else "error"
    except Exception:
        return "offline"


def call_chat_blocking(goal: str, session_id: str) -> dict:
    """POST /chat — returns {session_id, response}."""
    try:
        r = requests.post(
            f"{API_BASE}/chat",
            json={"goal": goal, "session_id": session_id},
            timeout=120,
        )
        r.raise_for_status()
        return {"ok": True, **r.json()}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def call_chat_stream(goal: str, session_id: str):
    """
    POST /chat/stream — manually parse SSE lines.
    FastAPI emits: data: <json>\n\n
    """
    try:
        r = requests.post(
            f"{API_BASE}/chat/stream",
            json={"goal": goal, "session_id": session_id},
            stream=True,
            timeout=120,
            headers={"Accept": "text/event-stream"},
        )
        r.raise_for_status()
        for raw_line in r.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            # SSE lines look like: "data: {...}" or "data: [DONE]"
            if raw_line.startswith("data:"):
                payload = raw_line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    yield json.loads(payload)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        yield {"event": "error", "data": str(e)}


def get_history(session_id: str):
    try:
        r = requests.get(f"{API_BASE}/history/{session_id}", timeout=5)
        return r.json().get("history", [])
    except Exception:
        return []


# ── Render helpers ────────────────────────────────────────────────────────────

def render_agent_chips(agents: list[str], active: list[str] = None):
    active = active or []
    chips = ""
    for ag in agents:
        meta = AGENT_META.get(ag, {"icon": "🤖", "color": "#8A96B4", "label": ag.title()})
        cls = "agent-chip active" if ag in active else "agent-chip"
        color_var = f'style="--chip-color:{meta["color"]}"' if ag in active else ""
        chips += f'<span class="{cls}" {color_var}>{meta["icon"]} {meta["label"]}</span>'
    return f'<div class="pipeline-label">Agents activated</div><div class="pipeline-row">{chips}</div>'


def render_user_msg(text: str, ts: str):
    st.markdown(f"""
    <div class="msg-user">
        <div class="msg-user-bubble">{text}</div>
    </div>
    """, unsafe_allow_html=True)


def render_assistant_msg(text: str, agents: list[str], ts: str):
    st.markdown(f"""
    <div class="msg-assistant">
        <div class="msg-avatar">⚡</div>
        <div class="msg-assistant-bubble">{text}</div>
    </div>
    """, unsafe_allow_html=True)
    if agents:
        st.markdown(render_agent_chips(agents, active=agents), unsafe_allow_html=True)
    st.markdown(f'<div class="timestamp">{ts}</div>', unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:16px 0 8px">
        <div style="font-size:20px;font-weight:800;color:#E8EAF0;letter-spacing:-0.5px;">
            ⚡ Productivity<span style="color:#7B61FF">OS</span>
        </div>
        <div style="font-size:12px;color:#4A5270;margin-top:4px;">Multi-agent assistant</div>
    </div>
    """, unsafe_allow_html=True)

    # Health check
    if st.button("↻ Check backend", use_container_width=True):
        st.session_state.api_status = check_health()

    status = st.session_state.api_status
    if status == "online":
        dot, label = "green", "Backend online"
    elif status == "offline":
        dot, label = "red", "Backend offline"
    else:
        dot, label = "grey", "Status unknown"

    st.markdown(f"""
    <div style="font-size:13px;color:#8A96B4;padding:6px 0;">
        <span class="status-dot {dot}"></span>{label}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sid-section">Session</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="sid-stat">
        <span>Session ID</span>
        <span class="sid-stat-val" style="font-size:11px;font-family:monospace;">
            {st.session_state.session_id[:8]}…
        </span>
    </div>
    <div class="sid-stat">
        <span>Queries sent</span>
        <span class="sid-stat-val">{st.session_state.total_queries}</span>
    </div>
    <div class="sid-stat">
        <span>Agents used</span>
        <span class="sid-stat-val">{len(st.session_state.agents_used)}/5</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sid-section">Mode</div>', unsafe_allow_html=True)
    st.session_state.use_streaming = st.toggle(
        "Streaming (SSE)", value=st.session_state.use_streaming
    )
    st.caption("Stream shows live agent pipeline. Blocking is simpler fallback.")

    st.markdown('<div class="sid-section">Try these</div>', unsafe_allow_html=True)
    for eg in EXAMPLE_GOALS:
        if st.button(eg, key=f"eg_{eg[:20]}", use_container_width=True):
            st.session_state["prefill"] = eg
            st.rerun()

    st.markdown('<div class="sid-section">Agents</div>', unsafe_allow_html=True)
    for key, meta in AGENT_META.items():
        used = key in st.session_state.agents_used
        indicator = f'<span style="color:{meta["color"]};font-weight:700;">●</span>' if used else '<span style="color:#2A3350;">○</span>'
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:8px;padding:5px 0;font-size:13px;color:#8A96B4;">
            {indicator} {meta["icon"]} {meta["label"]}
        </div>
        """, unsafe_allow_html=True)

    if st.button("🗑 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.total_queries = 0
        st.session_state.agents_used = set()
        st.rerun()


# ── Main panel ────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pos-header">
    <div class="pos-logo">Productivity<span>OS</span></div>
    <div class="pos-badge">Multi-Agent</div>
</div>
""", unsafe_allow_html=True)

# Render history
chat_container = st.container()
with chat_container:
    if not st.session_state.messages:
        st.markdown("""
        <div style="text-align:center;padding:60px 20px;color:#2A3350;">
            <div style="font-size:48px;margin-bottom:16px;">⚡</div>
            <div style="font-size:18px;font-weight:600;color:#4A5270;margin-bottom:8px;">
                What do you need to get done?
            </div>
            <div style="font-size:14px;color:#2A3350;">
                I'll break it down across email, calendar, docs, tasks, and the web.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for m in st.session_state.messages:
            if m["role"] == "user":
                render_user_msg(m["content"], m.get("ts", ""))
            else:
                render_assistant_msg(m["content"], m.get("agents", []), m.get("ts", ""))


# ── Input ─────────────────────────────────────────────────────────────────────

prefill = st.session_state.pop("prefill", "")
goal = st.chat_input("Describe what you want to get done…", key="chat_input")

# Use prefill if set
if prefill and not goal:
    goal = prefill

if goal:
    ts = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": goal, "ts": ts})
    st.session_state.total_queries += 1

    # Show user message immediately
    render_user_msg(goal, ts)

    if st.session_state.use_streaming:
        # ── Streaming path ──────────────────────────────────────────────────
        thinking_ph = st.empty()
        pipeline_ph = st.empty()
        response_ph = st.empty()

        thinking_ph.markdown("""
        <div class="thinking-row">
            <div class="dot-pulse"><span></span><span></span><span></span></div>
            <span>Orchestrator is planning…</span>
        </div>
        """, unsafe_allow_html=True)

        active_agents = []
        final_text = ""
        error_text = ""

        for evt in call_chat_stream(goal, st.session_state.session_id):
            etype = evt.get("event", "")

            if etype == "node_start":
                node = evt.get("node", "")
                label = "Planning sub-tasks…" if node == "plan" else "Synthesising results…"
                thinking_ph.markdown(f"""
                <div class="thinking-row">
                    <div class="dot-pulse"><span></span><span></span><span></span></div>
                    <span>{label}</span>
                </div>
                """, unsafe_allow_html=True)

            elif etype == "plan_ready":
                agents = evt.get("agents", [])
                active_agents = agents
                st.session_state.agents_used.update(agents)
                thinking_ph.markdown(f"""
                <div class="thinking-row">
                    <div class="dot-pulse"><span></span><span></span><span></span></div>
                    <span>Running {len(agents)} agent(s)…</span>
                </div>
                """, unsafe_allow_html=True)
                pipeline_ph.markdown(
                    render_agent_chips(list(AGENT_META.keys()), active=agents),
                    unsafe_allow_html=True,
                )

            elif etype == "final_response":
                final_text = evt.get("data", "")

            elif etype == "error":
                error_text = evt.get("data", "Unknown error")

        thinking_ph.empty()

        if error_text:
            final_text = f"⚠️ Could not reach backend.\n\nStart it with:\n  uvicorn api:app --reload --port 8000\n\nDetails: {error_text}"
            active_agents = []

        if not final_text:
            final_text = "No response received. The backend may be starting up — try again in a moment."

        response_ts = datetime.now().strftime("%H:%M")
        pipeline_ph.empty()
        render_assistant_msg(final_text, active_agents, response_ts)

        st.session_state.messages.append({
            "role": "assistant",
            "content": final_text,
            "agents": active_agents,
            "ts": response_ts,
        })

    else:
        # ── Blocking path ───────────────────────────────────────────────────
        with st.spinner("Agents working…"):
            result = call_chat_blocking(goal, st.session_state.session_id)

        response_ts = datetime.now().strftime("%H:%M")

        if result.get("ok"):
            text = result.get("response", "")
            # Try to pull agent list from Redis history for display
            history = get_history(st.session_state.session_id)
            agents_used: list[str] = []
        else:
            text = f"⚠️ Backend error: {result.get('error', 'unknown')}"
            agents_used = []

        render_assistant_msg(text, agents_used, response_ts)
        st.session_state.messages.append({
            "role": "assistant",
            "content": text,
            "agents": agents_used,
            "ts": response_ts,
        })

    st.rerun()