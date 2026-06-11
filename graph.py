"""
graph.py — LangGraph graph definition for the Productivity OS.

Graph shape:
  START → plan → [email | calendar | doc | task | web] (parallel) → synthesise → END

The router dispatches to every agent whose name appears in the plan.
Agents that aren't needed are skipped automatically.
"""

from __future__ import annotations

import uuid

from langgraph.graph import StateGraph, START, END

from state import AgentState
from agents.orchestrator import plan_node, synthesise_node
from agents.email_agent import email_agent_node
from agents.calendar_agent import calendar_agent_node
from agents.doc_agent import doc_agent_node
from agents.web_agent import web_agent_node
from agents.task_agent import task_agent_node

# ── Stub nodes for agents not yet implemented ─────────────────────────────────
# Replace each stub with its real module as you build week by week.


# ── Router: decides which sub-agents to run in parallel ───────────────────────

AGENT_NODES = ["email_agent", "calendar_agent", "doc_agent", "task_agent", "web_agent"]

def router(state: AgentState) -> list[str]:
    """
    Conditional edge: returns the list of agent node names that appear in the plan.
    LangGraph runs them all in parallel (fan-out), then converges at synthesise.
    """
    plan = state.get("plan", [])
    needed = {task.agent for task in plan}
    mapping = {
        "email":    "email_agent",
        "calendar": "calendar_agent",
        "doc":      "doc_agent",
        "task":     "task_agent",
        "web":      "web_agent",
    }
    targets = [mapping[a] for a in needed if a in mapping]
    return targets if targets else ["synthesise"]


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("plan", plan_node)
    g.add_node("email_agent", email_agent_node)
    g.add_node("calendar_agent", calendar_agent_node)
    g.add_node("doc_agent", doc_agent_node)
    g.add_node("task_agent", task_agent_node)
    g.add_node("web_agent", web_agent_node)
    g.add_node("synthesise", synthesise_node)

    # Edges
    g.add_edge(START, "plan")

    # Fan-out from plan to whichever agents are needed
    g.add_conditional_edges(
        "plan",
        router,
        {node: node for node in AGENT_NODES} | {"synthesise": "synthesise"},
    )

    # All agents converge into synthesise
    for node in AGENT_NODES:
        g.add_edge(node, "synthesise")

    g.add_edge("synthesise", END)

    return g.compile()


# ── Entrypoint ────────────────────────────────────────────────────────────────

def run(user_goal: str, session_id: str | None = None) -> str:
    """Run the graph for a user goal, return the final response string."""
    graph = build_graph()
    session_id = session_id or str(uuid.uuid4())

    initial_state: AgentState = {
        "messages": [],
        "user_goal": user_goal,
        "plan": [],
        "results": [],
        "final_response": "",
        "session_id": session_id,
        "memory_context": "",
    }

    final_state = graph.invoke(initial_state)
    return final_state["final_response"]


if __name__ == "__main__":
    # Quick smoke test — replace with your actual goal
    response = run("Summarise my last 5 emails and check if any need urgent replies.")
    print(response)