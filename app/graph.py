"""LangGraph workflow: multi-agent legal intelligence with critique loops."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from app.agents import (
    critic_agent,
    filing_agent,
    intake_agent,
    outcome_predictor_agent,
    precedent_analyst_agent,
    research_agent,
    strategy_agent,
    supervisor_agent,
)
from app.state import LegalCaseState


def _route_after_supervisor(state: LegalCaseState) -> str:
    if state.get("approved"):
        return END
    return state.get("rewrite_target") or "research"


def build_legal_intelligence_graph(*, with_checkpointer: bool = True):
    """Compile the multi-agent legal workflow graph with critique revision loops."""
    graph = StateGraph(LegalCaseState)

    graph.add_node("intake", intake_agent)
    graph.add_node("research", research_agent)
    graph.add_node("precedent_analyst", precedent_analyst_agent)
    graph.add_node("outcome_predictor", outcome_predictor_agent)
    graph.add_node("strategy", strategy_agent)
    graph.add_node("filing", filing_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("supervisor", supervisor_agent)

    graph.add_edge(START, "intake")
    graph.add_edge("intake", "research")
    graph.add_edge("research", "precedent_analyst")
    graph.add_edge("precedent_analyst", "outcome_predictor")
    graph.add_edge("outcome_predictor", "strategy")
    graph.add_edge("strategy", "filing")
    graph.add_edge("filing", "critic")
    graph.add_edge("critic", "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_after_supervisor,
        {
            "research": "research",
            "strategy": "strategy",
            END: END,
        },
    )

    checkpointer = MemorySaver() if with_checkpointer else None
    return graph.compile(checkpointer=checkpointer)


LEGAL_GRAPH = build_legal_intelligence_graph()
