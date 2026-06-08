"""Shared graph state for the legal intelligence workflow."""

from __future__ import annotations

from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages


def append_lists(
    left: list[Any] | None, right: list[Any] | None
) -> list[Any]:
    base = list(left or [])
    if right:
        base.extend(right)
    return base


def append_critique_history(
    left: list[dict[str, Any]] | None, right: list[dict[str, Any]] | None
) -> list[dict[str, Any]]:
    base = list(left or [])
    if right:
        base.extend(right)
    return base


RewriteTarget = Literal["research", "strategy"]


class LegalCaseState(TypedDict, total=False):
    """State passed between specialized legal agents."""

    case_input: dict[str, Any]
    structured_case: dict[str, Any]
    retrieved_precedents: list[dict[str, Any]]
    precedent_analysis: dict[str, Any]
    predicted_outcomes: dict[str, Any]
    win_strategies: list[dict[str, Any]]
    favorable_judgment_tactics: list[dict[str, Any]]
    filing_package: dict[str, Any]
    critique_report: dict[str, Any]
    critique_history: Annotated[list[dict[str, Any]], append_critique_history]
    revision_count: int
    rewrite_target: RewriteTarget
    quality_score: float
    approved: bool
    legal_caveats: Annotated[list[str], append_lists]
    agent_notes: Annotated[list[str], append_lists]
    errors: Annotated[list[str], append_lists]
    final_report: dict[str, Any]
    messages: Annotated[list, add_messages]

    # ---- Tracing / observability ----
    trace_id: str
    trace_file: str

    # ---- Escalation signals (set by critic, consumed by supervisor) ----
    needs_escalation: bool
    escalation_reasons: Annotated[list[str], append_lists]
    human_review_required: bool

