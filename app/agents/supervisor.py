"""Supervisor agent: critique gate, revision routing, and final report assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.messages import agent_message
from app.config import LEGAL_DISCLAIMER, MAX_REVISIONS
from app.llm import default_caveats, invoke_structured
from app.state import LegalCaseState


def _build_final_report(state: LegalCaseState, *, forced: bool = False) -> dict[str, Any]:
    case = state.get("structured_case") or {}
    critique = state.get("critique_report") or {}

    summary = invoke_structured(
        (
            "Write a concise executive summary (3-4 sentences) for the client covering: "
            "case posture, likely outcome, top strategy, and filing readiness. "
            "Include that this is not legal advice."
        ),
        {
            "case": case,
            "predicted_outcomes": state.get("predicted_outcomes"),
            "quality_score": state.get("quality_score"),
            "strategy_titles": [s.get("title") for s in (state.get("win_strategies") or [])[:3]],
        },
        temperature=0.3,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_id": case.get("case_id"),
        "title": case.get("title"),
        "disclaimer": LEGAL_DISCLAIMER,
        "executive_summary": summary,
        "quality_score": state.get("quality_score"),
        "revision_count": state.get("revision_count") or 0,
        "forced_approval": forced,
        "critique_summary": {
            "final_score": critique.get("quality_score"),
            "recommendations": critique.get("recommendations"),
            "partner_critique": critique.get("partner_critique"),
            "llm_critique": critique.get("llm_critique", True),
        },
        "critique_history": state.get("critique_history") or [],
        "structured_case": case,
        "precedent_analysis": state.get("precedent_analysis"),
        "predicted_outcomes": state.get("predicted_outcomes"),
        "win_strategies": state.get("win_strategies"),
        "favorable_judgment_tactics": state.get("favorable_judgment_tactics"),
        "filing_package": state.get("filing_package"),
        "top_precedents": (state.get("retrieved_precedents") or [])[:5],
        "agent_trace": state.get("agent_notes") or [],
        "agent_messages": [
            {"agent": getattr(m, "name", "unknown"), "content": getattr(m, "content", str(m))}
            for m in (state.get("messages") or [])
        ],
        "validation_errors": state.get("errors") or [],
        "data_sources": [
            "Caselaw Access Project (https://case.law/)",
            "Local corpus in data/cases/",
        ],
    }


def supervisor_agent(state: LegalCaseState) -> dict[str, Any]:
    critique = state.get("critique_report") or {}
    revision = state.get("revision_count") or 0
    errors = list(state.get("errors") or [])

    approved_by_critic = critique.get("approved_by_critic", False)
    forced = revision >= MAX_REVISIONS and not approved_by_critic
    approved = approved_by_critic or forced

    caveats = default_caveats()
    if errors:
        caveats.append(
            "Intake validation reported issues; outputs may be incomplete until required fields are provided."
        )
    if forced:
        caveats.append(
            f"Maximum revision limit ({MAX_REVISIONS}) reached; report approved with unresolved quality flags."
        )
    if not approved:
        caveats.append(
            f"Revision {revision + 1} requested: {critique.get('suggested_rewrite_target', 'research')}."
        )

    result: dict[str, Any] = {
        "approved": approved,
        "legal_caveats": caveats,
    }

    if approved:
        result["final_report"] = _build_final_report(state, forced=forced)
        result["messages"] = [
            agent_message(
                "supervisor",
                f"APPROVED final report (score {critique.get('quality_score')}, revisions {revision}).",
            )
        ]
        result["agent_notes"] = [
            "Supervisor approved and finalized report with LLM executive summary."
            + (" Forced after max revisions." if forced else "")
        ]
    else:
        new_revision = revision + 1
        target = critique.get("suggested_rewrite_target") or state.get("rewrite_target") or "research"
        result["revision_count"] = new_revision
        result["rewrite_target"] = target
        result["messages"] = [
            agent_message(
                "supervisor",
                f"REVISION {new_revision}: send work back to '{target}' agent. "
                + "; ".join(critique.get("recommendations") or [])[:500],
            )
        ]
        result["agent_notes"] = [
            f"Supervisor requested revision {new_revision} → {target}."
        ]

    return result
