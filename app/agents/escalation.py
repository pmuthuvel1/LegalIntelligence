"""Escalation agent: deterministic sample-mode fallback when the live LLM fails.

This agent is invoked by :func:`app.service.run_case_analysis` when an
``LLMError`` or ``ConfigurationError`` is raised mid-pipeline (or before the
graph can be invoked). It produces a *degraded but valid* final report using
only deterministic logic — no LLM calls — so the system stays usable when
quota is exhausted, the Compass endpoint is down, or required env vars are
missing.

The returned report is flagged ``forced_approval=True`` and
``human_review_required=True``, and its ``escalation_reasons`` list explains
the fallback. The structured trace logger receives an explicit ``escalate``
span so operators see the live→sample handoff.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

from app.config import LEGAL_DISCLAIMER
from app.logging_utils import log_event
from app.tools.registry import get_cap_retriever, is_warmed, warmup


__all__ = ["escalate_to_sample_mode"]


# ---------------------------------------------------------------------------
# Helpers — pure-Python, no LLM calls.
# ---------------------------------------------------------------------------

def _structure_case(raw: dict[str, Any]) -> dict[str, Any]:
    parties = raw.get("parties") or {}
    if not isinstance(parties, dict):
        parties = {"plaintiff": "", "defendant": ""}
    claims = raw.get("claims") or []
    facts = raw.get("key_facts") or []
    relief = raw.get("relief_sought") or []
    clauses = raw.get("contract_clauses") or raw.get("sample_clauses") or []

    return {
        "case_id": raw.get("case_id")
        or f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "title": (raw.get("title") or "").strip(),
        "jurisdiction": (raw.get("jurisdiction") or "").strip(),
        "court_type": (raw.get("court_type") or "").strip(),
        "venue": (raw.get("venue") or "").strip(),
        "parties": {
            "plaintiff": str(parties.get("plaintiff", "")).strip(),
            "defendant": str(parties.get("defendant", "")).strip(),
        },
        "claims": [str(c).strip() for c in claims if c],
        "legal_issues": [
            str(i).strip() for i in (raw.get("legal_issues") or claims) if i
        ],
        "relief_sought": [str(r).strip() for r in relief] if isinstance(relief, list) else [str(relief)],
        "key_facts": [str(f).strip() for f in facts] if isinstance(facts, list) else [str(facts)],
        "contract_clauses": clauses if isinstance(clauses, list) else [str(clauses)],
        "procedural_posture": raw.get("procedural_posture", "pre-filing"),
        "your_role": raw.get("your_role", "plaintiff"),
        "intake_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def _safe_retrieve(case: dict[str, Any], limit: int = 6) -> list[dict[str, Any]]:
    """Run the BM25 retriever without ever propagating exceptions."""
    try:
        if not is_warmed():
            warmup()
        retriever = get_cap_retriever()
        query_parts = [
            case.get("title", ""),
            " ".join(case.get("claims", [])),
            " ".join(case.get("legal_issues", [])),
            " ".join(case.get("key_facts", [])),
        ]
        return retriever.search(
            " ".join(p for p in query_parts if p).strip(),
            jurisdiction=case.get("jurisdiction") or None,
            legal_issues=case.get("legal_issues") or None,
            limit=limit,
        )
    except Exception:  # noqa: BLE001
        return []


def _analyze(precedents: list[dict[str, Any]]) -> dict[str, Any]:
    outcomes = [
        p.get("outcome_for_plaintiff")
        for p in precedents
        if p.get("outcome_for_plaintiff") is not None
    ]
    win_rate = (
        round(sum(1 for o in outcomes if o) / len(outcomes), 3)
        if outcomes
        else None
    )
    issue_counter: Counter[str] = Counter()
    for p in precedents:
        for issue in p.get("legal_issues") or []:
            issue_counter[str(issue).lower()] += 1

    cases = [
        {
            "citation": p.get("citation"),
            "name": p.get("name"),
            "court": p.get("court"),
            "decision_date": p.get("decision_date"),
            "relevance_score": p.get("relevance_score"),
            "holding_summary": (p.get("headnotes") or p.get("snippet") or "")[:280],
            "outcome_for_plaintiff": p.get("outcome_for_plaintiff"),
            "source_url": p.get("source_url"),
        }
        for p in precedents[:8]
    ]
    return {
        "precedent_count": len(precedents),
        "plaintiff_win_rate_in_sample": win_rate,
        "top_issues_in_corpus": issue_counter.most_common(5),
        "cases": cases,
        "favorable_precedents": [c for c in cases if c.get("outcome_for_plaintiff") is True],
        "unfavorable_precedents": [c for c in cases if c.get("outcome_for_plaintiff") is False],
    }


def _predict_outcomes(analysis: dict[str, Any], role: str) -> dict[str, Any]:
    win_rate = analysis.get("plaintiff_win_rate_in_sample") or 0.5
    most_likely = win_rate if role == "plaintiff" else round(1.0 - win_rate, 3)
    confidence = (
        "moderate"
        if analysis.get("precedent_count", 0) >= 4
        else "low"
        if analysis.get("precedent_count", 0) >= 2
        else "very_low"
    )
    return {
        "methodology": "Deterministic sample-mode estimate from CAP precedent win rates.",
        "confidence": confidence,
        "precedent_sample_size": analysis.get("precedent_count", 0),
        "role_analyzed": role,
        "scenarios": [
            {
                "scenario": "most_likely",
                "description": "Outcome aligned with the precedent sample win rate.",
                "estimated_probability": round(most_likely, 3),
            },
            {
                "scenario": "best_case",
                "description": "Full requested relief granted.",
                "estimated_probability": round(min(0.95, most_likely + 0.15), 3),
            },
            {
                "scenario": "worst_case",
                "description": "Dismissal or judgment against you.",
                "estimated_probability": round(max(0.05, 1.0 - most_likely - 0.05), 3),
            },
        ],
        "likely_judgment_summary": (
            "Sample-mode estimate; not validated by reasoning model."
        ),
        "not_legal_advice": True,
    }


def _template_strategies(case: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "priority": 1,
            "title": "Anchor claims to strongest favorable precedent",
            "actions": [
                "Lead briefing with top 2-3 favorable CAP cases by relevance score.",
                "Distinguish unfavorable precedents on facts and procedural posture.",
            ],
            "rationale": "Controlling precedent reduces outcome variance.",
        },
        {
            "priority": 2,
            "title": "Tighten the fact record",
            "actions": [
                "Map each claim element to documentary proof.",
                "Identify gaps opposing counsel will exploit.",
            ],
            "rationale": "Early dispositive motions turn on factual completeness.",
        },
        {
            "priority": 3,
            "title": "Procedural positioning",
            "actions": [
                f"Confirm venue and court_type ({case.get('court_type', 'court')}) "
                "satisfy jurisdictional rules.",
                "Calendar response deadlines and meet-and-confer obligations.",
            ],
            "rationale": "Procedural defects defeat otherwise meritorious claims.",
        },
    ]


def _template_filing(case: dict[str, Any]) -> dict[str, Any]:
    parties = case.get("parties") or {}
    return {
        "status": "draft_for_attorney_review",
        "complaint_outline": [
            {
                "section": "Caption",
                "content": f"{parties.get('plaintiff', '')} v. {parties.get('defendant', '')}",
            },
            {
                "section": "Jurisdiction & Venue",
                "content": f"{case.get('jurisdiction', '')} — {case.get('venue', '')}".strip(" —"),
            },
            {
                "section": "Claims for Relief",
                "content": "; ".join(case.get("claims") or []),
            },
            {
                "section": "Statement of Facts",
                "content": "Numbered paragraphs mapping each element to exhibits.",
            },
            {
                "section": "Prayer for Relief",
                "content": "; ".join(case.get("relief_sought") or []),
            },
        ],
        "pre_filing_checklist": [
            "Confirm statute of limitations for each claim.",
            "Verify filing fee and e-filing account for court.",
            "Review formatting requirements (margins, line numbering, word limits).",
        ],
        "note": (
            "Generated in sample-mode fallback. Attorney review required before "
            "any filing."
        ),
    }


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def escalate_to_sample_mode(
    case_input: dict[str, Any],
    *,
    trace_id: str,
    trace_file: str,
    reason: str,
    original_error: BaseException | None = None,
) -> dict[str, Any]:
    """Build a deterministic final-report state when the live LLM is unavailable.

    Returns the same shape as a successful LangGraph terminal state — with a
    populated ``final_report`` — so :mod:`app.service` can wrap it in the
    standard success envelope. ``human_review_required`` is set so callers know
    the result was not validated by the reasoning model.
    """
    state: dict[str, Any] = {"trace_id": trace_id, "trace_file": trace_file}

    log_event(
        state,
        agent_name="escalation_agent",
        action="escalate",
        target_agent="sample_runner",
        confidence=0.0,
        status="warning",
        output_summary=f"live LLM unavailable: {reason}",
        extra={
            "from": "live_api",
            "to": "sample_mode",
            "error_type": type(original_error).__name__ if original_error else None,
        },
    )

    structured = _structure_case(case_input)
    log_event(
        state,
        agent_name="escalation_agent",
        action="delegate",
        target_agent="cap_retriever",
        output_summary="sample-mode retrieval (no LLM query expansion)",
    )
    precedents = _safe_retrieve(structured)
    analysis = _analyze(precedents)
    predicted = _predict_outcomes(analysis, structured.get("your_role") or "plaintiff")
    strategies = _template_strategies(structured)
    filing = _template_filing(structured)

    escalation_reasons = [
        f"Live API unavailable; sample-mode fallback used ({reason}).",
        "Outputs are deterministic templates, not LLM-validated.",
        "Human attorney review is required before any reliance.",
    ]

    final_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_id": structured.get("case_id"),
        "title": structured.get("title"),
        "disclaimer": LEGAL_DISCLAIMER,
        "executive_summary": (
            "Sample-mode fallback report. The live reasoning model was "
            "unavailable, so this analysis was produced from deterministic "
            "templates plus the local CAP precedent index. Treat as a starting "
            "point for human attorney review — not as legal advice."
        ),
        "quality_score": None,
        "revision_count": 0,
        "forced_approval": True,
        "human_review_required": True,
        "escalation_reasons": escalation_reasons,
        "degraded_mode": "sample",
        "trace_id": trace_id,
        "trace_file": trace_file,
        "critique_summary": {
            "final_score": None,
            "recommendations": [
                "Re-run with the live API once the underlying issue is resolved.",
            ],
            "partner_critique": None,
            "llm_critique": False,
        },
        "critique_history": [],
        "structured_case": structured,
        "precedent_analysis": analysis,
        "predicted_outcomes": predicted,
        "win_strategies": strategies,
        "favorable_judgment_tactics": [
            {
                "tactic": "Engage counsel before relying on this output",
                "detail": "Sample mode bypasses LLM peer review and critique loops.",
            }
        ],
        "filing_package": filing,
        "top_precedents": precedents[:5],
        "agent_trace": [
            "Escalation agent activated: live API unavailable.",
            f"Retriever returned {len(precedents)} sample precedents.",
            "Template strategies and filing scaffolding produced without LLM.",
        ],
        "agent_messages": [
            {
                "agent": "escalation_agent",
                "content": f"Escalated to sample mode: {reason}",
            }
        ],
        "validation_errors": [],
        "data_sources": [
            "Caselaw Access Project (https://case.law/)",
            "Local corpus in data/cases/",
        ],
    }

    log_event(
        state,
        agent_name="escalation_agent",
        action="complete_task",
        target_agent="client",
        output_summary=(
            f"sample-mode report assembled with {len(precedents)} precedents"
        ),
        extra={"degraded_mode": "sample", "human_review_required": True},
    )

    return {
        "structured_case": structured,
        "retrieved_precedents": precedents,
        "precedent_analysis": analysis,
        "predicted_outcomes": predicted,
        "win_strategies": strategies,
        "filing_package": filing,
        "final_report": final_report,
        "quality_score": None,
        "revision_count": 0,
        "approved": True,
        "human_review_required": True,
        "escalation_reasons": escalation_reasons,
        "trace_id": trace_id,
        "trace_file": trace_file,
        "errors": [f"Escalated to sample mode: {reason}"],
        "critique_report": {
            "approved_by_critic": False,
            "quality_score": None,
            "recommendations": escalation_reasons,
            "partner_critique": None,
            "llm_critique": False,
        },
        "critique_history": [],
        "agent_notes": [
            "Escalation agent produced sample-mode fallback report.",
        ],
        "messages": [],
    }

