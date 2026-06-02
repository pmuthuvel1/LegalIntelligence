"""Precedent analyst: extract holdings, patterns, and plaintiff win rates."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.agents.messages import agent_message
from app.llm import invoke_json, invoke_structured
from app.state import LegalCaseState


def _extract_holding(snippet: str, headnotes: str | None) -> str:
    text = (headnotes or snippet or "").strip()
    if not text:
        return "Holding not available in excerpt."
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    return ". ".join(sentences[:2]) + ("." if sentences else "")


def _llm_holding_summary(case: dict[str, Any], precedent: dict[str, Any]) -> str:
    raw = invoke_structured(
        (
            "Summarize this case's holding in one sentence as it applies to the user's claims. "
            "Be factual; do not overstate relevance."
        ),
        {"user_case": case, "precedent": precedent},
        temperature=0.1,
    )
    return raw.strip()


def precedent_analyst_agent(state: LegalCaseState) -> dict[str, Any]:
    precedents = state.get("retrieved_precedents") or []
    case = state.get("structured_case") or {}

    outcomes = [
        p.get("outcome_for_plaintiff")
        for p in precedents
        if p.get("outcome_for_plaintiff") is not None
    ]
    win_rate = round(sum(1 for o in outcomes if o) / len(outcomes), 3) if outcomes else None

    issue_counter: Counter[str] = Counter()
    for p in precedents:
        for issue in p.get("legal_issues") or []:
            issue_counter[issue.lower()] += 1

    analyzed = []
    for p in precedents[:8]:
        holding = _extract_holding(p.get("snippet", ""), p.get("headnotes"))
        if len(holding) < 80:
            holding = _llm_holding_summary(case, p)

        analyzed.append(
            {
                "citation": p.get("citation"),
                "name": p.get("name"),
                "court": p.get("court"),
                "decision_date": p.get("decision_date"),
                "relevance_score": p.get("relevance_score"),
                "holding_summary": holding,
                "outcome_for_plaintiff": p.get("outcome_for_plaintiff"),
                "source_url": p.get("source_url"),
            }
        )

    enriched = invoke_json(
        (
            "You are a legal research analyst. Return JSON with keys: "
            "pattern_summary (string), distinguishing_factors (list), "
            "risk_factors (list). Base analysis only on provided precedents."
        ),
        {"case": case, "precedents": analyzed},
        model_type="reasoning",
    )

    analysis: dict[str, Any] = {
        "precedent_count": len(precedents),
        "plaintiff_win_rate_in_sample": win_rate,
        "top_issues_in_corpus": issue_counter.most_common(5),
        "cases": analyzed,
        "favorable_precedents": [
            c for c in analyzed if c.get("outcome_for_plaintiff") is True
        ],
        "unfavorable_precedents": [
            c for c in analyzed if c.get("outcome_for_plaintiff") is False
        ],
        "llm_pattern_summary": enriched,
    }

    note = (
        f"Precedent analyst reviewed {len(precedents)} cases; "
        f"sample plaintiff win rate: {win_rate if win_rate is not None else 'n/a'}. "
        "LLM synthesized pattern analysis."
    )

    return {
        "precedent_analysis": analysis,
        "agent_notes": [note],
        "messages": [agent_message("precedent_analyst", note)],
    }
