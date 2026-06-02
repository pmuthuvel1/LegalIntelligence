"""Strategy agent: actionable paths to win and obtain favorable judgments."""

from __future__ import annotations

from typing import Any

from app.agents.messages import agent_message
from app.exceptions import LLMError
from app.llm import invoke_json
from app.state import LegalCaseState


def _critique_revisions(
    strategies: list[dict[str, Any]], critique: dict[str, Any]
) -> list[dict[str, Any]]:
    revision_actions = []
    for rec in critique.get("recommendations") or []:
        revision_actions.append(f"Address critic feedback: {rec}")
    for issue in critique.get("issues") or []:
        if issue.get("area") == "strategy":
            revision_actions.append(issue.get("message", ""))

    if revision_actions:
        strategies.insert(
            0,
            {
                "priority": 0,
                "title": "Respond to peer critique",
                "actions": revision_actions[:5],
                "rationale": "Revision pass incorporating critic agent feedback.",
            },
        )
    return strategies


def _parse_strategies(result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = result.get("strategies")
    if not isinstance(raw, list) or len(raw) < 3:
        raise LLMError("Strategy agent: LLM returned fewer than 3 strategies.")
    out = []
    for idx, s in enumerate(raw[:6], start=1):
        if isinstance(s, dict) and s.get("title"):
            out.append(
                {
                    "priority": idx,
                    "title": s["title"],
                    "actions": s.get("actions") or [],
                    "rationale": s.get("rationale", "LLM-generated; validate with counsel."),
                }
            )
    if len(out) < 3:
        raise LLMError("Strategy agent: insufficient valid strategies from LLM.")
    return out


def strategy_agent(state: LegalCaseState) -> dict[str, Any]:
    case = state.get("structured_case") or {}
    analysis = state.get("precedent_analysis") or {}
    predicted = state.get("predicted_outcomes") or {}
    critique = state.get("critique_report") or {}
    revision = state.get("revision_count") or 0

    strategies = _parse_strategies(
        invoke_json(
            (
                "You are a litigation strategist. Return JSON with key strategies: a list of objects "
                "each having title, actions (list of strings), rationale. Provide 3-5 concrete strategies "
                "to win and obtain favorable judgment. Address adverse precedents if present."
            ),
            {
                "case": case,
                "precedent_analysis": analysis,
                "predicted_outcomes": predicted,
                "critique": critique,
            },
            temperature=0.3,
            model_type="reasoning",
        )
    )

    if critique and revision > 0:
        strategies = _critique_revisions(strategies, critique)

    tactic_out = invoke_json(
        (
            "Return JSON with key tactics: list of {tactic, detail} objects (3-4 items) "
            "for obtaining favorable judgment in this specific case."
        ),
        {"case": case, "predicted_outcomes": predicted, "strategies": strategies},
        temperature=0.3,
        model_type="reasoning",
    )
    tactics = tactic_out.get("tactics")
    if not isinstance(tactics, list) or not tactics:
        raise LLMError("Strategy agent: LLM did not return favorable-judgment tactics.")

    note = (
        f"Strategy agent produced {len(strategies)} win strategies and {len(tactics)} tactics via LLM."
        + (f" Revision {revision}." if revision else "")
    )

    return {
        "win_strategies": strategies,
        "favorable_judgment_tactics": tactics,
        "agent_notes": [note],
        "messages": [agent_message("strategy", note)],
    }
