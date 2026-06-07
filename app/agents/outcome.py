"""Outcome predictor: estimate likely judgment bands from precedent signals."""

from __future__ import annotations

from typing import Any

from app.agents.messages import agent_message
from app.compass import reasoning_model_name
from app.exceptions import LLMError
from app.llm import invoke_json
from app.state import LegalCaseState


def _confidence_label(score: float) -> str:
    if score >= 0.7:
        return "moderate"
    if score >= 0.45:
        return "low"
    return "very_low"


def outcome_predictor_agent(state: LegalCaseState) -> dict[str, Any]:
    analysis = state.get("precedent_analysis") or {}
    case = state.get("structured_case") or {}
    role = (case.get("your_role") or "plaintiff").lower()

    win_rate = analysis.get("plaintiff_win_rate_in_sample")
    favorable = len(analysis.get("favorable_precedents") or [])
    unfavorable = len(analysis.get("unfavorable_precedents") or [])
    total = favorable + unfavorable
    precedent_count = analysis.get("precedent_count") or 0

    if win_rate is None and total == 0:
        base_prob = 0.5
    elif win_rate is not None:
        base_prob = win_rate
    else:
        base_prob = favorable / total if total else 0.5

    if role == "defendant":
        base_prob = 1.0 - base_prob

    llm_out = invoke_json(
        (
            "You are a litigation risk analyst. Given case facts and precedent statistics, "
            "return JSON with keys: scenarios (list of 3 objects with scenario, description, "
            "estimated_probability as 0-1 float), likely_judgment_summary (string), "
            "confidence (one of: moderate, low, very_low). "
            "Calibrate probabilities to precedent win rates; do not guarantee outcomes."
        ),
        {
            "case": case,
            "precedent_analysis": {
                "precedent_count": precedent_count,
                "plaintiff_win_rate": win_rate,
                "favorable_count": favorable,
                "unfavorable_count": unfavorable,
                "pattern_summary": analysis.get("llm_pattern_summary"),
            },
            "baseline_probability": base_prob,
            "party_role": role,
        },
        temperature=0.2,
        model_name=reasoning_model_name(),
    )

    scenarios = llm_out.get("scenarios")
    if not isinstance(scenarios, list) or len(scenarios) < 3:
        raise LLMError("Outcome predictor: LLM did not return three judgment scenarios.")

    confidence = llm_out.get("confidence") or _confidence_label(base_prob)

    predicted = {
        "methodology": "LLM-calibrated estimate from CAP precedent sample and case facts.",
        "confidence": confidence,
        "precedent_sample_size": precedent_count,
        "role_analyzed": role,
        "scenarios": scenarios[:3],
        "likely_judgment_summary": llm_out.get("likely_judgment_summary") or scenarios[0].get("description", ""),
        "not_legal_advice": True,
        "llm_enhanced": True,
    }

    note = (
        f"Outcome predictor: most-likely probability ~{scenarios[0].get('estimated_probability')} "
        f"(confidence: {confidence}). LLM refined scenarios."
    )

    return {
        "predicted_outcomes": predicted,
        "agent_notes": [note],
        "messages": [agent_message("outcome_predictor", note)],
    }
