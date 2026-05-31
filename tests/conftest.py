"""Pytest fixtures — mock LLM for tests without live API credentials."""

from __future__ import annotations

from contextlib import ExitStack
from unittest.mock import patch

import pytest


def _mock_invoke_json(system_prompt: str, user_payload: dict, **kwargs) -> dict:
    prompt = system_prompt.lower()
    if "intake" in prompt:
        draft = user_payload.get("draft") or {}
        return {
            "legal_issues": draft.get("legal_issues") or draft.get("claims") or ["breach of contract"],
            "key_facts": draft.get("key_facts") or ["Material breach occurred."],
            "claims": draft.get("claims") or ["breach of contract"],
            "intake_summary": "Mock intake summary.",
        }
    if "search_terms" in prompt or "librarian" in prompt:
        return {"search_terms": ["breach of contract", "anticipatory breach", "damages"]}
    if "pattern_summary" in prompt or "research analyst" in prompt:
        return {
            "pattern_summary": "Mock precedent pattern.",
            "distinguishing_factors": ["Different fact pattern"],
            "risk_factors": ["Thin sample"],
        }
    if "scenarios" in prompt and "risk analyst" in prompt:
        return {
            "scenarios": [
                {"scenario": "most_likely", "description": "Partial success.", "estimated_probability": 0.55},
                {"scenario": "best_case", "description": "Full relief.", "estimated_probability": 0.7},
                {"scenario": "worst_case", "description": "Dismissal.", "estimated_probability": 0.15},
            ],
            "likely_judgment_summary": "Partial success on core claims.",
            "confidence": "moderate",
        }
    if "strategist" in prompt and "strategies" in prompt:
        return {
            "strategies": [
                {
                    "title": "Anchor favorable precedent",
                    "actions": ["Lead with top CAP cases", "Distinguish adverse authority"],
                    "rationale": "Controlling precedent reduces variance.",
                },
                {
                    "title": "Strengthen fact record",
                    "actions": ["Chronology exhibit", "Map elements to proof"],
                    "rationale": "Pleading-stage completeness matters.",
                },
                {
                    "title": "Neutralize adverse precedent",
                    "actions": ["Distinguish adverse case on facts"],
                    "rationale": "Opposing counsel will cite adverse authority.",
                },
            ]
        }
    if "tactics" in prompt:
        return {
            "tactics": [
                {"tactic": "Early dispositive framing", "detail": "Focus strongest claim."},
                {"tactic": "Remedy calibration", "detail": "Tier requested relief."},
            ]
        }
    if "drafting assistant" in prompt or "statement_of_facts" in prompt:
        return {
            "statement_of_facts": "1. Parties entered a contract. 2. Defendant breached. [VERIFY]",
            "pre_filing_checklist": ["Confirm limitations period", "Verify e-filing account"],
        }
    if "peer review" in prompt or "partner" in prompt:
        baseline = user_payload.get("baseline_scores") or {}
        issues = baseline.get("issues") or []
        critical = any(i.get("severity") == "critical" for i in issues)
        score = 82.0 if not critical else 45.0
        return {
            "quality_score": score,
            "approved": score >= 70 and not critical,
            "issues": issues,
            "recommendations": ["Mock recommendation."],
            "partner_critique": "Mock partner critique: analysis is adequate for review.",
        }
    return {"ok": True}


def _mock_invoke_structured(system_prompt: str, user_payload: dict, **kwargs) -> str:
    if "holding" in system_prompt.lower() or "summarize this case" in system_prompt.lower():
        return "Mock one-sentence holding summary."
    return "Mock executive summary for the client. This is not legal advice."


_INVOKE_JSON_PATCHES = [
    "app.agents.intake.invoke_json",
    "app.agents.research.invoke_json",
    "app.agents.precedent.invoke_json",
    "app.agents.outcome.invoke_json",
    "app.agents.strategy.invoke_json",
    "app.agents.filing.invoke_json",
    "app.agents.critic.invoke_json",
]

_INVOKE_STRUCTURED_PATCHES = [
    "app.agents.precedent.invoke_structured",
    "app.agents.supervisor.invoke_structured",
]


@pytest.fixture(autouse=True)
def mock_llm_for_tests(request):
    if request.node.get_closest_marker("no_llm_mock"):
        yield
        return

    with ExitStack() as stack:
        stack.enter_context(patch("app.llm.verify_llm_connectivity"))
        stack.enter_context(patch("app.service.verify_llm_connectivity"))
        stack.enter_context(patch("app.service.validate_llm_config"))
        stack.enter_context(patch("app.config.validate_llm_config"))
        for target in _INVOKE_JSON_PATCHES:
            stack.enter_context(patch(target, side_effect=_mock_invoke_json))
        for target in _INVOKE_STRUCTURED_PATCHES:
            stack.enter_context(patch(target, side_effect=_mock_invoke_structured))
        yield
