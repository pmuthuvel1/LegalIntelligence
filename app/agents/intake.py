"""Intake agent: validate and structure a case for downstream agents."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.messages import agent_message
from app.llm import invoke_json
from app.state import LegalCaseState


REQUIRED_FIELDS = ("title", "jurisdiction", "court_type", "parties", "claims")


def _normalize_parties(parties: Any) -> dict[str, str]:
    if isinstance(parties, dict):
        return {
            "plaintiff": str(parties.get("plaintiff", "")).strip(),
            "defendant": str(parties.get("defendant", "")).strip(),
        }
    return {"plaintiff": "", "defendant": ""}


def _llm_enrich_structured(raw: dict[str, Any], structured: dict[str, Any]) -> dict[str, Any]:
    result = invoke_json(
        (
            "You are a legal intake specialist. Given raw case input, return JSON with keys: "
            "legal_issues (list of strings), key_facts (refined list), "
            "claims (refined list), intake_summary (one sentence). "
            "Do not invent facts not supported by the input."
        ),
        {"raw_input": raw, "draft": structured},
        temperature=0.1,
    )
    if result.get("legal_issues"):
        structured["legal_issues"] = result["legal_issues"]
    if result.get("key_facts"):
        structured["key_facts"] = result["key_facts"]
    if result.get("claims"):
        structured["claims"] = result["claims"]
    if result.get("intake_summary"):
        structured["intake_summary"] = result["intake_summary"]
    return structured


def intake_agent(state: LegalCaseState) -> dict[str, Any]:
    raw = state.get("case_input") or {}
    errors: list[str] = []

    for field in REQUIRED_FIELDS:
        if not raw.get(field):
            errors.append(f"Missing required field: {field}")

    parties = _normalize_parties(raw.get("parties"))
    if not parties["plaintiff"] or not parties["defendant"]:
        errors.append("Both plaintiff and defendant must be specified.")

    claims = raw.get("claims") or []
    if not isinstance(claims, list) or not claims:
        errors.append("At least one claim or cause of action is required.")

    relief = raw.get("relief_sought") or []
    facts = raw.get("key_facts") or []
    clauses = raw.get("contract_clauses") or raw.get("sample_clauses") or []

    structured = {
        "case_id": raw.get("case_id") or f"CASE-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        "title": raw.get("title", "").strip(),
        "jurisdiction": raw.get("jurisdiction", "").strip(),
        "court_type": raw.get("court_type", "").strip(),
        "venue": raw.get("venue", "").strip(),
        "parties": parties,
        "claims": [str(c).strip() for c in claims],
        "legal_issues": [str(i).strip() for i in (raw.get("legal_issues") or claims)],
        "relief_sought": [str(r).strip() for r in relief] if isinstance(relief, list) else [str(relief)],
        "key_facts": [str(f).strip() for f in facts] if isinstance(facts, list) else [str(facts)],
        "contract_clauses": clauses if isinstance(clauses, list) else [str(clauses)],
        "procedural_posture": raw.get("procedural_posture", "pre-filing"),
        "your_role": raw.get("your_role", "plaintiff"),
        "intake_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    if not errors:
        structured = _llm_enrich_structured(raw, structured)

    note = "Intake agent structured the matter (LLM-enriched)."
    if errors:
        note += f" Warnings: {'; '.join(errors)}"

    return {
        "structured_case": structured,
        "errors": errors,
        "agent_notes": [note],
        "messages": [agent_message("intake", note)],
    }
