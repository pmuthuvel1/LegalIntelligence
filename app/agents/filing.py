"""Filing agent: assemble a practical pre-filing package (not filed on your behalf)."""

from __future__ import annotations

from typing import Any

from app.agents.messages import agent_message
from app.exceptions import LLMError
from app.llm import invoke_json
from app.state import LegalCaseState


def _complaint_outline(case: dict[str, Any], facts_section: str) -> list[dict[str, str]]:
    parties = case.get("parties") or {}
    return [
        {"section": "Caption", "content": f"{parties.get('plaintiff')} v. {parties.get('defendant')}"},
        {"section": "Jurisdiction & Venue", "content": f"{case.get('jurisdiction')} — {case.get('venue') or 'TBD'}"},
        {"section": "Parties", "content": f"Plaintiff: {parties.get('plaintiff')}; Defendant: {parties.get('defendant')}"},
        {"section": "Claims for Relief", "content": "; ".join(case.get("claims") or [])},
        {"section": "Statement of Facts", "content": facts_section},
        {
            "section": "Prayer for Relief",
            "content": "; ".join(case.get("relief_sought") or ["Damages and other appropriate relief"]),
        },
    ]


def filing_agent(state: LegalCaseState) -> dict[str, Any]:
    case = state.get("structured_case") or {}
    precedents = state.get("retrieved_precedents") or []

    draft = invoke_json(
        (
            "You are a litigation drafting assistant. Return JSON with keys: "
            "statement_of_facts (numbered fact paragraphs as a single string), "
            "pre_filing_checklist (list of jurisdiction-specific checklist items). "
            "Use only facts from the input; mark gaps as [VERIFY]."
        ),
        {"case": case, "key_precedents": precedents[:3]},
        temperature=0.2,
    )

    facts_section = draft.get("statement_of_facts")
    checklist = draft.get("pre_filing_checklist")
    if not facts_section:
        raise LLMError("Filing agent: LLM did not return statement_of_facts.")
    if not isinstance(checklist, list) or not checklist:
        raise LLMError("Filing agent: LLM did not return pre_filing_checklist.")

    exhibits = [
        {"id": "Exhibit A", "description": "Chronology of material events"},
        {"id": "Exhibit B", "description": "Operative contract and referenced clauses"},
        {"id": "Exhibit C", "description": "Correspondence showing notice and breach"},
    ]
    for idx, p in enumerate(precedents[:3], start=4):
        exhibits.append(
            {
                "id": f"Exhibit {chr(64 + idx)}",
                "description": f"Key precedent: {p.get('citation') or p.get('name')}",
            }
        )

    package = {
        "status": "draft_for_attorney_review",
        "complaint_outline": _complaint_outline(case, facts_section),
        "proposed_exhibits": exhibits,
        "pre_filing_checklist": checklist,
        "how_to_file": {
            "steps": [
                f"Open e-filing portal for {case.get('court_type', 'target court')}.",
                "Select complaint initiating civil action.",
                "Upload PDF with embedded exhibits; pay filing fee.",
                "Serve defendant per Rule 4 / local equivalent within required period.",
            ],
            "note": "This system does not e-file on your behalf.",
        },
        "llm_drafted": True,
    }

    note = "Filing agent assembled draft filing package with LLM-drafted facts."

    return {
        "filing_package": package,
        "agent_notes": [note],
        "messages": [agent_message("filing", note)],
    }
