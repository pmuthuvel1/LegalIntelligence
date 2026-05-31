"""Research agent: retrieve CAP precedents and optional CourtListener hits."""

from __future__ import annotations

from typing import Any

from app.agents.messages import agent_message
from app.config import MAX_PRECEDENTS
from app.llm import invoke_json
from app.state import LegalCaseState
from app.tools.registry import get_cap_retriever, get_courtlistener


def _llm_expand_query(case: dict[str, Any], critique: dict[str, Any], revision: int) -> list[str]:
    payload = {
        "case": {
            "title": case.get("title"),
            "claims": case.get("claims"),
            "legal_issues": case.get("legal_issues"),
            "key_facts": case.get("key_facts"),
        },
        "critique_recommendations": critique.get("recommendations") if critique else [],
        "revision": revision,
    }
    result = invoke_json(
        (
            "You are a legal research librarian. Return JSON with key search_terms: "
            "a list of 5-10 concise keyword phrases to find on-point caselaw "
            "(include synonyms, doctrine names, and cause-of-action elements)."
        ),
        payload,
        temperature=0.2,
    )
    terms = result.get("search_terms") or []
    if not isinstance(terms, list) or not terms:
        raise ValueError("LLM returned no search_terms")
    return [str(t) for t in terms if t]


def research_agent(state: LegalCaseState) -> dict[str, Any]:
    case = state.get("structured_case") or state.get("case_input") or {}
    critique = state.get("critique_report") or {}
    revision = state.get("revision_count") or 0

    query_parts = [
        case.get("title", ""),
        " ".join(case.get("claims", [])),
        " ".join(case.get("legal_issues", [])),
        " ".join(case.get("key_facts", [])),
        " ".join(
            c.get("text", str(c)) if isinstance(c, dict) else str(c)
            for c in case.get("contract_clauses", [])
        ),
    ]

    expanded = _llm_expand_query(case, critique, revision)
    query_parts.extend(expanded)

    if critique and revision > 0:
        query_parts.extend(critique.get("recommendations") or [])
        for issue in critique.get("issues") or []:
            if issue.get("area") == "research":
                query_parts.append(issue.get("message", ""))

    query = " ".join(p for p in query_parts if p).strip()
    limit = MAX_PRECEDENTS + (revision * 2)

    retriever = get_cap_retriever()
    precedents = retriever.search(
        query,
        jurisdiction=case.get("jurisdiction"),
        legal_issues=case.get("legal_issues"),
        limit=limit,
    )

    cl = get_courtlistener()
    external: list[dict[str, Any]] = []
    if cl.enabled:
        external = cl.search_opinions(query, limit=min(3 + revision, 6))
        for row in external:
            row["source"] = "courtlistener"

    combined = precedents + external
    note = (
        f"Research agent retrieved {len(precedents)} CAP precedents"
        + (f" and {len(external)} CourtListener results." if external else ".")
        + f" LLM expanded search with {len(expanded)} terms."
        + (f" (revision {revision})." if revision else "")
    )

    return {
        "retrieved_precedents": combined,
        "agent_notes": [note],
        "messages": [agent_message("research", note)],
    }
