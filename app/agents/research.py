"""Research agent: retrieve CAP precedents and optional CourtListener hits."""

from __future__ import annotations

from typing import Any

from app.agents.messages import agent_message
from app.config import (
    BROAD_SIMILARITY_THRESHOLD,
    LOW_PRECEDENT_FLOOR,
    MAX_PRECEDENTS,
    RETRIEVER_MAX_RETRIES,
    SIMILARITY_THRESHOLD,
)
from app.llm import invoke_json
from app.logging_utils import log_event
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


def _retrieve_with_broadening(
    state: LegalCaseState,
    *,
    retriever,
    query: str,
    jurisdiction: str | None,
    legal_issues: list[str] | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int, list[dict[str, Any]]]:
    """Call the CAP retriever and progressively broaden on sparse results.

    Returns ``(results, attempts, attempt_log)`` so the caller can record how
    many widening passes were required. Each pass past the first emits an
    explicit ``retry`` trace span pointing at the ``cap_retriever`` tool.
    """
    attempt_log: list[dict[str, Any]] = []

    # Attempt 1 — full filters, default threshold.
    results = retriever.search(
        query,
        jurisdiction=jurisdiction,
        legal_issues=legal_issues,
        limit=limit,
    )
    attempt_log.append(
        {
            "attempt": 1,
            "jurisdiction": jurisdiction,
            "threshold": SIMILARITY_THRESHOLD,
            "result_count": len(results),
        }
    )
    if len(results) >= LOW_PRECEDENT_FLOOR or RETRIEVER_MAX_RETRIES < 1:
        return results, 1, attempt_log

    # Attempt 2 — drop jurisdiction filter.
    log_event(
        state,
        agent_name="research",
        action="retry",
        target_agent="cap_retriever",
        retry_count=1,
        output_summary=(
            f"sparse results ({len(results)} < {LOW_PRECEDENT_FLOOR}); "
            "broadening: drop jurisdiction filter"
        ),
        extra={
            "previous_count": len(results),
            "floor": LOW_PRECEDENT_FLOOR,
            "broaden": "drop_jurisdiction",
        },
    )
    results = retriever.search(
        query,
        jurisdiction=None,
        legal_issues=legal_issues,
        limit=limit,
    )
    attempt_log.append(
        {
            "attempt": 2,
            "jurisdiction": None,
            "threshold": SIMILARITY_THRESHOLD,
            "result_count": len(results),
        }
    )
    if len(results) >= LOW_PRECEDENT_FLOOR or RETRIEVER_MAX_RETRIES < 2:
        return results, 2, attempt_log

    # Attempt 3 — also lower the similarity threshold floor.
    log_event(
        state,
        agent_name="research",
        action="retry",
        target_agent="cap_retriever",
        retry_count=2,
        output_summary=(
            f"still sparse ({len(results)} < {LOW_PRECEDENT_FLOOR}); "
            f"broadening: lower threshold to {BROAD_SIMILARITY_THRESHOLD}"
        ),
        extra={
            "previous_count": len(results),
            "floor": LOW_PRECEDENT_FLOOR,
            "broaden": "lower_threshold",
            "threshold": BROAD_SIMILARITY_THRESHOLD,
        },
    )
    results = retriever.search(
        query,
        jurisdiction=None,
        legal_issues=legal_issues,
        limit=limit,
        similarity_threshold=BROAD_SIMILARITY_THRESHOLD,
    )
    attempt_log.append(
        {
            "attempt": 3,
            "jurisdiction": None,
            "threshold": BROAD_SIMILARITY_THRESHOLD,
            "result_count": len(results),
        }
    )
    return results, 3, attempt_log


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
    precedents, attempts, attempt_log = _retrieve_with_broadening(
        state,
        retriever=retriever,
        query=query,
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
    note_parts = [
        f"Research agent retrieved {len(precedents)} CAP precedents",
    ]
    if external:
        note_parts.append(f"and {len(external)} CourtListener results")
    note_parts.append(f"LLM expanded search with {len(expanded)} terms")
    if attempts > 1:
        note_parts.append(f"after {attempts} retriever attempts (broadening on sparse results)")
    if revision:
        note_parts.append(f"(revision {revision})")
    note = "; ".join(note_parts) + "."

    return {
        "retrieved_precedents": combined,
        "agent_notes": [note],
        "messages": [agent_message("research", note)],
    }
