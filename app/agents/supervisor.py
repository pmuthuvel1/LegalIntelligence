"""Supervisor agent: critique gate, revision routing, and final report assembly."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.agents.messages import agent_message
from app.config import LEGAL_DISCLAIMER, MAX_REVISIONS
from app.llm import default_caveats, invoke_structured
from app.logging_utils import log_event
from app.state import LegalCaseState


def _build_final_report(
    state: LegalCaseState,
    *,
    forced: bool = False,
    human_review_required: bool = False,
) -> dict[str, Any]:
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
        "human_review_required": human_review_required,
        "escalation_reasons": list(state.get("escalation_reasons") or []),
        "trace_id": state.get("trace_id"),
        "trace_file": state.get("trace_file"),
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
    needs_escalation = bool(state.get("needs_escalation"))
    escalation_reasons = list(state.get("escalation_reasons") or [])

    approved_by_critic = critique.get("approved_by_critic", False)
    forced = revision >= MAX_REVISIONS and not approved_by_critic
    approved = approved_by_critic or forced
    # If we approve while escalation flags are still raised, surface the case
    # for human attorney review in the final report.
    human_review_required = approved and needs_escalation

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
    if human_review_required:
        caveats.append(
            "Critic escalated this matter — human attorney review required before relying on outputs."
        )

    result: dict[str, Any] = {
        "approved": approved,
        "legal_caveats": caveats,
        "human_review_required": human_review_required,
    }

    if approved:
        if human_review_required:
            log_event(
                state,
                agent_name="supervisor",
                action="escalate",
                target_agent="human_attorney",
                confidence=(state.get("quality_score") or 0) / 100.0,
                output_summary="; ".join(escalation_reasons)
                or "Critic flagged low confidence.",
                extra={
                    "reasons": escalation_reasons,
                    "forced_approval": forced,
                    "revision": revision,
                },
            )
        log_event(
            state,
            agent_name="supervisor",
            action="approve",
            target_agent="client",
            confidence=(state.get("quality_score") or 0) / 100.0,
            extra={
                "forced": forced,
                "revision": revision,
                "human_review_required": human_review_required,
            },
        )

        result["final_report"] = _build_final_report(
            state, forced=forced, human_review_required=human_review_required
        )
        approval_note = (
            f"APPROVED final report (score {critique.get('quality_score')}, "
            f"revisions {revision})."
        )
        if human_review_required:
            approval_note += " ⚠ HUMAN REVIEW REQUIRED."
        result["messages"] = [agent_message("supervisor", approval_note)]
        result["agent_notes"] = [
            "Supervisor approved and finalized report with LLM executive summary."
            + (" Forced after max revisions." if forced else "")
            + (" Escalated for human review." if human_review_required else "")
        ]
    else:
        new_revision = revision + 1
        target = (
            critique.get("suggested_rewrite_target")
            or state.get("rewrite_target")
            or "research"
        )
        result["revision_count"] = new_revision
        result["rewrite_target"] = target

        # Role authority: supervisor (acting on critic verdict) blocks final
        # report assembly until the quality threshold is met. Emit an explicit
        # ``reject`` span so the trace shows the gate, separately from the
        # delegation that follows.
        log_event(
            state,
            agent_name="supervisor",
            action="reject",
            target_agent="filing",
            confidence=(state.get("quality_score") or 0) / 100.0,
            output_summary=(
                f"quality {state.get('quality_score')} below threshold; "
                f"blocking final report and routing to {target}"
            ),
            extra={
                "revision": new_revision,
                "max_revisions": MAX_REVISIONS,
                "rewrite_target": target,
                "blocked_role": "report_writer",
            },
        )

        log_event(
            state,
            agent_name="supervisor",
            action="delegate",
            target_agent=target,
            confidence=(state.get("quality_score") or 0) / 100.0,
            retry_count=new_revision,
            output_summary="; ".join(critique.get("recommendations") or [])[:300],
            extra={
                "revision": new_revision,
                "max_revisions": MAX_REVISIONS,
                "issue_count": len(critique.get("issues") or []),
            },
        )

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
