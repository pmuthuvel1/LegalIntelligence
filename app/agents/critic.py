"""Critic agent: peer review of research, predictions, and strategy quality."""

from __future__ import annotations

import os
from typing import Any

from app.agents.messages import agent_message
from app.config import CRITIQUE_APPROVAL_THRESHOLD
from app.llm import invoke_json
from app.state import LegalCaseState, RewriteTarget


def _score_precedents(analysis: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    count = analysis.get("precedent_count") or 0
    if count < 3:
        issues.append(
            {
                "severity": "critical",
                "area": "research",
                "message": f"Only {count} precedents retrieved; need broader CAP search.",
                "rewrite_target": "research",
            }
        )
        return 40.0, issues
    if count < 5:
        issues.append(
            {
                "severity": "warning",
                "area": "research",
                "message": "Precedent sample is thin; consider expanding query terms.",
                "rewrite_target": "research",
            }
        )
        return 65.0, issues
    return 85.0, issues


def _score_outcomes(predicted: dict[str, Any], analysis: dict[str, Any]) -> tuple[float, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    confidence = predicted.get("confidence", "very_low")
    win_rate = analysis.get("plaintiff_win_rate_in_sample")
    scenarios = predicted.get("scenarios") or []
    likely = next((s for s in scenarios if s.get("scenario") == "most_likely"), {})
    prob = likely.get("estimated_probability", 0.5)

    if confidence == "very_low":
        issues.append(
            {
                "severity": "warning",
                "area": "outcome",
                "message": "Outcome prediction confidence is very low; strategies may be miscalibrated.",
                "rewrite_target": "strategy",
            }
        )
        return 55.0, issues

    if win_rate is not None and win_rate < 0.35 and prob > 0.6:
        issues.append(
            {
                "severity": "critical",
                "area": "outcome",
                "message": "Predicted probability exceeds precedent win rate; prediction may be overconfident.",
                "rewrite_target": "strategy",
            }
        )
        return 45.0, issues

    return 80.0, issues


def _score_strategy(
    strategies: list[dict[str, Any]], analysis: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    issues: list[dict[str, Any]] = []
    if len(strategies) < 3:
        issues.append(
            {
                "severity": "critical",
                "area": "strategy",
                "message": "Insufficient win strategies generated.",
                "rewrite_target": "strategy",
            }
        )
        return 50.0, issues

    unfavorable = analysis.get("unfavorable_precedents") or []
    strategy_text = " ".join(
        s.get("title", "") + " " + " ".join(s.get("actions") or [])
        for s in strategies
    ).lower()

    if unfavorable and "distinguish" not in strategy_text and "adverse" not in strategy_text:
        issues.append(
            {
                "severity": "critical",
                "area": "strategy",
                "message": "Adverse precedents exist but strategy lacks explicit distinguishing tactics.",
                "rewrite_target": "strategy",
            }
        )
        return 48.0, issues

    return 82.0, issues


def _pick_rewrite_target(issues: list[dict[str, Any]]) -> RewriteTarget:
    critical = [i for i in issues if i.get("severity") == "critical"]
    if not critical:
        return "strategy"
    research_count = sum(1 for i in critical if i.get("rewrite_target") == "research")
    return "research" if research_count >= len(critical) / 2 else "strategy"


def critic_agent(state: LegalCaseState) -> dict[str, Any]:
    analysis = state.get("precedent_analysis") or {}
    predicted = state.get("predicted_outcomes") or {}
    strategies = state.get("win_strategies") or []
    revision = state.get("revision_count") or 0

    p_score, p_issues = _score_precedents(analysis)
    o_score, o_issues = _score_outcomes(predicted, analysis)
    s_score, s_issues = _score_strategy(strategies, analysis)

    all_issues = p_issues + o_issues + s_issues
    baseline_score = round((p_score * 0.35 + o_score * 0.30 + s_score * 0.35), 1)
    rewrite_target = _pick_rewrite_target(all_issues)

    baseline = {
        "dimension_scores": {"precedents": p_score, "outcomes": o_score, "strategy": s_score},
        "issues": all_issues,
        "baseline_score": baseline_score,
    }

    llm_review = invoke_json(
        (
            "You are a senior litigation partner conducting peer review. Return JSON with keys: "
            "quality_score (0-100 number), approved (boolean), issues (list of "
            "{severity, area, message, rewrite_target}), recommendations (list of strings), "
            "partner_critique (2-3 sentence narrative). "
            f"Approve only if score >= {CRITIQUE_APPROVAL_THRESHOLD} and no critical issues."
        ),
        {
            "case": state.get("structured_case"),
            "precedent_analysis": analysis,
            "predicted_outcomes": predicted,
            "win_strategies": strategies,
            "baseline_scores": baseline,
        },
        temperature=0.2,
        model_name=os.getenv("OPENAI_REASONING_MODEL", "gpt-5.1"),
    )

    quality_score = float(llm_review.get("quality_score", baseline_score))
    all_issues = llm_review.get("issues") or all_issues
    rewrite_target = _pick_rewrite_target(all_issues)

    critique_report: dict[str, Any] = {
        "revision": revision,
        "quality_score": quality_score,
        "approval_threshold": CRITIQUE_APPROVAL_THRESHOLD,
        "dimension_scores": baseline["dimension_scores"],
        "issues": all_issues,
        "recommendations": llm_review.get("recommendations")
        or [i["message"] for i in all_issues if i.get("severity") in ("critical", "warning")]
        or ["Analysis meets quality thresholds."],
        "suggested_rewrite_target": rewrite_target,
        "approved_by_critic": bool(llm_review.get("approved"))
        if llm_review.get("approved") is not None
        else (
            quality_score >= CRITIQUE_APPROVAL_THRESHOLD
            and not any(i.get("severity") == "critical" for i in all_issues)
        ),
        "partner_critique": llm_review.get("partner_critique"),
        "llm_critique": True,
    }

    summary = (
        f"Critic review (rev {revision}): score {quality_score}/100, "
        f"{'PASS' if critique_report['approved_by_critic'] else 'NEEDS REVISION'} "
        f"→ {rewrite_target}. LLM peer review applied."
    )

    return {
        "critique_report": critique_report,
        "critique_history": [critique_report],
        "quality_score": quality_score,
        "rewrite_target": rewrite_target,
        "messages": [agent_message("critic", summary)],
        "agent_notes": [summary],
    }
