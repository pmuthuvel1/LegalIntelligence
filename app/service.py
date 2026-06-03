"""High-level service API for CLI and HTTP layers."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import message_to_dict

from app.config import LOGS_DIR, validate_config
from app.exceptions import AnalysisError, ConfigurationError, LLMError
from app.graph import LEGAL_GRAPH
from app.llm import llm_available
from app.tools.registry import is_warmed, warmup

logger = logging.getLogger(__name__)


def _ensure_llm_ready() -> None:
    """Validate LLM configuration is available."""
    if not llm_available():
        raise ConfigurationError(
            "Required LLM configuration missing: OPENAI_API_KEY and OPENAI_BASE_URL. "
            "Set both in the environment or .env file."
        )


def initialize() -> dict[str, Any]:
    """Startup hook: warm retrievers, validate config, verify LLM."""
    stats = warmup()
    _ensure_llm_ready()
    warnings = validate_config()
    if warnings:
        for w in warnings:
            logger.warning("Config warning: %s", w)
    stats["warnings"] = warnings
    stats["llm_verified"] = True
    return stats


def readiness() -> dict[str, Any]:
    warnings = validate_config()
    warmed = is_warmed()
    if not warmed:
        try:
            stats = warmup()
        except Exception as exc:
            raise ConfigurationError(str(exc)) from exc
    else:
        stats = warmup()

    try:
        _ensure_llm_ready()
        llm_ready = True
    except (ConfigurationError, LLMError) as exc:
        raise ConfigurationError(str(exc)) from exc

    return {
        "ready": stats.get("case_index_count", 0) > 0 and llm_ready,
        "case_index_count": stats.get("case_index_count", 0),
        "courtlistener_enabled": stats.get("courtlistener_enabled", False),
        "llm_ready": llm_ready,
        "warnings": warnings,
    }


def run_case_analysis(
    case_input: dict[str, Any],
    *,
    thread_id: str | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Execute the full multi-agent pipeline with critique loops."""
    _ensure_llm_ready()

    if not is_warmed():
        warmup()

    tid = thread_id or f"case-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": tid}}

    initial: dict[str, Any] = {
        "case_input": case_input,
        "revision_count": 0,
        "approved": False,
    }

    try:
        result = LEGAL_GRAPH.invoke(initial, config=config)
    except (ConfigurationError, LLMError):
        raise
    except Exception as exc:
        logger.exception("Pipeline failed for thread %s", tid)
        raise AnalysisError(f"Analysis pipeline failed: {exc}") from exc

    report = result.get("final_report") or {}
    if not report:
        raise AnalysisError("Pipeline completed without a final report.")

    if log:
        _write_log(case_input, report, result, thread_id=tid)

    return {
        "report": report,
        "legal_caveats": result.get("legal_caveats") or [],
        "errors": result.get("errors") or [],
        "quality_score": result.get("quality_score"),
        "revision_count": result.get("revision_count") or 0,
        "critique_history": result.get("critique_history") or [],
        "thread_id": tid,
    }


def _write_log(
    case_input: dict[str, Any],
    report: dict[str, Any],
    result: dict[str, Any],
    *,
    thread_id: str,
) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_id = (report.get("case_id") or case_input.get("case_id") or "unknown").replace("/", "-")
    path = LOGS_DIR / f"{stamp}_{case_id}.json"
    messages = []
    for m in result.get("messages") or []:
        try:
            messages.append(message_to_dict(m))
        except Exception:
            messages.append({"content": str(m)})

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "case_input": case_input,
        "quality_score": result.get("quality_score"),
        "revision_count": result.get("revision_count"),
        "critique_history": result.get("critique_history"),
        "agent_notes": result.get("agent_notes"),
        "agent_messages": messages,
        "report_summary": {
            "title": report.get("title"),
            "predicted_outcomes": report.get("predicted_outcomes"),
            "strategy_count": len(report.get("win_strategies") or []),
        },
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote interaction log to %s", path)
