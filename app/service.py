"""High-level service API for CLI and HTTP layers.

`run_case_analysis` is the single entry point used by `run.py` and the FastAPI
endpoint. It always returns a dict that matches one of two canonical shapes:

Success::

    {
      "run_id": "...",
      "status": "success",
      "output": {"summary": "...", "recommendations": [...], "artifacts": [...]},
      "agents": [{"name": "...", "role": "..."}, ...],
      "trace_id": "...",
      "log_file": "...",
      "execution_time_seconds": 42.7
    }

Error::

    {
      "run_id": "...",
      "status": "error",
      "error": {"type": "...", "message": "...", "recoverable": true|false},
      "trace_id": "...",
      "log_file": "...",
      "execution_time_seconds": 1.2
    }
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import message_to_dict

from app.config import LEGAL_FALLBACK_TO_SAMPLE, LOGS_DIR, PROJECT_ROOT, validate_config
from app.exceptions import AnalysisError, ConfigurationError, LLMError
from app.graph import LEGAL_GRAPH
from app.agents import escalate_to_sample_mode
from app.llm import llm_available
from app.logging_utils import new_trace_file, new_trace_id
from app.tools.registry import is_warmed, warmup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Agent registry (from metadata.json — single source of truth)
# ---------------------------------------------------------------------------

@lru_cache
def _agents_from_metadata() -> list[dict[str, str]]:
    path = PROJECT_ROOT / "metadata.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [
            {"name": a.get("id", ""), "role": a.get("role", "")}
            for a in data.get("agents") or []
        ]
    except (OSError, json.JSONDecodeError):
        return []


# ---------------------------------------------------------------------------
# Lifecycle hooks
# ---------------------------------------------------------------------------

def _ensure_llm_ready() -> None:
    if not llm_available():
        raise ConfigurationError(
            "Required LLM configuration missing: OPENAI_API_KEY. "
            "Set it in the environment or .env file."
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
    if not is_warmed():
        try:
            stats = warmup()
        except Exception as exc:  # noqa: BLE001
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


# ---------------------------------------------------------------------------
# Run envelope builders
# ---------------------------------------------------------------------------

_RECOVERABLE_ERRORS = (LLMError,)  # transient upstream / quota issues


def _classify_error(exc: BaseException) -> tuple[str, bool]:
    """Return ``(error_type, recoverable)`` for the error envelope."""
    name = type(exc).__name__
    # Heuristic: surface CompassQuotaError if the message hints at quota / rate limits.
    msg = str(exc).lower()
    if isinstance(exc, LLMError) and any(
        s in msg for s in ("quota", "rate limit", "429", "too many requests")
    ):
        return "CompassQuotaError", True
    return name, isinstance(exc, _RECOVERABLE_ERRORS)


def _success_envelope(
    *,
    run_id: str,
    started_at: float,
    report: dict[str, Any],
    result: dict[str, Any],
    log_file: str | None,
) -> dict[str, Any]:
    critique = result.get("critique_report") or {}
    recommendations = critique.get("recommendations") or []
    if not isinstance(recommendations, list):
        recommendations = [str(recommendations)]

    artifacts: list[dict[str, Any]] = [
        {"name": "final_report", "type": "json", "data": report},
    ]
    filing = report.get("filing_package") or result.get("filing_package")
    if filing:
        artifacts.append({"name": "filing_package", "type": "json", "data": filing})
    predicted = report.get("predicted_outcomes")
    if predicted:
        artifacts.append(
            {"name": "predicted_outcomes", "type": "json", "data": predicted}
        )
    if log_file:
        artifacts.append({"name": "interaction_log", "type": "jsonl", "path": log_file})

    summary = (
        report.get("executive_summary")
        or report.get("critique_summary", {}).get("partner_critique")
        or "Multi-agent legal intelligence pipeline completed."
    )

    return {
        "run_id": run_id,
        "status": "success",
        "output": {
            "summary": str(summary),
            "recommendations": [str(r) for r in recommendations],
            "artifacts": artifacts,
        },
        "agents": _agents_from_metadata(),
        "trace_id": result.get("trace_id"),
        "log_file": result.get("trace_file") or log_file,
        "execution_time_seconds": round(time.perf_counter() - started_at, 3),
    }


def _error_envelope(
    *,
    run_id: str,
    started_at: float,
    exc: BaseException,
    trace_id: str | None,
    log_file: str | None,
) -> dict[str, Any]:
    err_type, recoverable = _classify_error(exc)
    return {
        "run_id": run_id,
        "status": "error",
        "error": {
            "type": err_type,
            "message": str(exc),
            "recoverable": recoverable,
        },
        "trace_id": trace_id,
        "log_file": log_file,
        "execution_time_seconds": round(time.perf_counter() - started_at, 3),
    }


def _run_sample_mode_fallback(
    case_input: dict[str, Any],
    *,
    rid: str,
    started_at: float,
    trace_id: str,
    trace_file: str,
    exc: BaseException,
    thread_id: str | None = None,
) -> dict[str, Any]:
    """Invoke the escalation agent and wrap its output in a success envelope.

    The envelope is still ``status='success'`` so HTTP clients get a usable
    report, but every downstream signal (``human_review_required``, the extra
    artifact, the partner critique recommendations) makes the degraded mode
    obvious. The trace JSONL contains the live→sample handoff span.
    """
    err_type, recoverable = _classify_error(exc)
    try:
        result = escalate_to_sample_mode(
            case_input,
            trace_id=trace_id,
            trace_file=trace_file,
            reason=f"{err_type}: {exc}",
            original_error=exc,
        )
    except Exception as inner:  # noqa: BLE001
        logger.exception("Sample-mode fallback itself failed for run %s", rid)
        return _error_envelope(
            run_id=rid,
            started_at=started_at,
            exc=AnalysisError(
                f"Sample-mode fallback failed after {err_type}: {inner}"
            ),
            trace_id=trace_id,
            log_file=trace_file,
        )

    report = result.get("final_report") or {}
    interaction_log = _write_interaction_log(
        case_input,
        report,
        result,
        thread_id=thread_id or f"case-{uuid.uuid4().hex[:12]}",
        run_id=rid,
    )

    envelope = _success_envelope(
        run_id=rid,
        started_at=started_at,
        report=report,
        result=result,
        log_file=interaction_log,
    )
    # Surface the degraded mode and the originating error to callers.
    envelope.setdefault("output", {}).setdefault("artifacts", []).append(
        {
            "name": "escalation",
            "type": "json",
            "data": {
                "degraded_mode": "sample",
                "error_type": err_type,
                "message": str(exc),
                "recoverable": recoverable,
                "human_review_required": True,
                "reasons": result.get("escalation_reasons"),
            },
        }
    )
    return envelope


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_case_analysis(
    case_input: dict[str, Any],
    *,
    thread_id: str | None = None,
    run_id: str | None = None,
    log: bool = True,
) -> dict[str, Any]:
    """Execute the multi-agent pipeline and return the canonical envelope."""
    started_at = time.perf_counter()
    rid = run_id or f"run-{uuid.uuid4().hex[:12]}"
    trace_id = new_trace_id()
    trace_file = new_trace_file(trace_id)

    # Pre-flight: missing LLM config triggers the escalation agent's sample-mode
    # fallback when LEGAL_FALLBACK_TO_SAMPLE is enabled, otherwise returns a
    # clean error envelope (never a 500).
    try:
        _ensure_llm_ready()
        if not is_warmed():
            warmup()
    except (ConfigurationError, LLMError) as exc:
        if LEGAL_FALLBACK_TO_SAMPLE:
            return _run_sample_mode_fallback(
                case_input,
                rid=rid,
                started_at=started_at,
                trace_id=trace_id,
                trace_file=trace_file,
                exc=exc,
            )
        return _error_envelope(
            run_id=rid,
            started_at=started_at,
            exc=exc,
            trace_id=trace_id,
            log_file=trace_file,
        )

    tid = thread_id or f"case-{uuid.uuid4().hex[:12]}"
    config = {"configurable": {"thread_id": tid}}
    initial: dict[str, Any] = {
        "case_input": case_input,
        "revision_count": 0,
        "approved": False,
        "trace_id": trace_id,
        "trace_file": trace_file,
    }

    try:
        result = LEGAL_GRAPH.invoke(initial, config=config)
    except (ConfigurationError, LLMError) as exc:
        if LEGAL_FALLBACK_TO_SAMPLE:
            return _run_sample_mode_fallback(
                case_input,
                rid=rid,
                started_at=started_at,
                trace_id=trace_id,
                trace_file=trace_file,
                exc=exc,
                thread_id=tid,
            )
        return _error_envelope(
            run_id=rid,
            started_at=started_at,
            exc=exc,
            trace_id=trace_id,
            log_file=trace_file,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline failed for run %s", rid)
        return _error_envelope(
            run_id=rid,
            started_at=started_at,
            exc=AnalysisError(f"Analysis pipeline failed: {exc}"),
            trace_id=trace_id,
            log_file=trace_file,
        )

    report = result.get("final_report") or {}
    if not report:
        return _error_envelope(
            run_id=rid,
            started_at=started_at,
            exc=AnalysisError("Pipeline completed without a final report."),
            trace_id=result.get("trace_id") or trace_id,
            log_file=result.get("trace_file") or trace_file,
        )

    interaction_log: str | None = None
    if log:
        interaction_log = _write_interaction_log(
            case_input, report, result, thread_id=tid, run_id=rid
        )

    return _success_envelope(
        run_id=rid,
        started_at=started_at,
        report=report,
        result=result,
        log_file=interaction_log,
    )


# ---------------------------------------------------------------------------
# Detailed interaction log (separate from per-span trace JSONL)
# ---------------------------------------------------------------------------

def _write_interaction_log(
    case_input: dict[str, Any],
    report: dict[str, Any],
    result: dict[str, Any],
    *,
    thread_id: str,
    run_id: str,
) -> str:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    case_id = (
        report.get("case_id") or case_input.get("case_id") or run_id
    ).replace("/", "-")
    path: Path = LOGS_DIR / f"{stamp}_{case_id}.json"

    messages = []
    for m in result.get("messages") or []:
        try:
            messages.append(message_to_dict(m))
        except Exception:  # noqa: BLE001
            messages.append({"content": str(m)})

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "thread_id": thread_id,
        "trace_id": result.get("trace_id"),
        "trace_file": result.get("trace_file"),
        "case_input": case_input,
        "quality_score": result.get("quality_score"),
        "revision_count": result.get("revision_count"),
        "human_review_required": result.get("human_review_required"),
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
    return str(path)

