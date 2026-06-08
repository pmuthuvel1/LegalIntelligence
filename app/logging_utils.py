"""Structured agent-to-agent trace logging.

Emits one JSON line per span to a per-run JSONL file under ``LOGS_DIR/traces``
**and** echoes the same line to stdout for evaluator visibility.

Wire format (matches the contract requested by the evaluator)::

    {"timestamp":"2026-05-24T10:15:31.420Z","traceid":"trace-...",
     "spanid":"span-...","agent_name":"PlannerAgent","action":"receive_task"}

Optional fields appear only when set: ``target_agent``, ``confidence``,
``retry_count``, ``status`` (other than ``success``), ``input_summary``,
``output_summary``, ``extra``.

This module also exposes a :func:`traced_agent` decorator that wraps a
LangGraph node so every invocation is bracketed by ``receive_task`` /
``complete_task`` (or ``error``) spans without touching agent bodies.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Callable, Dict, Mapping, Optional

from app.config import LOGS_DIR

TRACE_DIR = LOGS_DIR / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Identifiers & timestamps
# ---------------------------------------------------------------------------

def utc_now_ms() -> str:
    """ISO-8601 timestamp with millisecond precision and ``Z`` suffix."""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def new_trace_id() -> str:
    return f"trace-{uuid.uuid4().hex[:12]}"


def new_span_id() -> str:
    return f"span-{uuid.uuid4().hex[:10]}"


def new_trace_file(trace_id: Optional[str] = None) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"_{trace_id}" if trace_id else ""
    return str(TRACE_DIR / f"agent_trace_{ts}{suffix}.jsonl")


# ---------------------------------------------------------------------------
# Low-level writer
# ---------------------------------------------------------------------------

def _summarize(value: Any, limit: int = 300) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value[:limit]
    try:
        return json.dumps(value, default=str, ensure_ascii=False)[:limit]
    except Exception:  # noqa: BLE001
        return str(value)[:limit]


def write_trace(
    *,
    trace_file: str,
    trace_id: str,
    span_id: str,
    agent_name: str,
    action: str,
    target_agent: Optional[str] = None,
    confidence: Optional[float] = None,
    retry_count: int = 0,
    status: str = "success",
    input_summary: str = "",
    output_summary: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write one structured trace record to ``trace_file`` and stdout."""
    record: Dict[str, Any] = {
        "timestamp": utc_now_ms(),
        "traceid": trace_id,
        "spanid": span_id,
        "agent_name": agent_name,
        "action": action,
    }
    if target_agent:
        record["target_agent"] = target_agent
    if confidence is not None:
        record["confidence"] = round(float(confidence), 4)
    if retry_count:
        record["retry_count"] = retry_count
    if status and status != "success":
        record["status"] = status
    if input_summary:
        record["input_summary"] = input_summary[:500]
    if output_summary:
        record["output_summary"] = output_summary[:800]
    if extra:
        record["extra"] = extra

    line = json.dumps(record, ensure_ascii=False)
    try:
        with open(trace_file, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    except OSError:
        # Never break the workflow because of a logging failure.
        pass
    print(line)
    return record


# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def ensure_trace_context(state: Mapping[str, Any]) -> tuple[str, str]:
    """Return ``(trace_id, trace_file)`` from state, creating them if absent."""
    trace_id = state.get("trace_id") or new_trace_id()
    trace_file = state.get("trace_file") or new_trace_file(trace_id)
    return trace_id, trace_file


def log_event(
    state: Mapping[str, Any],
    *,
    agent_name: str,
    action: str,
    target_agent: Optional[str] = None,
    confidence: Optional[float] = None,
    retry_count: int = 0,
    status: str = "success",
    input_summary: str = "",
    output_summary: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit an ad-hoc span (e.g. ``delegate``, ``validate``, ``escalate``)."""
    trace_id, trace_file = ensure_trace_context(state)
    write_trace(
        trace_file=trace_file,
        trace_id=trace_id,
        span_id=new_span_id(),
        agent_name=agent_name,
        action=action,
        target_agent=target_agent,
        confidence=confidence,
        retry_count=retry_count,
        status=status,
        input_summary=input_summary,
        output_summary=output_summary,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# Node decorator
# ---------------------------------------------------------------------------

def traced_agent(name: str) -> Callable:
    """Wrap a LangGraph node to emit ``receive_task`` and ``complete_task``.

    The wrapped function is also responsible for seeding ``trace_id`` /
    ``trace_file`` on the shared state for the rest of the workflow.
    """

    def decorator(func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> Callable:
        @wraps(func)
        def wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
            trace_id, trace_file = ensure_trace_context(state)
            span_id = new_span_id()

            write_trace(
                trace_file=trace_file,
                trace_id=trace_id,
                span_id=span_id,
                agent_name=name,
                action="receive_task",
                target_agent=name,
                retry_count=state.get("revision_count") or 0,
                input_summary=_summarize(
                    {
                        "state_keys": sorted(state.keys()),
                        "revision": state.get("revision_count"),
                        "approved": state.get("approved"),
                    }
                ),
            )

            try:
                result = func(state) or {}
            except Exception as exc:  # noqa: BLE001
                write_trace(
                    trace_file=trace_file,
                    trace_id=trace_id,
                    span_id=span_id,
                    agent_name=name,
                    action="error",
                    status="error",
                    output_summary=f"{type(exc).__name__}: {exc}",
                )
                raise

            # Propagate trace context through the shared state.
            result.setdefault("trace_id", trace_id)
            result.setdefault("trace_file", trace_file)

            messages = result.get("messages") or []
            last_msg = ""
            if messages:
                tail = messages[-1]
                last_msg = getattr(tail, "content", str(tail))

            write_trace(
                trace_file=trace_file,
                trace_id=trace_id,
                span_id=span_id,
                agent_name=name,
                action="complete_task",
                output_summary=_summarize(
                    {"keys": list(result.keys()), "message": last_msg}
                ),
            )
            return result

        return wrapper

    return decorator

