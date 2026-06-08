"""Smoke-test the structured trace wire format.

This script is safe to run with no external services. It exercises
:func:`app.logging_utils.write_trace` and :func:`log_event` and writes
the captured JSON lines plus a PASS/FAIL summary to ``smoke_out.txt``
in the repo root so the editor/agent can read the result back.
"""

from __future__ import annotations

import io
import json
import os
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    out = ROOT / "smoke_out.txt"
    lines: list[str] = []

    try:
        from app.logging_utils import (  # noqa: E402  (path manipulated above)
            ensure_trace_context,
            log_event,
            new_span_id,
            new_trace_file,
            new_trace_id,
            write_trace,
        )

        trace_id = new_trace_id()
        trace_file = new_trace_file(trace_id)
        state: dict = {"trace_id": trace_id, "trace_file": trace_file}

        # Capture the stdout the logger echoes so we can validate the wire
        # format without parsing the JSONL file.
        buf = io.StringIO()
        with redirect_stdout(buf):
            write_trace(
                trace_file=trace_file,
                trace_id=trace_id,
                span_id=new_span_id(),
                agent_name="PlannerAgent",
                action="receive_task",
            )
            log_event(
                state,
                agent_name="PlannerAgent",
                action="delegate",
                target_agent="ResearchAgent",
                confidence=0.42,
                retry_count=1,
                output_summary="hand off to research",
                extra={"reason": "smoke"},
            )
            log_event(
                state,
                agent_name="CriticAgent",
                action="escalate",
                target_agent="SupervisorAgent",
                confidence=0.31,
                status="warning",
                output_summary="quality below threshold",
            )

        emitted = [json.loads(line) for line in buf.getvalue().strip().splitlines()]
        lines.append(f"emitted_spans={len(emitted)}")

        required_top = ["timestamp", "traceid", "spanid", "agent_name", "action"]
        for i, rec in enumerate(emitted):
            for key in required_top:
                if key not in rec:
                    lines.append(f"FAIL span#{i} missing required key {key}")
                    out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
                    return 1
            lines.append(f"span#{i} keys=" + ",".join(rec.keys()))

        # First span: bare minimum, must have no optional keys.
        if any(k not in required_top for k in emitted[0]):
            lines.append(
                "FAIL span#0 has unexpected optional fields: "
                + ",".join(k for k in emitted[0] if k not in required_top)
            )
            out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
            return 1

        # Second span: delegate with optional fields.
        s1 = emitted[1]
        for key in ["target_agent", "confidence", "retry_count", "output_summary", "extra"]:
            if key not in s1:
                lines.append(f"FAIL span#1 missing optional key {key}")
                out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
                return 1

        # Third span: status surfaced because not success.
        s2 = emitted[2]
        if s2.get("status") != "warning":
            lines.append(f"FAIL span#2 status expected 'warning' got {s2.get('status')!r}")
            out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
            return 1

        # Verify legacy underscore-less aliases are gone.
        forbidden = ["agentname", "targetagent", "retrycount", "inputsummary", "outputsummary"]
        for i, rec in enumerate(emitted):
            for bad in forbidden:
                if bad in rec:
                    lines.append(f"FAIL span#{i} contains legacy key {bad}")
                    out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
                    return 1

        # Verify file contains the same content.
        on_disk = Path(trace_file).read_text(encoding="utf-8").strip().splitlines()
        lines.append(f"jsonl_lines={len(on_disk)}")
        if len(on_disk) != len(emitted):
            lines.append("FAIL stdout/file line count mismatch")
            out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
            return 1

        # Print a sample line for visual inspection.
        lines.append("sample_line=" + on_disk[0])
        out.write_text("\n".join(lines) + "\nRESULT=PASS\n")
        return 0

    except Exception:  # noqa: BLE001
        lines.append("EXCEPTION:")
        lines.append(traceback.format_exc())
        out.write_text("\n".join(lines) + "\nRESULT=FAIL\n")
        return 2


if __name__ == "__main__":
    sys.exit(main())

