#!/usr/bin/env python3
"""Standard entry point for Legal Intelligence multi-agent system."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.logging_config import setup_logging
from app.service import initialize, run_case_analysis


def _load_input(path: Path | None) -> dict:
    if path:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.load(sys.stdin)


def main() -> int:
    setup_logging()
    initialize()

    parser = argparse.ArgumentParser(
        description="Run LangGraph legal intelligence pipeline with critique loops."
    )
    parser.add_argument("-i", "--input", type=Path, help="Path to case JSON input (default: stdin)")
    parser.add_argument("-o", "--output", type=Path, help="Write full report JSON to file (default: stdout)")
    parser.add_argument("--thread-id", help="Checkpoint thread ID for session continuity")
    parser.add_argument("--no-log", action="store_true", help="Skip writing logs/ interaction trace")
    args = parser.parse_args()

    case_input = _load_input(args.input)
    result = run_case_analysis(case_input, thread_id=args.thread_id, log=not args.no_log)

    payload = {
        "report": result["report"],
        "legal_caveats": result["legal_caveats"],
        "errors": result["errors"],
        "quality_score": result.get("quality_score"),
        "revision_count": result.get("revision_count"),
        "critique_history": result.get("critique_history"),
        "thread_id": result.get("thread_id"),
    }
    text = json.dumps(payload, indent=2, default=str)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote report to {args.output}", file=sys.stderr)
    else:
        print(text)

    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
