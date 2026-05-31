#!/usr/bin/env python3
"""
Build case_index.json from CAP-style JSON files in data/cases/.

Download bulk CAP data from https://static.case.law/ (see https://case.law/docs/)
and place JSON case files under data/cases/, then run:

    python scripts/build_cap_index.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "data" / "cases"
INDEX_PATH = PROJECT_ROOT / "data" / "case_index.json"


def _snippet(text: str, max_len: int = 400) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean[:max_len] + ("..." if len(clean) > max_len else "")


def _row_from_case(path: Path, data: dict) -> dict:
    opinions = (data.get("casebody") or {}).get("opinions") or []
    text = " ".join(o.get("text", "") for o in opinions)
    citations = data.get("citations") or []
    cite = citations[0].get("cite") if citations else ""
    court = data.get("court") or {}
    jurisdiction = (data.get("jurisdiction") or {}).get("name") or ""

    return {
        "id": data.get("id") or path.stem,
        "file": path.name,
        "name": data.get("name", path.stem),
        "name_abbreviation": data.get("name_abbreviation", ""),
        "citation": cite,
        "court": court.get("name", "") if isinstance(court, dict) else str(court),
        "jurisdiction": jurisdiction,
        "decision_date": data.get("decision_date", ""),
        "outcome_for_plaintiff": data.get("outcome_for_plaintiff"),
        "legal_issues": data.get("legal_issues") or [],
        "headnotes": data.get("headnotes", ""),
        "snippet": _snippet(text or data.get("headnotes", "")),
        "source_url": data.get("source", "https://case.law/"),
    }


def main() -> None:
    rows = []
    for path in sorted(CASES_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_row_from_case(path, data))
    INDEX_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} entries to {INDEX_PATH}")


if __name__ == "__main__":
    main()
