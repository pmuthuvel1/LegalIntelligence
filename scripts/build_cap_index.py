#!/usr/bin/env python3
"""Build ``data/case_index.json`` from CAP-style JSON files in ``data/cases/``.

Handles every CAP file shape encountered in the wild:

* A single case dict (the legacy CAP API export and our hand-authored samples).
* A list of case dicts (bulk ``CasesMetadata.json`` / volume manifests from
  https://static.case.law/).
* Modern ``casebody.data.opinions`` (static.case.law) AND legacy
  ``casebody.opinions`` (CAP REST API) layouts.

Bad / empty records are skipped with a warning instead of aborting the build.

Usage::

    # 1. Download a CAP volume (or full reporter) from https://static.case.law/
    # 2. Drop the per-case JSON files into data/cases/
    # 3. python scripts/build_cap_index.py
"""

from __future__ import annotations

import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "data" / "cases"
INDEX_PATH = PROJECT_ROOT / "data" / "case_index.json"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _snippet(text: str, max_len: int = 400) -> str:
    clean = re.sub(r"\s+", " ", text or "").strip()
    return clean[:max_len] + ("..." if len(clean) > max_len else "")


def _opinion_text(casebody: Any) -> str:
    """Extract concatenated opinion text from any known casebody shape."""
    if not isinstance(casebody, dict):
        return casebody if isinstance(casebody, str) else ""

    # Modern static.case.law: casebody.data.opinions[*].text
    data = casebody.get("data")
    opinions = None
    if isinstance(data, dict):
        opinions = data.get("opinions")
    # Legacy CAP REST: casebody.opinions[*].text
    if not opinions:
        opinions = casebody.get("opinions")
    if not isinstance(opinions, list):
        return ""
    return " ".join(
        (o.get("text") or "") for o in opinions if isinstance(o, dict)
    )


def _head_matter(casebody: Any) -> str:
    if isinstance(casebody, dict):
        data = casebody.get("data")
        if isinstance(data, dict):
            return data.get("head_matter") or ""
    return ""


def _first_citation(case: dict) -> str:
    citations = case.get("citations") or []
    if isinstance(citations, list) and citations:
        first = citations[0]
        if isinstance(first, dict):
            return first.get("cite", "") or ""
        return str(first)
    return ""


def _court_name(case: dict) -> str:
    court = case.get("court")
    if isinstance(court, dict):
        return court.get("name") or court.get("name_abbreviation") or ""
    return str(court) if court else ""


def _jurisdiction_name(case: dict) -> str:
    j = case.get("jurisdiction")
    if isinstance(j, dict):
        return j.get("name") or j.get("name_long") or ""
    return str(j) if j else ""


def _looks_like_case(obj: Any) -> bool:
    """Heuristic: a dict that contains case-like fields."""
    if not isinstance(obj, dict):
        return False
    return any(
        k in obj
        for k in ("casebody", "decision_date", "citations", "name_abbreviation")
    )


def _iter_cases(payload: Any) -> Iterable[dict]:
    """Yield every case-shaped dict found in a parsed JSON payload.

    Accepts a single case dict, a list of case dicts, or a wrapper object
    that contains a ``cases`` / ``results`` list (some CAP exports).
    """
    if isinstance(payload, dict):
        if _looks_like_case(payload):
            yield payload
            return
        for key in ("cases", "results", "data"):
            inner = payload.get(key)
            if isinstance(inner, list):
                for item in inner:
                    if _looks_like_case(item):
                        yield item
                return
        # Last resort: dict but doesn't look like a case → skip.
        return
    if isinstance(payload, list):
        for item in payload:
            yield from _iter_cases(item)


def _row_from_case(source_file: Path, case_index_in_file: int, data: dict) -> dict | None:
    casebody = data.get("casebody")
    # static.case.law marks unavailable text with status != "ok".
    if isinstance(casebody, dict) and casebody.get("status") and casebody.get("status") != "ok":
        return None

    opinion_text = _opinion_text(casebody)
    head = _head_matter(casebody)
    headnotes = data.get("headnotes") or head or ""

    snippet_source = opinion_text or headnotes or data.get("name", "")
    if not snippet_source.strip():
        return None  # nothing useful to index

    case_id = data.get("id") or f"{source_file.stem}-{case_index_in_file}"
    return {
        "id": str(case_id),
        "file": source_file.name,
        "case_index_in_file": case_index_in_file,
        "name": data.get("name") or data.get("name_abbreviation") or source_file.stem,
        "name_abbreviation": data.get("name_abbreviation", ""),
        "citation": _first_citation(data),
        "court": _court_name(data),
        "jurisdiction": _jurisdiction_name(data),
        "decision_date": data.get("decision_date", ""),
        "outcome_for_plaintiff": data.get("outcome_for_plaintiff"),
        "legal_issues": data.get("legal_issues") or [],
        "headnotes": headnotes,
        "snippet": _snippet(snippet_source),
        "source_url": data.get("frontend_url")
        or data.get("source")
        or "https://case.law/",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not CASES_DIR.exists():
        print(f"[ERROR] {CASES_DIR} does not exist.", file=sys.stderr)
        sys.exit(1)

    rows: list[dict] = []
    files_seen = 0
    files_skipped = 0
    cases_skipped = 0

    for path in sorted(CASES_DIR.glob("*.json")):
        files_seen += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            files_skipped += 1
            print(f"[skip] {path.name}: cannot parse JSON ({exc})", file=sys.stderr)
            continue

        found_any = False
        for idx, case in enumerate(_iter_cases(payload)):
            try:
                row = _row_from_case(path, idx, case)
            except Exception:  # noqa: BLE001 — never let one bad case kill the build
                cases_skipped += 1
                traceback.print_exc(limit=1)
                continue
            if row is None:
                cases_skipped += 1
                continue
            rows.append(row)
            found_any = True

        if not found_any:
            files_skipped += 1
            print(f"[skip] {path.name}: no case-shaped records found", file=sys.stderr)

    INDEX_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(
        f"Wrote {len(rows)} entries to {INDEX_PATH}  "
        f"(files seen: {files_seen}, files skipped: {files_skipped}, "
        f"cases skipped: {cases_skipped})"
    )


if __name__ == "__main__":
    main()

