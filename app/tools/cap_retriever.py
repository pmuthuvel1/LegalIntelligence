"""Retrieve and rank cases from local Caselaw Access Project (CAP) samples."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import CASE_INDEX_PATH, DATA_DIR, MAX_PRECEDENTS, SIMILARITY_THRESHOLD


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]{3,}", text.lower())


class CAPCaseRetriever:
    """BM25-style retrieval over indexed CAP case excerpts."""

    def __init__(
        self,
        data_dir: Path | None = None,
        index_path: Path | None = None,
    ) -> None:
        self.data_dir = data_dir or DATA_DIR
        self.index_path = index_path or CASE_INDEX_PATH
        self._index: list[dict[str, Any]] = []
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.index_path.exists():
            self._index = json.loads(self.index_path.read_text(encoding="utf-8"))
        else:
            self._index = []
        self._loaded = True

    def search(
        self,
        query: str,
        *,
        jurisdiction: str | None = None,
        legal_issues: list[str] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        self.load()
        limit = limit or MAX_PRECEDENTS
        terms = _tokenize(query)
        if legal_issues:
            for issue in legal_issues:
                terms.extend(_tokenize(issue))

        if not terms:
            return []

        doc_freq: Counter[str] = Counter()
        doc_tokens: list[list[str]] = []
        for row in self._index:
            body = " ".join(
                [
                    row.get("name_abbreviation", ""),
                    row.get("name", ""),
                    row.get("court", ""),
                    " ".join(row.get("legal_issues", [])),
                    row.get("headnotes", ""),
                    row.get("snippet", ""),
                ]
            )
            tokens = _tokenize(body)
            doc_tokens.append(tokens)
            doc_freq.update(set(tokens))

        n_docs = max(len(self._index), 1)
        avg_dl = sum(len(t) for t in doc_tokens) / n_docs or 1.0
        k1, b = 1.2, 0.75

        scored: list[tuple[float, dict[str, Any]]] = []
        for idx, row in enumerate(self._index):
            if jurisdiction:
                j = (row.get("jurisdiction") or "").lower()
                if jurisdiction.lower() not in j and j not in jurisdiction.lower():
                    continue

            tokens = doc_tokens[idx]
            if not tokens:
                continue
            tf = Counter(tokens)
            dl = len(tokens)
            score = 0.0
            for term in set(terms):
                if term not in tf:
                    continue
                idf = math.log((n_docs - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5) + 1)
                freq = tf[term]
                score += idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * dl / avg_dl))

            if score >= SIMILARITY_THRESHOLD:
                scored.append((score, row))

        scored.sort(key=lambda x: x[0], reverse=True)
        results: list[dict[str, Any]] = []
        for score, row in scored[:limit]:
            case_path = self.data_dir / "cases" / row["file"]
            case_body: dict[str, Any] = {}
            if case_path.exists():
                case_body = json.loads(case_path.read_text(encoding="utf-8"))
            results.append(
                {
                    "relevance_score": round(score, 4),
                    "cap_id": row.get("id"),
                    "citation": row.get("citation"),
                    "name": row.get("name"),
                    "court": row.get("court"),
                    "decision_date": row.get("decision_date"),
                    "jurisdiction": row.get("jurisdiction"),
                    "outcome_for_plaintiff": row.get("outcome_for_plaintiff"),
                    "legal_issues": row.get("legal_issues", []),
                    "headnotes": row.get("headnotes"),
                    "snippet": row.get("snippet"),
                    "source_url": row.get("source_url"),
                    "full_case": case_body,
                }
            )
        return results
