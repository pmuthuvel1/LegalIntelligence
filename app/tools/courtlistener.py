"""Optional CourtListener API client (CAP data is also hosted there)."""

from __future__ import annotations

from typing import Any

import httpx

from app.config import COURTLISTENER_API_TOKEN, COURTLISTENER_BASE


class CourtListenerClient:
    """Thin REST client for supplemental precedent search."""

    def __init__(self, token: str | None = None, base_url: str | None = None) -> None:
        self.token = token or COURTLISTENER_API_TOKEN
        self.base_url = (base_url or COURTLISTENER_BASE).rstrip("/")

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    def search_opinions(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not self.enabled:
            return []

        headers = {"Authorization": f"Token {self.token}"}
        params = {"q": query, "order_by": "score desc", "page_size": limit}
        url = f"{self.base_url}/search/"
        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.get(url, headers=headers, params=params)
                resp.raise_for_status()
                payload = resp.json()
        except (httpx.HTTPError, ValueError):
            return []

        results: list[dict[str, Any]] = []
        for item in payload.get("results", [])[:limit]:
            results.append(
                {
                    "relevance_score": item.get("score"),
                    "name": item.get("caseName") or item.get("case_name"),
                    "citation": item.get("citation"),
                    "court": item.get("court"),
                    "decision_date": item.get("dateFiled") or item.get("date_filed"),
                    "snippet": item.get("snippet"),
                    "source_url": item.get("absolute_url"),
                    "source": "courtlistener",
                }
            )
        return results
