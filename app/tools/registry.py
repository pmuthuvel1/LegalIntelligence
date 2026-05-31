"""Singleton retriever registry with startup warmup."""

from __future__ import annotations

import logging
from threading import Lock

from app.tools.cap_retriever import CAPCaseRetriever
from app.tools.courtlistener import CourtListenerClient

logger = logging.getLogger(__name__)

_lock = Lock()
_cap_retriever: CAPCaseRetriever | None = None
_courtlistener: CourtListenerClient | None = None
_warmed = False


def get_cap_retriever() -> CAPCaseRetriever:
    global _cap_retriever
    if _cap_retriever is None:
        with _lock:
            if _cap_retriever is None:
                _cap_retriever = CAPCaseRetriever()
    return _cap_retriever


def get_courtlistener() -> CourtListenerClient:
    global _courtlistener
    if _courtlistener is None:
        with _lock:
            if _courtlistener is None:
                _courtlistener = CourtListenerClient()
    return _courtlistener


def warmup() -> dict[str, int | bool]:
    """Load indexes at startup; return readiness stats."""
    global _warmed
    retriever = get_cap_retriever()
    retriever.load()
    count = len(retriever._index)
    _warmed = True
    logger.info("CAP retriever warmed: %d indexed cases", count)
    return {
        "case_index_count": count,
        "courtlistener_enabled": get_courtlistener().enabled,
        "warmed": _warmed,
    }


def is_warmed() -> bool:
    return _warmed
