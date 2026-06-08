"""Application configuration loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("LEGAL_DATA_DIR", PROJECT_ROOT / "data"))
CASE_INDEX_PATH = DATA_DIR / "case_index.json"
LOGS_DIR = Path(os.getenv("LEGAL_LOGS_DIR", PROJECT_ROOT / "logs"))
CHECKPOINT_DIR = Path(os.getenv("LEGAL_CHECKPOINT_DIR", PROJECT_ROOT / "logs" / "checkpoints"))

APP_ENV = os.getenv("APP_ENV", "development")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Compass API configuration.
# All values come from environment variables; see .env.example for canonical
# defaults. The :mod:`app.compass` module is the runtime single source of truth
# (it raises ConfigurationError when a required value is missing).
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
COMPASS_CHAT_MODEL = os.getenv("COMPASS_CHAT_MODEL", "")
COMPASS_REASONING_MODEL = os.getenv("COMPASS_REASONING_MODEL", "")
COMPASS_EMBEDDING_MODEL = os.getenv("COMPASS_EMBEDDING_MODEL", "")
COMPASS_WHISPER_MODEL = os.getenv("COMPASS_WHISPER_MODEL", "")
SAMPLE_MODE = os.getenv("SAMPLE_MODE", "false").lower() == "true"

# Optional: supplemental search via CourtListener (CAP data is hosted there)
COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")
COURTLISTENER_BASE = os.getenv(
    "COURTLISTENER_API_BASE", "https://www.courtlistener.com/api/rest/v4"
)

MAX_PRECEDENTS = int(os.getenv("MAX_PRECEDENTS", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.12"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
CRITIQUE_APPROVAL_THRESHOLD = float(os.getenv("CRITIQUE_APPROVAL_THRESHOLD", "70"))
ESCALATION_QUALITY_THRESHOLD = float(os.getenv("ESCALATION_QUALITY_THRESHOLD", "40"))

# Retriever retry / broadening behaviour.
# When the first BM25 pass returns fewer than ``LOW_PRECEDENT_FLOOR`` hits the
# research agent will retry with broader parameters (jurisdiction dropped,
# threshold relaxed) up to ``RETRIEVER_MAX_RETRIES`` times, emitting an explicit
# ``retry`` trace span for each attempt.
LOW_PRECEDENT_FLOOR = int(os.getenv("LOW_PRECEDENT_FLOOR", "3"))
RETRIEVER_MAX_RETRIES = int(os.getenv("RETRIEVER_MAX_RETRIES", "2"))
BROAD_SIMILARITY_THRESHOLD = float(
    os.getenv("BROAD_SIMILARITY_THRESHOLD", str(max(SIMILARITY_THRESHOLD / 3.0, 0.01)))
)

# Escalation Agent: if a live LLM call fails (quota, network, missing key) the
# service layer can fall back to a deterministic, no-LLM "sample mode" run that
# still produces a valid final report flagged ``human_review_required``.
LEGAL_FALLBACK_TO_SAMPLE = (
    os.getenv("LEGAL_FALLBACK_TO_SAMPLE", "true").lower() == "true"
)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8001"))
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:8001,http://127.0.0.1:8001").split(",")
    if o.strip()
]

LEGAL_DISCLAIMER = (
    "This system provides legal information and research assistance only. "
    "It is not a law firm, does not provide legal advice, and does not "
    "create an attorney-client relationship. Outcomes, strategies, and "
    "predictions are probabilistic estimates based on historical cases from "
    "the Caselaw Access Project (https://case.law/) and may be incomplete or "
    "incorrect. Consult a licensed attorney in your jurisdiction before filing "
    "or relying on any output."
)


def validate_config() -> list[str]:
    """Return non-fatal warnings for optional misconfiguration."""
    warnings: list[str] = []
    if not CASE_INDEX_PATH.exists():
        warnings.append(f"Case index missing: {CASE_INDEX_PATH}")
    return warnings
