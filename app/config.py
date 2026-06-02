"""Application configuration loaded from environment."""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

from app.exceptions import ConfigurationError

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("LEGAL_DATA_DIR", PROJECT_ROOT / "data"))
CASE_INDEX_PATH = DATA_DIR / "case_index.json"
LOGS_DIR = Path(os.getenv("LEGAL_LOGS_DIR", PROJECT_ROOT / "logs"))
CHECKPOINT_DIR = Path(os.getenv("LEGAL_CHECKPOINT_DIR", PROJECT_ROOT / "logs" / "checkpoints"))

APP_ENV = os.getenv("APP_ENV", "development")
APP_VERSION = os.getenv("APP_VERSION", "2.0.0")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TIMEOUT = float(os.getenv("OPENAI_TIMEOUT", "60"))
OPENAI_MAX_TOKENS = int(os.getenv("OPENAI_MAX_TOKENS", "2048"))

# COMPASS Model Selection
COMPASS_CHAT_MODEL = os.getenv("COMPASS_CHAT_MODEL", "gpt-4o-mini")
COMPASS_REASONING_MODEL = os.getenv("COMPASS_REASONING_MODEL", "gpt-4o")
COMPASS_EMBEDDING_MODEL = os.getenv("COMPASS_EMBEDDING_MODEL", "text-embedding-3-large")
COMPASS_WHISPER_MODEL = os.getenv("COMPASS_WHISPER_MODEL", "whisper-1")
SAMPLE_MODE = os.getenv("SAMPLE_MODE", "false").lower() == "true"

COURTLISTENER_API_TOKEN = os.getenv("COURTLISTENER_API_TOKEN", "")
COURTLISTENER_BASE = os.getenv(
    "COURTLISTENER_API_BASE", "https://www.courtlistener.com/api/rest/v4"
)

MAX_PRECEDENTS = int(os.getenv("MAX_PRECEDENTS", "8"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.12"))
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "2"))
CRITIQUE_APPROVAL_THRESHOLD = float(os.getenv("CRITIQUE_APPROVAL_THRESHOLD", "70"))

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


def validate_llm_config() -> None:
    """Raise ConfigurationError if required LLM environment variables are missing."""
    missing: list[str] = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not OPENAI_BASE_URL:
        missing.append("OPENAI_BASE_URL")
    if missing:
        raise ConfigurationError(
            f"Required LLM configuration missing: {', '.join(missing)}. "
            "Set both OPENAI_API_KEY and OPENAI_BASE_URL in the environment or .env file."
        )

    parsed = urlparse(OPENAI_BASE_URL)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigurationError(
            f"OPENAI_BASE_URL is invalid: {OPENAI_BASE_URL!r}. "
            "Expected a full URL such as https://api.openai.com/v1"
        )


def validate_config() -> list[str]:
    """Return non-fatal warnings for optional misconfiguration."""
    warnings: list[str] = []
    if not CASE_INDEX_PATH.exists():
        warnings.append(f"Case index missing: {CASE_INDEX_PATH}")
    return warnings
