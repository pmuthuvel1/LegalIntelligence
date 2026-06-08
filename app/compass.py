"""Compass API client helpers.

Centralized factory for LangChain ``ChatOpenAI`` / ``OpenAIEmbeddings`` clients
configured against the Compass (Core42) endpoint.

All configuration is sourced from environment variables (loaded via
``python-dotenv``). There are **no hardcoded model names or URLs** in this
module — the canonical defaults live in ``.env.example``. Copy that file to
``.env`` (or set the variables in your environment) before running the app.

Required environment variables:
    OPENAI_API_KEY            Compass API key
    OPENAI_BASE_URL           Compass endpoint (e.g. ``https://api.core42.ai/v1``)
    COMPASS_CHAT_MODEL        Chat completion model
    COMPASS_REASONING_MODEL   Reasoning model for complex steps
    COMPASS_EMBEDDING_MODEL   Embedding model for RAG workflows
    COMPASS_WHISPER_MODEL     Whisper / audio transcription model
"""

from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.exceptions import ConfigurationError

load_dotenv()


def _required_env(name: str) -> str:
    """Return required env var or raise :class:`ConfigurationError`."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file (see .env.example) or pass it at runtime."
        )
    return value


# ---------------------------------------------------------------------------
# Name resolvers — pure env lookups, raise if not configured.
# ---------------------------------------------------------------------------

def base_url() -> str:
    """Compass base URL (required env)."""
    return _required_env("OPENAI_BASE_URL").rstrip("/")


def chat_model_name() -> str:
    """Configured chat model name (required env)."""
    return _required_env("COMPASS_CHAT_MODEL")


def reasoning_model_name() -> str:
    """Configured reasoning model name (required env)."""
    return _required_env("COMPASS_REASONING_MODEL")


def embedding_model_name() -> str:
    """Configured embedding model name (required env)."""
    return _required_env("COMPASS_EMBEDDING_MODEL")


def whisper_model_name() -> str:
    """Configured Whisper / audio model name (required env)."""
    return _required_env("COMPASS_WHISPER_MODEL")


# Backwards-compatible alias.
get_whisper_model_name = whisper_model_name


def compass_available() -> bool:
    """True when the API key is present (other settings validated lazily)."""
    return bool(os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Client factories — cached, fully env-driven.
# ---------------------------------------------------------------------------

@lru_cache
def get_chat_model(
    model_name: str | None = None,
    temperature: float = 0.2,
) -> ChatOpenAI:
    """Return a LangChain ``ChatOpenAI`` client configured for Compass."""
    return ChatOpenAI(
        model=model_name or chat_model_name(),
        api_key=_required_env("OPENAI_API_KEY"),
        base_url=base_url(),
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


@lru_cache
def get_reasoning_model(temperature: float = 0.1) -> ChatOpenAI:
    """Reasoning model for complex steps — use sparingly to manage quota."""
    return get_chat_model(
        model_name=reasoning_model_name(),
        temperature=temperature,
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    """Return Compass embeddings client for RAG workflows."""
    return OpenAIEmbeddings(
        model=embedding_model_name(),
        api_key=_required_env("OPENAI_API_KEY"),
        base_url=base_url(),
    )

