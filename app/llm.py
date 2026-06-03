"""LLM client for Compass API — requires OPENAI_API_KEY and OPENAI_BASE_URL."""

from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.exceptions import ConfigurationError, LLMError

load_dotenv()

logger = logging.getLogger(__name__)

LEGAL_DISCLAIMER = (
    "This system provides legal information and research assistance only. "
    "It is not a law firm, does not provide legal advice, and does not "
    "create an attorney-client relationship. Outcomes, strategies, and "
    "predictions are probabilistic estimates based on historical cases from "
    "the Caselaw Access Project (https://case.law/) and may be incomplete or "
    "incorrect. Consult a licensed attorney in your jurisdiction before filing "
    "or relying on any output."
)


def _required_env(name: str) -> str:
    """Get required environment variable or raise RuntimeError."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"Missing required environment variable: {name}. "
            f"Set it in your .env file or pass it at runtime."
        )
    return value


def llm_available() -> bool:
    """True when both required LLM settings are present (does not check connectivity)."""
    return bool(os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"))


@lru_cache
def get_chat_model(model_name: str | None = None, temperature: float = 0.2) -> ChatOpenAI:
    """
    Returns a LangChain ChatOpenAI client configured for Compass.
    
    Args:
        model_name: Model name (defaults to OPENAI_MODEL env var)
        temperature: Temperature for generation (0.0-2.0)
    """
    api_key = _required_env("OPENAI_API_KEY")
    base_url = _required_env("OPENAI_BASE_URL")
    model = model_name or os.getenv("OPENAI_MODEL", "gpt-4.1")
    
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        temperature=temperature,
        timeout=60,
        max_retries=2,
    )


@lru_cache
def get_reasoning_model() -> ChatOpenAI:
    """
    Use Compass reasoning model for complex reasoning steps.
    This helps manage quota responsibly.
    """
    return get_chat_model(
        model_name=os.getenv("OPENAI_REASONING_MODEL", "gpt-5.1"),
    )


@lru_cache
def get_embedding_model() -> OpenAIEmbeddings:
    """
    Returns Compass embeddings for RAG workflows.
    """
    api_key = _required_env("OPENAI_API_KEY")
    base_url = _required_env("OPENAI_BASE_URL")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        api_key=api_key,
        base_url=base_url.rstrip("/"),
    )


def invoke_structured(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    temperature: float = 0.2,
    model_name: str | None = None,
) -> str:
    """Call the LLM and return text content; raises on config or runtime failure.
    
    Args:
        system_prompt: System prompt for the LLM
        user_payload: User input payload (will be JSON-serialized)
        temperature: Temperature for generation
        model_name: Model name (defaults to OPENAI_MODEL)
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        model = get_chat_model(model_name=model_name, temperature=temperature)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, indent=2, default=str)),
        ]
        response = model.invoke(messages)
    except ConfigurationError:
        raise
    except Exception as exc:
        raise LLMError(f"LLM request failed: {exc}") from exc

    content = getattr(response, "content", None)
    if not content or not str(content).strip():
        raise LLMError("LLM returned an empty response.")
    logger.debug("LLM response received (%d chars)", len(content))
    return str(content)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Parse JSON object from LLM output (raw or fenced code block)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                parsed = json.loads(text[start : end + 1])
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass
    raise LLMError("LLM response did not contain valid JSON.")


def invoke_json(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    temperature: float = 0.2,
    model_name: str | None = None,
) -> dict[str, Any]:
    """Call the LLM and parse a JSON object; raises on failure.
    
    Args:
        system_prompt: System prompt for the LLM
        user_payload: User input payload (will be JSON-serialized)
        temperature: Temperature for generation
        model_name: Model name (defaults to OPENAI_MODEL)
    """
    raw = invoke_structured(
        system_prompt + "\n\nRespond with valid JSON only.",
        user_payload,
        temperature=temperature,
        model_name=model_name,
    )
    return _extract_json_block(raw)


def default_caveats() -> list[str]:
    return [
        LEGAL_DISCLAIMER,
        "Predictions are not guarantees of court outcomes.",
        "Local rules, judges, and recent unpublished decisions may differ from CAP corpus.",
        "Verify all citations and procedural requirements with primary sources.",
    ]
