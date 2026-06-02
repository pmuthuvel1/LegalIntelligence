"""LLM client — requires OPENAI_API_KEY and OPENAI_BASE_URL; fails if unreachable."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from app.config import (
    LEGAL_DISCLAIMER,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TIMEOUT,
    COMPASS_CHAT_MODEL,
    COMPASS_REASONING_MODEL,
    COMPASS_EMBEDDING_MODEL,
    COMPASS_WHISPER_MODEL,
    validate_llm_config,
)
from app.exceptions import ConfigurationError, LLMError

logger = logging.getLogger(__name__)

_llm_verified = False


def llm_available() -> bool:
    """True when both required LLM settings are present (does not check connectivity)."""
    return bool(OPENAI_API_KEY and OPENAI_BASE_URL)


def verify_llm_connectivity(*, force: bool = False) -> None:
    """
    Validate credentials and confirm the OpenAI-compatible endpoint responds.

    Raises ConfigurationError for missing/invalid config or auth failures.
    Raises LLMError when the endpoint cannot be reached.
    """
    global _llm_verified
    if _llm_verified and not force:
        return

    validate_llm_config()

    base = OPENAI_BASE_URL.rstrip("/")
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    models_url = f"{base}/models"
    connect_timeout = min(OPENAI_TIMEOUT, 15.0)

    try:
        with httpx.Client(timeout=connect_timeout) as client:
            resp = client.get(models_url, headers=headers)
    except httpx.RequestError as exc:
        raise LLMError(
            f"OPENAI_BASE_URL not reachable at {models_url}: {exc}"
        ) from exc

    if resp.status_code == 401:
        raise ConfigurationError(
            "OPENAI_API_KEY was rejected by the LLM endpoint (HTTP 401 Unauthorized)."
        )
    if resp.status_code == 403:
        raise ConfigurationError(
            "OPENAI_API_KEY lacks permission for the LLM endpoint (HTTP 403 Forbidden)."
        )
    if resp.status_code >= 500:
        raise LLMError(
            f"LLM endpoint error at {models_url} (HTTP {resp.status_code})."
        )
    if resp.status_code == 404:
        # Some compatible servers omit /models; verify base URL responds.
        try:
            with httpx.Client(timeout=connect_timeout) as client:
                base_resp = client.get(base, headers=headers)
        except httpx.RequestError as exc:
            raise LLMError(f"OPENAI_BASE_URL not reachable at {base}: {exc}") from exc
        if base_resp.status_code >= 500:
            raise LLMError(
                f"LLM endpoint error at {base} (HTTP {base_resp.status_code})."
            )
        if base_resp.status_code in (401, 403):
            raise ConfigurationError(
                f"OPENAI_API_KEY rejected by LLM endpoint (HTTP {base_resp.status_code})."
            )
    elif resp.status_code >= 400:
        raise LLMError(
            f"Unexpected response from LLM endpoint {models_url} (HTTP {resp.status_code})."
        )

    _llm_verified = True
    logger.info("LLM connectivity verified for %s", base)


def get_chat_model(*, temperature: float = 0.2, model_type: str = "chat"):
    """Build ChatOpenAI client; raises if configuration is missing.
    
    Args:
        temperature: Temperature for generation (0.0-2.0)
        model_type: Type of model - "chat" (default), "reasoning", or specific model name
    """
    validate_llm_config()

    from langchain_openai import ChatOpenAI

    # Select the appropriate model based on model_type
    if model_type == "reasoning":
        model_name = COMPASS_REASONING_MODEL
    elif model_type == "chat":
        model_name = COMPASS_CHAT_MODEL
    elif model_type == "embedding":
        model_name = COMPASS_EMBEDDING_MODEL
    else:
        # Treat as explicit model name
        model_name = model_type

    return ChatOpenAI(
        model=model_name,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL.rstrip("/"),
        temperature=temperature,
        timeout=OPENAI_TIMEOUT,
        max_tokens=OPENAI_MAX_TOKENS,
    )


def invoke_structured(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    temperature: float = 0.2,
    model_type: str = "chat",
) -> str:
    """Call the LLM and return text content; raises on config or runtime failure.
    
    Args:
        system_prompt: System prompt for the LLM
        user_payload: User input payload (will be JSON-serialized)
        temperature: Temperature for generation
        model_type: Type of model - "chat", "reasoning", or specific model name
    """
    verify_llm_connectivity()

    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        model = get_chat_model(temperature=temperature, model_type=model_type)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, indent=2, default=str)),
        ]
        response = model.invoke(messages)
    except (ConfigurationError, LLMError):
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
    model_type: str = "chat",
) -> dict[str, Any]:
    """Call the LLM and parse a JSON object; raises on failure.
    
    Args:
        system_prompt: System prompt for the LLM
        user_payload: User input payload (will be JSON-serialized)
        temperature: Temperature for generation
        model_type: Type of model - "chat", "reasoning", or specific model name
    """
    raw = invoke_structured(
        system_prompt + "\n\nRespond with valid JSON only.",
        user_payload,
        temperature=temperature,
        model_type=model_type,
    )
    return _extract_json_block(raw)


def default_caveats() -> list[str]:
    return [
        LEGAL_DISCLAIMER,
        "Predictions are not guarantees of court outcomes.",
        "Local rules, judges, and recent unpublished decisions may differ from CAP corpus.",
        "Verify all citations and procedural requirements with primary sources.",
    ]
