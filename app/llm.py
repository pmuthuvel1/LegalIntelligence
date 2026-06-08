"""High-level LLM helpers — thin wrappers around `app.compass` factories.

All model/client configuration lives in :mod:`app.compass`. This module focuses
on prompt invocation, response parsing, and shared disclaimers.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

# Re-export factories so existing imports `from app.llm import ...` keep working.
from app.compass import (  # noqa: F401
    _required_env,
    compass_available as llm_available,
    get_chat_model,
    get_embedding_model,
    get_reasoning_model,
    whisper_model_name as get_whisper_model_name,
)
from app.exceptions import ConfigurationError, LLMError

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


def invoke_structured(
    system_prompt: str,
    user_payload: dict[str, Any],
    *,
    temperature: float = 0.2,
    model_name: str | None = None,
) -> str:
    """Call the chat model and return its text content.

    Raises ``ConfigurationError`` for missing env vars or ``LLMError`` for
    runtime / empty-response failures.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        model = get_chat_model(model_name=model_name, temperature=temperature)
        response = model.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=json.dumps(user_payload, indent=2, default=str)),
        ])
    except ConfigurationError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise LLMError(f"LLM request failed: {exc}") from exc

    content = getattr(response, "content", None)
    if not content or not str(content).strip():
        raise LLMError("LLM returned an empty response.")
    logger.debug("LLM response received (%d chars)", len(content))
    return str(content)


def _extract_json_block(text: str) -> dict[str, Any]:
    """Parse a JSON object from LLM output (raw or fenced code block)."""
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
    """Call the chat model and parse a JSON object from the response."""
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

