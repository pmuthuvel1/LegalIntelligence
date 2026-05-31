"""Tests for LLM client configuration and validation."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from app import llm
from app.config import validate_llm_config
from app.exceptions import ConfigurationError, LLMError


@pytest.mark.no_llm_mock
def test_get_chat_model_passes_base_url():
    with patch("app.config.OPENAI_API_KEY", "test-key"), patch(
        "app.config.OPENAI_BASE_URL", "https://api.example.com/v1"
    ), patch.object(llm, "OPENAI_API_KEY", "test-key"), patch.object(
        llm, "OPENAI_BASE_URL", "https://api.example.com/v1"
    ), patch.object(llm, "OPENAI_MODEL", "gpt-test"), patch(
        "langchain_openai.ChatOpenAI"
    ) as mock_chat:
        llm.get_chat_model()
        kwargs = mock_chat.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "https://api.example.com/v1"
        assert kwargs["model"] == "gpt-test"


def test_invoke_json_parses_fenced_json():
    with patch.object(llm, "invoke_structured", return_value='```json\n{"ok": true}\n```'):
        assert llm.invoke_json("sys", {"a": 1}) == {"ok": True}


@pytest.mark.no_llm_mock
def test_validate_llm_config_missing_key():
    with patch("app.config.OPENAI_API_KEY", ""), patch(
        "app.config.OPENAI_BASE_URL", "https://api.example.com/v1"
    ):
        with pytest.raises(ConfigurationError, match="OPENAI_API_KEY"):
            validate_llm_config()


@pytest.mark.no_llm_mock
def test_validate_llm_config_missing_base_url():
    with patch("app.config.OPENAI_API_KEY", "sk-test"), patch("app.config.OPENAI_BASE_URL", ""):
        with pytest.raises(ConfigurationError, match="OPENAI_BASE_URL"):
            validate_llm_config()


@pytest.mark.no_llm_mock
def test_verify_llm_connectivity_unreachable():
    with patch.object(llm, "_llm_verified", False), patch(
        "app.config.OPENAI_API_KEY", "sk-test"
    ), patch("app.config.OPENAI_BASE_URL", "https://api.example.com/v1"), patch(
        "httpx.Client"
    ) as mock_client:
        mock_client.return_value.__enter__.return_value.get.side_effect = httpx.ConnectError(
            "Connection refused", request=MagicMock()
        )
        with pytest.raises(LLMError, match="not reachable"):
            llm.verify_llm_connectivity(force=True)


@pytest.mark.no_llm_mock
def test_verify_llm_connectivity_unauthorized():
    with patch.object(llm, "_llm_verified", False), patch(
        "app.config.OPENAI_API_KEY", "bad-key"
    ), patch("app.config.OPENAI_BASE_URL", "https://api.example.com/v1"), patch(
        "httpx.Client"
    ) as mock_client:
        mock_resp = MagicMock()
        mock_resp.status_code = 401
        mock_client.return_value.__enter__.return_value.get.return_value = mock_resp
        with pytest.raises(ConfigurationError, match="401"):
            llm.verify_llm_connectivity(force=True)
