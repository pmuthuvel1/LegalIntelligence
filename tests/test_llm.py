"""Tests for LLM client configuration and validation."""

from unittest.mock import MagicMock, patch

import pytest

from app import llm
from app.exceptions import ConfigurationError


@pytest.mark.no_llm_mock
def test_get_chat_model_passes_base_url():
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.core42.ai/v1",
        "COMPASS_CHAT_MODEL": "gpt-test",
    }), patch("langchain_openai.ChatOpenAI") as mock_chat:
        # Clear the lru_cache before calling
        llm.get_chat_model.cache_clear()
        llm.get_chat_model()
        kwargs = mock_chat.call_args.kwargs
        assert kwargs["api_key"] == "test-key"
        assert kwargs["base_url"] == "https://api.core42.ai/v1"
        assert kwargs["model"] == "gpt-test"


def test_invoke_json_parses_fenced_json():
    with patch.object(llm, "invoke_structured", return_value='```json\n{"ok": true}\n```'):
        assert llm.invoke_json("sys", {"a": 1}) == {"ok": True}


@pytest.mark.no_llm_mock
def test_required_env_missing():
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ConfigurationError, match="Missing required environment variable"):
            llm._required_env("NONEXISTENT")


@pytest.mark.no_llm_mock
def test_llm_available_with_both_env_vars():
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.core42.ai/v1",
    }):
        assert llm.llm_available() is True


@pytest.mark.no_llm_mock
def test_llm_available_missing_key():
    with patch.dict("os.environ", {
        "OPENAI_BASE_URL": "https://api.core42.ai/v1",
    }, clear=True):
        assert llm.llm_available() is False


@pytest.mark.no_llm_mock
def test_llm_available_with_default_base_url():
    """Base URL has a sensible default; only the API key is strictly required."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
    }, clear=True):
        assert llm.llm_available() is True


@pytest.mark.no_llm_mock
def test_env_base_url_takes_precedence_over_default():
    """Value from environment must always take precedence over the built-in default."""
    with patch.dict("os.environ", {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_BASE_URL": "https://api.core42.ai/v1",
    }, clear=True), patch("langchain_openai.ChatOpenAI") as mock_chat:
        llm.get_chat_model.cache_clear()
        llm.get_chat_model()
        assert mock_chat.call_args.kwargs["base_url"] == "https://api.core42.ai/v1"
