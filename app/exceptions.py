"""Application-specific exceptions."""

from __future__ import annotations


class LegalIntelligenceError(Exception):
    """Base error for the legal intelligence service."""


class ConfigurationError(LegalIntelligenceError):
    """Raised when required configuration is missing or invalid."""


class LLMError(LegalIntelligenceError):
    """Raised when the LLM endpoint is unreachable or returns an error."""


class AnalysisError(LegalIntelligenceError):
    """Raised when the agent pipeline fails."""
