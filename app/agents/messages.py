"""Helpers for inter-agent message passing."""

from __future__ import annotations

from langchain_core.messages import AIMessage


def agent_message(agent_name: str, content: str) -> AIMessage:
    return AIMessage(content=content, name=agent_name)
