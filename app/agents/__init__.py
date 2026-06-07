"""Specialized legal intelligence agents."""

from app.agents.critic import critic_agent
from app.agents.escalation import escalate_to_sample_mode
from app.agents.filing import filing_agent
from app.agents.intake import intake_agent
from app.agents.outcome import outcome_predictor_agent
from app.agents.precedent import precedent_analyst_agent
from app.agents.research import research_agent
from app.agents.strategy import strategy_agent
from app.agents.supervisor import supervisor_agent

__all__ = [
    "intake_agent",
    "research_agent",
    "precedent_analyst_agent",
    "outcome_predictor_agent",
    "strategy_agent",
    "filing_agent",
    "critic_agent",
    "supervisor_agent",
    "escalate_to_sample_mode",
]
