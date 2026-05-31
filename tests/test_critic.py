"""Tests for critic agent quality scoring."""

from app.agents.critic import critic_agent


def test_critic_passes_strong_analysis():
    state = {
        "revision_count": 0,
        "precedent_analysis": {
            "precedent_count": 6,
            "plaintiff_win_rate_in_sample": 0.6,
            "unfavorable_precedents": [{"citation": "X v. Y"}],
        },
        "predicted_outcomes": {
            "confidence": "moderate",
            "scenarios": [{"scenario": "most_likely", "estimated_probability": 0.55}],
        },
        "win_strategies": [
            {
                "title": "Neutralize adverse precedent",
                "actions": ["Distinguish adverse case X v. Y on facts."],
            },
            {"title": "A", "actions": ["a"]},
            {"title": "B", "actions": ["b"]},
        ],
    }
    result = critic_agent(state)
    assert result["critique_report"]["quality_score"] >= 70
    assert result["critique_report"]["approved_by_critic"] is True


def test_critic_fails_thin_precedents():
    state = {
        "revision_count": 0,
        "precedent_analysis": {"precedent_count": 1},
        "predicted_outcomes": {"confidence": "very_low", "scenarios": []},
        "win_strategies": [{"title": "Only one", "actions": []}],
    }
    result = critic_agent(state)
    assert result["critique_report"]["approved_by_critic"] is False
    assert result["rewrite_target"] in ("research", "strategy")
