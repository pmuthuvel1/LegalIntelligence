"""Integration tests for LangGraph pipeline."""

from app.graph import build_legal_intelligence_graph
from app.service import initialize


def test_pipeline_produces_report_with_critique():
    initialize()
    graph = build_legal_intelligence_graph(with_checkpointer=False)
    case = {
        "title": "Test Co v. Example LLC",
        "jurisdiction": "Arkansas",
        "court_type": "Circuit Court",
        "parties": {"plaintiff": "Test Co", "defendant": "Example LLC"},
        "claims": ["breach of contract"],
        "key_facts": ["Defendant refused delivery."],
    }
    result = graph.invoke({"case_input": case, "revision_count": 0, "approved": False})
    assert result.get("final_report")
    assert result.get("critique_history")
    assert result.get("quality_score") is not None
    assert "critique_summary" in result["final_report"]
