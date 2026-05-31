"""API smoke tests."""

from fastapi.testclient import TestClient

from app.api.app import create_app


def test_health_and_ready():
    client = TestClient(create_app())
    health = client.get("/v1/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    ready = client.get("/v1/ready")
    assert ready.status_code == 200
    body = ready.json()
    assert body["case_index_count"] >= 1


def test_analyze_endpoint():
    client = TestClient(create_app())
    payload = {
        "case": {
            "title": "API Test v. Defendant",
            "jurisdiction": "Arkansas",
            "court_type": "Circuit Court",
            "parties": {"plaintiff": "API Test", "defendant": "Defendant"},
            "claims": ["breach of contract"],
            "key_facts": ["Written contract breached."],
        },
        "skip_log": True,
    }
    resp = client.post("/v1/analyze", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["report"]["title"] == "API Test v. Defendant"
    assert data["quality_score"] is not None
    assert "legal_caveats" in data
