import pytest


@pytest.mark.asyncio
async def test_root(client):
    response = await client.get("/")
    assert response.status_code == 200
    body = response.json()
    assert body["service"] == "llmops-platform"


@pytest.mark.asyncio
async def test_metrics_endpoint_exposes_prometheus_format(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert b"llm_inference_latency_seconds" in response.content


@pytest.mark.asyncio
async def test_ready_endpoint(client):
    response = await client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"ready": True}


@pytest.mark.asyncio
async def test_evaluation_score_endpoint(client):
    payload = {
        "prompt": "What is the capital of France?",
        "completion": "The capital of France is Paris.",
        "reference_answer": "Paris is the capital of France.",
    }
    response = await client.post("/evaluation/score", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert "relevance_score" in body
    assert "hallucination_score" in body
    assert 0.0 <= body["hallucination_score"] <= 1.0
