from fastapi.testclient import TestClient

import docintel.api as api
from docintel.agents import Pipeline
from tests.test_pipeline import NoLLM


def test_api_endpoints(index):
    api.PIPELINE = Pipeline(index, NoLLM())
    c = TestClient(api.app)
    assert c.get("/health").json()["chunks"] == len(index.chunks)
    assert "bert.pdf" in c.get("/documents").json()
    r = c.post("/query", json={"question": "What objectives is BERT pre-trained with?"})
    assert r.status_code == 200 and r.json()["citations"]
    assert c.post("/query", json={"question": "hi"}).status_code == 422
