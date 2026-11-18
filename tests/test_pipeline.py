from docintel.agents import Pipeline
from docintel.llm import LLM


class NoLLM(LLM):
    def __init__(self):
        self.key = None


def test_pipeline_grounded_and_cited(index):
    r = Pipeline(index, NoLLM()).ask("How many attention heads does the Transformer use?")
    assert r["mode"] == "extractive"
    assert r["citations"] and "[" in r["answer"]
    assert r["status"] == "verified"
    assert r["verification"]["grounded_ratio"] >= 0.7
    assert r["citations"][0]["doc"] == "transformer.pdf"
    assert [t.split(":")[0] for t in r["trace"][:4]] == ["planner", "researcher", "analyst", "verifier"]


def test_verifier_flags_unsupported_claims(index):
    class Fake(NoLLM):
        available = True

        def complete(self, system, prompt, max_tokens=800):
            if "JSON" in system:
                return '{"sub_queries": ["attention heads"]}'
            return "Quantum blockchains replace all databases entirely [1]."

    r = Pipeline(index, Fake()).ask("How many attention heads does the Transformer use?")
    assert r["status"] == "low_confidence"
    assert r["attempts"] == 2  # retry loop fired, then gave up
    assert r["verification"]["claims"][0]["supported"] is False


def test_uncited_claims_unsupported(index):
    class Fake(NoLLM):
        available = True

        def complete(self, system, prompt, max_tokens=800):
            return "Transformers use eight attention heads." if "Answer" in system else '{"sub_queries": []}'

    r = Pipeline(index, Fake()).ask("How many attention heads does the Transformer use?")
    assert r["verification"]["grounded_ratio"] == 0.0
