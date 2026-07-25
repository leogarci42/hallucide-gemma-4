from hallucide import RiskTier, apply_risk_floor, advance_retrieval, RetrievalProvider
from hallucide.core_types.exceptions import RetrievalError
from hallucide.core_types.types import Intent, Passage, RetrievalState


class DummyRetrievalProvider:
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        return Passage(
            source_id=query["source_id"],
            source_type="normatif",
            opposable=True,
            text="Extrait de texte officiel.",
            metadata={"query": query},
        )


def test_risk_floor_elevated_when_condition_present() -> None:
    result = apply_risk_floor(RiskTier.FAIBLE, [False, True, False])
    assert result == RiskTier.ÉLEVÉ


def test_risk_floor_preserves_llm_elevated() -> None:
    result = apply_risk_floor(RiskTier.ÉLEVÉ, [False, False])
    assert result == RiskTier.ÉLEVÉ


def test_retrieval_advances_state_until_max_hops() -> None:
    provider = DummyRetrievalProvider()
    intent = Intent(id="1", question="Quel est le texte ?")
    state = RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=3)

    passage, next_state = advance_retrieval(intent, provider, {"source_id": "doc1"}, state, max_hops=3)
    assert passage.source_id == "doc1"
    assert next_state.hop_count == 1
    assert "doc1" in next_state.visited_documents

    passage, next_state = advance_retrieval(intent, provider, {"source_id": "doc2"}, next_state, max_hops=3)
    assert next_state.hop_count == 2


def test_retrieval_raises_when_budget_exhausted() -> None:
    provider = DummyRetrievalProvider()
    intent = Intent(id="1", question="Quel est le texte ?")
    state = RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=0)

    try:
        advance_retrieval(intent, provider, {"source_id": "doc1"}, state, max_hops=3)
        assert False, "Expected RetrievalError"
    except RetrievalError:
        assert True
