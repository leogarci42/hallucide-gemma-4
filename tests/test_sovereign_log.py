import pytest

from hallucide import Orchestrator, RiskTier
from hallucide.audit.audit import build_compliance_log
from hallucide.audit.sovereign_log import (
    NonCorrelationViolation,
    SovereignLogStore,
    assert_compliance_entry_is_anonymous,
    build_access_log_entry,
    generate_session_ref,
)
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage, RetrievalState


class DummyDecomposer:
    def decompose(self, message: str):
        return [Intent(id="1", question=message)]


class DummyIntentGenerator:
    def generate_claims(self, intent: Intent, passage: Passage):
        return [Claim(ref=passage.text, status=ClaimStatus.AUTHENTIFIÉ)]


class DummyRetrievalProvider:
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
        return Passage(
            source_id=query["source_id"],
            source_type="normatif",
            opposable=True,
            text="Passage authentique.",
            metadata={"query": query},
        )


def _run_orchestrator(message: str = "Quelle est la règle ?"):
    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    return orchestrator.run(
        message=message,
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )


def test_generate_session_ref_is_random_and_opaque() -> None:
    a = generate_session_ref()
    b = generate_session_ref()

    assert a != b
    assert len(a) == 32  # uuid4().hex


def test_confidential_compliance_entry_passes_anonymity_check() -> None:
    orchestration_result = _run_orchestrator()
    entries = build_compliance_log(
        orchestration_result,
        provider="local-ollama",
        model="llama3",
        message="Question confidentielle d'un parlementaire",
        session_ref=generate_session_ref(),
        confidential=True,
    )

    assert_compliance_entry_is_anonymous(entries[0])  # ne lève pas


def test_non_confidential_compliance_entry_with_query_is_rejected() -> None:
    # Garde-fou §13.4 : un appelant qui oublie confidential=True ne doit pas
    # pouvoir faire fuiter le texte de la question dans le journal Conformité.
    orchestration_result = _run_orchestrator()
    entries = build_compliance_log(
        orchestration_result,
        provider="local-ollama",
        model="llama3",
        message="Question confidentielle d'un parlementaire",
        confidential=False,
    )

    with pytest.raises(NonCorrelationViolation):
        assert_compliance_entry_is_anonymous(entries[0])


def test_sovereign_log_store_keeps_logs_separate() -> None:
    store = SovereignLogStore()
    orchestration_result = _run_orchestrator()
    compliance_entries = build_compliance_log(
        orchestration_result,
        provider="local-ollama",
        model="llama3",
        message="Question",
        session_ref=generate_session_ref(),
        confidential=True,
    )
    for entry in compliance_entries:
        store.record_compliance(entry)

    access_entry = build_access_log_entry(pseudonymized_identity="dep-hash-abc123", request_count=3)
    store.record_access(access_entry)

    assert len(store.compliance_entries) == 1
    assert len(store.access_entries) == 1
    # Aucune structure ne joint les deux par identité commune.
    assert not hasattr(store, "_joined")


def test_sovereign_log_store_rejects_leaking_compliance_entry() -> None:
    store = SovereignLogStore()
    orchestration_result = _run_orchestrator()
    leaking_entries = build_compliance_log(
        orchestration_result,
        provider="local-ollama",
        model="llama3",
        message="Question sensible",
        confidential=False,
    )

    with pytest.raises(NonCorrelationViolation):
        store.record_compliance(leaking_entries[0])

    assert len(store.compliance_entries) == 0


def test_access_log_entry_never_carries_query_or_claims() -> None:
    entry = build_access_log_entry(pseudonymized_identity="dep-hash-abc123", request_count=5)
    payload = entry.to_dict()

    assert "query" not in payload
    assert "claims" not in payload
    assert payload["pseudonymized_identity"] == "dep-hash-abc123"
