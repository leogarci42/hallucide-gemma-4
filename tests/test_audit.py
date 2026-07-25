from hallucide import Orchestrator, RiskTier
from hallucide.audit.audit import build_compliance_log, passage_hash, verify_replay
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage, RetrievalState


class DummyDecomposer:
    def decompose(self, message: str):
        return [Intent(id="1", question=message)]


class DummyIntentGenerator:
    def generate_claims(self, intent: Intent, passage: Passage):
        return [Claim(ref=passage.text, status=ClaimStatus.AUTHENTIFIÉ)]


class NoClaimIntentGenerator:
    def generate_claims(self, intent: Intent, passage: Passage):
        return []


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


def test_compliance_log_has_required_fields() -> None:
    orchestration_result = _run_orchestrator()
    entries = build_compliance_log(orchestration_result, provider="local-ollama", model="llama3", message="Quelle est la règle ?")

    assert len(entries) == 1
    entry = entries[0]
    assert entry.provider == "local-ollama"
    assert entry.model == "llama3"
    assert entry.risk_tier == "faible"
    assert entry.mcp_calls == ("doc1",)
    assert entry.verbatim_check == "PASS"
    assert entry.compliance_status == "VALIDATED"
    assert entry.human_validation == "n/a"
    assert entry.governance_version == "v4"
    assert entry.query == "Quelle est la règle ?"
    assert len(entry.passage_hashes) == 1
    assert len(entry.passage_hashes[0]) == 64  # sha256 hex digest


def test_compliance_log_pending_validation_when_risk_elevated() -> None:
    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    orchestration_result = orchestrator.run(
        message="Question sensible",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[True],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )
    entries = build_compliance_log(orchestration_result, provider="api", model="gemini-1.5-pro")

    assert entries[0].risk_tier == "élevé"
    assert entries[0].human_validation == "pending"


def test_confidential_mode_omits_query_per_13_4() -> None:
    orchestration_result = _run_orchestrator()
    entries = build_compliance_log(
        orchestration_result,
        provider="local-ollama",
        model="llama3",
        message="Question confidentielle de parlementaire",
        session_ref="opaque-token-123",
        confidential=True,
    )

    entry = entries[0]
    assert entry.query is None
    assert entry.session_ref == "opaque-token-123"
    assert "query" not in entry.to_dict()


def test_compliance_status_is_no_answer_when_zero_claims() -> None:
    # §8 : verify_claims([]) donne verbatim_check="PASS" par vacuité (all([])),
    # ce qui rendrait compliance_status="VALIDATED" trompeur si on ne le
    # distinguait pas explicitement (piège C1 -- source hors-sujet, le modèle
    # ne produit aucune affirmation -- voir ANALYSE_TEST_JURISPRUDENCE.md).
    orchestrator = Orchestrator(
        model_provider=object(), decomposer=DummyDecomposer(), intent_generator=NoClaimIntentGenerator()
    )
    orchestration_result = orchestrator.run(
        message="Question hors corpus",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )
    entries = build_compliance_log(orchestration_result, provider="api", model="mistral-small")

    assert entries[0].claims == ()
    assert entries[0].verbatim_check == "PASS"
    assert entries[0].compliance_status == "NO_ANSWER"


def test_passage_hash_is_replayable() -> None:
    text = "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits."
    digest = passage_hash(text)

    assert digest == passage_hash(text)
    assert verify_replay(text, digest) is True
    assert verify_replay("Un autre texte.", digest) is False


def test_passage_hash_uses_normalized_text() -> None:
    spaced = "Les  contrats   légalement formés…"
    collapsed = "Les contrats légalement formés…"
    assert passage_hash(spaced) == passage_hash(collapsed)
