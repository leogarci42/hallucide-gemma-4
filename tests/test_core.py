from hallucide import AskResult, MockModelProvider, RiskTier, Hallucide, SovereignLogStore
from hallucide.validation.human_validation import ValidationDecision, ValidationKey
from hallucide.core_types.types import Intent, Passage, RetrievalState


class FakeRetrievalProvider:
    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        return Passage(
            source_id="doc-1",
            source_type="normatif",
            opposable=True,
            text="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
            metadata={},
        )


def _build_guard() -> Hallucide:
    model_provider = MockModelProvider(
        responses={
            "decompose": '[{"id": "1", "question": "Que dit l\'article 1103 ?"}]',
            "claims": (
                '[{"ref": "Les contrats légalement formés tiennent lieu de loi '
                'à ceux qui les ont faits.", "status": "AUTHENTIFIÉ"}]'
            ),
        }
    )
    guard = Hallucide(model_provider=model_provider)
    guard.retrieval_provider = FakeRetrievalProvider()
    return guard


def test_ask_runs_full_pipeline_and_returns_orchestration_result() -> None:
    guard = _build_guard()
    result = guard.ask(message="Que dit l'article 1103 ?", query={"route": "code_article"})

    assert isinstance(result, AskResult)
    assert result.orchestration.results[0].verification.verbatim_check == "PASS"
    assert result.orchestration.results[0].risk_tier == RiskTier.FAIBLE


def test_ask_records_compliance_entry_in_log_store() -> None:
    guard = _build_guard()
    result = guard.ask(message="Que dit l'article 1103 ?", query={"route": "code_article"})

    assert len(guard.log_store.compliance_entries) == 1
    assert guard.log_store.compliance_entries[0].session_ref == result.session_ref


def test_ask_never_logs_the_question_in_compliance_entry() -> None:
    # §13.4 : utiliser Hallucide implique le mode souverain par
    # construction -- le journal Conformité ne contient jamais `query`.
    guard = _build_guard()
    guard.ask(message="Question sensible de parlementaire", query={"route": "code_article"})

    entry = guard.log_store.compliance_entries[0]
    assert entry.query is None
    assert "query" not in entry.to_dict()


def test_ask_uses_provided_session_ref() -> None:
    guard = _build_guard()
    result = guard.ask(
        message="Que dit l'article 1103 ?",
        query={"route": "code_article"},
        session_ref="custom-session-ref",
    )

    assert result.session_ref == "custom-session-ref"


def test_ask_marks_high_risk_intent_as_not_published_pending_validation() -> None:
    # §4 étape 9 : aucune décision encore enregistrée -> pas publiable, mais
    # pas d'exception non plus -- l'appelant décide quoi afficher en attendant.
    guard = _build_guard()
    result = guard.ask(
        message="Que dit l'article 1103 ?",
        query={"route": "code_article"},
        floor_conditions=[True],  # force le risque élevé
    )

    assert result.orchestration.results[0].risk_tier == RiskTier.ÉLEVÉ
    assert result.published == (False,)
    assert result.compliance_entries[0].human_validation == "pending"


def test_ask_publishes_after_human_approval() -> None:
    guard = _build_guard()
    first = guard.ask(
        message="Que dit l'article 1103 ?",
        query={"route": "code_article"},
        floor_conditions=[True],
    )
    assert first.published == (False,)

    key = ValidationKey.from_result(first.orchestration.results[0])
    guard.validation_registry.record_decision(key, ValidationDecision.APPROVED, validator_ref="dep-hash-abc")

    second = guard.ask(
        message="Que dit l'article 1103 ?",
        query={"route": "code_article"},
        floor_conditions=[True],
    )
    assert second.published == (True,)
    assert second.compliance_entries[0].human_validation == "approved"


def test_guard_uses_shared_log_store_across_calls() -> None:
    store = SovereignLogStore()
    model_provider = MockModelProvider(
        responses={
            "decompose": '[{"id": "1", "question": "Q1"}]',
            "claims": '[{"ref": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.", "status": "AUTHENTIFIÉ"}]',
        }
    )
    guard = Hallucide(model_provider=model_provider, log_store=store)
    guard.retrieval_provider = FakeRetrievalProvider()

    guard.ask(message="Q1", query={"route": "code_article"})
    guard.ask(message="Q1", query={"route": "code_article"})

    assert len(store.compliance_entries) == 2
