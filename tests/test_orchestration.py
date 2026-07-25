from hallucide import Orchestrator, OrchestrationResult, RiskTier
from hallucide.core_types.exceptions import RetrievalError, VerificationError
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage, RetrievalState
from hallucide.retrieval.retrieval import RetrievalProvider


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


def test_orchestrator_runs_single_intent() -> None:
    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert isinstance(result, OrchestrationResult)
    assert result.results[0].verification.verbatim_check == "PASS"


def test_orchestrator_skips_echo_back_for_single_intent() -> None:
    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.echo_back is None
    assert result.coverage_passed is True


def test_orchestrator_builds_echo_back_when_multiple_intents() -> None:
    class TwoIntentDecomposer:
        def decompose(self, message: str):
            return [
                Intent(id="1", question="Quel est le délai de rétractation ?"),
                Intent(id="2", question="Quelle est la sanction en cas de non-respect ?"),
            ]

    class PerIntentRetrievalProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id=f"doc-{intent.id}",
                source_type="normatif",
                opposable=True,
                text="Passage authentique.",
                metadata={"query": query},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=TwoIntentDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quel est le délai de rétractation et quelle est la sanction en cas de non-respect ?",
        retrieval_provider=PerIntentRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=2,
    )

    assert result.echo_back is not None
    assert "1. Quel est le délai de rétractation ?" in result.echo_back
    assert "2. Quelle est la sanction en cas de non-respect ?" in result.echo_back
    assert result.coverage_passed is True


def test_orchestrator_flags_low_coverage_when_intent_forgotten() -> None:
    # Piège E4 (§10) : la décomposition oublie une des deux questions du message.
    class ForgetfulDecomposer:
        def decompose(self, message: str):
            return [Intent(id="1", question="Quel est le délai de rétractation ?")]

    orchestrator = Orchestrator(model_provider=object(), decomposer=ForgetfulDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quel est le délai de rétractation et quelle est la sanction en cas de non-respect ?",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.coverage_passed is False
    assert "sanction" in result.coverage_missing_tokens
    # §2 : le plancher de risque inclut la couverture, l'appelant ne peut pas l'oublier.
    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_elevates_risk_when_slot_inferred() -> None:
    # Piège A3 (§10) : le passage signale un slot inféré (numéro deviné, pas
    # copié depuis la question) -- le risque doit s'élever pour cette intention.
    class SlotInferredRetrievalProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id=query["source_id"],
                source_type="normatif",
                opposable=True,
                text="Passage authentique.",
                metadata={"slot_inferred": True},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=SlotInferredRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_keeps_risk_low_when_slot_copied() -> None:
    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=DummyRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.results[0].risk_tier == RiskTier.FAIBLE


def test_orchestrator_elevates_risk_when_citation_possibly_truncated() -> None:
    # Piège B2 (§7) : la citation omet l'exception ("sauf...") qui suit
    # immédiatement dans le passage source -- le risque doit s'élever.
    class TruncatingRetrievalProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id=query["source_id"],
                source_type="normatif",
                opposable=True,
                text="Le salarié bénéficie d'un délai de préavis de deux mois sauf en cas de faute grave.",
                metadata={},
            )

    class TruncatingIntentGenerator:
        def generate_claims(self, intent: Intent, passage: Passage):
            return [Claim(ref="Le salarié bénéficie d'un délai de préavis de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=TruncatingIntentGenerator())
    result = orchestrator.run(
        message="Quel est le délai de préavis ?",
        retrieval_provider=TruncatingRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.results[0].verification.claims[0].truncation_flagged is True
    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_elevates_risk_when_pertinence_non_garantie() -> None:
    # Piège C1 (§10) : route texte libre, pertinence non garantie -- le
    # risque doit s'élever automatiquement, sans que l'appelant l'oublie.
    class FreeTextRetrievalProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id=query["source_id"],
                source_type="normatif",
                opposable=False,
                text="Ordonnance n°62-91 relative au congé spécial de certains fonctionnaires",
                metadata={"pertinence_non_garantie": True},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="congé menstruel",
        retrieval_provider=FreeTextRetrievalProvider(),
        retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={"source_id": "doc1"},
        max_hops=1,
    )

    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_two_intents_may_rely_on_same_source_without_conflict() -> None:
    # Régression : deux questions distinctes sur le même article ne doivent
    # pas faire échouer le pipeline. visited_documents borne le multi-saut AU
    # SEIN d'une intention (§4ter), pas entre intentions.
    class TwoIntentDecomposer:
        def decompose(self, message: str):
            return [Intent(id="1", question="Q1 sur 1103"), Intent(id="2", question="Q2 sur 1103")]

    class SameSourceProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(source_id="ARTI-1103", source_type="normatif", opposable=True, text="Passage authentique.", metadata={})

    orchestrator = Orchestrator(model_provider=object(), decomposer=TwoIntentDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Deux questions sur le même article",
        retrieval_provider=SameSourceProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=3,
    )

    assert len(result.results) == 2
    assert all(r.passage.source_id == "ARTI-1103" for r in result.results)


def test_shared_budget_is_consumed_across_intents() -> None:
    # Le budget global reste partagé : deux intentions consomment chacune un
    # hop du budget total (contrairement à visited_documents, réinitialisé).
    class TwoIntentDecomposer:
        def decompose(self, message: str):
            return [Intent(id="1", question="Q1"), Intent(id="2", question="Q2")]

    class PerIntentProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(source_id=f"doc-{intent.id}", source_type="normatif", opposable=True, text="Passage authentique.", metadata={})

    orchestrator = Orchestrator(model_provider=object(), decomposer=TwoIntentDecomposer(), intent_generator=DummyIntentGenerator())
    shared_state = RetrievalState(remaining_budget=1)  # budget pour un seul hop
    try:
        orchestrator.run(
            message="Deux questions",
            retrieval_provider=PerIntentProvider(),
            retrieval_state=shared_state,
            floor_conditions=[False],
            llm_risk=RiskTier.FAIBLE,
            query={},
            max_hops=3,
        )
        assert False, "Expected RetrievalError (budget épuisé après la 1re intention)"
    except RetrievalError:
        assert True


def test_refused_intent_is_logged_not_crashed_and_others_continue() -> None:
    # §4 étape 8 / §7bis / §8 : un refus de vérification produit un résultat
    # BLOCKED journalisable pour CETTE intention, sans faire échouer les autres.
    class TwoIntentDecomposer:
        def decompose(self, message: str):
            return [Intent(id="1", question="refusée"), Intent(id="2", question="valide")]

    class SourceProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(source_id=f"doc-{intent.id}", source_type="normatif", opposable=True, text="Le texte officiel exact.", metadata={})

    class MixedGenerator:
        def generate_claims(self, intent: Intent, passage: Passage):
            if intent.id == "1":
                return [Claim(ref="Citation inventée absente du passage.", status=ClaimStatus.AUTHENTIFIÉ)]
            return [Claim(ref="Le texte officiel exact.", status=ClaimStatus.AUTHENTIFIÉ)]

    orchestrator = Orchestrator(model_provider=object(), decomposer=TwoIntentDecomposer(), intent_generator=MixedGenerator())
    result = orchestrator.run(
        message="deux intentions",
        retrieval_provider=SourceProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=3,
    )

    assert len(result.results) == 2
    assert result.results[0].verification.verbatim_check == "FAIL"
    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ  # un refus force le risque élevé
    assert result.results[1].verification.verbatim_check == "PASS"


def test_orchestrator_elevates_risk_when_claim_is_interpretation() -> None:
    # §2/INV-011 : une INTERPRÉTATION est "bornée puis déléguée à l'humain"
    # (§10, B3/D1) -- le plancher doit la faire passer en risque élevé, jamais
    # en publication directe à risque faible.
    class InterpretationGenerator:
        def generate_claims(self, intent: Intent, passage: Passage):
            return [Claim(ref="Une reformulation fidèle du passage authentique.", status=ClaimStatus.INTERPRÉTATION)]

    class SourceProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id="doc1", source_type="normatif", opposable=True,
                text="Une reformulation très fidèle du même passage authentique original.",
                metadata={},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=InterpretationGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=SourceProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=1,
    )

    assert result.results[0].verification.claims[0].status == ClaimStatus.INTERPRÉTATION
    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_elevates_risk_when_claim_is_cite_non_opposable() -> None:
    # §2/INV-011 : un verbatim exact sur source non opposable (F1) reste
    # exact mais pas opposable -- délégué à l'humain via le plancher.
    class NonOpposableProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id="debat-1", source_type="normatif", opposable=False,
                text="Passage authentique.", metadata={},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=NonOpposableProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=1,
    )

    assert result.results[0].verification.claims[0].status == ClaimStatus.CITÉ_NON_OPPOSABLE
    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_elevates_risk_when_single_query_serves_multiple_intents() -> None:
    # §4 : "1 intention -> 1 requête". L'étape 3 (formulation de requête par
    # intention) n'étant pas implémentée, une query unique partagée entre N>1
    # intentions recrée le piège E1 -- mode dégradé borné : risque élevé pour
    # toutes les intentions, donc validation humaine avant publication.
    class TwoIntentDecomposer:
        def decompose(self, message: str):
            return [Intent(id="1", question="Quelle règle ?"), Intent(id="2", question="Quelle sanction ?")]

    class SharedQueryProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(source_id="doc-partagé", source_type="normatif", opposable=True, text="Passage authentique.", metadata={})

    orchestrator = Orchestrator(model_provider=object(), decomposer=TwoIntentDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle règle ? Quelle sanction ?",
        retrieval_provider=SharedQueryProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=1,
    )

    assert len(result.results) == 2
    assert all(r.risk_tier == RiskTier.ÉLEVÉ for r in result.results)


def test_orchestrator_elevates_risk_when_selection_ambiguous() -> None:
    # Piège A3 (variante) : plusieurs candidats matchaient la requête et le
    # provider en a choisi un silencieusement -- le signal metadata doit être
    # lu par l'orchestration et élever le risque.
    class AmbiguousProvider:
        def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]):
            return Passage(
                source_id="doc1", source_type="normatif", opposable=True,
                text="Passage authentique.",
                metadata={"selection_ambiguous": True, "candidate_count": 2},
            )

    orchestrator = Orchestrator(model_provider=object(), decomposer=DummyDecomposer(), intent_generator=DummyIntentGenerator())
    result = orchestrator.run(
        message="Quelle est la règle ?",
        retrieval_provider=AmbiguousProvider(),
        retrieval_state=RetrievalState(),
        floor_conditions=[False],
        llm_risk=RiskTier.FAIBLE,
        query={},
        max_hops=1,
    )

    assert result.results[0].risk_tier == RiskTier.ÉLEVÉ


def test_orchestrator_raises_when_no_intents() -> None:
    class EmptyDecomposer:
        def decompose(self, message: str):
            return []

    orchestrator = Orchestrator(model_provider=object(), decomposer=EmptyDecomposer(), intent_generator=DummyIntentGenerator())

    try:
        orchestrator.run(
            message="",
            retrieval_provider=DummyRetrievalProvider(),
            retrieval_state=RetrievalState(hop_count=0, visited_documents=set(), remaining_budget=2),
            floor_conditions=[False],
            llm_risk=RiskTier.FAIBLE,
            query={"source_id": "doc1"},
            max_hops=1,
        )
        assert False, "Expected ValueError"
    except ValueError:
        assert True
