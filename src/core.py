from __future__ import annotations

from dataclasses import dataclass, field

from .audit.audit import ComplianceLogEntry, build_compliance_log
from .coverage.coverage import DEFAULT_COVERAGE_THRESHOLD
from .validation.human_validation import HumanValidationRegistry, is_publishable, resolve_human_validation_status
from .decomposition.llm import ModelProvider, PromptBasedDecomposer, PromptBasedIntentGenerator
from .retrieval.multi_source import MultiSourceRetrievalProvider
from .decomposition.orchestration import Orchestrator
from .audit.sovereign_log import SovereignLogStore, generate_session_ref
from .triage.triage import RiskTier
from .core_types.types import OrchestrationResult, RetrievalState


@dataclass(frozen=True)
class AskResult:
    """Réponse complète d'un appel Hallucide.ask : le résultat de
    l'orchestration (§4) accompagné de son entrée de journal de conformité
    (§8/§13.4), déjà enregistrée dans le store fourni.

    `published` indique, par intention (même ordre que orchestration.results),
    si le résultat peut être montré à l'utilisateur final (§4 étape 9) :
    toujours vrai en risque faible, conditionné à une approbation humaine
    explicite en risque élevé -- jamais d'exception, l'appelant décide quoi
    afficher en attendant.
    """

    orchestration: OrchestrationResult
    compliance_entries: tuple[ComplianceLogEntry, ...]
    session_ref: str
    published: tuple[bool, ...]


@dataclass
class Hallucide:
    """Façade qui assemble les briques démontrées séparément (Orchestrator,
    MultiSourceRetrievalProvider, SovereignLogStore) en un point d'entrée
    unique : poser une question, recevoir une réponse gouvernée et
    journalisée. N'introduit aucune nouvelle règle de gouvernance -- elle ne
    fait que câbler celles qui existent déjà.

    Le log_store implémente §13.4 (cloisonnement conformité/identité) : il
    refuse par construction toute entrée contenant `query`. Utiliser
    Hallucide, c'est donc être en mode souverain par construction -- pas
    une option cosmétique (§13.1 : le passage au mode souverain est un
    changement d'exploitation, pas d'architecture). `query` n'est donc
    jamais journalisée dans le journal Conformité.
    """

    model_provider: ModelProvider
    log_store: SovereignLogStore = field(default_factory=SovereignLogStore)
    retrieval_provider: MultiSourceRetrievalProvider = field(default_factory=MultiSourceRetrievalProvider)
    validation_registry: HumanValidationRegistry = field(default_factory=HumanValidationRegistry)
    max_hops: int = 3

    def __post_init__(self) -> None:
        self.orchestrator = Orchestrator(
            model_provider=self.model_provider,
            decomposer=PromptBasedDecomposer(self.model_provider),
            intent_generator=PromptBasedIntentGenerator(self.model_provider),
        )

    def ask(
        self,
        message: str,
        query: dict[str, str],
        llm_risk: RiskTier = RiskTier.FAIBLE,
        floor_conditions: list[bool] | None = None,
        coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
        session_ref: str | None = None,
    ) -> AskResult:
        """Exécute le flux complet (§4) : décomposition -> récupération réelle
        (Moulineuse ou data.gouv selon `query`, §6bis/§6ter) -> vérification
        déterministe (§7) -> journalisation de conformité (§8/§13.4).
        """
        session_ref = session_ref or generate_session_ref()

        orchestration_result = self.orchestrator.run(
            message=message,
            retrieval_provider=self.retrieval_provider,
            retrieval_state=RetrievalState(),
            floor_conditions=floor_conditions or [],
            llm_risk=llm_risk,
            query=query,
            max_hops=self.max_hops,
            coverage_threshold=coverage_threshold,
        )

        # §4 étape 9 : le statut effectif (n/a | pending | approved | rejected)
        # tient compte d'une éventuelle décision déjà enregistrée pour cette
        # intention -- jamais "pending" par défaut si un humain a déjà tranché.
        human_validations = {
            index: resolve_human_validation_status(result, self.validation_registry)
            for index, result in enumerate(orchestration_result.results)
        }
        published = tuple(
            is_publishable(result, self.validation_registry) for result in orchestration_result.results
        )

        provider_name = type(self.model_provider).__name__
        model_name = getattr(self.model_provider, "model", provider_name)
        entries = build_compliance_log(
            orchestration_result,
            provider=provider_name,
            model=model_name,
            message=message,
            session_ref=session_ref,
            confidential=True,
            human_validations=human_validations,
        )
        for entry in entries:
            self.log_store.record_compliance(entry)

        return AskResult(
            orchestration=orchestration_result,
            compliance_entries=entries,
            session_ref=session_ref,
            published=published,
        )
