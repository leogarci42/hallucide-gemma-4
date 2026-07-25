from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hallucide.validation.document import verify_document
from hallucide.core_types.exceptions import InvalidClaimError, VerificationError
from hallucide.triage.triage import RiskTier, apply_risk_floor
from hallucide.core_types.types import Claim, ClaimStatus, DocumentDraft, Passage
from hallucide.verification.verifier import verify_claims

REFUS = "REFUS_VÉRIFICATION"


@dataclass(frozen=True)
class TrapCase:
    """Un cas du banc de mesure §12 : entrée connue, statut attendu connu.

    `trap_type` reprend la taxonomie §10 (ex. "A1", "B2", "F1") pour permettre
    des métriques par type de piège plutôt qu'un seul taux agrégé. `is_answerable`
    distingue les cas piégés (on attend un blocage/refus/statut dégradé) des cas
    répondables (on attend une publication normale) -- nécessaire pour calculer
    le taux de sur-refus séparément du taux de blocage.
    """

    id: str
    trap_type: str
    is_answerable: bool
    claim: Claim
    passage: Passage
    expected_status: Optional[ClaimStatus]  # None si l'on attend un refus (NON_AUTHENTIFIÉ)


@dataclass(frozen=True)
class CaseResult:
    case: TrapCase
    actual_status: str  # valeur de ClaimStatus, ou REFUS
    correct: bool


def evaluate_case(case: TrapCase) -> CaseResult:
    """Exécute un cas contre le vérificateur déterministe (§7), sans LLM (INV-014)."""
    try:
        result = verify_claims([case.claim], case.passage)
        actual = result.claims[0].status.value
    except (VerificationError, InvalidClaimError):
        actual = REFUS

    expected = case.expected_status.value if case.expected_status is not None else REFUS
    return CaseResult(case=case, actual_status=actual, correct=actual == expected)


@dataclass(frozen=True)
class TrapTypeMetrics:
    trap_type: str
    total: int
    correct: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 1.0


@dataclass(frozen=True)
class MeasurementReport:
    """Rapport §12 : taux de blocage correct, taux de sur-refus, métriques
    par type de piège. Mesure le vérificateur déterministe uniquement -- la
    cible que la spec dit garantie par le code, pas par le modèle (§2/INV-014).
    """

    results: tuple[CaseResult, ...]

    @property
    def total_cases(self) -> int:
        return len(self.results)

    @property
    def trap_results(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if not r.case.is_answerable)

    @property
    def answerable_results(self) -> tuple[CaseResult, ...]:
        return tuple(r for r in self.results if r.case.is_answerable)

    @property
    def correct_blocking_rate(self) -> float:
        """Taux de blocage correct (§12) : sur les pièges, proportion correctement traitée."""
        traps = self.trap_results
        if not traps:
            return 1.0
        return sum(1 for r in traps if r.correct) / len(traps)

    @property
    def over_refusal_rate(self) -> float:
        """Taux de sur-refus (§12) = répondables bloqués / répondables.

        Un vérificateur trop prudent est un problème de disponibilité, pas
        seulement de confort -- la spec le traite comme une propriété de
        sécurité (§12).
        """
        answerable = self.answerable_results
        if not answerable:
            return 0.0
        wrongly_blocked = sum(1 for r in answerable if r.actual_status == REFUS)
        return wrongly_blocked / len(answerable)

    @property
    def metrics_by_trap_type(self) -> tuple[TrapTypeMetrics, ...]:
        by_type: dict[str, list[CaseResult]] = {}
        for r in self.results:
            by_type.setdefault(r.case.trap_type, []).append(r)

        return tuple(
            TrapTypeMetrics(
                trap_type=trap_type,
                total=len(rs),
                correct=sum(1 for r in rs if r.correct),
            )
            for trap_type, rs in sorted(by_type.items())
        )


def run_measurement(cases: list[TrapCase]) -> MeasurementReport:
    return MeasurementReport(results=tuple(evaluate_case(c) for c in cases))


@dataclass(frozen=True)
class TriageCase:
    """Cas du banc de triage (§12) : item à risque connu, triage LLM simulé,
    conditions plancher déterministes -- on mesure si le plancher (§2/INV-011)
    rattrape un triage LLM qui aurait classé l'item à tort en risque faible.
    """

    id: str
    llm_risk: RiskTier
    floor_conditions: tuple[bool, ...]
    expected_risk: RiskTier


@dataclass(frozen=True)
class TriageReport:
    total: int
    false_negatives: int

    @property
    def false_negative_rate(self) -> float:
        """Taux de faux négatifs de triage (§12) : items à risque connu élevé
        classés faible. Doit être structurellement 0% par construction du
        plancher déterministe (INV-011) -- ce banc le démontre plutôt que
        de le présumer.
        """
        return self.false_negatives / self.total if self.total else 0.0


def run_triage_measurement(cases: list[TriageCase]) -> TriageReport:
    false_negatives = 0
    for case in cases:
        actual = apply_risk_floor(case.llm_risk, list(case.floor_conditions))
        if case.expected_risk == RiskTier.ÉLEVÉ and actual != RiskTier.ÉLEVÉ:
            false_negatives += 1
    return TriageReport(total=len(cases), false_negatives=false_negatives)


# --- Mesure par mode document (§12, v4) ---


@dataclass(frozen=True)
class DocumentCase:
    """Cas du banc documentaire (§12 v4) : un DocumentDraft à verdict connu.

    `is_answerable=True` : on attend une publication (blocage = sur-refus).
    `is_answerable=False` : cas piège (ex. B5), on attend un blocage.
    """

    id: str
    trap_type: str
    is_answerable: bool
    draft: DocumentDraft
    source: Passage


@dataclass(frozen=True)
class DocumentCaseResult:
    case: DocumentCase
    publishable: bool
    correct: bool


@dataclass(frozen=True)
class DocumentMeasurementReport:
    """§12 (v4) : le taux de sur-refus se mesure PAR MODE. L'absence de
    verbatim est suspecte en production, attendue en synthèse -- un seuil
    unique calibré pour la production bloquerait toute synthèse, et
    réciproquement laisserait tout passer.
    """

    results: tuple[DocumentCaseResult, ...]

    def _by_mode(self) -> dict[str, list[DocumentCaseResult]]:
        grouped: dict[str, list[DocumentCaseResult]] = {}
        for r in self.results:
            grouped.setdefault(r.case.draft.mode.value, []).append(r)
        return grouped

    @property
    def over_refusal_rate_by_mode(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for mode, results in sorted(self._by_mode().items()):
            answerable = [r for r in results if r.case.is_answerable]
            if not answerable:
                rates[mode] = 0.0
                continue
            wrongly_blocked = sum(1 for r in answerable if not r.publishable)
            rates[mode] = wrongly_blocked / len(answerable)
        return rates

    @property
    def correct_blocking_rate_by_mode(self) -> dict[str, float]:
        rates: dict[str, float] = {}
        for mode, results in sorted(self._by_mode().items()):
            traps = [r for r in results if not r.case.is_answerable]
            if not traps:
                rates[mode] = 1.0
                continue
            rates[mode] = sum(1 for r in traps if r.correct) / len(traps)
        return rates


def run_document_measurement(cases: list[DocumentCase]) -> DocumentMeasurementReport:
    """Exécute chaque cas contre verify_document (§7ter), sans LLM (INV-014)."""
    results: list[DocumentCaseResult] = []
    for case in cases:
        outcome = verify_document(case.draft, case.source)
        expected_publishable = case.is_answerable
        results.append(
            DocumentCaseResult(
                case=case,
                publishable=outcome.publishable,
                correct=outcome.publishable == expected_publishable,
            )
        )
    return DocumentMeasurementReport(results=tuple(results))
