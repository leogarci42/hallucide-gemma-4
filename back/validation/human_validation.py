from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from hallucide.audit.audit import passage_hash
from hallucide.core_types.types import IntentExecutionResult, RiskTier


class ValidationDecision(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True)
class ValidationKey:
    """Clé stable identifiant la décision attendue pour une intention (§4
    étape 9) : intent.id + passage_hash du contenu réellement vérifié --
    pas l'identité du passage seule, car deux intentions distinctes peuvent
    s'appuyer sur le même passage et appeler des décisions différentes.
    """

    intent_id: str
    passage_hash: str

    @classmethod
    def from_result(cls, result: IntentExecutionResult) -> "ValidationKey":
        return cls(intent_id=result.intent.id, passage_hash=passage_hash(result.passage.text))


@dataclass(frozen=True)
class ValidationRecord:
    decision: ValidationDecision
    validator_ref: str  # identité pseudonymisée du validateur (§13.4), jamais en clair
    timestamp: str
    comment: str | None = None


class HumanValidationRegistry:
    """Registre séparé des décisions humaines (§4 étape 9), indexé par
    ValidationKey. Les entrées de conformité (§8) restent immuables une fois
    journalisées (ComplianceLogEntry est frozen) ; le statut `human_validation`
    effectif se recalcule à la lecture en consultant ce registre, plutôt que
    de muter une entrée déjà écrite -- une preuve déjà journalisée reste
    traçable telle qu'elle était au moment de l'écriture.
    """

    def __init__(self) -> None:
        self._records: dict[ValidationKey, ValidationRecord] = {}

    def record_decision(
        self,
        key: ValidationKey,
        decision: ValidationDecision,
        validator_ref: str,
        comment: str | None = None,
    ) -> ValidationRecord:
        record = ValidationRecord(
            decision=decision,
            validator_ref=validator_ref,
            timestamp=datetime.now(timezone.utc).isoformat(),
            comment=comment,
        )
        self._records[key] = record
        return record

    def get_decision(self, key: ValidationKey) -> ValidationRecord | None:
        return self._records.get(key)


def resolve_human_validation_status(
    result: IntentExecutionResult, registry: HumanValidationRegistry
) -> str:
    """Statut effectif (§8 : "n/a | pending | approved") en tenant compte
    d'une éventuelle décision de rejet, que la spec ne nomme pas explicitement
    mais qui doit être distinguable de "approved" pour ne pas publier un
    contenu rejeté sous une étiquette de validation positive.
    """
    if result.risk_tier != RiskTier.ÉLEVÉ:
        return "n/a"

    record = registry.get_decision(ValidationKey.from_result(result))
    if record is None:
        return "pending"
    if record.decision == ValidationDecision.APPROVED:
        return "approved"
    return "rejected"


def is_publishable(result: IntentExecutionResult, registry: HumanValidationRegistry) -> bool:
    """§4 étape 9 : une intention à risque élevé n'est publiable qu'après
    approbation explicite -- ni en silence (pending), ni si elle a été
    rejetée. Risque faible : publiable directement (rien à attendre).
    """
    status = resolve_human_validation_status(result, registry)
    return status in ("n/a", "approved")
