from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Set


class ClaimStatus(str, Enum):
    AUTHENTIFIÉ = "AUTHENTIFIÉ"
    CITÉ_NON_OPPOSABLE = "CITÉ_NON_OPPOSABLE"
    INTERPRÉTATION = "INTERPRÉTATION"
    NON_AUTHENTIFIÉ = "NON_AUTHENTIFIÉ"
    DONNÉE_TRACÉE = "DONNÉE_TRACÉE"


class RiskTier(str, Enum):
    FAIBLE = "faible"
    ÉLEVÉ = "élevé"


@dataclass(frozen=True)
class Intent:
    id: str
    question: str


@dataclass(frozen=True)
class Passage:
    source_id: str
    source_type: str
    opposable: bool
    text: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class Claim:
    ref: str
    status: ClaimStatus
    truncation_flagged: bool = False


@dataclass(frozen=True)
class VerificationResult:
    verbatim_check: str
    claims: tuple[Claim, ...]


@dataclass(frozen=True)
class IntentExecutionResult:
    intent: Intent
    passage: Passage
    verification: VerificationResult
    risk_tier: RiskTier


@dataclass(frozen=True)
class OrchestrationResult:
    intents: tuple[Intent, ...]
    results: tuple[IntentExecutionResult, ...]
    echo_back: Optional[str] = None
    coverage_passed: Optional[bool] = None
    coverage_ratio: Optional[float] = None
    coverage_missing_tokens: tuple[str, ...] = ()


# --- Mode document (§7ter, v4) ---


class DocumentMode(str, Enum):
    ANALYSE = "analyse"
    SYNTHÈSE = "synthèse"
    PRODUCTION = "production"


@dataclass(frozen=True)
class CoverageMap:
    """§7ter/INV-017 : mapping de couverture d'une synthèse. `source_units`
    est segmenté par le CODE (jamais par le LLM) ; `covered` mappe chaque
    unité vers les refs des claims qui la couvrent (existence vérifiée par
    lookup) ; `omitted` liste les omissions EXPLICITES -- une unité absente
    des deux est une omission silencieuse (piège B5) et bloque la publication.
    """

    source_units: tuple[str, ...]
    covered: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    omitted: tuple[str, ...] = ()


@dataclass(frozen=True)
class DocumentDraft:
    """§7ter : un document est une liste ordonnée de claims, chacun vérifié
    individuellement (§7). Aucun statut agrégé n'existe (INV-015) -- le
    document expose la mosaïque des statuts de ses claims.
    """

    mode: DocumentMode
    claims: tuple[Claim, ...]
    coverage: Optional[CoverageMap] = None  # requis si mode == SYNTHÈSE (INV-017)


@dataclass
class RetrievalState:
    hop_count: int = 0
    visited_documents: Set[str] = None
    remaining_budget: int = 1000

    def __post_init__(self) -> None:
        if self.visited_documents is None:
            self.visited_documents = set()
