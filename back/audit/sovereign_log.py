from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from hallucide.audit.audit import ComplianceLogEntry

# §13.4 : tension centrale -- savoir quel utilisateur recherche quelle loi
# est politiquement sensible. Principe : séparer la preuve de conformité de
# l'identité de l'auteur, via deux journaux distincts non corrélables par
# défaut. Le jeton de session est un UUID aléatoire, sans dérivation
# réversible depuis l'identité -- sans accès simultané aux deux journaux ET
# à la table de correspondance ci-dessous, aucune des deux parties ne permet
# de relier une question à une identité.
_FIELDS_FORBIDDEN_IN_COMPLIANCE_LOG = ("query", "identity", "user_id", "author")


def generate_session_ref() -> str:
    """Jeton opaque (§13.4), sans lien dérivable avec l'identité de l'auteur."""
    return uuid.uuid4().hex


@dataclass(frozen=True)
class AccessLogEntry:
    """Journal Accès (§13.4, restreint) : authentification, volumétrie par
    session -- jamais le contenu d'une question ni un statut de conformité.
    Pseudonymisé : l'identité figure ici (sous forme pseudonymisée), jamais
    dans le journal Conformité.
    """

    timestamp: str
    session_ref: str
    pseudonymized_identity: str
    request_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=False)


def build_access_log_entry(
    pseudonymized_identity: str,
    request_count: int,
    session_ref: str | None = None,
) -> AccessLogEntry:
    """Construit une entrée du journal Accès (§13.4), distincte et stockée
    séparément du journal Conformité (cf. SovereignLogStore).
    """
    return AccessLogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_ref=session_ref or generate_session_ref(),
        pseudonymized_identity=pseudonymized_identity,
        request_count=request_count,
    )


class NonCorrelationViolation(Exception):
    """Levée si une entrée de conformité porte un champ interdit (§13.4)."""


def assert_compliance_entry_is_anonymous(entry: ComplianceLogEntry) -> None:
    """Garde-fou déterministe (§13.4) : le journal Conformité ne doit jamais
    contenir le texte de la question ni une identité, même par erreur d'un
    appelant qui aurait oublié `confidential=True`. Lève plutôt que de
    journaliser silencieusement une fuite.
    """
    payload = entry.to_dict()
    for forbidden_field in _FIELDS_FORBIDDEN_IN_COMPLIANCE_LOG:
        if forbidden_field in payload:
            raise NonCorrelationViolation(
                f"Compliance log entry leaks forbidden field '{forbidden_field}' (§13.4)."
            )


class SovereignLogStore:
    """Stockage séparé des deux journaux (§13.4) : aucune méthode ne permet
    de joindre Conformité et Accès par construction -- ils vivent dans deux
    listes distinctes, jamais dans une structure commune indexée par identité.
    """

    def __init__(self) -> None:
        self._compliance_entries: list[ComplianceLogEntry] = []
        self._access_entries: list[AccessLogEntry] = []

    def record_compliance(self, entry: ComplianceLogEntry) -> None:
        assert_compliance_entry_is_anonymous(entry)
        self._compliance_entries.append(entry)

    def record_access(self, entry: AccessLogEntry) -> None:
        self._access_entries.append(entry)

    @property
    def compliance_entries(self) -> tuple[ComplianceLogEntry, ...]:
        return tuple(self._compliance_entries)

    @property
    def access_entries(self) -> tuple[AccessLogEntry, ...]:
        return tuple(self._access_entries)
