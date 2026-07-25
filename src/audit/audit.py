from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from hallucide.verification.normalization import normalize_text
from hallucide.core_types.types import IntentExecutionResult, OrchestrationResult, RiskTier

GOVERNANCE_VERSION = "v4"


def passage_hash(passage_text: str) -> str:
    """sha256 of the normalized passage text (§7), for replayable proof (§8)."""
    normalized = normalize_text(passage_text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ComplianceLogEntry:
    timestamp: str
    provider: str
    model: str
    risk_tier: str
    mcp_calls: tuple[str, ...]
    passage_hashes: tuple[str, ...]
    claims: tuple[dict[str, str], ...]
    verbatim_check: str
    compliance_status: str
    human_validation: str
    governance_version: str
    query: str | None = None
    session_ref: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=False)


def _compliance_status(result: IntentExecutionResult) -> str:
    # §8 : NO_ANSWER distingue "aucune affirmation produite" de "affirmations
    # vérifiées avec succès". Sans ça, verify_claims([]) donne verbatim_check
    # PASS par vacuité (all([]) == True) et produit un VALIDATED trompeur
    # quand le modèle n'a en réalité rien affirmé (ex. source hors-sujet,
    # piège C1 -- voir ANALYSE_TEST_JURISPRUDENCE.md).
    if not result.verification.claims:
        return "NO_ANSWER"
    return "VALIDATED" if result.verification.verbatim_check == "PASS" else "BLOCKED"


def _default_human_validation(risk_tier: RiskTier) -> str:
    return "pending" if risk_tier == RiskTier.ÉLEVÉ else "n/a"


def build_compliance_log_entry(
    result: IntentExecutionResult,
    provider: str,
    model: str,
    query: str | None = None,
    session_ref: str | None = None,
    confidential: bool = False,
    human_validation: str | None = None,
) -> ComplianceLogEntry:
    """Build one §8/§13.4 compliance log entry for a single intent's result.

    When confidential=True (§13.4 sovereign deployment), `query` is omitted
    even if provided and `session_ref` should carry an opaque token instead
    of an identity -- the compliance log never contains the question text
    nor the author's identity in that mode.

    `human_validation` lets the caller pass the resolved §4 step 9 status
    (via human_validation.resolve_human_validation_status against a
    HumanValidationRegistry); defaults to the risk-tier-only status
    ("pending"/"n/a") when omitted, so this module never has to import the
    validation registry itself.
    """
    claims = tuple({"ref": c.ref, "status": c.status.value} for c in result.verification.claims)
    mcp_calls = (result.passage.source_id,)
    hashes = (passage_hash(result.passage.text),)

    return ComplianceLogEntry(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        risk_tier=result.risk_tier.value,
        mcp_calls=mcp_calls,
        passage_hashes=hashes,
        claims=claims,
        verbatim_check=result.verification.verbatim_check,
        compliance_status=_compliance_status(result),
        human_validation=human_validation or _default_human_validation(result.risk_tier),
        governance_version=GOVERNANCE_VERSION,
        query=None if confidential else query,
        session_ref=session_ref,
    )


def build_compliance_log(
    orchestration_result: OrchestrationResult,
    provider: str,
    model: str,
    message: str | None = None,
    session_ref: str | None = None,
    confidential: bool = False,
    human_validations: dict[int, str] | None = None,
) -> tuple[ComplianceLogEntry, ...]:
    """One compliance log entry per intent result (§8), replayable via passage_hashes.

    `human_validations` optionally maps result index -> resolved status
    (see build_compliance_log_entry); omit to fall back to risk-tier-only.
    """
    human_validations = human_validations or {}
    return tuple(
        build_compliance_log_entry(
            result,
            provider=provider,
            model=model,
            query=message,
            session_ref=session_ref,
            confidential=confidential,
            human_validation=human_validations.get(index),
        )
        for index, result in enumerate(orchestration_result.results)
    )


def verify_replay(passage_text: str, expected_hash: str) -> bool:
    """INV-rejouabilité (§8): recompute the hash from a reconstituted passage."""
    return passage_hash(passage_text) == expected_hash
