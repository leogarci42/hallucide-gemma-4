from hallucide.validation.human_validation import (
    HumanValidationRegistry,
    ValidationDecision,
    ValidationKey,
    is_publishable,
    resolve_human_validation_status,
)
from hallucide.core_types.types import Claim, ClaimStatus, Intent, IntentExecutionResult, Passage, RiskTier, VerificationResult


def _result(risk_tier: RiskTier, intent_id: str = "1", text: str = "Passage authentique.") -> IntentExecutionResult:
    intent = Intent(id=intent_id, question="?")
    passage = Passage(source_id="doc1", source_type="normatif", opposable=True, text=text, metadata={})
    verification = VerificationResult(
        verbatim_check="PASS", claims=(Claim(ref=text, status=ClaimStatus.AUTHENTIFIÉ),)
    )
    return IntentExecutionResult(intent=intent, passage=passage, verification=verification, risk_tier=risk_tier)


def test_low_risk_is_always_na_and_publishable() -> None:
    registry = HumanValidationRegistry()
    result = _result(RiskTier.FAIBLE)

    assert resolve_human_validation_status(result, registry) == "n/a"
    assert is_publishable(result, registry) is True


def test_high_risk_without_decision_is_pending_and_not_publishable() -> None:
    registry = HumanValidationRegistry()
    result = _result(RiskTier.ÉLEVÉ)

    assert resolve_human_validation_status(result, registry) == "pending"
    assert is_publishable(result, registry) is False


def test_high_risk_approved_becomes_publishable() -> None:
    registry = HumanValidationRegistry()
    result = _result(RiskTier.ÉLEVÉ)
    key = ValidationKey.from_result(result)
    registry.record_decision(key, ValidationDecision.APPROVED, validator_ref="dep-hash-abc")

    assert resolve_human_validation_status(result, registry) == "approved"
    assert is_publishable(result, registry) is True


def test_high_risk_rejected_stays_unpublishable() -> None:
    registry = HumanValidationRegistry()
    result = _result(RiskTier.ÉLEVÉ)
    key = ValidationKey.from_result(result)
    registry.record_decision(key, ValidationDecision.REJECTED, validator_ref="dep-hash-abc")

    assert resolve_human_validation_status(result, registry) == "rejected"
    assert is_publishable(result, registry) is False


def test_validation_key_distinguishes_intents_sharing_same_passage() -> None:
    registry = HumanValidationRegistry()
    result_a = _result(RiskTier.ÉLEVÉ, intent_id="1", text="même passage")
    result_b = _result(RiskTier.ÉLEVÉ, intent_id="2", text="même passage")

    registry.record_decision(
        ValidationKey.from_result(result_a), ValidationDecision.APPROVED, validator_ref="dep-hash-abc"
    )

    assert resolve_human_validation_status(result_a, registry) == "approved"
    assert resolve_human_validation_status(result_b, registry) == "pending"


def test_validator_ref_is_never_required_to_be_a_plain_identity() -> None:
    # §13.4 : le validateur est référencé sous forme pseudonymisée, jamais en clair.
    registry = HumanValidationRegistry()
    result = _result(RiskTier.ÉLEVÉ)
    record = registry.record_decision(
        ValidationKey.from_result(result), ValidationDecision.APPROVED, validator_ref="dep-hash-9f3a21"
    )

    assert record.validator_ref == "dep-hash-9f3a21"
    assert record.comment is None
