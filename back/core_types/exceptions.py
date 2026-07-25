from __future__ import annotations

class HallucideError(Exception):
    """Base exception for hallucide failures."""


class VerificationError(HallucideError):
    """Raised when a verification step fails for a claim or a result.

    Carries the full VerificationResult (verbatim_check="FAIL" + per-claim
    statuses) so a caller that wants to log the refusal (§8, compliance_status
    "BLOCKED") can do so instead of losing the detail to the exception.
    """

    def __init__(self, message: str, result: object | None = None) -> None:
        super().__init__(message)
        self.result = result


class RetrievalError(HallucideError):
    """Raised during retrieval state or document handling issues."""


class InvalidClaimError(HallucideError, ValueError):
    """Raised when a claim is invalid or cannot be parsed."""