from __future__ import annotations

from enum import Enum
from typing import Iterable


class RiskTier(str, Enum):
    FAIBLE = "faible"
    ÉLEVÉ = "élevé"


def apply_risk_floor(llm_risk: RiskTier, floor_conditions: Iterable[bool]) -> RiskTier:
    """Combine LLM risk with deterministic floor conditions.

    The deterministic floor cannot allow the result to go below elevated
    when any condition is present.
    """
    if llm_risk == RiskTier.ÉLEVÉ or any(floor_conditions):
        return RiskTier.ÉLEVÉ
    return RiskTier.FAIBLE
