from __future__ import annotations

import re
from dataclasses import dataclass

from hallucide.verification.normalization import normalize_text
from hallucide.core_types.types import Intent

_TOKEN_PATTERN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)

# Mots vides : leur absence dans une intention ne signale jamais un oubli (§4 étape 1bis).
_STOPWORDS = frozenset(
    {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "est",
        "que", "qui", "quoi", "ce", "cette", "ces", "à", "au", "aux", "en",
        "dans", "sur", "pour", "par", "avec", "sans", "se", "sa", "son", "ses",
        "il", "elle", "on", "je", "tu", "nous", "vous", "ils", "elles",
        "sont", "es", "suis", "sommes", "êtes", "été", "être",
        "d", "l", "qu", "s", "n",
    }
)

DEFAULT_COVERAGE_THRESHOLD = 0.8


def _is_significant(token: str) -> bool:
    if token in _STOPWORDS:
        return False
    # Un chiffre isolé (article 6, alinéa 2) est significatif : ne jamais le
    # filtrer, sinon une intention entière sur une référence à un chiffre
    # devient invisible au contrôle de couverture E4 (§4 étape 1bis).
    if token.isdigit():
        return True
    # Sinon, écarter les fragments d'un seul caractère (initiales/élisions
    # parasites qui échappent aux stopwords, ex. "j", "m", "c").
    return len(token) > 1


def _significant_tokens(text: str) -> set[str]:
    normalized = normalize_text(text).lower()
    tokens = _TOKEN_PATTERN.findall(normalized)
    return {t for t in tokens if _is_significant(t)}


@dataclass(frozen=True)
class CoverageResult:
    """Contrôle de couverture (§4 étape 1bis) : la concaténation des intentions
    couvre-t-elle le message source ? Heuristique lexicale, déterministe,
    mais pas une preuve sémantique -- piège E4 'borné puis délégué' (§10),
    pas verrouillé : un faux négatif de couverture reste possible sur
    synonymes ou reformulations fortes.
    """

    ratio: float
    threshold: float
    missing_tokens: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.ratio >= self.threshold


def check_coverage(
    message: str,
    intents: list[Intent],
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> CoverageResult:
    message_tokens = _significant_tokens(message)
    if not message_tokens:
        return CoverageResult(ratio=1.0, threshold=threshold, missing_tokens=())

    intent_tokens: set[str] = set()
    for intent in intents:
        intent_tokens |= _significant_tokens(intent.question)

    covered = message_tokens & intent_tokens
    missing = message_tokens - intent_tokens
    ratio = len(covered) / len(message_tokens)

    return CoverageResult(
        ratio=ratio,
        threshold=threshold,
        missing_tokens=tuple(sorted(missing)),
    )


def build_echo_back(intents: list[Intent]) -> str:
    """Restitution des N intentions à l'utilisateur avant réponse (§4 étape 1bis),
    requise dès que N>1.
    """
    lines = [f"{i + 1}. {intent.question}" for i, intent in enumerate(intents)]
    return "\n".join(lines)
