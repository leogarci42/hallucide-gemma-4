from __future__ import annotations

import re
from dataclasses import dataclass

from hallucide.verification.normalization import normalize_text

# §4bis, piège A3 : un numéro existant mais faux, deviné par le LLM, récupère
# un article réel qui matche verbatim et serait exposé AUTHENTIFIÉ à tort.
# Discriminateur déterministe : la valeur du slot figure-t-elle dans la
# question de l'utilisateur ? Copiée -> confiance ; inférée -> escalade.
_TOKEN_SPLIT_PATTERN = re.compile(r"[\s.,;:()\[\]«»\"']+")
_LOOSE_NORMALIZE_PATTERN = re.compile(r"[^A-Z0-9-]")


def _loose_normalize(value: str) -> str:
    return _LOOSE_NORMALIZE_PATTERN.sub("", normalize_text(value).upper())


def _question_tokens(question: str) -> set[str]:
    """Tokens de la question, normalisés individuellement mais SANS fusion
    entre eux -- pour comparer le slot à des mots entiers, jamais à une
    sous-chaîne d'un mot voisin (ex. l'article '16' ne doit pas matcher
    l'année '2016', piège A3 §10)."""
    raw_tokens = _TOKEN_SPLIT_PATTERN.split(normalize_text(question))
    return {_loose_normalize(tok) for tok in raw_tokens if tok}


@dataclass(frozen=True)
class SlotProvenance:
    slot_name: str
    slot_value: str
    copied: bool

    @property
    def inferred(self) -> bool:
        return not self.copied


def check_slot_provenance(question: str, slot_name: str, slot_value: str) -> SlotProvenance:
    """Détermine si `slot_value` a été copié depuis `question` ou inféré par le LLM.

    Comparaison déterministe après normalisation souple (casse, accents,
    ponctuation, espaces) afin que "L. 1232-6" et "L1232-6" soient reconnus
    comme la même valeur copiée. Ceci ne prouve jamais la bonne référence ;
    il isole le sous-ensemble dangereux (piège A3, §10), pour escalade.
    """
    if not slot_value:
        return SlotProvenance(slot_name=slot_name, slot_value=slot_value, copied=False)

    question_tokens = _question_tokens(question)
    collapsed_slot = _loose_normalize(slot_value)  # "L. 1232-6" -> "L1232-6"

    raw_slot_tokens = _TOKEN_SPLIT_PATTERN.split(normalize_text(slot_value))
    slot_tokens = {_loose_normalize(tok) for tok in raw_slot_tokens if tok}
    slot_tokens.discard("")

    # Copié si :
    #  - la forme concaténée du slot est un mot entier de la question
    #    ("L. 1232-6" reconnu même si écrit "L1232-6"), OU
    #  - chaque token significatif du slot est un mot entier de la question
    #    (slot multi-mots comme "code civil").
    # Comparaison par mot entier dans les deux cas, jamais par sous-chaîne
    # (l'article "16" ne matche plus l'année "2016", piège A3 §10).
    copied = (
        bool(collapsed_slot) and collapsed_slot in question_tokens
    ) or (
        bool(slot_tokens) and slot_tokens.issubset(question_tokens)
    )

    return SlotProvenance(slot_name=slot_name, slot_value=slot_value, copied=copied)
