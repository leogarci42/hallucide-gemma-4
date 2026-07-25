from __future__ import annotations

import re
import unicodedata
from typing import Pattern


_NON_BREAKING_SPACE = "\u00A0"
_SPACE_PATTERN: Pattern[str] = re.compile(r"[ \t\u00A0]+")
_PUNCTUATION_NORMALIZATION = str.maketrans({
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
})


def normalize_text(text: str) -> str:
    """Normalize text for deterministic verbatim comparison."""
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.translate(_PUNCTUATION_NORMALIZATION)
    normalized = normalized.replace(_NON_BREAKING_SPACE, " ")
    normalized = _SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def normalize_numeric(text: str) -> str:
    """Normalize numeric strings for strict tracked data comparison.

    INV-013 : "14,5" et "14.5" sont la m\u00EAme valeur -- la virgule d\u00E9cimale
    fran\u00E7aise ne doit pas produire un refus face \u00E0 un point d\u00E9cimal (le
    corpus est fran\u00E7ais : la virgule est d\u00E9cimale, pas un s\u00E9parateur de
    milliers ; les milliers sont des espaces, retir\u00E9s juste au-dessus).
    """
    normalized = normalize_text(text)
    normalized = normalized.replace("\u202F", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace(",", ".")
    return normalized
