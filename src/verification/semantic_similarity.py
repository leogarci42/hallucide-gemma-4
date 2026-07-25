from __future__ import annotations

import re
from typing import Iterable

from hallucide.verification.normalization import normalize_text
from hallucide.core_types.types import ClaimStatus, Passage, VerificationResult

# Étape 8, Path A -- similarité "sémantique" DÉTERMINISTE.
#
# Contrat de sûreté (à ne jamais violer) : cette couche ne peut que SIGNALER
# une reformulation douteuse (augmenter le risque). Elle n'authentifie JAMAIS
# un claim, ne rattrape JAMAIS un NON_AUTHENTIFIÉ, ne remplace pas le
# vérificateur verbatim (§7). Elle vient APRÈS le déterministe, pas à sa place.
#
# Volontairement SANS embeddings, SANS modèle, SANS appel réseau : deux mesures
# lexicales purement calculables (donc 100% reproductibles et testables), qui
# approximent la proximité sans introduire le "flou" d'un score de modèle. Le
# nom "sémantique" est un raccourci de vocabulaire : le calcul reste lexical.

_TOKEN_PATTERN = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)
_STOPWORDS = frozenset(
    {
        "le", "la", "les", "un", "une", "des", "de", "du", "et", "ou", "est",
        "que", "qui", "quoi", "ce", "cette", "ces", "à", "au", "aux", "en",
        "dans", "sur", "pour", "par", "avec", "sans", "se", "sa", "son", "ses",
        "il", "elle", "on", "ne", "pas", "plus", "leur", "leurs", "sont", "a",
        "d", "l", "qu", "s", "n", "the",
    }
)

# En dessous de ce score, une reformulation (INTERPRÉTATION) est jugée trop
# éloignée de la source pour passer sans regard humain. Réglable ; documenté
# comme un plancher de sûreté, pas comme un seuil d'acceptation (il ne sert
# jamais à valider, seulement à alerter).
DEFAULT_DISTANCE_THRESHOLD = 0.30

# Poids des deux mesures dans le score combiné. Les tokens capturent le
# vocabulaire partagé ; les trigrammes de caractères capturent la proximité
# morphologique (accords, flexions) que le token exact rate.
_TOKEN_WEIGHT = 0.5
_TRIGRAM_WEIGHT = 0.5


def _content_tokens(text: str) -> set[str]:
    tokens = _TOKEN_PATTERN.findall(normalize_text(text).lower())
    return {t for t in tokens if t not in _STOPWORDS and (t.isdigit() or len(t) > 1)}


def _char_trigrams(text: str) -> set[str]:
    # Trigrammes sur le texte normalisé, espaces compris (les frontières de
    # mots comptent). Une chaîne de moins de 3 caractères n'a pas de trigramme :
    # on retombe alors sur elle-même pour ne pas renvoyer un ensemble vide.
    normalized = normalize_text(text).lower()
    if len(normalized) < 3:
        return {normalized} if normalized else set()
    return {normalized[i : i + 3] for i in range(len(normalized) - 2)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def similarity_score(candidate: str, source: str) -> float:
    """Proximité lexicale déterministe entre une reformulation et sa source,
    dans [0, 1] (1 = identique après normalisation, 0 = aucun terme commun).

    Combinaison de deux mesures purement calculables :
      - Jaccard des tokens de contenu (vocabulaire partagé),
      - Jaccard des trigrammes de caractères (proximité morphologique).

    Aucun modèle, aucun aléa : deux appels sur les mêmes entrées donnent
    toujours le même nombre.
    """
    token_sim = _jaccard(_content_tokens(candidate), _content_tokens(source))
    trigram_sim = _jaccard(_char_trigrams(candidate), _char_trigrams(source))
    return _TOKEN_WEIGHT * token_sim + _TRIGRAM_WEIGHT * trigram_sim


def is_distant_reformulation(
    candidate: str, source: str, threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> bool:
    """True si la reformulation est trop éloignée de la source pour passer
    sans regard humain (score de proximité strictement sous le seuil).

    Ne dit PAS "c'est faux" -- dit "c'est assez loin pour mériter une
    validation humaine". C'est un signal de risque, pas un verdict.
    """
    return similarity_score(candidate, source) < threshold


def semantic_floor_conditions(
    verification: VerificationResult,
    passage: Passage,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> list[bool]:
    """Conditions de plancher de risque dérivées de la proximité sémantique,
    une par claim (même ordre que `verification.claims`).

    True = "reformulation éloignée -> forcer le risque élevé". Conçu pour être
    passé (par un OU logique) dans le paramètre `floor_conditions` existant de
    `Hallucide.ask` : additif, ne modifie aucune logique du moteur.

    Ne s'applique qu'aux INTERPRÉTATION (les seules reformulations). Les claims
    verbatim (AUTHENTIFIÉ / CITÉ_NON_OPPOSABLE) et les statuts déjà bloquants ne
    sont jamais concernés : cette couche ne peut qu'ajouter du risque, pas en
    retirer.
    """
    conditions: list[bool] = []
    for claim in verification.claims:
        if claim.status == ClaimStatus.INTERPRÉTATION:
            conditions.append(is_distant_reformulation(claim.ref, passage.text, threshold))
        else:
            conditions.append(False)
    return conditions


def any_distant_reformulation(
    verification: VerificationResult,
    passage: Passage,
    threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> bool:
    """Raccourci : au moins un claim est-il une reformulation éloignée ?"""
    return any(semantic_floor_conditions(verification, passage, threshold))
