from hallucide.core_types.types import Claim, ClaimStatus, Passage, VerificationResult
from hallucide.verification.semantic_similarity import (
    DEFAULT_DISTANCE_THRESHOLD,
    any_distant_reformulation,
    is_distant_reformulation,
    semantic_floor_conditions,
    similarity_score,
)


# --- similarity_score : bornes et déterminisme ---

def test_identical_text_scores_one() -> None:
    text = "Le consommateur dispose d'un délai de rétractation de 10 jours."
    assert similarity_score(text, text) == 1.0


def test_disjoint_text_scores_zero() -> None:
    # Aucun token de contenu ni trigramme commun.
    assert similarity_score("abcdef", "zyxwvu") == 0.0


def test_score_is_bounded() -> None:
    a = "La bonne foi préside à la négociation des contrats."
    b = "Les parties négocient de bonne foi."
    score = similarity_score(a, b)
    assert 0.0 <= score <= 1.0


def test_score_is_deterministic() -> None:
    # Propriété centrale : aucun aléa, aucun modèle. Deux appels identiques.
    a = "Le contrat engage les parties qui l'ont signé."
    b = "Les signataires sont engagés par le contrat."
    assert similarity_score(a, b) == similarity_score(a, b)


def test_close_paraphrase_scores_higher_than_unrelated() -> None:
    source = "Le consommateur dispose d'un délai de rétractation de dix jours."
    close = "Le délai de rétractation du consommateur est de dix jours."
    far = "Le maire préside le conseil municipal de la commune."
    assert similarity_score(close, source) > similarity_score(far, source)


def test_empty_strings_score_one_together() -> None:
    # Deux vides sont "identiques" (Jaccard de deux ensembles vides = 1).
    assert similarity_score("", "") == 1.0


# --- is_distant_reformulation : le signal de risque ---

def test_close_paraphrase_is_not_distant() -> None:
    source = "Le consommateur dispose d'un délai de rétractation de dix jours."
    close = "Le délai de rétractation du consommateur est de dix jours."
    assert is_distant_reformulation(close, source) is False


def test_unrelated_text_is_distant() -> None:
    source = "Le consommateur dispose d'un délai de rétractation de dix jours."
    far = "Le maire préside le conseil municipal de la commune."
    assert is_distant_reformulation(far, source) is True


def test_threshold_is_tunable() -> None:
    source = "Le consommateur dispose d'un délai de rétractation de dix jours."
    candidate = "Le délai de rétractation est de dix jours pour le consommateur."
    # Un seuil à 1.0 rend (presque) tout "distant" ; un seuil à 0.0 rend tout
    # "proche". Le paramètre gouverne bien la décision.
    assert is_distant_reformulation(candidate, source, threshold=1.0) is True
    assert is_distant_reformulation(candidate, source, threshold=0.0) is False


# --- semantic_floor_conditions : contrat de sûreté (ne fait qu'AJOUTER du risque) ---

def _passage(text: str) -> Passage:
    return Passage(source_id="s", source_type="normatif", opposable=True, text=text, metadata={})


def test_floor_only_applies_to_interpretation() -> None:
    # Un claim AUTHENTIFIÉ (verbatim) n'est JAMAIS marqué par cette couche,
    # même si on lui passe une source différente : elle ne touche pas au verbatim.
    passage = _passage("texte source de référence")
    verification = VerificationResult(
        verbatim_check="PASS",
        claims=(
            Claim(ref="tout autre chose sans rapport aucun", status=ClaimStatus.AUTHENTIFIÉ),
        ),
    )
    assert semantic_floor_conditions(verification, passage) == [False]


def test_floor_flags_distant_interpretation() -> None:
    passage = _passage("Le consommateur dispose d'un délai de rétractation de dix jours.")
    verification = VerificationResult(
        verbatim_check="PASS",
        claims=(
            Claim(ref="Le maire préside le conseil municipal de la commune.", status=ClaimStatus.INTERPRÉTATION),
        ),
    )
    assert semantic_floor_conditions(verification, passage) == [True]


def test_floor_does_not_flag_close_interpretation() -> None:
    passage = _passage("Le consommateur dispose d'un délai de rétractation de dix jours.")
    verification = VerificationResult(
        verbatim_check="PASS",
        claims=(
            Claim(ref="Le délai de rétractation du consommateur est de dix jours.", status=ClaimStatus.INTERPRÉTATION),
        ),
    )
    assert semantic_floor_conditions(verification, passage) == [False]


def test_floor_conditions_preserve_claim_order() -> None:
    passage = _passage("Le consommateur dispose d'un délai de rétractation de dix jours.")
    verification = VerificationResult(
        verbatim_check="PASS",
        claims=(
            Claim(ref="Le délai de rétractation du consommateur est de dix jours.", status=ClaimStatus.INTERPRÉTATION),
            Claim(ref="Le maire préside le conseil municipal.", status=ClaimStatus.INTERPRÉTATION),
            Claim(ref="citation verbatim", status=ClaimStatus.AUTHENTIFIÉ),
        ),
    )
    assert semantic_floor_conditions(verification, passage) == [False, True, False]


def test_any_distant_reformulation() -> None:
    passage = _passage("Le consommateur dispose d'un délai de rétractation de dix jours.")
    verification = VerificationResult(
        verbatim_check="PASS",
        claims=(
            Claim(ref="Le maire préside le conseil municipal.", status=ClaimStatus.INTERPRÉTATION),
        ),
    )
    assert any_distant_reformulation(verification, passage) is True


def test_no_claims_means_no_flags() -> None:
    passage = _passage("texte")
    verification = VerificationResult(verbatim_check="PASS", claims=())
    assert semantic_floor_conditions(verification, passage) == []
    assert any_distant_reformulation(verification, passage) is False


def test_default_threshold_is_exposed() -> None:
    # Garde-fou : le seuil par défaut reste un plancher de sûreté raisonnable.
    assert 0.0 < DEFAULT_DISTANCE_THRESHOLD < 1.0
