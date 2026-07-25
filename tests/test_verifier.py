from hallucide.core_types.exceptions import VerificationError
from hallucide.core_types.types import Claim, ClaimStatus, Passage
from hallucide.verification.verifier import verify_claims


def test_faithful_paraphrase_marked_interpretation_is_accepted() -> None:
    # §7 : une reformulation (INTERPRÉTATION) dont les termes sont ancrés dans
    # le passage est publiée en INTERPRÉTATION (non opposable), pas bloquée.
    passage = Passage(
        source_id="q1", source_type="normatif", opposable=False,
        text="Mme Obono interroge le ministre sur la réduction progressive des contrats aidés pour les associations.",
        metadata={},
    )
    claims = [Claim(ref="La question porte sur la réduction des contrats aidés pour les associations.", status=ClaimStatus.INTERPRÉTATION)]
    result = verify_claims(claims, passage)

    assert result.verbatim_check == "PASS"
    assert result.claims[0].status == ClaimStatus.INTERPRÉTATION


def test_unanchored_interpretation_is_refused() -> None:
    # Une reformulation dont les termes n'ont AUCUN ancrage dans le passage
    # est une invention pure -> NON_AUTHENTIFIÉ (blocage), même marquée INTERPRÉTATION.
    passage = Passage(
        source_id="q2", source_type="normatif", opposable=False,
        text="Mme Obono interroge le ministre sur la réduction des contrats aidés pour les associations.",
        metadata={},
    )
    claims = [Claim(ref="La question mentionne la fermeture de la trésorerie municipale de Saint-Étienne.", status=ClaimStatus.INTERPRÉTATION)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError as exc:
        assert exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ


def test_paraphrase_labelled_authentifie_is_refused() -> None:
    # §7 : "le mot authentifié ne touche jamais une paraphrase". Un claim
    # marqué AUTHENTIFIÉ mais qui n'est PAS un verbatim exact est bloqué.
    passage = Passage(
        source_id="q3", source_type="normatif", opposable=True,
        text="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
        metadata={},
    )
    claims = [Claim(ref="Les contrats formés ont force de loi.", status=ClaimStatus.AUTHENTIFIÉ)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError as exc:
        assert exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ


def test_verify_claims_authenticated_passage() -> None:
    passage = Passage(
        source_id="doc1",
        source_type="normatif",
        opposable=True,
        text="Article 1103 du Code civil : Les contrats légalement formés tiennent lieu de loi entre les parties.",
        metadata={},
    )
    claims = [Claim(ref="Les contrats légalement formés tiennent lieu de loi entre les parties.", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.verbatim_check == "PASS"
    assert result.claims[0].status == ClaimStatus.AUTHENTIFIÉ


def test_verify_claims_cite_non_opposable() -> None:
    passage = Passage(
        source_id="doc2",
        source_type="normatif",
        opposable=False,
        text="Exposé des motifs : Cette disposition est proposée pour clarifier la procédure.",
        metadata={},
    )
    claims = [Claim(ref="Cette disposition est proposée pour clarifier la procédure.", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.verbatim_check == "PASS"
    assert result.claims[0].status == ClaimStatus.CITÉ_NON_OPPOSABLE


def test_verify_claims_fail_on_non_contiguous_segment() -> None:
    passage = Passage(
        source_id="doc3",
        source_type="normatif",
        opposable=True,
        text="Le contrat est nul si le consentement a été vicié. La loi protège les parties.",
        metadata={},
    )
    claims = [Claim(ref="Le contrat est nul si le consentement a été vicié. La loi protège les parties.", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.verbatim_check == "PASS"
    assert result.claims[0].status == ClaimStatus.AUTHENTIFIÉ


def test_verify_claims_fail_on_spliced_segment() -> None:
    passage = Passage(
        source_id="doc4",
        source_type="normatif",
        opposable=True,
        text="Le contrat est nul si le consentement a été vicié. La loi protège les parties.",
        metadata={},
    )
    claims = [Claim(ref="Le contrat est nul si la loi protège les parties.", status=ClaimStatus.AUTHENTIFIÉ)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError:
        assert True


def test_verification_error_carries_the_failed_result() -> None:
    # §8 : l'exception de refus porte le VerificationResult complet pour que
    # l'orchestrateur puisse le journaliser (compliance_status BLOCKED).
    passage = Passage(source_id="d", source_type="normatif", opposable=True, text="Texte officiel.", metadata={})
    claims = [Claim(ref="Citation inventée absente du passage.", status=ClaimStatus.AUTHENTIFIÉ)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError as exc:
        assert exc.result is not None
        assert exc.result.verbatim_check == "FAIL"
        assert exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ


def test_verify_claims_flags_truncation_when_exception_omitted_immediately_after() -> None:
    # Piège B2 (§7) : "sauf" suit immédiatement la portion citée, mais le
    # claim s'arrête juste avant -- l'exception a été omise par la citation.
    passage = Passage(
        source_id="doc5",
        source_type="normatif",
        opposable=True,
        text="Le salarié bénéficie d'un délai de préavis de deux mois sauf en cas de faute grave.",
        metadata={},
    )
    claims = [Claim(ref="Le salarié bénéficie d'un délai de préavis de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.AUTHENTIFIÉ
    assert result.claims[0].truncation_flagged is True


def test_verify_claims_does_not_flag_when_citation_includes_the_restriction() -> None:
    passage = Passage(
        source_id="doc6",
        source_type="normatif",
        opposable=True,
        text="Le salarié bénéficie d'un délai de préavis de deux mois sauf en cas de faute grave.",
        metadata={},
    )
    claims = [
        Claim(
            ref="Le salarié bénéficie d'un délai de préavis de deux mois sauf en cas de faute grave.",
            status=ClaimStatus.AUTHENTIFIÉ,
        )
    ]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is False


def test_verify_claims_does_not_flag_when_no_restriction_follows() -> None:
    passage = Passage(
        source_id="doc7",
        source_type="normatif",
        opposable=True,
        text="Le salarié bénéficie d'un délai de préavis de deux mois. Le contrat se poursuit normalement.",
        metadata={},
    )
    claims = [Claim(ref="Le salarié bénéficie d'un délai de préavis de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is False


def test_verify_claims_does_not_flag_restriction_far_from_citation() -> None:
    # Portée explicitement limitée à l'adjacence textuelle immédiate (§7) :
    # un "sauf" qui n'est pas collé à la citation échappe à l'heuristique.
    passage = Passage(
        source_id="doc8",
        source_type="normatif",
        opposable=True,
        text="Le salarié bénéficie d'un délai de préavis de deux mois. Une autre phrase ici. Sauf disposition contraire.",
        metadata={},
    )
    claims = [Claim(ref="Le salarié bénéficie d'un délai de préavis de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is False


def test_verify_claims_flags_truncation_on_any_occurrence_not_just_first() -> None:
    # Régression : si la citation apparaît deux fois dans le passage, seule
    # la seconde occurrence étant suivie d'une restriction omise, le drapeau
    # doit tout de même se déclencher (ne pas s'arrêter à la 1ère occurrence).
    passage = Passage(
        source_id="doc9",
        source_type="normatif",
        opposable=True,
        text=(
            "Le salarié bénéficie d'un délai de préavis de deux mois. "
            "Ailleurs dans le même texte, le salarié bénéficie d'un délai de préavis de deux mois "
            "sauf en cas de faute grave."
        ),
        metadata={},
    )
    claims = [Claim(ref="le salarié bénéficie d'un délai de préavis de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is True


def test_verify_claims_does_not_flag_word_merely_starting_with_connector() -> None:
    # Régression B2 : un mot plus long partageant le préfixe d'un connecteur
    # (ex. "sauferie" pour "sauf") ne doit pas déclencher un faux drapeau.
    passage = Passage(
        source_id="doc10",
        source_type="normatif",
        opposable=True,
        text="Le délai est de deux mois sauferie diverse.",
        metadata={},
    )
    claims = [Claim(ref="Le délai est de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is False


def test_interpretation_with_inverted_negation_is_refused() -> None:
    # Ancre dure (§7, B3) : « n'est pas valide » ne doit pas s'ancrer comme
    # « est valide » -- les marqueurs de négation de la reformulation doivent
    # exister dans la source, même s'ils sont des stopwords pour le seuil de 60%.
    passage = Passage(
        source_id="neg1", source_type="normatif", opposable=False,
        text="Le contrat conclu entre les parties est valide et produit tous ses effets.",
        metadata={},
    )
    claims = [Claim(ref="Le contrat conclu entre les parties n'est pas valide.", status=ClaimStatus.INTERPRÉTATION)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError as exc:
        assert exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ


def test_interpretation_with_faithful_negation_is_accepted() -> None:
    # Contrôle de sur-refus : une reformulation qui reprend la négation
    # présente dans la source reste acceptée ("n'" élidé ≡ "ne" plein).
    passage = Passage(
        source_id="neg2", source_type="normatif", opposable=False,
        text="Le contrat conclu entre les parties ne produit pas ses effets à l'égard des tiers.",
        metadata={},
    )
    claims = [Claim(ref="Le contrat conclu entre les parties n'a pas d'effets à l'égard des tiers.", status=ClaimStatus.INTERPRÉTATION)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.INTERPRÉTATION


def test_interpretation_with_substituted_number_is_refused() -> None:
    # Ancre dure (§7, B3) : un chiffre substitué (« 14 jours » au lieu de
    # « 10 jours ») noyé dans une phrase par ailleurs fidèle est une
    # distorsion, pas un ancrage -- tout token chiffré doit exister dans la source.
    passage = Passage(
        source_id="num1", source_type="normatif", opposable=False,
        text="Le consommateur dispose d'un délai de rétractation de 10 jours à compter de la signature.",
        metadata={},
    )
    claims = [Claim(ref="Le consommateur dispose d'un délai de rétractation de 14 jours après la signature.", status=ClaimStatus.INTERPRÉTATION)]
    try:
        verify_claims(claims, passage)
        assert False, "Expected VerificationError"
    except VerificationError as exc:
        assert exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ


def test_interpretation_with_faithful_number_is_accepted() -> None:
    passage = Passage(
        source_id="num2", source_type="normatif", opposable=False,
        text="Le consommateur dispose d'un délai de rétractation de 10 jours à compter de la signature.",
        metadata={},
    )
    claims = [Claim(ref="Le consommateur dispose d'un délai de rétractation de 10 jours après la signature.", status=ClaimStatus.INTERPRÉTATION)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.INTERPRÉTATION


def test_verbatim_comparison_is_case_insensitive() -> None:
    # §7 : la casse ne change pas la fidélité d'une citation -- une majuscule
    # de début de phrase ne doit pas produire un NON_AUTHENTIFIÉ (sur-refus, §12).
    passage = Passage(
        source_id="case1", source_type="normatif", opposable=True,
        text="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
        metadata={},
    )
    claims = [Claim(ref="les contrats légalement formés tiennent lieu de loi", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.AUTHENTIFIÉ


def test_verbatim_comparison_ignores_edge_punctuation() -> None:
    # §7 : un point final ajouté par la citation (le passage continue par une
    # virgule) est de la ponctuation de bord, pas une infidélité -- et la
    # troncature adjacente doit toujours être détectée sur la citation nettoyée.
    passage = Passage(
        source_id="punct1", source_type="normatif", opposable=True,
        text="Le salarié bénéficie d'un délai de préavis de deux mois, sauf en cas de faute grave.",
        metadata={},
    )
    claims = [Claim(ref="Le salarié bénéficie d'un délai de préavis de deux mois.", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.AUTHENTIFIÉ
    assert result.claims[0].truncation_flagged is True


def test_opposable_passage_with_abrogated_etat_is_downgraded() -> None:
    # Défense en profondeur C2 (§7) : le vérificateur re-contrôle le cycle de
    # vie -- un Passage mal construit par un provider (opposable=True mais
    # etat=ABROGE) ne produit jamais un AUTHENTIFIÉ.
    passage = Passage(
        source_id="c2-defense", source_type="normatif", opposable=True,
        text="Le présent article fixe le délai à deux mois.",
        metadata={"etat": "ABROGE"},
    )
    claims = [Claim(ref="Le présent article fixe le délai à deux mois.", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.CITÉ_NON_OPPOSABLE


def test_traced_data_accepts_decimal_comma_vs_point() -> None:
    # INV-013 : "14,5" et "14.5" sont la même valeur -- pas de refus sur la
    # convention décimale.
    passage = Passage(
        source_id="data1", source_type="donnee", opposable=True,
        text="14,5", metadata={},
    )
    claims = [Claim(ref="14.5", status=ClaimStatus.DONNÉE_TRACÉE)]
    result = verify_claims(claims, passage)

    assert result.claims[0].status == ClaimStatus.DONNÉE_TRACÉE


def test_verify_claims_flags_connector_at_end_of_passage() -> None:
    # Un connecteur en toute fin de passage (frontière = fin de chaîne) est
    # une omission tout aussi réelle et doit être signalée.
    passage = Passage(
        source_id="doc11",
        source_type="normatif",
        opposable=True,
        text="Le délai est de deux mois toutefois",
        metadata={},
    )
    claims = [Claim(ref="Le délai est de deux mois", status=ClaimStatus.AUTHENTIFIÉ)]
    result = verify_claims(claims, passage)

    assert result.claims[0].truncation_flagged is True
