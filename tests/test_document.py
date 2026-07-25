from hallucide.validation.document import (
    check_documentary_coverage,
    segment_source_units,
    verify_document,
)
from hallucide.analysis.measurement import DocumentCase, run_document_measurement
from hallucide.triage.triage import RiskTier
from hallucide.core_types.types import Claim, ClaimStatus, CoverageMap, DocumentDraft, DocumentMode, Passage

_LOI_TEXT = (
    "Article 1er\n"
    "Le délai de rétractation est de dix jours.\n"
    "\n"
    "Article 2\n"
    "La sanction du non-respect est la nullité du contrat.\n"
    "\n"
    "Article 3\n"
    "Les modalités d'application sont fixées par décret.\n"
)


def _loi_passage(opposable: bool = True) -> Passage:
    return Passage(
        source_id="LOI-TEST", source_type="normatif", opposable=opposable,
        text=_LOI_TEXT, metadata={"etat": "VIGUEUR"},
    )


# --- Segmentation déterministe (§7ter : le CODE segmente, jamais le LLM) ---


def test_segmentation_detects_article_headings() -> None:
    units = segment_source_units(_LOI_TEXT)
    assert units == ("Article 1er", "Article 2", "Article 3")


def test_segmentation_falls_back_to_paragraphs() -> None:
    text = "Premier bloc de texte.\n\nSecond bloc de texte.\n\nTroisième bloc."
    assert segment_source_units(text) == ("§1", "§2", "§3")


def test_segmentation_single_block_is_one_unit() -> None:
    assert segment_source_units("Un texte d'un seul tenant, sans structure.") == ("§1",)


# --- INV-015 : aucun statut agrégé, chaque claim garde le sien ---


def test_document_result_has_no_aggregated_status() -> None:
    passage = _loi_passage()
    draft = DocumentDraft(
        mode=DocumentMode.ANALYSE,
        claims=(
            Claim(ref="Le délai de rétractation est de dix jours.", status=ClaimStatus.AUTHENTIFIÉ),
            Claim(ref="Un décret fixera les modalités d'application du délai.", status=ClaimStatus.INTERPRÉTATION),
        ),
    )
    result = verify_document(draft, passage)

    # La mosaïque des statuts est préservée claim par claim...
    assert result.verification.claims[0].status == ClaimStatus.AUTHENTIFIÉ
    assert result.verification.claims[1].status == ClaimStatus.INTERPRÉTATION
    # ...et le résultat n'expose aucun champ de statut agrégé « document vérifié ».
    assert not hasattr(result, "status")
    assert not hasattr(result, "document_status")


# --- Mode ANALYSE ---


def test_analyse_blocks_citation_absent_from_source() -> None:
    # On ne peut pas « expliquer » un texte en citant des phrases qui n'y figurent pas.
    passage = _loi_passage()
    draft = DocumentDraft(
        mode=DocumentMode.ANALYSE,
        claims=(Claim(ref="Le délai de rétractation est de quatorze jours.", status=ClaimStatus.AUTHENTIFIÉ),),
    )
    result = verify_document(draft, passage)

    assert result.publishable is False
    assert result.verification.verbatim_check == "FAIL"
    assert result.risk_tier == RiskTier.ÉLEVÉ


def test_analyse_with_interpretation_elevates_risk_but_publishes() -> None:
    passage = _loi_passage()
    draft = DocumentDraft(
        mode=DocumentMode.ANALYSE,
        claims=(Claim(ref="Un décret fixera les modalités d'application du délai.", status=ClaimStatus.INTERPRÉTATION),),
    )
    result = verify_document(draft, passage)

    assert result.publishable is True
    assert result.risk_tier == RiskTier.ÉLEVÉ  # statut faible -> plancher §2


# --- INV-016 : plancher par mode ---


def test_production_mode_is_always_high_risk() -> None:
    # Même un document 100% verbatim exact sur source opposable en vigueur
    # reste à risque élevé en mode production (INV-016).
    passage = _loi_passage()
    draft = DocumentDraft(
        mode=DocumentMode.PRODUCTION,
        claims=(Claim(ref="Le délai de rétractation est de dix jours.", status=ClaimStatus.AUTHENTIFIÉ),),
    )
    result = verify_document(draft, passage)

    assert result.publishable is True  # rien de bloquant...
    assert result.risk_tier == RiskTier.ÉLEVÉ  # ...mais jamais sans humain


def test_synthesis_of_normative_source_is_high_risk() -> None:
    passage = _loi_passage()
    coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={
            "Article 1er": ("Le délai de rétractation est de dix jours.",),
            "Article 2": ("Le délai de rétractation est de dix jours.",),
            "Article 3": ("Le délai de rétractation est de dix jours.",),
        },
    )
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref="Le délai de rétractation est de dix jours.", status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is True
    assert result.risk_tier == RiskTier.ÉLEVÉ  # source normative (INV-016)


def test_synthesis_of_data_source_can_stay_low_risk() -> None:
    # Contre-exemple qui prouve que le plancher est bien PAR MODE + type de
    # source : une donnée tracée exacte, hors normatif, reste à risque faible.
    passage = Passage(source_id="res-1", source_type="donnee", opposable=True, text="43 328 508", metadata={})
    coverage = CoverageMap(source_units=("§1",), covered={"§1": ("43 328 508",)})
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref="43 328 508", status=ClaimStatus.DONNÉE_TRACÉE),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is True
    assert result.risk_tier == RiskTier.FAIBLE


# --- INV-017 / piège B5 : contrôle de couverture documentaire ---


def test_synthesis_without_coverage_map_is_not_publishable() -> None:
    passage = _loi_passage()
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref="Le délai de rétractation est de dix jours.", status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=None,
    )
    result = verify_document(draft, passage)

    assert result.publishable is False
    assert any("INV-017" in v for v in result.coverage_violations)


def test_silent_omission_is_detected_b5() -> None:
    # Piège B5 : la synthèse couvre les articles 1 et 2 mais l'article 3
    # (le chapitre gênant) disparaît sans être déclaré omis -> blocage.
    passage = _loi_passage()
    ref = "Le délai de rétractation est de dix jours."
    coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={"Article 1er": (ref,), "Article 2": (ref,)},
        omitted=(),
    )
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is False
    assert any("B5" in v and "Article 3" in v for v in result.coverage_violations)


def test_explicit_omission_is_publishable() -> None:
    # La même synthèse avec l'omission NOMMÉE est publiable (risque élevé,
    # source normative) : l'omission déclarée est un choix, pas un trou.
    passage = _loi_passage()
    ref = "Le délai de rétractation est de dix jours."
    coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={"Article 1er": (ref,), "Article 2": (ref,)},
        omitted=("Article 3",),
    )
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is True
    assert result.coverage_violations == ()
    assert result.risk_tier == RiskTier.ÉLEVÉ


def test_coverage_units_must_match_code_segmentation() -> None:
    # Le LLM ne choisit pas le périmètre : des unités inventées ou omises de
    # la segmentation du code sont des violations.
    passage = _loi_passage()
    ref = "Le délai de rétractation est de dix jours."
    coverage = CoverageMap(
        source_units=("Article 1er", "Article 2"),  # Article 3 escamoté du périmètre
        covered={"Article 1er": (ref,), "Article 2": (ref,)},
    )
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is False
    assert any("absentes du mapping" in v for v in result.coverage_violations)


def test_coverage_refs_must_exist_in_document() -> None:
    passage = _loi_passage()
    ref = "Le délai de rétractation est de dix jours."
    coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={
            "Article 1er": (ref,),
            "Article 2": ("Un claim fantôme qui n'existe pas dans le document.",),
            "Article 3": (ref,),
        },
    )
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),),
        coverage=coverage,
    )
    result = verify_document(draft, passage)

    assert result.publishable is False
    assert any("inexistantes" in v for v in result.coverage_violations)


def test_check_documentary_coverage_flags_unit_both_covered_and_omitted() -> None:
    draft = DocumentDraft(
        mode=DocumentMode.SYNTHÈSE,
        claims=(Claim(ref="r1", status=ClaimStatus.INTERPRÉTATION),),
    )
    coverage = CoverageMap(
        source_units=("§1", "§2"),
        covered={"§1": ("r1",), "§2": ("r1",)},
        omitted=("§2",),
    )
    violations = check_documentary_coverage(coverage, ("§1", "§2"), draft)

    assert any("incohérent" in v for v in violations)


# --- §12 (v4) : mesure du sur-refus PAR MODE ---


def test_document_measurement_reports_over_refusal_by_mode() -> None:
    passage = _loi_passage()
    ref = "Le délai de rétractation est de dix jours."
    full_coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={"Article 1er": (ref,), "Article 2": (ref,)},
        omitted=("Article 3",),
    )
    b5_coverage = CoverageMap(
        source_units=("Article 1er", "Article 2", "Article 3"),
        covered={"Article 1er": (ref,), "Article 2": (ref,)},
        omitted=(),
    )
    cases = [
        # Production répondable, publiée -> pas de sur-refus en production.
        DocumentCase(
            id="prod-ok", trap_type="", is_answerable=True,
            draft=DocumentDraft(mode=DocumentMode.PRODUCTION, claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),)),
            source=passage,
        ),
        # Synthèse répondable avec couverture conforme -> publiée.
        DocumentCase(
            id="synth-ok", trap_type="", is_answerable=True,
            draft=DocumentDraft(mode=DocumentMode.SYNTHÈSE, claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),), coverage=full_coverage),
            source=passage,
        ),
        # Synthèse répondable SANS mapping (erreur d'appelant) -> bloquée = sur-refus en synthèse.
        DocumentCase(
            id="synth-sans-mapping", trap_type="", is_answerable=True,
            draft=DocumentDraft(mode=DocumentMode.SYNTHÈSE, claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),), coverage=None),
            source=passage,
        ),
        # Piège B5 : omission silencieuse -> blocage attendu et obtenu.
        DocumentCase(
            id="synth-b5", trap_type="B5", is_answerable=False,
            draft=DocumentDraft(mode=DocumentMode.SYNTHÈSE, claims=(Claim(ref=ref, status=ClaimStatus.AUTHENTIFIÉ),), coverage=b5_coverage),
            source=passage,
        ),
    ]
    report = run_document_measurement(cases)

    # Les taux sont bien calculés PAR MODE, indépendamment l'un de l'autre.
    assert report.over_refusal_rate_by_mode["production"] == 0.0
    assert report.over_refusal_rate_by_mode["synthèse"] == 0.5
    assert report.correct_blocking_rate_by_mode["synthèse"] == 1.0
