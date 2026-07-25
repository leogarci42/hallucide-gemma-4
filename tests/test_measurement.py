from hallucide.analysis.measurement import (
    REFUS,
    TrapCase,
    TriageCase,
    evaluate_case,
    run_document_measurement,
    run_measurement,
    run_triage_measurement,
)
from hallucide.analysis.trap_dataset import DOCUMENT_TRAP_DATASET, TRAP_DATASET, TRIAGE_DATASET
from hallucide.triage.triage import RiskTier
from hallucide.core_types.types import Claim, ClaimStatus, DocumentMode, Passage


def test_evaluate_case_correct_for_exact_citation() -> None:
    passage = Passage(source_id="d1", source_type="normatif", opposable=True, text="Le texte exact.", metadata={})
    case = TrapCase(
        id="t1",
        trap_type="B1",
        is_answerable=True,
        claim=Claim(ref="Le texte exact.", status=ClaimStatus.AUTHENTIFIÉ),
        passage=passage,
        expected_status=ClaimStatus.AUTHENTIFIÉ,
    )
    result = evaluate_case(case)

    assert result.correct is True
    assert result.actual_status == "AUTHENTIFIÉ"


def test_evaluate_case_detects_wrong_expectation() -> None:
    passage = Passage(source_id="d2", source_type="normatif", opposable=True, text="Le texte exact.", metadata={})
    case = TrapCase(
        id="t2",
        trap_type="B1",
        is_answerable=True,
        claim=Claim(ref="Le texte exact.", status=ClaimStatus.AUTHENTIFIÉ),
        passage=passage,
        expected_status=ClaimStatus.CITÉ_NON_OPPOSABLE,  # mauvaise attente volontaire
    )
    result = evaluate_case(case)

    assert result.correct is False


def test_evaluate_case_maps_verification_error_to_refus() -> None:
    passage = Passage(source_id="d3", source_type="normatif", opposable=True, text="Autre chose.", metadata={})
    case = TrapCase(
        id="t3",
        trap_type="A1",
        is_answerable=False,
        claim=Claim(ref="Texte absent du passage.", status=ClaimStatus.AUTHENTIFIÉ),
        passage=passage,
        expected_status=None,
    )
    result = evaluate_case(case)

    assert result.actual_status == REFUS
    assert result.correct is True


def test_full_trap_dataset_achieves_perfect_blocking_and_zero_over_refusal() -> None:
    # Le jeu de test piège (§12) doit démontrer que le vérificateur traite
    # correctement chaque cas connu -- sinon le dataset ou le code a un défaut.
    report = run_measurement(TRAP_DATASET)

    assert report.correct_blocking_rate == 1.0
    assert report.over_refusal_rate == 0.0
    assert all(r.correct for r in report.results)


def test_metrics_by_trap_type_cover_expected_types() -> None:
    report = run_measurement(TRAP_DATASET)
    trap_types = {m.trap_type for m in report.metrics_by_trap_type}

    for expected_type in ("A1", "B1", "B2", "B3", "B4", "C2", "F1"):
        assert expected_type in trap_types

    for m in report.metrics_by_trap_type:
        assert m.accuracy == 1.0


def test_triage_dataset_has_zero_false_negatives() -> None:
    # INV-011 : le plancher déterministe ne peut jamais descendre sous élevé.
    report = run_triage_measurement(TRIAGE_DATASET)

    assert report.false_negative_rate == 0.0
    assert report.false_negatives == 0


def test_document_trap_dataset_blocks_all_traps_without_over_refusal() -> None:
    # §12 (v4) : le jeu de test documentaire de référence doit démontrer un
    # blocage correct de 100% (dont B5) et 0% de sur-refus, mode par mode.
    report = run_document_measurement(DOCUMENT_TRAP_DATASET)

    assert all(r.correct for r in report.results)
    for mode, rate in report.over_refusal_rate_by_mode.items():
        assert rate == 0.0, f"sur-refus en mode {mode}"
    for mode, rate in report.correct_blocking_rate_by_mode.items():
        assert rate == 1.0, f"blocage incorrect en mode {mode}"


def test_document_trap_dataset_covers_all_three_modes_and_b5() -> None:
    modes = {c.draft.mode for c in DOCUMENT_TRAP_DATASET}
    assert modes == {DocumentMode.ANALYSE, DocumentMode.SYNTHÈSE, DocumentMode.PRODUCTION}
    assert any(c.trap_type == "B5" for c in DOCUMENT_TRAP_DATASET)


def test_triage_measurement_detects_a_real_false_negative() -> None:
    # Contre-épreuve : si le plancher était cassé (ex. ignoré), ce cas le révélerait.
    broken_case = TriageCase(
        id="broken",
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(False,),  # aucune condition plancher déclenchée
        expected_risk=RiskTier.ÉLEVÉ,  # mais on attendait élevé quand même
    )
    report = run_triage_measurement([broken_case])

    assert report.false_negatives == 1
    assert report.false_negative_rate == 1.0
