import pytest

from hallucide.analysis.calibration import Annotation, compute_cohen_kappa, run_calibration


def test_perfect_agreement_gives_kappa_one() -> None:
    a = [Annotation(case_id=f"c{i}", annotator_id="a1", judged_status=s) for i, s in enumerate(["AUTHENTIFIÉ", "REFUS", "INTERPRÉTATION", "REFUS"])]
    b = [Annotation(case_id=f"c{i}", annotator_id="a2", judged_status=s) for i, s in enumerate(["AUTHENTIFIÉ", "REFUS", "INTERPRÉTATION", "REFUS"])]

    result = compute_cohen_kappa(a, b)

    assert result.kappa == pytest.approx(1.0)
    assert result.agreement_count == 4
    assert result.total_count == 4


def test_systematic_disagreement_gives_low_or_negative_kappa() -> None:
    # a1 dit toujours AUTHENTIFIÉ, a2 dit toujours REFUS -- désaccord total,
    # mais chacun est cohérent avec lui-même (une seule catégorie chacun).
    a = [Annotation(case_id=f"c{i}", annotator_id="a1", judged_status="AUTHENTIFIÉ") for i in range(5)]
    b = [Annotation(case_id=f"c{i}", annotator_id="a2", judged_status="REFUS") for i in range(5)]

    result = compute_cohen_kappa(a, b)

    assert result.agreement_count == 0
    assert result.kappa <= 0.0


def test_partial_agreement_gives_intermediate_kappa() -> None:
    a = [Annotation(case_id=f"c{i}", annotator_id="a1", judged_status=s) for i, s in enumerate(
        ["AUTHENTIFIÉ", "AUTHENTIFIÉ", "REFUS", "REFUS", "INTERPRÉTATION"]
    )]
    b = [Annotation(case_id=f"c{i}", annotator_id="a2", judged_status=s) for i, s in enumerate(
        ["AUTHENTIFIÉ", "REFUS", "REFUS", "REFUS", "INTERPRÉTATION"]
    )]

    result = compute_cohen_kappa(a, b)

    assert 0.0 < result.kappa < 1.0
    assert result.agreement_count == 4


def test_raises_when_no_common_cases() -> None:
    a = [Annotation(case_id="c1", annotator_id="a1", judged_status="AUTHENTIFIÉ")]
    b = [Annotation(case_id="c2", annotator_id="a2", judged_status="REFUS")]

    with pytest.raises(ValueError, match="No common case_id"):
        compute_cohen_kappa(a, b)


def test_ignores_cases_not_judged_by_both_annotators() -> None:
    a = [
        Annotation(case_id="c1", annotator_id="a1", judged_status="AUTHENTIFIÉ"),
        Annotation(case_id="c2", annotator_id="a1", judged_status="REFUS"),
    ]
    b = [Annotation(case_id="c1", annotator_id="a2", judged_status="AUTHENTIFIÉ")]  # ne juge pas c2

    result = compute_cohen_kappa(a, b)

    assert result.total_count == 1
    assert result.agreement_count == 1


def test_run_calibration_computes_all_pairwise_kappas() -> None:
    annotations = [
        Annotation(case_id="c1", annotator_id="a1", judged_status="AUTHENTIFIÉ"),
        Annotation(case_id="c2", annotator_id="a1", judged_status="REFUS"),
        Annotation(case_id="c1", annotator_id="a2", judged_status="AUTHENTIFIÉ"),
        Annotation(case_id="c2", annotator_id="a2", judged_status="REFUS"),
        Annotation(case_id="c1", annotator_id="a3", judged_status="REFUS"),
        Annotation(case_id="c2", annotator_id="a3", judged_status="AUTHENTIFIÉ"),
    ]

    report = run_calibration(annotations)

    assert len(report.pairwise_kappas) == 3  # C(3,2) paires
    pair_ids = {(a, b) for a, b, _ in report.pairwise_kappas}
    assert ("a1", "a2") in pair_ids
    assert ("a1", "a3") in pair_ids
    assert ("a2", "a3") in pair_ids

    kappa_a1_a2 = next(r for a, b, r in report.pairwise_kappas if (a, b) == ("a1", "a2"))
    assert kappa_a1_a2.kappa == pytest.approx(1.0)

    kappa_a1_a3 = next(r for a, b, r in report.pairwise_kappas if (a, b) == ("a1", "a3"))
    assert kappa_a1_a3.kappa <= 0.0


def test_run_calibration_with_no_annotations_has_empty_report() -> None:
    report = run_calibration([])
    assert report.pairwise_kappas == ()
    assert report.mean_kappa == 1.0
