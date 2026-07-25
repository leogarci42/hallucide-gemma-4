from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

# §12 : calibration inter-annotateur du gold standard, "pour valider le jeu
# de test, pas l'opération" -- ce module ne juge jamais une exécution réelle
# du système, seulement l'accord entre humains sur les statuts attendus d'un
# TrapCase (trap_dataset.py) avant de faire confiance au jeu de test lui-même.


@dataclass(frozen=True)
class Annotation:
    """Jugement d'un annotateur humain sur un cas du gold standard (§12).

    `judged_status` reprend la même valeur que TrapCase.expected_status
    (une valeur de ClaimStatus, ou REFUS pour un refus attendu) -- catégoriel,
    jamais un score continu, cohérent avec §1bis (5 statuts + refus, rien de plus).
    """

    case_id: str
    annotator_id: str
    judged_status: str


@dataclass(frozen=True)
class CohenKappaResult:
    kappa: float
    agreement_count: int
    total_count: int
    categories: tuple[str, ...]

    @property
    def observed_agreement(self) -> float:
        return self.agreement_count / self.total_count if self.total_count else 1.0


def _paired_judgments(
    annotations_a: list[Annotation], annotations_b: list[Annotation]
) -> list[tuple[str, str]]:
    """Associe les jugements des deux annotateurs par case_id, en ignorant
    les cas non jugés par les deux -- le kappa n'a de sens que sur l'intersection.
    """
    by_case_a = {a.case_id: a.judged_status for a in annotations_a}
    by_case_b = {a.case_id: a.judged_status for a in annotations_b}
    common_case_ids = sorted(set(by_case_a) & set(by_case_b))
    return [(by_case_a[case_id], by_case_b[case_id]) for case_id in common_case_ids]


def compute_cohen_kappa(
    annotations_a: list[Annotation], annotations_b: list[Annotation]
) -> CohenKappaResult:
    """Kappa de Cohen (§12) entre deux annotateurs sur les cas jugés par les
    deux. κ = (accord_observé - accord_attendu_au_hasard) / (1 - accord_attendu_au_hasard).

    κ=1 : accord parfait. κ=0 : pas mieux que le hasard. κ<0 : pire que le hasard.
    N'a de sens que sur des jugements catégoriels (ici : statut attendu) --
    jamais utilisé pour juger une exécution réelle du système, seulement le
    jeu de test lui-même (INV-014 : aucun juge-LLM, et ceci n'est pas un juge
    du tout, seulement une mesure d'accord humain⇄humain).
    """
    pairs = _paired_judgments(annotations_a, annotations_b)
    if not pairs:
        raise ValueError("No common case_id between the two annotators; nothing to compare.")

    categories = tuple(sorted({status for pair in pairs for status in pair}))
    total = len(pairs)
    agreement_count = sum(1 for a, b in pairs if a == b)
    observed_agreement = agreement_count / total

    marginal_a = {cat: sum(1 for a, _ in pairs if a == cat) / total for cat in categories}
    marginal_b = {cat: sum(1 for _, b in pairs if b == cat) / total for cat in categories}
    expected_agreement = sum(marginal_a[cat] * marginal_b[cat] for cat in categories)

    if expected_agreement >= 1.0:
        # Accord parfait attendu par construction (une seule catégorie utilisée
        # par les deux annotateurs) -- kappa indéfini par la formule standard,
        # mais l'accord observé est nécessairement total dans ce cas.
        kappa = 1.0
    else:
        kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)

    return CohenKappaResult(
        kappa=kappa,
        agreement_count=agreement_count,
        total_count=total,
        categories=categories,
    )


@dataclass(frozen=True)
class CalibrationReport:
    """Rapport de calibration (§12) sur un sous-ensemble annoté du gold
    standard par plusieurs annotateurs (2 ou plus -- kappa moyen par paire
    si plus de deux, comme recommandé pour l'accord multi-annotateurs).
    """

    pairwise_kappas: tuple[tuple[str, str, CohenKappaResult], ...]

    @property
    def mean_kappa(self) -> float:
        if not self.pairwise_kappas:
            return 1.0
        return sum(result.kappa for _, _, result in self.pairwise_kappas) / len(self.pairwise_kappas)


def run_calibration(annotations: list[Annotation]) -> CalibrationReport:
    """Calibration inter-annotateur (§12) : calcule le kappa de Cohen pour
    chaque paire d'annotateurs distincts présents dans `annotations`.
    """
    by_annotator: dict[str, list[Annotation]] = {}
    for annotation in annotations:
        by_annotator.setdefault(annotation.annotator_id, []).append(annotation)

    annotator_ids = sorted(by_annotator)
    pairwise = tuple(
        (id_a, id_b, compute_cohen_kappa(by_annotator[id_a], by_annotator[id_b]))
        for id_a, id_b in combinations(annotator_ids, 2)
    )
    return CalibrationReport(pairwise_kappas=pairwise)
