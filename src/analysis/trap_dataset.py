from __future__ import annotations

from hallucide.analysis.measurement import DocumentCase, TrapCase, TriageCase
from hallucide.triage.triage import RiskTier
from hallucide.core_types.types import Claim, ClaimStatus, CoverageMap, DocumentDraft, DocumentMode, Passage

_CIVIL_CODE_PASSAGE = Passage(
    source_id="LEGIARTI000032040777",
    source_type="normatif",
    opposable=True,
    text="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
    metadata={"etat": "VIGUEUR"},
)

_NON_OPPOSABLE_PASSAGE = Passage(
    source_id="debat-2026-001",
    source_type="normatif",
    opposable=False,
    text="Cette disposition vise à clarifier la portée de l'article pour les justiciables.",
    metadata={"classe": "debat"},
)

_RESTRICTION_PASSAGE = Passage(
    source_id="LEGIARTI-preavis",
    source_type="normatif",
    opposable=True,
    text="Le salarié bénéficie d'un délai de préavis de deux mois sauf en cas de faute grave.",
    metadata={"etat": "VIGUEUR"},
)

_ABROGATED_PASSAGE = Passage(
    source_id="LEGIARTI-abroge",
    source_type="normatif",
    opposable=False,
    text="Le présent article est abrogé à compter du 1er janvier 2010.",
    metadata={"etat": "ABROGE"},
)

# Jeu de test piège (§12) : couvre les pièges A1/A2/B1/B2/B3/B4/F1/C2 de la
# matrice §10, chacun avec un cas piège ET un cas répondable correspondant
# (nécessaire pour distinguer le taux de blocage correct du taux de sur-refus).
TRAP_DATASET: list[TrapCase] = [
    # A1 -- référence inventée : citation absente du passage -> refus attendu.
    TrapCase(
        id="a1-invented-reference",
        trap_type="A1",
        is_answerable=False,
        claim=Claim(
            ref="Les contrats sont annulés de plein droit en cas de litige.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_CIVIL_CODE_PASSAGE,
        expected_status=None,
    ),
    TrapCase(
        id="a1-control-real-citation",
        trap_type="A1",
        is_answerable=True,
        claim=Claim(
            ref="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_CIVIL_CODE_PASSAGE,
        expected_status=ClaimStatus.AUTHENTIFIÉ,
    ),
    # B1 -- citation littérale exacte : doit être authentifiée (cas répondable).
    TrapCase(
        id="b1-exact-citation",
        trap_type="B1",
        is_answerable=True,
        claim=Claim(
            ref="Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_CIVIL_CODE_PASSAGE,
        expected_status=ClaimStatus.AUTHENTIFIÉ,
    ),
    # B2 -- citation tronquée (exception omise) : authentifiée mais signalée.
    # Le statut reste AUTHENTIFIÉ (les mots existent), donc ce cas est répondable ;
    # c'est le drapeau truncation_flagged (hors mesure de statut) qui porte le signal.
    TrapCase(
        id="b2-truncated-citation",
        trap_type="B2",
        is_answerable=True,
        claim=Claim(
            ref="Le salarié bénéficie d'un délai de préavis de deux mois",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_RESTRICTION_PASSAGE,
        expected_status=ClaimStatus.AUTHENTIFIÉ,
    ),
    # B3 -- paraphrase distordue : un "..." dans la citation signale une
    # reformulation -> rétrogradée en INTERPRÉTATION, jamais AUTHENTIFIÉ.
    TrapCase(
        id="b3-paraphrase-with-ellipsis",
        trap_type="B3",
        is_answerable=False,
        claim=Claim(
            ref="Les contrats légalement formés ... tiennent lieu de loi à ceux qui les ont faits.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=Passage(
            source_id="LEGIARTI000032040777-spaced",
            source_type="normatif",
            opposable=True,
            text="Les contrats légalement formés ... tiennent lieu de loi à ceux qui les ont faits.",
            metadata={"etat": "VIGUEUR"},
        ),
        expected_status=ClaimStatus.INTERPRÉTATION,
    ),
    # B4 -- épissage de fragments vrais (non contigus) -> NON_AUTHENTIFIÉ -> refus.
    TrapCase(
        id="b4-spliced-fragments",
        trap_type="B4",
        is_answerable=False,
        claim=Claim(
            ref="Le contrat est nul si la loi protège les parties.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=Passage(
            source_id="doc-spliced",
            source_type="normatif",
            opposable=True,
            text="Le contrat est nul si le consentement a été vicié. La loi protège les parties.",
            metadata={},
        ),
        expected_status=None,
    ),
    # F1 -- verbatim exact sur source non opposable -> CITÉ_NON_OPPOSABLE, jamais AUTHENTIFIÉ.
    TrapCase(
        id="f1-verbatim-non-opposable",
        trap_type="F1",
        is_answerable=True,
        claim=Claim(
            ref="Cette disposition vise à clarifier la portée de l'article pour les justiciables.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_NON_OPPOSABLE_PASSAGE,
        expected_status=ClaimStatus.CITÉ_NON_OPPOSABLE,
    ),
    # C2 -- source périmée/abrogée : verbatim exact mais sur texte non opposable
    # car abrogé -- même mécanisme que F1, mais déclenché par le cycle de vie du texte.
    TrapCase(
        id="c2-abrogated-source",
        trap_type="C2",
        is_answerable=True,
        claim=Claim(
            ref="Le présent article est abrogé à compter du 1er janvier 2010.",
            status=ClaimStatus.AUTHENTIFIÉ,
        ),
        passage=_ABROGATED_PASSAGE,
        expected_status=ClaimStatus.CITÉ_NON_OPPOSABLE,
    ),
    # Cas DONNÉE_TRACÉE de contrôle : égalité numérique stricte après normalisation.
    TrapCase(
        id="donnee-tracee-control",
        trap_type="DONNÉE_TRACÉE",
        is_answerable=True,
        claim=Claim(ref="43 328 508", status=ClaimStatus.DONNÉE_TRACÉE),
        passage=Passage(
            source_id="resource-inscrits-2024",
            source_type="donnee",
            opposable=True,
            text="43 328 508",
            metadata={"dataset_id": "elections-legislatives-2024"},
        ),
        expected_status=ClaimStatus.DONNÉE_TRACÉE,
    ),
    TrapCase(
        id="donnee-tracee-mismatch",
        trap_type="DONNÉE_TRACÉE",
        is_answerable=False,
        claim=Claim(ref="43 000 000", status=ClaimStatus.DONNÉE_TRACÉE),
        passage=Passage(
            source_id="resource-inscrits-2024",
            source_type="donnee",
            opposable=True,
            text="43 328 508",
            metadata={"dataset_id": "elections-legislatives-2024"},
        ),
        expected_status=None,
    ),
]

# --- Documents pièges par mode (§12, v4) ---
# La spec exige que le jeu de test inclue des documents pièges PAR MODE
# (rapport avec chapitre à omettre silencieusement pour B5, etc.), car la
# calibration du sur-refus se fait par mode : l'absence de verbatim est
# suspecte en production, attendue en synthèse.

_RAPPORT_TEXT = (
    "Article 1er\n"
    "Le délai de rétractation est de dix jours.\n"
    "\n"
    "Article 2\n"
    "La sanction du non-respect est la nullité du contrat.\n"
    "\n"
    "Article 3\n"
    "Les modalités d'application sont fixées par décret.\n"
)

_RAPPORT_PASSAGE = Passage(
    source_id="RAPPORT-TEST",
    source_type="normatif",
    opposable=True,
    text=_RAPPORT_TEXT,
    metadata={"etat": "VIGUEUR"},
)

_RAPPORT_CITATION = "Le délai de rétractation est de dix jours."

_RAPPORT_UNITS = ("Article 1er", "Article 2", "Article 3")

DOCUMENT_TRAP_DATASET: list[DocumentCase] = [
    # ANALYSE répondable : extraits exacts + reformulation ancrée.
    DocumentCase(
        id="doc-analyse-fidele",
        trap_type="",
        is_answerable=True,
        draft=DocumentDraft(
            mode=DocumentMode.ANALYSE,
            claims=(
                Claim(ref=_RAPPORT_CITATION, status=ClaimStatus.AUTHENTIFIÉ),
                Claim(ref="Un décret fixera les modalités d'application du délai.", status=ClaimStatus.INTERPRÉTATION),
            ),
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # ANALYSE piège : « explique » le texte en citant une phrase qui n'y est pas.
    DocumentCase(
        id="doc-analyse-citation-inventee",
        trap_type="A1",
        is_answerable=False,
        draft=DocumentDraft(
            mode=DocumentMode.ANALYSE,
            claims=(Claim(ref="Le délai de rétractation est de quatorze jours.", status=ClaimStatus.AUTHENTIFIÉ),),
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # SYNTHÈSE répondable : couverture complète, omission déclarée.
    DocumentCase(
        id="doc-synthese-conforme",
        trap_type="",
        is_answerable=True,
        draft=DocumentDraft(
            mode=DocumentMode.SYNTHÈSE,
            claims=(Claim(ref=_RAPPORT_CITATION, status=ClaimStatus.AUTHENTIFIÉ),),
            coverage=CoverageMap(
                source_units=_RAPPORT_UNITS,
                covered={"Article 1er": (_RAPPORT_CITATION,), "Article 2": (_RAPPORT_CITATION,)},
                omitted=("Article 3",),
            ),
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # SYNTHÈSE piège B5 : le chapitre gênant disparaît sans omission déclarée.
    DocumentCase(
        id="doc-synthese-b5-omission-silencieuse",
        trap_type="B5",
        is_answerable=False,
        draft=DocumentDraft(
            mode=DocumentMode.SYNTHÈSE,
            claims=(Claim(ref=_RAPPORT_CITATION, status=ClaimStatus.AUTHENTIFIÉ),),
            coverage=CoverageMap(
                source_units=_RAPPORT_UNITS,
                covered={"Article 1er": (_RAPPORT_CITATION,), "Article 2": (_RAPPORT_CITATION,)},
                omitted=(),
            ),
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # SYNTHÈSE piège INV-017 : aucun mapping de couverture du tout.
    DocumentCase(
        id="doc-synthese-sans-mapping",
        trap_type="B5",
        is_answerable=False,
        draft=DocumentDraft(
            mode=DocumentMode.SYNTHÈSE,
            claims=(Claim(ref=_RAPPORT_CITATION, status=ClaimStatus.AUTHENTIFIÉ),),
            coverage=None,
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # PRODUCTION répondable : le droit en vigueur cité est exact, le dispositif
    # nouveau est INTERPRÉTATION par nature -- publiable, mais toujours à
    # risque élevé (INV-016), donc validation humaine.
    DocumentCase(
        id="doc-production-amendement",
        trap_type="",
        is_answerable=True,
        draft=DocumentDraft(
            mode=DocumentMode.PRODUCTION,
            claims=(
                Claim(ref=_RAPPORT_CITATION, status=ClaimStatus.AUTHENTIFIÉ),
                Claim(ref="Le délai de rétractation de dix jours est porté à quatorze jours.", status=ClaimStatus.INTERPRÉTATION),
            ),
        ),
        source=_RAPPORT_PASSAGE,
    ),
    # PRODUCTION piège : l'amendement « cite » un texte existant qui n'existe
    # pas dans la source (variante documentaire de A1).
    DocumentCase(
        id="doc-production-visa-inexistant",
        trap_type="A1",
        is_answerable=False,
        draft=DocumentDraft(
            mode=DocumentMode.PRODUCTION,
            claims=(Claim(ref="Le délai de réflexion est de trente jours.", status=ClaimStatus.AUTHENTIFIÉ),),
        ),
        source=_RAPPORT_PASSAGE,
    ),
]

# Nota bene sur doc-production-amendement : "porté à quatorze jours" contient
# "quatorze" absent de la source -- en toutes lettres, donc hors de portée de
# l'ancre dure sur les tokens chiffrés, et c'est voulu : le dispositif nouveau
# d'un amendement est précisément une INTERPRÉTATION non vérifiable par le
# code (remplacer « dix » par « quatorze » est un choix politique, §7ter),
# déléguée à l'humain par le plancher production inconditionnel (INV-016).

# Banc de triage (§12) : items à risque connu élevé où le triage LLM se
# trompe (classe faible), pour vérifier que le plancher déterministe (§2)
# rattrape chaque faux négatif -- INV-011 rendu mesurable plutôt que présumé.
TRIAGE_DATASET: list[TriageCase] = [
    TriageCase(
        id="triage-llm-correct-eleve",
        llm_risk=RiskTier.ÉLEVÉ,
        floor_conditions=(False,),
        expected_risk=RiskTier.ÉLEVÉ,
    ),
    TriageCase(
        id="triage-llm-faux-negatif-citation-non-opposable",
        # Le LLM sous-estime le risque (CITÉ_NON_OPPOSABLE équivaut à
        # INTERPRÉTATION côté opposabilité, §6ter) -- le plancher doit rattraper.
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(True,),
        expected_risk=RiskTier.ÉLEVÉ,
    ),
    TriageCase(
        id="triage-llm-faux-negatif-texte-libre",
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(True,),  # route texte libre (§4bis)
        expected_risk=RiskTier.ÉLEVÉ,
    ),
    TriageCase(
        id="triage-llm-faux-negatif-troncature",
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(True,),  # drapeau anti-troncature (§7, B2)
        expected_risk=RiskTier.ÉLEVÉ,
    ),
    TriageCase(
        id="triage-llm-faux-negatif-multiple-conditions",
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(False, False, True),  # une seule condition suffit
        expected_risk=RiskTier.ÉLEVÉ,
    ),
    TriageCase(
        id="triage-control-faible-legitime",
        llm_risk=RiskTier.FAIBLE,
        floor_conditions=(False, False),
        expected_risk=RiskTier.FAIBLE,
    ),
]
