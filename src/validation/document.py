"""Mode document (§7ter, v4) : analyse / synthèse / production.

Un document est une liste ordonnée de claims, chacun vérifié individuellement
par le vérificateur déterministe (§7). Ce module n'introduit AUCUN statut
agrégé (INV-015) : le résultat expose la mosaïque des statuts, un plancher de
risque par mode (INV-016) et le contrôle de couverture documentaire qui ferme
le piège B5 (INV-017). Aucun appel LLM (INV-007) -- le LLM propose le mapping
de couverture, le code le vérifie.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from hallucide.core_types.exceptions import VerificationError
from hallucide.triage.triage import RiskTier, apply_risk_floor
from hallucide.core_types.types import ClaimStatus, CoverageMap, DocumentDraft, DocumentMode, Passage, VerificationResult
from hallucide.verification.verifier import verify_claims

# §7ter : la segmentation en unités structurelles est faite par le CODE,
# jamais par le LLM -- sinon le LLM pourrait "segmenter" de façon à faire
# disparaître le chapitre gênant du périmètre de couverture (B5 réintroduit
# en amont). Marqueurs structurels des textes officiels français, détectés
# en début de ligne.
_UNIT_HEADING_PATTERN = re.compile(
    r"^(?:article\s+[\w.\-]+|chapitre\s+[\w]+|section\s+[\w]+|titre\s+[\w]+|annexe\s*[\w]*|livre\s+[\w]+)\s*(?:[-–—.:]|$)",
    re.IGNORECASE,
)
_PARAGRAPH_SPLIT_PATTERN = re.compile(r"\n\s*\n")


def segment_source_units(source_text: str) -> tuple[str, ...]:
    """Segmente une source en unités structurelles (§7ter), déterministe.

    Stratégie : les en-têtes structurels ("Article 3", "Chapitre II",
    "Section 1", "Titre IV", "Livre I", "Annexe") en début de ligne délimitent
    les unités ; l'étiquette de l'unité est la ligne d'en-tête normalisée. À
    défaut d'au moins deux en-têtes, repli sur les paragraphes (blocs séparés
    par une ligne vide), étiquetés "§1", "§2", ... Une source sans structure
    détectable est UNE unité ("§1") -- le contrôle de couverture reste alors
    trivialement exigeant (l'unité doit être couverte ou omise explicitement).
    """
    lines = source_text.splitlines()
    headings: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped and _UNIT_HEADING_PATTERN.match(stripped):
            headings.append(re.sub(r"\s+", " ", stripped.rstrip(" -–—.:")).strip())

    if len(headings) >= 2:
        return tuple(headings)

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT_PATTERN.split(source_text) if p.strip()]
    if not paragraphs:
        return ()
    return tuple(f"§{i}" for i in range(1, len(paragraphs) + 1))


@dataclass(frozen=True)
class DocumentVerificationResult:
    """Résultat §7ter. Volontairement SANS statut agrégé (INV-015) : les
    statuts vivent claim par claim dans `verification.claims`. `publishable`
    n'est pas un statut de qualité mais une porte de conformité : False
    signifie qu'un verrou structurel (refus §7bis ou couverture INV-017)
    bloque la publication -- et True n'exempte jamais de la validation
    humaine quand `risk_tier` est élevé (§4 étape 9).
    """

    mode: DocumentMode
    verification: VerificationResult
    risk_tier: RiskTier
    publishable: bool
    coverage_violations: tuple[str, ...] = ()


def check_documentary_coverage(coverage: CoverageMap | None, source_units: tuple[str, ...], draft: DocumentDraft) -> tuple[str, ...]:
    """Contrôle de couverture documentaire (§7ter, INV-017, piège B5).

    Le mapping est fourni par le LLM mais vérifié ici par lookup :
    - une synthèse sans mapping n'est pas publiable ;
    - les unités du mapping doivent être EXACTEMENT celles segmentées par le
      code (le LLM ne choisit pas le périmètre) ;
    - chaque unité est couverte ou nommée dans les omissions explicites --
      une unité absente des deux est une omission silencieuse ;
    - chaque ref de claim citée par le mapping doit exister dans le document.

    Retourne la liste des violations (vide = couverture conforme).
    """
    if coverage is None:
        return ("INV-017 : synthèse sans mapping de couverture -- non publiable.",)

    violations: list[str] = []

    declared_units = set(coverage.source_units)
    code_units = set(source_units)
    if declared_units != code_units:
        invented = sorted(declared_units - code_units)
        dropped = sorted(code_units - declared_units)
        if invented:
            violations.append(f"Unités absentes de la segmentation du code : {invented}.")
        if dropped:
            violations.append(f"Unités segmentées par le code mais absentes du mapping : {dropped}.")

    claim_refs = {c.ref for c in draft.claims}
    covered_units = set(coverage.covered.keys())
    omitted_units = set(coverage.omitted)

    silently_missing = sorted(code_units - covered_units - omitted_units)
    if silently_missing:
        # Piège B5 : le chapitre gênant a disparu sans être déclaré omis.
        violations.append(f"Omission silencieuse (B5) : unités ni couvertes ni déclarées omises : {silently_missing}.")

    both = sorted(covered_units & omitted_units)
    if both:
        violations.append(f"Unités à la fois couvertes et omises (mapping incohérent) : {both}.")

    for unit, refs in coverage.covered.items():
        unknown = sorted(set(refs) - claim_refs)
        if unknown:
            violations.append(f"Unité '{unit}' : refs de claims inexistantes dans le document : {unknown}.")
        if not refs:
            violations.append(f"Unité '{unit}' : couverte par aucun claim (couverture vide).")

    return tuple(violations)


def verify_document(draft: DocumentDraft, source: Passage, llm_risk: RiskTier = RiskTier.FAIBLE) -> DocumentVerificationResult:
    """Vérifie un DocumentDraft contre sa source (§7ter).

    - Mode ANALYSE : la source est le passage -- on ne peut pas « expliquer »
      un texte en citant des phrases qui n'y figurent pas ; l'anti-troncature
      (B2) élève le risque.
    - Mode SYNTHÈSE : contrôle de couverture documentaire (INV-017) ; plancher
      élevé si la source est normative (INV-016) ; les chiffres repris sont
      tenus par les ancres dures du vérificateur (§7).
    - Mode PRODUCTION : plancher élevé inconditionnel (INV-016) -- le
      dispositif nouveau est INTERPRÉTATION par nature, seul le texte existant
      cité à l'appui est vérifiable.

    Limite documentée : le lookup d'existence de l'article/alinéa visé par un
    amendement (mode production) exige une récupération -- il relève de
    l'appelant via les routes §6bis ; ici le plancher élevé garantit qu'aucune
    production ne contourne l'humain en attendant.
    """
    # §7bis : un refus de vérification (claim NON_AUTHENTIFIÉ) bloque la
    # publication du document mais reste journalisable -- même schéma que
    # l'orchestrateur (§4 étape 8).
    try:
        verification = verify_claims(draft.claims, source)
        refused = False
    except VerificationError as exc:
        verification = exc.result
        refused = True

    coverage_violations: tuple[str, ...] = ()
    if draft.mode == DocumentMode.SYNTHÈSE:
        coverage_violations = check_documentary_coverage(draft.coverage, segment_source_units(source.text), draft)

    # §2 (v4) / INV-016 : plancher par mode, cumulé aux conditions communes
    # (statuts faibles, troncature) -- jamais en remplacement.
    production_mode = draft.mode == DocumentMode.PRODUCTION
    synthesis_of_normative = draft.mode == DocumentMode.SYNTHÈSE and source.source_type == "normatif"
    weak_status_present = any(
        c.status in (ClaimStatus.INTERPRÉTATION, ClaimStatus.CITÉ_NON_OPPOSABLE) for c in verification.claims
    )
    truncation_flagged = any(c.truncation_flagged for c in verification.claims)

    risk_tier = apply_risk_floor(
        llm_risk,
        [
            production_mode,
            synthesis_of_normative,
            weak_status_present,
            truncation_flagged,
            refused,
            bool(coverage_violations),
        ],
    )

    publishable = not refused and not coverage_violations

    return DocumentVerificationResult(
        mode=draft.mode,
        verification=verification,
        risk_tier=risk_tier,
        publishable=publishable,
        coverage_violations=coverage_violations,
    )
