#!/usr/bin/env python3
"""Pipeline complet du hackathon (étages 1+2+3), branché sur les vrais
endpoints MCP Alien (get.alien.club/gemma4-hackathon, cluster MedRxiv).
Aucune modification du moteur Hallucide : on réutilise Hallucide.ask() tel
quel avec GemmaModelProvider + AlienRetrievalProvider (ou un stub si
ALIEN_API_TOKEN est absent).

Usage : python -m scripts.ask_medical "Quelle est l'efficacité du traitement X ?"
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE))

env_path = WORKSPACE / ".env"
if env_path.exists():
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

from hallucide import AlienRetrievalProvider, GemmaModelProvider, Hallucide  # noqa: E402
from hallucide.core_types.exceptions import HallucideError  # noqa: E402
from hallucide.core_types.types import Intent, Passage, RetrievalState  # noqa: E402
from hallucide.decomposition.routing import DomainRouter  # noqa: E402

# Étage 1 : liste FERMÉE de domaines (noms lisibles pour le prompt de Gemma).
# Sous-ensemble factuel/chiffré du cluster MedRxiv (§6 : "vérité facile à
# vérifier"). Étage 2 : chaque nom est mappé vers son ID numérique de dataset
# MedRxiv réel (confirmé via datacluster_list_datasets, 2026-07-25).
DOMAINS = {
    "Oncology": "Essais cliniques et études en oncologie (cancers)",
    "Cardiovascular_Medicine": "Études sur les maladies cardiovasculaires",
    "Neurology": "Études en neurologie",
    "Infectious_Diseases_(except_HIV_AIDS)": "Études sur les maladies infectieuses (hors VIH/sida)",
    "Endocrinology": "Études en endocrinologie (diabète, hormones, métabolisme)",
}
DATASET_IDS = {
    "Oncology": "30",
    "Cardiovascular_Medicine": "4",
    "Neurology": "25",
    "Infectious_Diseases_(except_HIV_AIDS)": "20",
    "Endocrinology": "8",
}

# Dataset de repli si le routage (étage 1) échoue techniquement
# (§5 : "le routing ne doit jamais casser la chaîne").
FIXED_DATASET_NAME = os.environ.get("ALIEN_DEFAULT_DATASET_NAME", "Oncology")


class _StubAlienRetrievalProvider:
    """Remplace AlienRetrievalProvider tant qu'on n'a pas ALIEN_API_TOKEN --
    permet de tester tout le reste du pipeline (Gemma génère + auto-décompose
    + vérif déterministe) sans accès réseau. Le texte est un passage
    plausible mais fictif, à ne jamais confondre avec une vraie source
    (source_id le signale explicitement)."""

    def retrieve(self, intent: Intent, state: RetrievalState, query: dict[str, str]) -> Passage:
        dataset_id = query.get("dataset_id", "0")
        text = (
            "Dans un essai randomisé contrôlé (n=412), le traitement X a réduit "
            "l'incidence des rechutes de 34% sur 12 mois par rapport au placebo "
            "(IC95% 21-45%, p<0.01). Aucun effet indésirable grave n'a été rapporté."
            "\n\n---\n\n"
            "Une méta-analyse portant sur 8 essais (n total=3120) confirme une "
            "réduction significative du risque relatif (RR=0.68), mais souligne "
            "une hétérogénéité modérée entre études (I²=42%)."
        )
        return Passage(
            source_id=f"STUB:{dataset_id}",
            source_type="etude_medicale",
            opposable=True,
            text=text,
            metadata={
                "dataset_id": dataset_id,
                "nb_passages": 2,
                "nb_chars": len(text),
                "source": "STUB -- pas de vraie donnée Alien (ALIEN_API_TOKEN absent)",
            },
        )


def _build_retrieval_provider():
    token = os.environ.get("ALIEN_API_TOKEN")
    if token:
        mcp_url = os.environ.get("ALIEN_MCP_URL")
        kwargs = {"api_token": token}
        if mcp_url:
            kwargs["mcp_url"] = mcp_url
        return AlienRetrievalProvider(**kwargs), True
    return _StubAlienRetrievalProvider(), False


def _route_dataset(model_provider: GemmaModelProvider, question: str, forced_name: str | None) -> str | None:
    """Étage 1. Renvoie le NOM de domaine choisi, ou None si le système doit
    refuser de répondre (§5, "les 3 chemins possibles") :
      - `forced_name` (argv[2]) court-circuite le routage -- pratique pour
        tester étage 2+3 isolément.
      - domaine trouvé par Gemma dans la liste fermée -> ce nom.
      - AUCUN (explicite ou hors-liste, garde-fou dans DomainRouter) -> None,
        refus explicite, PAS de fallback (§5 : le refus est voulu, pas une panne).
      - panne technique du routage lui-même -> fallback sur FIXED_DATASET_NAME
        (§5 : "le routing ne doit jamais casser la chaîne").
    """
    if forced_name:
        return forced_name

    router = DomainRouter(model_provider, DOMAINS)
    try:
        chosen = router.route(question)
    except HallucideError as exc:
        print(f"[!] Routage étage 1 en panne ({exc}) -> repli sur le dataset fixe '{FIXED_DATASET_NAME}'.\n")
        return FIXED_DATASET_NAME

    if chosen is None:
        print("[REFUS] Aucun domaine du corpus ne couvre cette question -> "
              "le système refuse de répondre plutôt que d'inventer une source.\n")
        return None

    print(f"[Étage 1] Domaine choisi par Gemma : {chosen}\n")
    return chosen


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "Le traitement X réduit-il le risque de rechute ?"
    forced_name = sys.argv[2] if len(sys.argv) > 2 else None

    model_provider = GemmaModelProvider(
        base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it"),
    )

    print(f"Question : {question}\n")

    domain_name = _route_dataset(model_provider, question, forced_name)
    if domain_name is None:
        return

    dataset_id = DATASET_IDS.get(domain_name, domain_name)

    retrieval_provider, is_real = _build_retrieval_provider()
    if not is_real:
        print("[!] ALIEN_API_TOKEN absent de .env -> utilisation d'un passage STUB "
              "(pas de vraie donnée). Colle ton token pour tester en réel.\n")

    guard = Hallucide(model_provider=model_provider, retrieval_provider=retrieval_provider)

    try:
        result = guard.ask(message=question, query={"dataset_id": dataset_id})
    except HallucideError as exc:
        print(f"[ERREUR] {type(exc).__name__}: {exc}")
        return

    for i, r in enumerate(result.orchestration.results):
        print(f"--- Intention {i + 1}: {r.intent.question}")
        print(f"    Risque : {r.risk_tier.value} | Publiable : {result.published[i]}")
        print(f"    Passages injectés : {r.passage.metadata.get('nb_passages', '?')} "
              f"({r.passage.metadata.get('nb_chars', len(r.passage.text))} caractères)")
        for c in r.verification.claims:
            flag = " [TRONCATURE ?]" if c.truncation_flagged else ""
            print(f"      [{c.status.value}]{flag} {c.ref}")
        print()


if __name__ == "__main__":
    main()
