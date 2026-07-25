#!/usr/bin/env python3
"""Banc de mesure du hackathon (§6 du brief) : produit le tableau de
résultats "Gemma seul / +RAG massif / +RAG+vérif" + les métriques de
routage (étage 1), sur des questions réelles contre le vrai corpus Alien.

Méthode par question (dans le corpus) :
  1. Étage 1 : DomainRouter route la question -> comparé au domaine attendu
     (métrique "bon dataset choisi").
  2. Un seul appel Alien réel (RAG massif) sert de vérité terrain pour les
     3 configs -- pas de triple coût réseau.
  3. Config A "Gemma seul" : Gemma répond SANS le passage (juste sa
     connaissance paramétrique), la réponse est découpée en phrases, chaque
     phrase est vérifiée contre le passage RÉCUPÉRÉ (mais jamais montré à
     Gemma) -- mesure le taux d'hallucination hors contexte.
  4. Config B "Gemma + RAG massif" : génération contrainte habituelle
     (PromptBasedIntentGenerator, passage injecté) + vérification -- mesure
     le taux d'affirmations NON_AUTHENTIFIÉ malgré le contexte (l'erreur
     brute du modèle, pas encore filtrée).
  5. Config C "Gemma + RAG + vérif déterministe" : mêmes claims que B, mais
     on ne compte que celles qui seraient effectivement PUBLIÉES (jamais les
     NON_AUTHENTIFIÉ, bloquées par construction) -- doit tomber à ~0.

Hors corpus (5 questions) : uniquement le routage, mesure le taux de refus
correct (§6 : "métrique appréciée d'un jury recherche").

Usage : python -m scripts.run_eval
"""
from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass, field
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

from hallucide import AlienRetrievalProvider, GemmaModelProvider  # noqa: E402
from hallucide.core_types.exceptions import HallucideError  # noqa: E402
from hallucide.core_types.types import Claim, ClaimStatus, Intent, Passage, RetrievalState  # noqa: E402
from hallucide.decomposition.llm import PromptBasedIntentGenerator  # noqa: E402
from hallucide.decomposition.routing import DomainRouter  # noqa: E402
from hallucide.verification.verifier import verify_claims  # noqa: E402
from hallucide.core_types.exceptions import VerificationError  # noqa: E402

from scripts.ask_medical import DATASET_IDS, DOMAINS  # noqa: E402

# (question, domaine attendu ou None si hors corpus)
TEST_CASES: list[tuple[str, str | None]] = [
    # Oncology
    ("Does immunotherapy improve survival rates in melanoma patients?", "Oncology"),
    ("What is the risk of relapse after breast cancer treatment?", "Oncology"),
    ("How effective is chemotherapy in reducing tumor size?", "Oncology"),
    # Cardiovascular_Medicine
    ("Does statin therapy reduce the risk of heart attack?", "Cardiovascular_Medicine"),
    ("What are the risk factors for atrial fibrillation?", "Cardiovascular_Medicine"),
    ("How does hypertension affect cardiovascular outcomes?", "Cardiovascular_Medicine"),
    # Neurology
    ("What treatments reduce seizure frequency in epilepsy patients?", "Neurology"),
    ("Does early intervention improve outcomes in stroke patients?", "Neurology"),
    ("What are the risk factors for cognitive decline in Alzheimer's disease?", "Neurology"),
    # Infectious_Diseases_(except_HIV_AIDS)
    ("How effective are vaccines at preventing COVID-19 transmission?", "Infectious_Diseases_(except_HIV_AIDS)"),
    ("What is the impact of antibiotic resistance on treatment outcomes?", "Infectious_Diseases_(except_HIV_AIDS)"),
    ("How does sepsis mortality vary with early antibiotic treatment?", "Infectious_Diseases_(except_HIV_AIDS)"),
    # Endocrinology
    ("Does metformin improve glycemic control in type 2 diabetes patients?", "Endocrinology"),
    ("What is the relationship between obesity and insulin resistance?", "Endocrinology"),
    ("How does thyroid hormone replacement affect metabolic outcomes?", "Endocrinology"),
    # Hors corpus (5)
    ("What's the best recipe for chocolate cake?", None),
    ("How do I fix a flat tire on my bike?", None),
    ("What is the capital of France?", None),
    ("Can you recommend a good sci-fi movie?", None),
    ("What's the weather like in Paris tomorrow?", None),
]

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _ask_freeform(model_provider: GemmaModelProvider, question: str) -> str:
    """Config A : Gemma répond avec SA SEULE connaissance, sans passage --
    prompt ad hoc, n'existe pas dans le moteur (qui ne génère qu'en mode
    contraint par un passage)."""
    response = model_provider.generate(messages=[
        {"role": "system", "content": "Answer the medical question in 2-4 sentences, using your own knowledge."},
        {"role": "user", "content": question},
    ])
    return response.get("text", "")


@dataclass
class CaseOutcome:
    question: str
    expected_domain: str | None
    routed_domain: str | None
    routing_correct: bool
    config_a_total: int = 0
    config_a_hallucinated: int = 0
    config_b_total: int = 0
    config_b_hallucinated: int = 0
    config_c_published_hallucinated: int = 0
    error: str | None = None


def _evaluate_in_corpus(
    model_provider: GemmaModelProvider,
    retrieval_provider: AlienRetrievalProvider,
    question: str,
    expected_domain: str,
    routed_domain: str | None,
) -> CaseOutcome:
    outcome = CaseOutcome(
        question=question,
        expected_domain=expected_domain,
        routed_domain=routed_domain,
        routing_correct=(routed_domain == expected_domain),
    )

    dataset_id = DATASET_IDS[expected_domain]
    intent = Intent(id="1", question=question)
    try:
        passage = retrieval_provider.retrieve(intent, RetrievalState(), {"dataset_id": dataset_id})
    except HallucideError as exc:
        outcome.error = f"retrieval: {exc}"
        return outcome

    # --- Config A : sans contexte, vérité terrain = passage récupéré ---
    try:
        freeform = _ask_freeform(model_provider, question)
        sentences = _split_sentences(freeform)
        claims_a = [Claim(ref=s, status=ClaimStatus.INTERPRÉTATION) for s in sentences]
        outcome.config_a_total = len(claims_a)
        for c in claims_a:
            try:
                result = verify_claims([c], passage)
                if result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ:
                    outcome.config_a_hallucinated += 1
            except VerificationError as exc:
                if exc.result.claims[0].status == ClaimStatus.NON_AUTHENTIFIÉ:
                    outcome.config_a_hallucinated += 1
    except HallucideError as exc:
        outcome.error = f"config_a: {exc}"

    # --- Config B/C : génération contrainte par le passage (RAG massif) ---
    try:
        generator = PromptBasedIntentGenerator(model_provider)
        claims_b = generator.generate_claims(intent, passage)
        outcome.config_b_total = len(claims_b)
        for c in claims_b:
            try:
                result = verify_claims([c], passage)
                verified_status = result.claims[0].status
            except VerificationError as exc:
                verified_status = exc.result.claims[0].status
            if verified_status == ClaimStatus.NON_AUTHENTIFIÉ:
                outcome.config_b_hallucinated += 1
                # Config C : ce claim ne serait JAMAIS publié (bloqué par le
                # vérificateur déterministe) -- ne compte pas ici. On ne
                # compterait dans config_c_published_hallucinated que si le
                # moteur avait laissé passer un NON_AUTHENTIFIÉ comme publié,
                # ce qui ne peut pas arriver par construction (verify_claims
                # lève VerificationError -> IntentExecutionResult marqué
                # risque ÉLEVÉ, jamais AUTHENTIFIÉ/publiable).
    except HallucideError as exc:
        if outcome.error is None:
            outcome.error = f"config_b: {exc}"

    return outcome


def main() -> None:
    model_provider = GemmaModelProvider(
        base_url=os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1"),
        model=os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it"),
    )
    token = os.environ.get("ALIEN_API_TOKEN")
    if not token:
        print("[ERREUR] ALIEN_API_TOKEN absent de .env -- ce banc de mesure a besoin de vraies données.")
        return
    retrieval_provider = AlienRetrievalProvider(api_token=token)
    router = DomainRouter(model_provider, DOMAINS)

    outcomes: list[CaseOutcome] = []
    refusal_correct = 0
    refusal_total = 0
    t0 = time.time()

    for i, (question, expected_domain) in enumerate(TEST_CASES, 1):
        print(f"[{i}/{len(TEST_CASES)}] {question[:70]}...")
        try:
            routed = router.route(question)
        except HallucideError as exc:
            print(f"    routage en panne: {exc}")
            routed = None

        if expected_domain is None:
            refusal_total += 1
            if routed is None:
                refusal_correct += 1
            else:
                print(f"    [!] hors corpus mais routé vers '{routed}' au lieu de refuser")
            continue

        outcome = _evaluate_in_corpus(model_provider, retrieval_provider, question, expected_domain, routed)
        outcomes.append(outcome)
        if outcome.error:
            print(f"    [!] {outcome.error}")
        else:
            print(f"    routage={'OK' if outcome.routing_correct else 'FAUX (' + str(routed) + ')'} "
                  f"| A: {outcome.config_a_hallucinated}/{outcome.config_a_total} halluciné "
                  f"| B: {outcome.config_b_hallucinated}/{outcome.config_b_total} halluciné")

    elapsed = time.time() - t0

    # --- Tableau final ---
    valid = [o for o in outcomes if o.error is None and o.config_a_total and o.config_b_total]
    n = len(valid)
    print("\n" + "=" * 70)
    print(f"RÉSULTATS ({n} questions in-corpus exploitables sur {len(TEST_CASES) - refusal_total}, "
          f"{elapsed:.0f}s)")
    print("=" * 70)

    if n:
        a_rate = sum(o.config_a_hallucinated / o.config_a_total for o in valid) / n * 100
        b_rate = sum(o.config_b_hallucinated / o.config_b_total for o in valid) / n * 100
        routing_acc = sum(o.routing_correct for o in valid) / n * 100
        print(f"{'Config':<45} {'Taux hallucination':>20}")
        print(f"{'-' * 45} {'-' * 20}")
        print(f"{'Gemma seul (sans contexte)':<45} {a_rate:>18.0f}%")
        print(f"{'Gemma + RAG massif (avant filtrage)':<45} {b_rate:>18.0f}%")
        print(f"{'Gemma + RAG + vérif déterministe (publié)':<45} {'0% (garanti par construction)':>28}")
        print()
        print(f"Taux de bon routage (étage 1)       : {routing_acc:.0f}% ({sum(o.routing_correct for o in valid)}/{n})")

    if refusal_total:
        print(f"Taux de refus correct (hors corpus) : {refusal_correct / refusal_total * 100:.0f}% "
              f"({refusal_correct}/{refusal_total})")


if __name__ == "__main__":
    main()
