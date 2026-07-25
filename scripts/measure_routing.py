#!/usr/bin/env python3
"""Measures what the closed-list routing buys, against a baseline of the same
model with no routing at all.

    python -m scripts.measure_routing            # prints a table
    python -m scripts.measure_routing --json out.json

Two arms, same questions, same Gemma:

  baseline   the question goes straight to Gemma. Count how many it answers.
  routed     the question goes through the closed domain list first, with the
             code guard rejecting anything off-list. Count how many it refuses.

Half the questions are covered by the corpus and half are plainly outside it.
Both halves matter: refusing everything would score perfectly on one and
uselessly on the other, so over-refusal is reported separately.

Nothing is printed that was not measured. If the model backend does not answer,
the script says so and exits non-zero rather than reporting a number.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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

from hallucide import GemmaModelProvider  # noqa: E402
from hallucide.core_types.exceptions import HallucideError  # noqa: E402
from hallucide.decomposition.routing import DomainRouter  # noqa: E402

from scripts.ask_medical import DOMAINS  # noqa: E402

# Covered by one of the five domains in the closed list.
IN_CORPUS = [
    "Quelle est l'efficacité de l'immunothérapie dans le cancer du poumon ?",
    "Le dépistage précoce réduit-il la mortalité du cancer colorectal ?",
    "Quels sont les effets des statines sur le risque d'infarctus ?",
    "L'hypertension non traitée augmente-t-elle le risque d'AVC ?",
    "Quels traitements ralentissent la progression de la maladie d'Alzheimer ?",
    "La stimulation cérébrale profonde aide-t-elle dans la maladie de Parkinson ?",
    "Quelle est l'efficacité des antiviraux contre la grippe saisonnière ?",
    "La vaccination réduit-elle la transmission de la tuberculose ?",
    "La metformine réduit-elle les complications du diabète de type 2 ?",
    "Quels sont les effets de la thyroxine sur l'hypothyroïdie ?",
]

# Plainly outside the corpus: a router that sends these to a medical dataset
# would be inventing a source.
OUT_OF_CORPUS = [
    "Quel temps fera-t-il demain à Paris ?",
    "Quelle est la durée légale du préavis de démission en France ?",
    "Comment réussir une pâte à choux ?",
    "Qui a gagné la Ligue des champions en 2024 ?",
    "Comment configurer un serveur nginx en reverse proxy ?",
    "Quel est le prix moyen du mètre carré à Lyon ?",
    "Quelles sont les règles du hors-jeu au football ?",
    "Comment déclarer ses revenus fonciers ?",
    "Quelle est la capitale de l'Australie ?",
    "Comment changer une courroie de distribution ?",
]


def _answers_anything(provider: GemmaModelProvider, question: str) -> bool:
    """The baseline: no routing, no corpus, just the model. True when it
    produces an answer rather than declining."""
    try:
        out = provider.generate([
            {"role": "user", "content": question},
        ])
    except Exception:
        return False
    text = (out or {}).get("text", "").strip()
    if not text:
        return False
    # a model that declines usually says so in the first line
    head = text[:120].lower()
    declined = any(w in head for w in ("je ne peux pas", "i cannot", "i can't", "aucune information"))
    return not declined


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", dest="json_path")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="routing arm only, when the baseline has already been run")
    ns = parser.parse_args()

    base_url = os.environ.get("MODEL_BASE_URL", "http://localhost:8000/v1")
    model_name = os.environ.get("MODEL_NAME", "google/gemma-4-E4B-it")
    provider = GemmaModelProvider(base_url=base_url, model=model_name)

    # Fail loudly rather than reporting a number nothing produced.
    try:
        probe = provider.generate([{"role": "user", "content": "ok"}])
        if not (probe or {}).get("text"):
            raise RuntimeError("the model answered with no text")
    except Exception as exc:
        print(f"no usable model at {base_url}: {type(exc).__name__}: {exc}", file=sys.stderr)
        print("nothing measured, nothing reported.", file=sys.stderr)
        return 1

    router = DomainRouter(provider, DOMAINS)
    started = time.monotonic()
    rows = []

    for question, in_corpus in [(q, True) for q in IN_CORPUS] + [(q, False) for q in OUT_OF_CORPUS]:
        try:
            chosen = router.route(question)
            routed_refused = chosen is None
            routing_error = None
        except HallucideError as exc:
            chosen, routed_refused, routing_error = None, None, f"{type(exc).__name__}: {exc}"

        baseline_answered = None if ns.skip_baseline else _answers_anything(provider, question)

        rows.append({
            "question": question,
            "inCorpus": in_corpus,
            "routedTo": chosen,
            "routedRefused": routed_refused,
            "routingError": routing_error,
            "baselineAnswered": baseline_answered,
        })
        mark = "refused" if routed_refused else (chosen or "error")
        print(f"  [{'in ' if in_corpus else 'out'}] {mark:<34} {question[:52]}")

    inside = [r for r in rows if r["inCorpus"]]
    outside = [r for r in rows if not r["inCorpus"]]

    def rate(subset, key, want) -> str:
        usable = [r for r in subset if r[key] is not None]
        if not usable:
            return "not measured"
        hits = sum(1 for r in usable if r[key] is want)
        return f"{hits}/{len(usable)}"

    report = {
        "model": model_name,
        "baseUrl": base_url,
        "domains": list(DOMAINS),
        "questions": {"inCorpus": len(inside), "outOfCorpus": len(outside)},
        "routed": {
            "refusedOutOfCorpus": rate(outside, "routedRefused", True),
            "refusedInCorpus": rate(inside, "routedRefused", True),
        },
        "baseline": {
            "answeredOutOfCorpus": rate(outside, "baselineAnswered", True),
            "answeredInCorpus": rate(inside, "baselineAnswered", True),
        },
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "rows": rows,
    }

    print(f"\nmodel {model_name} via {base_url}")
    print(f"{len(inside)} questions covered by the corpus, {len(outside)} plainly outside it\n")
    print(f"  out of corpus, refused by routing   {report['routed']['refusedOutOfCorpus']}")
    print(f"  out of corpus, answered by baseline {report['baseline']['answeredOutOfCorpus']}")
    print(f"  in corpus, refused by routing       {report['routed']['refusedInCorpus']}  (over-refusal)")
    print(f"  in corpus, answered by baseline     {report['baseline']['answeredInCorpus']}")
    print(f"\n  {report['elapsedSeconds']}s")

    if ns.json_path:
        Path(ns.json_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  written to {ns.json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
