#!/usr/bin/env python3
"""Standalone: measures what closed-list routing buys, against a baseline of
the same model answering with no routing at all.

No clone, no install: standard library only, talks to an OpenAI-compatible
endpoint. The routing prompt and the code guard are copied verbatim from
src/decomposition/routing.py so the figure describes the shipped behaviour.

    python measure_on_brev.py                       # localhost:8000
    python measure_on_brev.py http://host:8000/v1 model-id

Prints a table and writes measurements.json. If the model does not answer, it
says so and reports nothing.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request

BASE_URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/v1").rstrip("/")
MODEL = sys.argv[2] if len(sys.argv) > 2 else None

DOMAINE_AUCUN = "AUCUN"

DOMAINS = {
    "Oncology": "Essais cliniques et études en oncologie (cancers)",
    "Cardiovascular_Medicine": "Études sur les maladies cardiovasculaires",
    "Neurology": "Études en neurologie",
    "Infectious_Diseases_(except_HIV_AIDS)": "Études sur les maladies infectieuses (hors VIH/sida)",
    "Endocrinology": "Études en endocrinologie (diabète, hormones, métabolisme)",
}

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


def http_json(url: str, payload: dict | None = None, timeout: int = 90) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"} if data else {}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def chat(messages: list[dict], max_tokens: int = 64) -> str:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0}
    out = http_json(f"{BASE_URL}/chat/completions", body)
    return (out["choices"][0]["message"].get("content") or "").strip()


# --- copied from src/decomposition/routing.py, prompt and guard both ---------

def routing_messages(question: str) -> list[dict]:
    listing = "\n".join(f"- {k}: {v}" for k, v in DOMAINS.items())
    return [
        {
            "role": "system",
            "content": (
                "Tu choisis un domaine de connaissance dans une liste FERMÉE pour "
                "répondre à une question médicale.\n"
                f"Domaines disponibles :\n{listing}\n\n"
                "Réponds UNIQUEMENT par la clé exacte d'un domaine ci-dessus (pas la "
                f"description, pas de phrase), ou par « {DOMAINE_AUCUN} » si aucun "
                "domaine ne correspond à la question. Aucun autre texte."
            ),
        },
        {"role": "user", "content": question},
    ]


def route(question: str) -> str | None:
    """None means refused: the model said none, or the code guard rejected an
    answer that is not exactly one of the keys."""
    text = chat(routing_messages(question), max_tokens=24)
    first = text.splitlines()[0].strip().strip("\"'.` ") if text.strip() else DOMAINE_AUCUN
    if first == DOMAINE_AUCUN or first not in DOMAINS:
        return None
    return first


def baseline_answers(question: str) -> bool:
    """No routing, no corpus: does the model answer at all?"""
    try:
        text = chat([{"role": "user", "content": question}], max_tokens=120)
    except Exception:
        return False
    if not text:
        return False
    head = text[:140].lower()
    return not any(w in head for w in ("je ne peux pas", "i cannot", "i can't", "aucune information"))


def main() -> int:
    global MODEL
    try:
        models = http_json(f"{BASE_URL}/models", timeout=15)["data"]
    except Exception as exc:
        print(f"no model list at {BASE_URL}: {type(exc).__name__}: {exc}")
        return 1

    ids = [m["id"] for m in models]
    if MODEL is None:
        gemma = [i for i in ids if "gemma" in i.lower()]
        MODEL = gemma[0] if gemma else ids[0]
    print(f"endpoint {BASE_URL}")
    print(f"models   {', '.join(ids)}")
    print(f"using    {MODEL}\n")

    try:
        chat([{"role": "user", "content": "ok"}], max_tokens=8)
    except Exception as exc:
        print(f"the model does not answer chat completions: {exc}")
        print("nothing measured, nothing reported.")
        return 1

    rows = []
    started = time.time()
    for question, inside in [(q, True) for q in IN_CORPUS] + [(q, False) for q in OUT_OF_CORPUS]:
        try:
            chosen = route(question)
            refused, error = chosen is None, None
        except Exception as exc:
            chosen, refused, error = None, None, f"{type(exc).__name__}: {exc}"
        answered = baseline_answers(question)
        rows.append({
            "question": question, "inCorpus": inside, "routedTo": chosen,
            "routedRefused": refused, "routingError": error, "baselineAnswered": answered,
        })
        print(f"  [{'in ' if inside else 'out'}] {('refused' if refused else (chosen or 'error')):<34} {question[:50]}")

    inside_rows = [r for r in rows if r["inCorpus"]]
    outside_rows = [r for r in rows if not r["inCorpus"]]

    def rate(subset, key, want):
        usable = [r for r in subset if r[key] is not None]
        return f"{sum(1 for r in usable if r[key] is want)}/{len(usable)}" if usable else "not measured"

    report = {
        "endpoint": BASE_URL, "model": MODEL, "domains": list(DOMAINS),
        "routed": {
            "refusedOutOfCorpus": rate(outside_rows, "routedRefused", True),
            "refusedInCorpus": rate(inside_rows, "routedRefused", True),
        },
        "baseline": {
            "answeredOutOfCorpus": rate(outside_rows, "baselineAnswered", True),
            "answeredInCorpus": rate(inside_rows, "baselineAnswered", True),
        },
        "elapsedSeconds": round(time.time() - started, 1),
        "rows": rows,
    }

    print(f"\nmodel {MODEL}")
    print(f"{len(inside_rows)} questions covered by the corpus, {len(outside_rows)} plainly outside it\n")
    print(f"  out of corpus, refused by routing   {report['routed']['refusedOutOfCorpus']}")
    print(f"  out of corpus, answered by baseline {report['baseline']['answeredOutOfCorpus']}")
    print(f"  in corpus, refused by routing       {report['routed']['refusedInCorpus']}  (over-refusal)")
    print(f"  in corpus, answered by baseline     {report['baseline']['answeredInCorpus']}")
    print(f"\n  {report['elapsedSeconds']}s")

    with open("measurements.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("  written to measurements.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
