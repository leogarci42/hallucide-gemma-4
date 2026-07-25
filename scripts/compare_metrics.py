#!/usr/bin/env python3
"""Writes metrics/README.md from the two measurement files, so the table can
never drift from the numbers that produced it.

    python scripts/compare_metrics.py

Reads metrics/baseline.json (the model alone, no routing, no retrieval, no
checking) and metrics/pipeline.json (the same model behind our pipeline). Every
figure printed is read from those files; nothing is typed in by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

METRICS = Path(__file__).resolve().parents[1] / "metrics"


def load(name: str) -> dict:
    path = METRICS / name
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the measurement first, do not write a number by hand.")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    base = load("baseline.json")
    pipe = load("pipeline.json")

    rows = []
    pipe_by_q = {r["question"]: r for r in pipe["rows"]}
    for r in base["rows"]:
        p = pipe_by_q.get(r["question"])
        rows.append((
            r["question"],
            r["inCorpus"],
            "answered" if r["baselineAnswered"] else "declined",
            (p or {}).get("verdict") or "—",
            (p or {}).get("publishedChars", 0),
        ))

    def block(title: str, inside: bool) -> str:
        head = (
            f"### {title}\n\n"
            "| question | model alone | our pipeline | published |\n"
            "|---|---|---|---|\n"
        )
        body = "".join(
            f"| {q} | {b} | {v} | {c} chars |\n"
            for q, i, b, v, c in rows if i is inside
        )
        return head + body

    withheld = sum(
        1 for r in pipe["rows"] if r["inCorpus"] and r.get("verdict") == "unsupported"
    )

    text = f"""# Two arms, same questions, same weights

Both arms ask `{base['model']}` the same twenty questions. The only difference
is what sits in front of the model.

- **baseline** — the question goes straight to the model. No routing, no
  retrieval, no checking. Produced by `scripts/measure_standalone.py`.
- **pipeline** — the question goes through the closed-list routing, Alien
  retrieval, generation, self-decomposition and the two checks. A claim that no
  passage backs is withheld rather than published. Produced by
  `scripts/measure_pipeline.py`.

Ten questions are covered by the corpus and ten are plainly outside it. Both
halves matter: refusing everything wins the first half and loses the second.

## The numbers

| | model alone | our pipeline |
|---|---|---|
| out of corpus, answered / published | {base['baseline']['answeredOutOfCorpus']} | {pipe['outOfCorpus']['publishedSomething']} |
| out of corpus, refused | {base['routed']['refusedOutOfCorpus']} (with routing) | {pipe['outOfCorpus']['refused']} |
| in corpus, answered / published | {base['baseline']['answeredInCorpus']} | {pipe['inCorpus']['publishedSomething']} |
| in corpus, refused | {base['routed']['refusedInCorpus']} (with routing) | {pipe['inCorpus']['refused']} |
| errors | — | {pipe['errors']} |
| wall clock | {base['elapsedSeconds']}s | {pipe['elapsedSeconds']}s |

Read the first row on its own: the model alone answers questions it has no
source for, in the same confident register it uses when it does. The pipeline
publishes only what a retrieved passage backs.

The pipeline is slower by construction — it retrieves, re-reads its own draft
and checks every claim. The cost is the point.

## Where it over-withholds

{withheld} of the ten covered questions ended `unsupported`: the pipeline
retrieved, drafted and decomposed, then withheld every claim. That is the
failure mode this design chooses — it publishes nothing rather than something
unbacked — but it is a failure, not a win, and it is reported here as one.

The known cause is decomposition: medRxiv passages carry inline citation
markers, and the splitter turns some of them into their own "claims", which no
passage can back. Those fragments drag the whole answer to `unsupported`.
Cleaning the markers before decomposition is the fix; it is not in this run.

## Per question

{block('Covered by the corpus', True)}
{block('Plainly outside the corpus', False)}

## Reproducing

```bash
python scripts/measure_standalone.py http://localhost:8000/v1   # baseline arm
python scripts/measure_pipeline.py   http://localhost:8100      # pipeline arm
python scripts/compare_metrics.py                               # this file
```

`baseline.json` and `pipeline.json` hold the per-question detail, including the
claim counts and each verdict.
"""

    out = METRICS / "README.md"
    out.write_text(text, encoding="utf-8")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
