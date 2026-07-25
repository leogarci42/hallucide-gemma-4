#!/usr/bin/env python3
"""Runs the same twenty questions through our own pipeline and records what it
publishes, claim by claim.

    python scripts/measure_pipeline.py                       # bridge on :8100
    python scripts/measure_pipeline.py http://host:8100 out.json

The companion arm is scripts/measure_standalone.py, which asks the same model
directly with no routing, no retrieval and no checking. Comparing the two files
is the whole point: same questions, same weights, one of them with the context
engineering in front.

Standard library only, so it runs anywhere the bridge is reachable. Nothing is
reported that the bridge did not return: a question that errors is recorded as
an error, not as a refusal.
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BRIDGE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8100").rstrip("/")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "pipeline.json")

# The same two sets as scripts/measure_standalone.py, kept identical on purpose:
# a comparison across different questions would not be a comparison.
IN_CORPUS = [
    "How effective is immunotherapy in non-small cell lung cancer?",
    "Does early screening reduce colorectal cancer mortality?",
    "What effect do statins have on the risk of myocardial infarction?",
    "Does untreated hypertension increase the risk of stroke?",
    "Which treatments slow the progression of Alzheimer's disease?",
    "Does deep brain stimulation help in Parkinson's disease?",
    "How effective are antivirals against seasonal influenza?",
    "Does vaccination reduce tuberculosis transmission?",
    "Does metformin reduce complications in type 2 diabetes?",
    "What are the effects of thyroxine on hypothyroidism?",
]

OUT_OF_CORPUS = [
    "What will the weather be in Paris tomorrow?",
    "What is the statutory notice period for resigning in France?",
    "How do you make choux pastry?",
    "Who won the Champions League in 2024?",
    "How do I configure nginx as a reverse proxy?",
    "What is the average price per square metre in Lyon?",
    "What are the offside rules in football?",
    "How do I declare rental income on my tax return?",
    "What is the capital of Australia?",
    "How do I replace a timing belt?",
]


def ask(question: str, timeout: int = 180) -> tuple[int, dict]:
    req = urllib.request.Request(
        f"{BRIDGE}/ask",
        data=json.dumps({"question": question}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        try:
            return exc.code, json.loads(body)
        except json.JSONDecodeError:
            return exc.code, {"error": body[:200]}


def main() -> int:
    try:
        with urllib.request.urlopen(f"{BRIDGE}/health", timeout=20) as r:
            health = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"no bridge at {BRIDGE}: {type(exc).__name__}: {exc}")
        print("nothing measured, nothing reported.")
        return 1

    print(f"bridge {BRIDGE}")
    print(f"model  {health.get('model')}\n")

    rows = []
    started = time.time()
    for question, inside in [(q, True) for q in IN_CORPUS] + [(q, False) for q in OUT_OF_CORPUS]:
        t0 = time.time()
        status, payload = ask(question)
        elapsed = round(time.time() - t0, 1)

        if status != 200:
            row = {
                "question": question, "inCorpus": inside, "verdict": None,
                "error": payload.get("error", f"HTTP {status}"), "seconds": elapsed,
            }
            print(f"  [{'in ' if inside else 'out'}] ERROR {row['error'][:60]}")
        else:
            claims = payload.get("claims") or []
            by_status: dict[str, int] = {}
            for c in claims:
                by_status[c.get("status", "?")] = by_status.get(c.get("status", "?"), 0) + 1
            row = {
                "question": question,
                "inCorpus": inside,
                "verdict": payload.get("verdict"),
                "dataset": payload.get("dataset"),
                "claims": len(claims),
                "claimsByStatus": by_status,
                # what the user is actually shown, once unsupported claims are
                # withheld: an empty string means the pipeline published nothing
                "publishedChars": len(payload.get("verified") or ""),
                "sources": len(payload.get("sources") or []),
                "error": None,
                "seconds": elapsed,
            }
            print(
                f"  [{'in ' if inside else 'out'}] {str(row['verdict']):<12}"
                f" claims {row['claims']:>2}  published {row['publishedChars']:>5} chars"
                f"  {elapsed:>5}s  {question[:44]}"
            )
        rows.append(row)

    inside_rows = [r for r in rows if r["inCorpus"]]
    outside_rows = [r for r in rows if not r["inCorpus"]]

    def count(subset, pred) -> str:
        usable = [r for r in subset if r["error"] is None]
        return f"{sum(1 for r in usable if pred(r))}/{len(usable)}" if usable else "not measured"

    report = {
        "arm": "pipeline",
        "bridge": BRIDGE,
        "model": health.get("model"),
        "producedBy": "scripts/measure_pipeline.py",
        "outOfCorpus": {
            "refused": count(outside_rows, lambda r: r["verdict"] == "refused"),
            "publishedSomething": count(outside_rows, lambda r: r.get("publishedChars", 0) > 0),
        },
        "inCorpus": {
            "refused": count(inside_rows, lambda r: r["verdict"] == "refused"),
            "publishedSomething": count(inside_rows, lambda r: r.get("publishedChars", 0) > 0),
        },
        "errors": sum(1 for r in rows if r["error"]),
        "elapsedSeconds": round(time.time() - started, 1),
        "rows": rows,
    }

    print(f"\n  out of corpus, refused              {report['outOfCorpus']['refused']}")
    print(f"  out of corpus, published something  {report['outOfCorpus']['publishedSomething']}")
    print(f"  in corpus, refused                  {report['inCorpus']['refused']}")
    print(f"  in corpus, published something      {report['inCorpus']['publishedSomething']}")
    print(f"  errors                              {report['errors']}")
    print(f"\n  {report['elapsedSeconds']}s")

    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  written to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
