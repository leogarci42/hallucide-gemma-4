#!/usr/bin/env python3
"""Runs the engine's trap suite and prints the result.

Deterministic: no model, no network, no key. Every case has a known expected
verdict, so the number is a count of agreements, not a judgement.

    python -m scripts.measure_verifier
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hallucide.analysis.measurement import run_measurement  # noqa: E402
from hallucide.analysis.trap_dataset import TRAP_DATASET  # noqa: E402


def main() -> int:
    report = run_measurement(TRAP_DATASET)
    results = report.results

    answerable = [r for r in results if r.case.is_answerable]
    traps = [r for r in results if not r.case.is_answerable]

    for r in results:
        kind = "answerable" if r.case.is_answerable else "trap      "
        print(f"  {'ok ' if r.correct else 'MISS'} [{kind}] {r.case.trap_type:<15} {r.case.id}")
        if not r.correct:
            expected = r.case.expected_status.value if r.case.expected_status else "REFUS_VÉRIFICATION"
            print(f"       expected {expected}, got {r.actual_status}")

    print(f"\n  traps caught          {sum(r.correct for r in traps)}/{len(traps)}")
    print(f"  answerable published  {sum(r.correct for r in answerable)}/{len(answerable)}  (over-refusal check)")
    print(f"  overall               {sum(r.correct for r in results)}/{len(results)}")
    return 0 if all(r.correct for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
