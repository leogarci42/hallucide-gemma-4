# Alien Hallucination

A local **Gemma 4** answers medical questions from research corpora served by
**Alien Intelligence**, and every sentence it writes is checked against the
passage it came from. Anything that does not hold is withheld and labelled, so
an unverified claim is never presented as a fact.

Built at the Gemma 4 Hackathon, 42 Paris, 25 July 2026, for the **Context
Engineering for SLMs** track.

![The interface, with the pipeline running](docs/demo.gif)

*Same recording as [`docs/demo.mp4`](docs/demo.mp4) if you want it full size.*

## The pipeline

```
 1  prompt                        the user asks something
 2  routing        Gemma          picks a domain from a closed list of Alien
                                  datasets, or answers none. A code guard also
                                  rejects any answer outside the list. Either
                                  way the pipeline stops and the question is
                                  refused
 3  search         Alien          semantic search inside the chosen dataset,
                                  returning many candidate passages
 4  injection                     those passages go into Gemma's prompt, which
                                  is told to answer from them and nothing else
 5  generation     Gemma          drafts an answer from those passages
 6  decomposition  Gemma          re-reads its own answer in the same context
                                  window and cuts it into claims, one sentence
                                  each
 7a semantic check                is the claim close enough to a source
                                  passage? a score against a threshold
 7b literal check                 do its figures and negations hold against the
                                  source? deterministic, no model judgement
 8  aggregation                   valid only if both checks pass; otherwise
                                  hallucinated, or unverifiable when a check
                                  could not settle it. The verdict comes from
                                  the source, never from the model judging
                                  itself
 9  output                        the answer annotated claim by claim, each one
                                  beside the passage it was checked against,
                                  with how much context was injected
```

Gemma 4 does the work at four of those steps: it routes, it writes, it splits
its own answer, and it never gets to decide whether it was right. Alien
Intelligence supplies the corpora and the retrieval.

## Layout

```
front/     the interface: Next.js, TypeScript, no CSS or state framework
src/       the engine: routing, Alien retrieval, Gemma, verbatim checking
scripts/   command-line entry points into the engine
bridge.py  serves the engine over HTTP, in the contract the interface reads
tests/     the engine's suite, run with pytest
```

## Running the interface

```bash
make both           # engine on :8000, interface on :3000
make front          # interface only
make bridge         # engine only
make ask Q="..."    # the pipeline straight from the shell, no HTTP
```

`bridge.py` is the seam between the two. It imports the closed domain list, the
dataset ids and the retrieval wiring from `scripts/ask_medical.py`, so nothing
in `src/` is duplicated or modified: a change there carries over. It answers
503 rather than serving anything invented — no model backend, no Alien token,
and it says which.

The engine counts characters, not tokens, so the interface displays
`contextChars`. Converting one into the other would turn a measurement into an
estimate.

It runs on <http://localhost:3000> and works with no backend: it reports the
engine as unreachable and every question fails with a retry button. Nothing is
faked to fill the gap.

To connect the engine:

```bash
ENGINE_ORIGIN=http://localhost:8000 make front
```

`ENGINE_ORIGIN` is read server-side only. Two routes are proxied.

| Route | Method | Body | Response |
|---|---|---|---|
| `/health` | GET | — | `{ "model": "gemma-4-E4B-it" }` |
| `/ask` | POST | `{ "question": "..." }` | below |

```jsonc
{
  "draft":    "what Gemma wrote, untouched",
  "verified": "the same answer with the claims that failed removed",
  "verdict":  "grounded | partial | unsupported | refused",
  "claims": [
    {
      "id": "c1",
      "text": "one sentence",
      "status": "grounded",              // or "hallucinated" | "unverifiable"
      "semanticPass": true,              // optional, check 7a
      "literalPass": true,               // optional, check 7b
      "source": {
        "id": "...",
        "title": "shown to the user",
        "url": "https://...",            // optional
        "passage": "the matched text"    // optional
      }
    }
  ],
  "model":            "gemma-4-E4B-it",  // optional
  "dataset":          "medRxiv",         // optional, what routing chose
  "contextPassages":  24,                // optional
  "contextChars":     18400,             // optional
  "contextTokens":    3102,              // optional, if your engine counts them
  "latencyMs":        2410               // optional
}
```

`model`, `dataset`, `contextPassages`, `contextChars`, `contextTokens` and
`latencyMs` are
optional and are shown **only when the engine sends them**. The interface never
computes, estimates or defaults a number. A field that is missing is a field
that is not displayed.

Unknown fields are ignored, and a response that cannot be read at all is
reported as such rather than rendered half-way. `verdict` is derived from the
claims when absent, and a claim `status` from the two checks; if either check
is missing the claim is treated as unverifiable rather than assumed sound.

## Measuring it

```bash
make measure
```

Two numbers, neither of them an opinion.

**The verifier, on the engine's trap suite.** Deterministic: no model, no
network, no key. Every case has a known expected verdict, so the figure is a
count of agreements. Both halves are reported, because refusing everything
would look perfect on one and useless on the other:

```
traps caught          4/4
answerable published  6/6   (over-refusal check)
overall               10/10
```

**Two arms, against a baseline of the same model with no routing.** Twenty
questions, ten covered by the closed domain list and ten plainly outside it
(weather, notice periods, football). Same weights on both sides — the only
difference is what sits in front of the model. Run on `google/gemma-4-E4B-it`,
vLLM on an NVIDIA Brev box:

| | model alone | our pipeline |
|---|---|---|
| out of corpus, answered / published | **8/10** | **0/10** |
| out of corpus, refused | 10/10 (routing) | 10/10 |
| in corpus, answered / published | 10/10 | **4/10** |
| in corpus, refused | 0/10 | 0/10 |
| errors | — | 0/20 |
| wall clock | 53.3s | 91.5s |

Read the first row alone: the model answers eight of the ten questions it has no
source for, in the register it uses when it does have one. It declines only the
two that are explicitly about live data. Routing refuses all ten and over-refuses
none of the ten it covers.

The second bold figure is the honest half. Six of the ten covered questions were
retrieved, drafted, decomposed — and then published nothing, because the
decomposer turns medRxiv inline citation markers into claims of their own that no
passage can back. The design withholds rather than guesses, which is the point;
withholding six correct answers out of ten is still a defect, and it is reported
as one.

What this does **not** show: whether the four published answers are factually
correct (there is no ground truth here, only "backed by a retrieved passage"),
and whether the six withheld ones were wrong. Ten questions per cell — read
`10/10` as ten out of ten, not as a percentage.

Per-question tables and the raw runs are in [`metrics/`](metrics/README.md).
Regenerate everything, no figure typed by hand:

```bash
python scripts/measure_standalone.py http://localhost:8000/v1   # baseline arm
python scripts/measure_pipeline.py   http://localhost:8100      # pipeline arm
python scripts/compare_metrics.py                               # the tables
```

Both arms need a reachable backend: with none, they say so and report nothing
rather than printing a number nothing produced.

## What happens when things break

A demo has to survive a flaky laptop, so failure is a designed state.

- requests time out after 60s and are retried up to three times with jittered
  exponential backoff
- only worthwhile failures are retried: offline, unreachable, timeout, 5xx. A
  rejected question or an unreadable response is reported, not retried
- any question can be cancelled while it runs, and any failed or cancelled one
  asked again from where it sits in the conversation
- a question left running when the tab closes comes back marked cancelled
  rather than spinning forever
- a render fault shows a recoverable panel; conversations live in
  `localStorage` and survive a reload

## Credits and licences

- **Gemma 4** — Google DeepMind, Apache 2.0. Not redistributed here.
- **Alien Intelligence** — corpora and retrieval.
- **Icons** — [Lucide](https://lucide.dev), ISC, inlined in
  `front/components/icons.tsx`.
- **Dopis** — the typeface files in `front/public/fonts` are used under the
  licence they were supplied with.
- **This repository** — MIT, see `LICENSE`.

### Prior work

The verification engine in `src/` started as **Hallucide**, an open-source
project written by the same team for the Assemblée nationale hackathon earlier
in 2026. Its history is public in this repository.

Written during this hackathon: the routing over Alien datasets, the two checks
and their aggregation, the whole interface in `front/`, and the contract
between the two.
