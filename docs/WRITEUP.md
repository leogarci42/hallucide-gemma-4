# Alien Hallucination

### A local Gemma 4 that is never allowed to judge its own work

**Track: Context Engineering for SLMs**

## The problem

Ask a small model a medical question and it answers. It answers when it knows,
and it answers when it does not, in the same confident register. In a domain
where a wrong dose or an inverted contraindication carries real cost, a fluent
answer with no traceable source is worse than no answer.

The usual fix is to retrieve passages and hope the model stays close to them.
Hope is not a mechanism. Our position is narrower: **the model may write, but
it may never rule on whether what it wrote is true.** That verdict comes from
the source, checked by code.

## The pipeline

```
 1  prompt                     the user asks something
 2  routing        Gemma       picks a domain from a closed list of Alien
                               datasets, or answers AUCUN
 3  search         Alien       semantic search inside that dataset only
 4  injection                  the retrieved passages fill Gemma's prompt
 5  generation     Gemma       drafts an answer from those passages
 6  decomposition  Gemma       re-reads its own answer, same context window,
                               and cuts it into claims, one sentence each
 7a semantic check             is the claim close enough to a source passage?
 7b literal check              do its figures and negations hold?
 8  aggregation                valid only if both pass
 9  output                     annotated claim by claim, each beside its passage
```

Gemma 4 does the work at four steps, and is excluded from the fifth. It routes,
it writes, it splits its own answer — and step 8 is arithmetic over two checks
it never sees. Swap Gemma for another model and the product still runs; swap out
the verification and there is no product.

## Three decisions that carry the design

**The routing list is closed, and the guard is code.** Gemma is shown five
dataset names and asked for exactly one, or `AUCUN`. Any answer that is not
character-for-character one of those keys is forced to `AUCUN` by a `dict`
lookup — not a fuzzy match, not a nearest neighbour. A hallucinated dataset name
cannot become a retrieval. The refusal path is the feature: a system that
declines when the corpus does not cover the question demonstrates the thesis
better than a lucky correct answer.

**Decomposition happens inside Gemma, in the same context window.** A separate
sentence splitter would produce claims in wording the model never used, and the
verbatim check would fail on punctuation rather than on substance. Asking Gemma
to re-read its own draft keeps the claims in the exact words it wrote, which is
what the checker needs.

**A claim is valid only if both checks pass.** Lane A scores semantic proximity
to the source passages. Lane B is deterministic: figures, quantities and
negations must line up. Failing either makes it a hallucination. When a lane
cannot settle it, the claim is `unverifiable` — stated as such, not rounded up
to fine. That third state is the honest one and the interface shows it with its
own marker.

## Context engineering, concretely

The interesting constraint was not the size of the window but what deserves to
be in it. Retrieval returns ten chunks and roughly nine thousand characters per
question; all of it goes in, with an instruction to answer from those passages
and nothing else. What we spend effort on is what happens *around* that window:
choosing the right corpus before spending a token, and re-reading the output
inside the same window so the checker and the writer agree on wording.

## What we measured

Numbers we did not measure are not in this writeup.

**The verifier, deterministically.** The engine ships a trap suite: cases with a
known expected verdict, run without any model, network or key.

```
traps caught          4/4
answerable published  6/6   (over-refusal check)
overall               10/10
```

Both halves are reported deliberately. A system that refuses everything scores
4/4 on the first line and 0/6 on the second; without the second, the first means
nothing. Reproduce with `make measure`.

**The routing, against a baseline.** Twenty questions, ten covered by the closed
list and ten plainly outside it (weather, notice periods, football). The routed
arm counts refusals; the baseline arm asks the same Gemma directly, with no
routing, and counts how many it answers anyway.

<!-- FILL FROM measurements.json ONCE THE BREV RUN COMPLETES; if it does not
     complete, delete this paragraph rather than estimating anything. -->

## What broke

**Three different causes, one symptom.** The pipeline diagram animates claims
travelling between steps, changing colour as the input changes hands. The
colours refused to change, three times over, for three unrelated reasons: a
global CSS transition on `fill` was smoothing and delaying every SMIL colour
change; SMIL silently disables an entire animation when a `values` list contains
an unresolvable `var(--token)`; and three successive regex edits had each
replaced only the first match, leaving stale duplicate elements painting the old
colours on top. Each looked like the same bug and none was.

**The demo has to survive a dead backend.** With no engine configured the
interface reports it as unreachable and every question fails with a retry
button. Nothing is faked to fill the gap: requests time out at 60s and retry
three times with jittered exponential backoff, only failures worth retrying are
retried, any question can be cancelled mid-flight, and a question left running
when the tab closes comes back marked cancelled rather than spinning forever.

**The measurement nearly did not happen.** The GPU box we were pointed at turned
out to be serving an embedding-only model — `/v1/embeddings` answered,
`/v1/chat/completions` was refused outright. Discovering that late is why the
routing figure landed last.

## Honest provenance

The verification engine in `src/` is **not new**. It began as Hallucide, an
open-source project the same team wrote for the Assemblée nationale hackathon
earlier in 2026: claim decomposition, verbatim checking, the risk floor, the
audit log. Its history is public in this repository.

Written during this hackathon: the closed-list routing and its code guard, the
Alien retrieval provider against the live MCP server, the Gemma 4 provider, the
two checks and their aggregation, the HTTP bridge, the whole interface, and the
measurement harnesses. We did not spend the day on plumbing, and that is why
there was time to measure anything at all.

## Running it

```bash
make measure   # the numbers above
make both      # engine on :8100, interface on :3000
```

The repository is MIT. Gemma 4 is Apache 2.0 and not redistributed here.
