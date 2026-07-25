# Two arms, same questions, same weights

Both arms ask `google/gemma-4-E4B-it` the same twenty questions. The only difference
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
| out of corpus, answered / published | 8/10 | 0/10 |
| out of corpus, refused | 10/10 (with routing) | 10/10 |
| in corpus, answered / published | 10/10 | 4/10 |
| in corpus, refused | 0/10 (with routing) | 0/10 |
| errors | — | 0 |
| wall clock | 53.3s | 91.5s |

Read the first row on its own: the model alone answers questions it has no
source for, in the same confident register it uses when it does. The pipeline
publishes only what a retrieved passage backs.

The pipeline is slower by construction — it retrieves, re-reads its own draft
and checks every claim. The cost is the point.

## Where it over-withholds

6 of the ten covered questions ended `unsupported`: the pipeline
retrieved, drafted and decomposed, then withheld every claim. That is the
failure mode this design chooses — it publishes nothing rather than something
unbacked — but it is a failure, not a win, and it is reported here as one.

The known cause is decomposition: medRxiv passages carry inline citation
markers, and the splitter turns some of them into their own "claims", which no
passage can back. Those fragments drag the whole answer to `unsupported`.
Cleaning the markers before decomposition is the fix; it is not in this run.

## Per question

### Covered by the corpus

| question | model alone | our pipeline | published |
|---|---|---|---|
| How effective is immunotherapy in non-small cell lung cancer? | answered | partial | 425 chars |
| Does early screening reduce colorectal cancer mortality? | answered | partial | 769 chars |
| What effect do statins have on the risk of myocardial infarction? | answered | unsupported | 0 chars |
| Does untreated hypertension increase the risk of stroke? | answered | partial | 407 chars |
| Which treatments slow the progression of Alzheimer's disease? | answered | partial | 2013 chars |
| Does deep brain stimulation help in Parkinson's disease? | answered | unsupported | 0 chars |
| How effective are antivirals against seasonal influenza? | answered | unsupported | 0 chars |
| Does vaccination reduce tuberculosis transmission? | answered | unsupported | 0 chars |
| Does metformin reduce complications in type 2 diabetes? | answered | unsupported | 0 chars |
| What are the effects of thyroxine on hypothyroidism? | answered | unsupported | 0 chars |

### Plainly outside the corpus

| question | model alone | our pipeline | published |
|---|---|---|---|
| What will the weather be in Paris tomorrow? | declined | refused | 0 chars |
| What is the statutory notice period for resigning in France? | answered | refused | 0 chars |
| How do you make choux pastry? | answered | refused | 0 chars |
| Who won the Champions League in 2024? | answered | refused | 0 chars |
| How do I configure nginx as a reverse proxy? | answered | refused | 0 chars |
| What is the average price per square metre in Lyon? | declined | refused | 0 chars |
| What are the offside rules in football? | answered | refused | 0 chars |
| How do I declare rental income on my tax return? | answered | refused | 0 chars |
| What is the capital of Australia? | answered | refused | 0 chars |
| How do I replace a timing belt? | answered | refused | 0 chars |


## Reproducing

```bash
python scripts/measure_standalone.py http://localhost:8000/v1   # baseline arm
python scripts/measure_pipeline.py   http://localhost:8100      # pipeline arm
python scripts/compare_metrics.py                               # this file
```

`baseline.json` and `pipeline.json` hold the per-question detail, including the
claim counts and each verdict.
