# Does context actually let us predict an ungauged site?

> **RESULT 2026-08-21, then CORRECTED the same day.** The first run found
> context worth exactly +0.0000 — but that test was MIS-SPECIFIED. It drew
> context from training regions only, which under leave-region-out forces
> context basins to be geographically distant. Measured: the nearest available
> context basin was a median **1.73°** away, while the nearest real gauge is
> **0.29°**. So it asked "can basins ~190 km away help?" and answered no —
> which is neither the interesting question nor the real PUB setting.
>
> Nearby gauged basins share STORMS with the query. With geographic retrieval
> from the full pool, the trivial baselines jump from `ctx_mean` 0.312 to
> **0.829** — the information was always there and the protocol excluded it.
> The corrected test is running; the flat result below stands only for
> *distant* context.

This is the load-bearing claim of the whole design. Everything else — the DEM
sampler, the merged site encoder, the StefaLand reuse — is scaffolding around
it. It has **not** been tested for time series.

## What has and has not been shown

| | tested? | what it showed |
|---|---|---|
| context helps for **point measurements** (unit D) | **yes** | +0.127 R² cross-variable, 9/9 across regions and seeds; own-site context raises velocity 0.575 → 0.906 |
| a single site's **streamflow** is predictable from forcings + attributes (unit A) | **yes** | R² 0.769 whole-site, leave-region-out, 124 unseen basins |
| **context basins** help predict an **ungauged basin's streamflow** | **NO** | unit A has no cross-site attention at all |

The third row is the claim. The first row is suggestive but is a different
modality, a different unit, and a much easier setting (the context is the query
site's *own* history, not other sites).

## Why it might fail — stated before running

1. **Unit A alone already reaches 0.769.** Context must beat a strong incumbent,
   not zero.
2. **The classic PUB baseline is strong.** Copying the most attribute-similar
   gauged basin's hydrograph is what operational PUB does, and it is hard to
   beat with 671 basins.
3. **We already measured this exact failure once.** In unit D, neighbour sites
   gave a *step*, not a curve: −0.145 → 0.582 on the first neighbour, then flat
   out to 16. The model calibrated off one neighbour and never aggregated. If
   that repeats here, "context works" is really "one donor works", which is
   nearest-neighbour regionalisation with extra steps.

## The ladder

Each rung is cheap and can kill the next.

### T-A · is the summary bottleneck binding?
No new architecture. Vary `k_summary` in unit A (1, 3, 8) and re-measure
whole-site R². `K · 256` numbers is everything a site can eventually say to the
connector, so if performance is already flat in K, the bottleneck is not the
limiting factor and we can stop worrying about it.

### T-B · the core test
Build the minimal connector. A task = **1 query basin + K context basins**.
The query has **every streamflow patch masked** (`whole_site`); the context
basins have theirs **visible**. Compare `K = 0` against `K > 0` on identical
query basins in held-out regions.

**Pass:** context improves whole-site R² over `K = 0`.
**Fail:** flat → the connector adds nothing over per-site prediction, and the
cross-site premise does not hold for this modality.

### T-C · the context-scaling curve
R² vs `K ∈ {0, 1, 2, 4, 8, 16, 32}`. A rising curve is the claim; a step at
K=1 that then flattens is the unit-D failure repeating and means "one donor",
not "in-context learning".

### T-D · baselines that must be beaten
| baseline | why it matters |
|---|---|
| **no context** (unit A alone) | the internal bar — 0.769 |
| **nearest-neighbour donor** | copy the most attribute-similar context basin's flow. THE standard PUB method. If we do not beat this, we have reinvented it expensively. |
| **context mean** | average the context basins' flow. Catches "the model just learned regional climatology." |
| regional LSTM | the U3 gate proper; deferred but required before any claim ships |

### T-E · retrieval ablation — does it matter WHICH basins?
Run T-C three ways: context basins chosen **at random**, by **attribute
similarity**, and by **geographic proximity**.

This is the sharpest test in the ladder. If random context works as well as
similar context, the model is not using site identity at all — it is just
seeing more data, and "retrieved context set" is a story we would be telling
ourselves. A gap between random and similar is what makes the retrieval step
real.

## Protocol

- Leave-region-out by USGS drainage-basin code, always. Context basins are
  drawn from **training regions only** — a context basin from the held-out
  region would leak.
- The query basin's own streamflow is never visible in any arm.
- Same query basins across every arm and every K, so no comparison is
  confounded by an easier population (the mistake that turned a real +0.024
  into an apparent +0.090 earlier in this project).
- Report per-mask, not just averaged.
- Multiple seeds before anything is quoted.

## What each outcome means

- **Rising curve, beats nearest-neighbour donor** → the central claim holds,
  and the connector earns the whole architecture.
- **Rising curve, loses to nearest-neighbour donor** → context is real but the
  model is a worse regionaliser than a simple donor; the design needs work
  before it is worth anything.
- **Step at K=1 then flat** → "one donor", not in-context learning. Fix the
  training-time context-size randomisation first, then re-test.
- **Flat from K=0** → the cross-site premise fails for time series. That is a
  publishable negative and should be recorded as loudly as a positive.


---

## RESULT (superseded) — DISTANT context adds nothing

40 epochs, leave-region-out (`01,11,17`), 124 held-out query basins, identical
query basins and evaluation window across every arm and every K.

| K | similar: model | ctx_mean | nn_donor | random: model |
|---|---|---|---|---|
| 0 | **0.5590** | — | — | **0.6234** |
| 1 | 0.4908 | 0.1517 | 0.1517 | 0.6217 |
| 2 | 0.5266 | 0.3119 | 0.1517 | 0.6227 |
| 4 | 0.5282 | 0.3930 | 0.1517 | 0.6219 |
| 8 | 0.5393 | 0.4653 | 0.1517 | 0.6216 |
| 16 | 0.5514 | 0.4655 | 0.1517 | 0.6217 |
| 32 | 0.5584 | 0.4333 | 0.1517 | 0.6215 |
| **gain** | **+0.0000** | | | **+0.0000** |

> **Read this section as: distant context adds nothing.** It does not settle
> the claim, because both arms here drew context from training regions only.

- **T-B FAIL for distant context.** Never beats K=0 in either arm.
- **T-C FAIL.** Not a rising curve, not even the "one donor" step. Flat.
- **T-E moot.** With no context effect to compare, similar-vs-random cannot
  discriminate. It did reveal something else though (below).
- **T-D partial pass, for the wrong reason.** The model beats `ctx_mean`
  (0.466) and `nn_donor` (0.152) — but through its per-site pathway, not
  through context. Beating a donor baseline without using donors is not the
  claim.

**Training with retrieved context DEGRADED the per-site path.** The random
arm's K=0 is 0.6234; the similar arm's is 0.5590. Same architecture, same data,
same seed — the only difference is which context basins were sampled during
training. Some of the model's capacity went into a pathway that turned out to
be worthless, and the useful pathway got worse for it.

### Why — the explanation that fits, and it was foreseeable

For a **fully** ungauged basin, ask what another basin's hydrograph could add.
The query's own forcings already carry its weather. Its attributes already say
what kind of basin it is. Training already taught how that kind of basin
responds to that weather. Context basins sit in **different regions with
different weather at the same instant**, so their streamflow at time *t* says
almost nothing about the query's streamflow at time *t*.

**Attribute-similar is not weather-correlated.** That is the crux, and it is
the structural difference from unit D, where the context was the query site's
OWN measurements — irreducibly site-specific information that no attribute
table contains. That is why unit D was positive and this is not.

### What this does and does not kill

Does not kill: **own-site context** (proposal use cases 1 and 2 — at-a-station
ratings and few-shot site updating). Unit D's +0.127 and its rising curve stand.

Kills, as currently framed: **PUB by neighbouring gauged basins** (use case 5).
For zero-gauge prediction, forcings + attributes + a trained model is the whole
story, and the connector is dead weight that also costs per-site accuracy.

### The flaw, found by asking "won't it need to be a NEIGHBOUR?"

Under leave-region-out, drawing context from training regions guarantees the
context is in a *different drainage region* from the query. Median distance to
the nearest context basin was 1.73° versus 0.29° to the nearest real gauge.
Attribute-similar basins on the far side of the continent see different weather
on the same day; a basin 32 km away sees the same storm.

The fix is `--retrieval geo --context-pool all`: geographic neighbours, drawn
from every basin except the query itself. **This is not leakage.** The model
still never trained on the held-out region; those neighbouring gauges exist in
reality and their records are available at inference; and the query's own
streamflow is never visible in any arm. It is exactly the operational PUB
setting — a region with some gauges and one ungauged catchment among them.

How strong the signal is, from the untrained smoke run at K=4:

| retrieval / pool | `nn_donor` | `ctx_mean` |
|---|---|---|
| attribute-similar, training pool (original) | 0.152 | 0.312 |
| **geographic, full pool (corrected)** | **0.785** | **0.829** |

`ctx_mean` at 0.829 is now a formidable baseline — averaging a few nearby
gauged hydrographs. The model has to beat THAT, not 0.31.

### The follow-up worth running

The interesting regime is not zero-gauge, it is **sparse-gauge**: give the
query basin a FEW of its own streamflow patches and hide the rest. That is
unit D's setting transposed to time series, and it is the one where context
carries information the attributes cannot. If that curve rises, the honest
claim becomes "in-context conditioning works on a site's own sparse
observations" — narrower than the original pitch, but true and still valuable.

Two secondary possibilities not yet excluded, in order of plausibility:
1. **Contemporaneous context is the wrong construction.** Neighbouring basins
   might inform via shared weather only if they are actually nearby. `geo`
   retrieval within the held-out region would leak, but geographic proximity
   among training basins for a training query is testable.
2. Architecture: one cross-attention layer and 3 summary tokens per site may be
   too thin. Weak, given that the effect is exactly zero rather than small.
