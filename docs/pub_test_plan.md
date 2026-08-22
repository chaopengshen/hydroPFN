# Does context actually let us predict an ungauged site?

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
