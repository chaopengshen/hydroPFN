> **METRIC WARNING (2026-08-30).** Every CAMELS NSE/R² in this document is on
> `log1p(QObs)`, z-scored, on a 671-basin internal protocol — NOT raw mm/day.
> These numbers are 0.05–0.15 higher than the same models score on the
> citable metric, and may not sit beside published values. The verified
> protocol, raw-scale numbers, and the current suite live in
> [docs/camels531_protocol.md](docs/camels531_protocol.md). The metric ladder
> for one model: 16-day pooled R² 0.91–0.96 → daily pooled R² ~0.90 →
> log median NSE 0.872 → **raw median NSE 0.800**. Internal comparisons on a
> shared protocol remain valid.

# Diagnosis — what the PUB result rests on, and what it does not yet establish

Written 2026-08-22 at commit `9f64373` (model and training code unchanged since
`cf0af3e`). Purpose: record the experiment precisely enough to reproduce, and
state plainly the one control that is **missing**.

## The headline, restated exactly

> An ungauged basin in a held-out **region**, every streamflow observation
> hidden, predicted at **R² 0.853** by conditioning on ~4 nearby gauged basins
> at inference, with no retraining. **+0.075** over a well-trained no-context
> model of the same architecture, **+0.090** over a regional LSTM.

Every one of those numbers is a **spatial** holdout only. Read the next section
before quoting any of them.

---

## RESOLVED 2026-08-22 — the temporal test was run, and it holds

> Train 1980–2004 (windows end day 9000), evaluate April 2006 (day 9600),
> **608-day gap**, all three arms on the identical training period.
>
> | arm | R² |
> |---|---|
> | LSTM **(a) CEILING** — test gauges INCLUDED in training | **0.8606** |
> | **ours** — never sees the query gauge at all | **0.8447** |
> | LSTM **(b) PUB** — test gauges excluded | 0.7553 |
>
> **Not period memorisation.** The split costs us 0.008 at the peak (0.8528 →
> 0.8447) and costs the LSTM control 0.007 (0.7625 → 0.7553) — the split is not
> intrinsically harder, and both models pay the same small price.
>
> **No ceiling violation.** We land 0.016 BELOW the gauged ceiling, which is
> what physics says should happen.
>
> **The interpretable statement.** The gap between not having a gauge (0.7553)
> and having it (0.8606) is 0.1053. Context recovers 0.0894 of it — **85%**.
> For a basin never gauged, real-time neighbours buy 85% of what installing a
> gauge and calibrating on its record would buy.

The original description of the gap is kept below, because it is what the
earlier numbers rested on.

## THE MISSING CONTROL (as it stood before 2026-08-22) — no temporal split

**Training samples windows uniformly from the entire 1980–2014 record.
Evaluation sits at patches 200–232 = days 3200–3712 = 1988-10-05 to
1990-03-01 — inside the training period.**

```python
# train_pub.py, training loop
s = int(rng.integers(0, max(1, N - win)))      # ANY window in the record
# train_pub.py, evaluation
fixed_start = a.eval_start                      # patch 200, i.e. 1988-10-05
```

So the model has seen that stretch of weather through the training-region
basins, and at evaluation the context basins report **at the same timesteps**
being predicted. Two distinct concerns follow, and only the first is
intentional:

1. **Real-time conditioning (intended).** Nearby gauges reporting *now* inform
   the ungauged basin *now*. This is the operational PUB setting and is the
   claim being made. It is legitimate.
2. **Period memorisation (NOT controlled).** The model may have learned the
   specific weather sequence of 1988–90 from training-region basins and be
   recalling it rather than inferring from context. Nothing in the current
   design rules this out.

The result is therefore evidence for **spatial** generalisation with real-time
neighbours, and is **silent** on temporal generalisation. It should not be
described as "prediction in ungauged basins" without that qualifier.

### The ceiling question

A regional LSTM **with the test gauge included in training** is the natural
upper bound for how well that gauge can be predicted. Our 0.853 has never been
compared against it. Exceeding it would not be automatically wrong — real-time
neighbour observations are a different information set from historical
calibration at the gauge, and could beat it on event timing — but the burden of
proof sits with us and has not been discharged.

### The rigorous test, specified

Train on one period, test on another, all models on the same training period:

| arm | trained on | tests |
|---|---|---|
| **ours** | training-region basins, period 1 | PUB + real-time neighbours |
| **LSTM (a)** | ALL basins **including the test gauge**, period 1 | the gauged CEILING |
| **LSTM (b)** | training-region basins only, period 1 | standard PUB |

Evaluation for all three in period 2, on the held-out basins. Expected
ordering if the result is real: `(b) < ours`, and `ours` vs `(a)` is the
interesting comparison. If ours collapses toward `(b)` under a temporal split,
the current number was period memorisation.

Implemented as `--train-end` / `--eval-start-day` in `train_pub.py` and
`--include-test-gauges` in `experiments/lstm_baseline.py`.

---

## Exact setup as run

**Data** `CAMELS_Frederik.nc`, 671 basins × 12,784 days (1980-01-01 to
2014-12-31), no missing values.
Series: `prcp_daymet, srad_daymet, tmax_daymet, tmin_daymet, vp_daymet` +
`QObs` (log1p). 26 static attributes. `lat/lon` used for retrieval only, never
as features.

**Split** leave-region-out by USGS drainage-basin code (first two digits of the
gage id), 14 regions.
Primary holdout `01,11,17` → 124 query basins, 547 training.
Secondary holdout `02,07,10` → 170 query basins, 501 training.

**Task** 1 query basin + K context basins, all sharing one time window.
Query: every `QObs` patch masked. Context: `QObs` visible.
Context retrieval: `geo` (k-nearest by lat/lon), `--context-pool all` so a
held-out query may use other basins in its own region. Not leakage — the model
never trained on them, those gauges exist operationally, and the query's own
streamflow is never visible.

**Windows** 16-day patches, 32 patches per window = 512 days.
Training: random start. Evaluation: fixed start at patch 200.

**Model** `PUBModel` = `SiteEncoder` → `CrossSiteConnector` → query decoder,
7.1M params.
`SiteEncoder`: d=256, 4 layers, 4 heads, `d_ffd`=512, `k_summary`=3.
Connector: 4 layers, 8 heads. Plus `--time-aligned` cross-attention (each query
patch attends to context patches at the same time index) — **this is the single
change that took the model from 0.556 to 0.853**.
Training: 40 epochs × 150 steps, batch 4 tasks, AdamW, OneCycle max_lr 3e-4,
K sampled from `{0,1,2,4,8,16}`.

**LSTM baseline** `experiments/lstm_baseline.py`, 0.30M params, hidden 256,
1 layer, dropout 0.4, 60 epochs × 200 steps, 120-day warmup excluded from both
loss and scoring, predictions aggregated to the same 16-day patches.

**Reproduce**
```bash
source gpuenv.sh && export PYTHONPATH=$PWD/src
python -m hydropfn.train.train_pub --nc data/CAMELS_Frederik.nc \
    --retrieval geo --context-pool all --time-aligned \
    --epochs 40 --steps 150 --seed 0 --holdout 01,11,17 --tag geo_all_ta
python experiments/lstm_baseline.py --nc data/CAMELS_Frederik.nc \
    --holdout 01,11,17 --epochs 60 --steps 200 --seed 0 --tag h1
python -m hydropfn.train.train_site_encoder --source camels \
    --nc data/CAMELS_Frederik.nc --holdout 01,11,17 --epochs 60 --steps 300 \
    --seed 0 --eval-start 200 --tag fixedwin
```

## Results as measured

| model | seeds | median | spread |
|---|---|---|---|
| PUB, K=4 with context | 0.8528 / 0.8528 / 0.8612 | **0.8528** | 0.008 |
| unit A standalone | 0.7879 / 0.7722 / 0.7782 | 0.7782 | 0.016 |
| regional LSTM | 0.7498 / 0.7625 / 0.7633 | 0.7625 | 0.014 |
| PUB, K=0 | 0.4714 (K sampled 1/6) · 0.5259 (3/8) | — | — |
| donor-averaging `ctx_mean`, K=4 | 0.8293 | — | — |
| nearest-neighbour donor | 0.7850 | — | — |

Second region set (`02,07,10`, single seed): ours 0.7278, LSTM 0.7074,
`ctx_mean` 0.6721, K=0 0.3601.

## Everything else that qualifies these numbers

- ~~16-day patch aggregates, not daily.~~ **WRONG — corrected 2026-08-22.**
  `train_pub.py` always scored DAILY values; `lstm_baseline.py` scored 16-day
  means. See "The scoring-resolution mismatch" below. Our numbers ARE daily
  and are comparable to published daily results on that axis.
- **The two-path design costs −0.25** in the no-context mode (0.778 → 0.526).
  Re-weighting K=0 from 1/6 to 3/8 of training recovers +0.055, essentially
  free, but nowhere near all. Deployment implication: ship two models.
- **Unit A vs the LSTM is a TIE** (+0.016 against spreads of 0.016 and 0.014),
  not a win — retracted after multi-seeding. So essentially all advantage over
  the LSTM comes from conditioning, not from a better hydrological model.
- **The attribute ablation is much more seed-sensitive than the headline**
  (`attr_gain` 0.183 vs ~0.114 across seeds). Do not quote it to one decimal
  from a single run.
- **Window 200 is slightly easier than average** for unit A (0.7879 fixed vs
  0.7631 random-window). All comparisons are at window 200 so none is affected,
  but absolute values are mildly optimistic.

## Four claims published and then corrected, for the record

| claim | what broke it |
|---|---|
| StefaLand's input paths transfer | reading the 485 tensors rather than the config |
| context is worth nothing | the protocol forced context ~190 km away; nearest real gauge is 32 km |
| context is worth +0.38 | the K=0 baseline was damaged by the very thing being measured |
| unit A beats the LSTM | three seeds instead of one |

Each was too favourable, and each fell to inspecting the underlying artefact
rather than reasoning about it. A pre-registered plan guards against moving the
goalposts, not against aiming at the wrong one — every one of these was a setup
error, invisible from inside the plan. **The missing temporal control at the
top of this document is the same class of error, still open.**


---

# CONSOLIDATED — after four fairness challenges

All numbers below: leave-region-out (`01,11,17`), **temporal split** (train
ends day 9000, evaluate day 9600, 608-day gap), scored on **DAILY** values.
(This section originally said "16-day patches" — wrong for our model, and the
source of the mismatch corrected on 2026-08-22.)

## The comparison that matters, on IDENTICAL 62 basins

| | neighbour information | R² |
|---|---|---|
| LSTM (b) | none | 0.7074 |
| LSTM (c) | **trained on the neighbours** | 0.7799 |
| nearest-neighbour donor | copy nearest neighbour's flow | 0.8010 |
| **`ctx_mean`** | **average the neighbours' flow** | **0.8311** |
| **ours** | **read them at inference** | **0.8546** |

**Reading neighbours at inference beats training on them by +0.075.** But
`ctx_mean` at 0.8311 is the strongest comparator, and our margin over IT is
**+0.024** — that is the number to lead with, not the +0.147 over LSTM(b).

## The mechanism — two context modes

| mode | context window | K=0 | peak | gain |
|---|---|---|---|---|
| **A — assimilation** | the query's CURRENT window | 0.4684 | **0.8584** | **+0.390** |
| **B — regionalisation** | an earlier TRAINING-period window | 0.5227 | 0.5265 | +0.004 |

A neighbour's *current flow* is worth +0.39. A neighbour's *history* is worth
~nothing extra. The effect is real-time state, not learned similarity — which
is why this is data assimilation, and why the time-aligned attention (which
matches patch n of the query to patch n of the context) is the component that
matters.

*(Mode B is being re-run with the training-context leak closed; the figure
above still carries it. Under the leak the neighbours' histories were partly
absorbed into the weights already, so the null was about REDUNDANCY, not about
the value of historical data — a distinction CS identified.)*

## Corrections forced by the four challenges

| challenge | verdict |
|---|---|
| no temporal split | **real** — now controlled; costs 0.008, result survives |
| exceeding the gauged ceiling | **not happening** — we sit 0.016 under it (0.8447 vs 0.8606) |
| LSTM undertrained | **no** — 5x budget makes it WORSE (0.7553 → 0.7467; it overfits) |
| held-out basins in training context | **real** — 70% of eval basins were exposed; now fixed, and fixing it slightly IMPROVED our result (0.8447 → 0.8584) |

### RETRACTED: "the two-path design costs −0.25"

That was a **training-budget artefact**, not an architectural cost. At 5x
budget K=0 goes 0.4539 → **0.6066** (+0.153) and the large-K decay disappears
(K=32: 0.8053 → 0.8432). Our model was undertrained — 24,000 query-windows
against the LSTM's 384,000 — and the no-context mode absorbed the shortfall.

| | K=0 | peak | K=32 |
|---|---|---|---|
| 1x budget | 0.4539 | 0.8447 | 0.8053 |
| **5x budget** | **0.6066** | **0.8642** | **0.8432** |

## The pattern worth naming

Five measurements in this session compared against something weaker than the
real alternative: the damaged K=0 baseline, the undertrained-looking LSTM, "no
neighbour info" when the fair comparator was training on them, a protocol that
forced context 190 km away, and a config table read instead of the tensors.
Each was individually defensible; together they biased consistently in one
direction. **Every one was caught by someone asking whether the setup matched
the real situation — none by the pre-registered plan.**


---

# 2026-08-22 — three verification findings, and what they cost

Prompted by three questions about the architecture. All three checks came back
badly. Recorded here because two of them invalidated numbers already reported.

## 1. The scoring-resolution mismatch (invalidated every head-to-head)

`train_pub.py` scored **512 daily values**; `lstm_baseline.py` scored **32
sixteen-day means**, under a comment reading *"matching the PUB evaluation
exactly"*. It did not. I wrote that comment and then repeatedly cited it as
grounds that the comparison was like-for-like.

Aggregation is worth **+0.023 R²**, measured on both LSTM arms:

| arm | R² daily | R² 16-day means |
|---|---|---|
| PUB LSTM (b) | 0.7383 | 0.7619 |
| gauged-ceiling LSTM | 0.8442 | 0.8670 |

Every LSTM number previously quoted was the right-hand column; every number of
ours was the left. Corrected, on daily throughout:

| | R² daily |
|---|---|
| PUB LSTM (b) — no test gauge, no neighbours | 0.7383 |
| gauged ceiling — test gauge IN training | 0.8442 |
| ours, leak closed | **0.8584** |
| ours, 5× budget | **0.8642** |

Two consequences. Our margin over the PUB LSTM is **+0.120, not +0.103**. And
**we exceed the "gauged ceiling"** by 0.014 (0.020 at 5× budget) — previously
reported as sitting just below it. The user raised exactly this suspicion when
the ceiling was first introduced; the scoring bug is what buried it.

**The ceiling is not a ceiling.** It is a different information set. Ours has
two things it lacks: concurrent discharge at neighbouring gauges, and
bidirectional attention. Never state "beats the gauged ceiling" bare — it
invites the reading that we beat having the gauge, which is not what happened.

Both scripts now report both resolutions, always.

## 2. No geospatial encoding in the connector

`CrossSiteConnector.forward` added **only a role embedding**. The architecture
figure has been claiming "add geo-encoding" throughout. The model chose context
by geographic proximity and then treated it as exchangeable — it could not tell
a gauge 20 km away from one 300 km away. That is the obvious suspect for the
decay at large K: with no distance, the only way to discount a far gauge is to
discount all of them.

Now implemented as opt-in `geo=True`: Fourier encoding of **displacement from
the query**. The first version scaled longitude by `cos(query_latitude)` — the
better distance metric, but it makes the encoding a function of ABSOLUTE
latitude, i.e. a region-identifying channel that would quietly undermine
leave-region-out. Replaced with a fixed reference latitude: ~10% distance
distortion across CONUS, exact translation invariance. Two tests pin both
halves (`test_connector_geo_translation_invariant` at atol 1e-3 — float32
rounding amplified by the 64 rad/deg top frequency, measured, not assumed;
`test_connector_geo_responds_to_distance`).

**Untrained. Whether it fixes the large-K decay is a hypothesis, not a result.**

## 3. The model is not causal

`nn.TransformerEncoder` with only a padding mask — fully bidirectional.
Predicting day *t* it sees the whole 512-day window, including days after *t*,
and the context sites' full windows. **The LSTM baseline is causal.** Ours is a
smoother; the LSTM is a filter. An asymmetry favouring us that was never stated
in any previous comparison.

This is not automatically wrong. For **PUB as historical reconstruction** it is
the correct tool — a hydrologist reconstructing 1990 has 1991 in hand. For the
**mode A data-assimilation claim** it is not acceptable. The fix is not to make
the model causal; it is to stop letting one number serve both claims.
A causal-mask ablation would partition the +0.120 margin. **Not yet run.**

## 4. Mode B baselines read the future (found while collecting mode B)

`nn_donor`/`ctx_mean` computed from `Xp[sites[1], sl]` — the **eval** slice,
always — regardless of `--context-period`. In mode B the model's context comes
from the training period but the baselines still read the neighbours'
*concurrent* discharge: the exact information mode B withholds from the model.
The tell was that the baseline columns were byte-identical between the mode A
and mode B result files.

Fixed: baselines now read the context window. The mode B table reverses.

| K | model | nn_donor | ctx_mean |
|---|---|---|---|
| 0 | 0.6359 | — | — |
| 4 | 0.6371 | −0.793 | −0.663 |
| 32 | 0.6392 | −0.793 | −0.510 |

Negative baselines are the *correct* answer: matching flows across different
years is worse than predicting the mean. Previously this table showed
`ctx_mean 0.8215` against our `0.5086`, i.e. a trivial baseline crushing us.
Entirely the bug.

Mode B must now be stated as **two separate facts**:

- **Context adds nothing in mode B**: K=0 → 0.6359, K=32 → 0.6392, gain
  **+0.003**. Survives every fix (leaked +0.004, leak-closed +0.006).
- **The model is far above any legitimate mode B baseline** (0.636 vs −0.79 to
  −0.51) — but that value is entirely the per-site pathway (forcings and
  statics), not the neighbours.

### Budget check — RESOLVED, mode B is genuinely dead

The retracted "two-path costs −0.25" was a pure budget artifact, so mode B was
re-run at 5× compute before being declared dead. It is not an artifact:

| | K=0 | K=32 | gain |
|---|---|---|---|
| 40 epochs | 0.6359 | 0.6392 | +0.003 |
| 200 epochs | 0.7492 | 0.7543 | +0.005 |

The absolute level WAS badly undertrained (+0.11 on K=0). **The context gain
was not** — it stays ~zero at 5× budget. Contrast mode A at **+0.390**: a ~78×
difference, robust to the leak fix, the baseline fix, and the budget.

Note where mode B K=0 lands: **0.7492 daily, against the PUB LSTM's 0.7383
daily** — a tie, and the same tie already established for unit A standalone.
This locates the whole result: **without concurrent neighbours we are a
regional LSTM.** Everything above that comes from reading neighbouring gauges
in the eval window.

Consequence for the "local stations not in training" case: a station absent
from training contributes ~nothing through its history. If it helps at all, it
helps through concurrent data — which makes it a mode A use case, not mode B.

## What this run of errors has in common

The four corrections at the top of this document were all **setup** errors. So
are these. Every one was invisible from inside the plan and visible instantly
on reading the artefact — the comment that claimed matching resolutions, the
`forward` that had no geo term, the byte-identical baseline columns.

The new entry in the pattern: **three of these four were errors in my own
instrumentation, not in the model** — and two of them ran *against* us, which
is why they survived so long. A bias that flatters gets challenged; a bias that
penalises just looks like an honest result.

---

# 2026-08-26 — the DEM arm: seven negatives, two positives, and one mechanism

The question: does terrain add anything the time-series arm does not already
have? Tested BEFORE building a connector, because building first conflates
"does DEM carry signal" with "did we wire it correctly" — the confound that
made the geo-encoding look useless when it had merely been attached to a dead
path.

## What was measured

Source: USGS 3DEP 1/3 arc-second (~10 m), streamed via `/vsicurl`. 655/671
CAMELS gauges, 4,402 HYDRoSWOT sites, 6,000 pretraining locations, 5,368
within-basin points.

| target | curated table exists? | DEM effect |
|---|---|---|
| CAMELS flow signatures, 1.28 / 12.8 / 51.2 km | yes (26 statics) | −0.046 … +0.025 |
| channel width (4,402 sites) | yes (NHDPlus) | +0.006 |
| channel depth | yes | −0.002 |
| streamflow in-model, K=0, 1.28 km | yes | **−0.009** |
| streamflow in-model, K=0, 12.8 km + noise-averaged | yes | **+0.016** |
| streamflow in-model, K=4/8 | yes | ~0 |
| **streamflow, statics WITHHELD 50%** | — | **−0.075** |
| **lithology (LITH1)** | **NO** | **0.593 → 0.644 acc** |
| **bedrock age (log10 Ma)** | **NO** | **0.367 → 0.438 R²** |

## The organising principle

**DEM helps only where no curated attribute table exists.** Every negative is
on a target where somebody already hand-built the terrain summary —
`elev_mean`, `slope_mean`, `MEANELEVSMO`, `log_slope`. We were repeatedly
asking a raw DEM to beat a curated summary *of itself*. Geology has no such
table, and that is the one place terrain wins.

## The mechanism, and why the substitution test FAILED

A prediction was made in advance and falsified. With curated attributes
withheld on 50% of tasks — the honest global case, where a station has a DEM
and little else — DEM was predicted to gain **+0.05 or more**. It **lost
0.075**, the largest DEM effect measured and the wrong sign.

The training loss went the other way (1.299 with DEM vs 1.539 without), so the
features fit the training basins *better* and generalised worse. Textbook
overfitting, with a specific suspected cause:

**Terrain features act as a LOCATION FINGERPRINT.** Terrain is strongly
spatially autocorrelated, so 768 DEM numbers largely encode *where you are*.
Within a region that is useful; under leave-**region**-out it is exactly the
wrong thing to memorise, because held-out regions carry terrain signatures
never seen in training.

This fits every result: DEM never helped streamflow, hurt *more* when the model
was forced to rely on it, and helped only on **geology** — where the label is
itself a function of location, so fingerprinting is not penalised.

It is also the failure mode the geo-encoding was explicitly designed against:
displacement was made translation-invariant so the connector could not identify
regions. **The DEM features have no such protection.**

**Not yet tested, and it decides the question:** do DEM features predict HUC2
region? If yes, the fingerprint is confirmed, the streamflow negatives are
explained, and the geology positive becomes suspect for the same reason.

## Side finding, independent of DEM and worth keeping

**Attribute dropout improves the time-series arm.** Withholding the curated
statics on 50% of tasks: K=0 goes **0.7142 → 0.7463 (+0.032)** under
leave-region-out. Always-available statics appear to cause overfitting; forcing
the model to sometimes work without them generalises better. Free improvement,
no DEM involved.

## Two corrections to earlier DEM claims

1. **"DEM actively hurts" (1.28 km, −0.009) was partly my extraction, not
   terrain.** Two independent flaws each cost ~0.012: a footprint far too small
   (1.28 km against a median 18 km basin) and a **stochastic feature
   extractor** — `q_sample` injects a random field, so a single draw makes each
   site's features a *sample* rather than an expectation. Ridge across
   thousands of sites averages that away, which is why the probes looked
   healthy while the in-model result did not. Averaging 8 draws and widening to
   12.8 km turned −0.009 into +0.016.

2. **"The U-Net has no embedding designed to be extracted" was true but the
   wrong conclusion.** Its bottleneck is a 128×16×16 *spatial* map. That is a
   liability for producing site tokens and exactly right for producing 2-D
   fields. The sampler was built to generate terrain, not describe it — and a
   future groundwater-level arm would want the field, not the vector. Its
   conditional-inpainting objective already matches "sparse wells → continuous
   water table".

## Multi-scale pretraining

17,500 patches at three footprints (1.28 km @10 m, 12.8 km @100 m, 51.2 km
@400 m), 2.03M-param U-Net, scale-conditioned on **continuous**
`[log10(m/px), log10(footprint km)]` so an unseen resolution *interpolates*
rather than falling off a lookup table — necessary because 3DEP 10 m is
CONUS-only and the global tier is 30 m (GLO-30) or 90 m (MERIT).

Improves the geology features across all three metrics (age R² 0.420 → 0.438).
Notably the probe evaluates a **single** scale, so the gain comes from having
been *trained* across three, not from being given three at inference.

## Open

- **the fingerprint check** — do DEM features predict HUC2? Decides everything
  above.
- SGMC circularity: geologists map unit contacts partly *from* topography.
  Visible in per-class recall — Sedimentary 0.72, Unconsolidated 0.58,
  Igneous **0.087**.
- SGMC is US-only; the global label is GLiM, much coarser.
- 30 m degradation: downsample CONUS and re-probe. No new data needed.
- within-basin patch bag (5,368 points, 8 per basin, with relative position
  and basin area) — fetched, not yet tested.

---

# 2026-08-28 — the anchor was the wrong checkpoint, and "the sixth element" is retracted

Chasing the garbage anchor row (psd 122 in my harness) forced a checkpoint
audit, and the metadata plus an own-harness re-run reframe the target:

| checkpoint | param | own-harness psd (rerank) | identity |
|---|---|---|---|
| `allfix.pt` | **v** | **0.592** | the STACKED "all improvements" run |
| `residual.pt` | **eps** | *re-running* | residual-only — the probable 0.810 champion |

The dev-log's own words: the winning recipe was "residual + best-of-K", and
the stacked run carried **two actively harmful arms** on top of it. `allfix`
("all fixes") is that stacked run — its own harness scores it 0.592, not
0.810.

**Consequences, in order of severity:**

1. **"v-prediction was the sixth lost element" is RETRACTED.** v was in the
   stacked run, i.e. plausibly one of the HARMFUL arms — and my own parity
   ablation independently agrees (parity-v: marginally better at fine scale,
   worse at 12.8 km, unstable val). The commit that reintroduced v as part of
   "the winning recipe" came from the same summary-not-source reading that
   lost the recipe in the first place. eps-parity is the configuration to
   carry forward.

2. **The anchor row's garbage in MY harness (psd 122) is a real harness bug,
   separate from checkpoint identity** — allfix decodes fine in its own
   harness. Undiagnosed: schedules are byte-identical, param handling is
   proven by parity-v, the architecture load is strict. Until my harness
   reproduces an external checkpoint, cross-harness rows stay out of every
   table.

3. **Where the comparison actually stands** (fine-scale psd, cross-harness so
   approximate): no-recipe multi-scale 0.45 → parity-eps best-of-8 **0.64**;
   allfix single-scale stacked 0.59; recorded champion **0.810** (residual.pt,
   repro pending). We have caught the stacked run and remain **well short of
   the champion**. Not recovered.

## Addendum, same day: the champion re-measured, and where we actually stand

`residual.pt` own-harness, today's eval draw: **psd 0.633, vario10 0.828,
vario80 0.927** (rerank). The recorded 0.810 does NOT reproduce — and the
harmonic baseline moved with it (elev 2.95 today vs 2.03 recorded), so the
drift is in the EVALUATION PROTOCOL (patch subset / mask draw), not the
checkpoint. Even the historical anchor is protocol-bound. Every anchor
number must carry its eval draw.

**The honest recovery table** (fine scale; cross-harness but both on
original-style holes):

| | psd | vario10 | vario80 |
|---|---|---|---|
| champion `residual.pt`, own harness, today | 0.633 | 0.828 | 0.927 |
| **ours: multi-scale parity-eps, best-of-8** | **0.636** | 0.741 | 0.899 |
| stacked `allfix.pt`, own harness | 0.592 | 0.749 | 0.917 |
| ours, no recipe (ms1) | 0.451 | 0.740 | 0.818 |

Against the champion as it measures TODAY: **psd matched (0.636 vs 0.633),
vario80 close (−0.03), vario10 short by −0.09** — and ours additionally
covers 12.8 km and 51.2 km (psd ~0.51-0.55) from one scale-conditioned net,
which the single-scale champion cannot do at all. Against the recorded-but-
unreproducible 0.810, still short; that number should no longer be quoted
without its (lost) protocol.

## Correction, same day: the record did NOT drift — my flag did

Re-run at the recorded protocol (`--n-eval 60`, defaults): `residual.pt`
scores **psd 0.824, vario10 0.838, vario80 0.964**, and harmonic's elev is
**2.028 — exactly the recorded value**. Corpus, metrics and eval code all
predate the checkpoints on disk; nothing changed. The "protocol drift" in the
previous addendum was **my own `--n-eval 40`**, worth 0.19 of psd via a
smaller, rougher-median eval subset. That addendum's claim is retracted: the
0.810 record is sound and reproducible.

**Corrected standing:** the champion reproduces at **0.824**; our multi-scale
parity-eps measures **0.636** — but on MY eval set (32 val patches, ported
masks), not the champion's 60 held-out tiles. The same subset effect that
just fooled me (residual.pt: 0.633 at n=40 vs 0.824 at n=60) means these two
numbers are NOT directly comparable, in either direction. The recovery
question is therefore OPEN, not answered: settling it requires one shared
protocol — either fix the external-checkpoint decode bug in my harness and
score both models on identical patches and masks, or evaluate the parity
checkpoint inside the original harness.

Lesson count for this anchor hunt alone: wrong checkpoint (allfix), wrong
n-eval (mine), one real harness bug (psd 122, still open), and one false
accusation against a sound record. Every one was a protocol error, and every
one was caught by re-measuring rather than reasoning.

## Resolution: shared protocol run — recovered, and exceeded

`eval_external.py` (a surgical one-block patch of the original harness — the
ckpt loader infers width/in_ch from weights and freezes the scale conditioning
at the corpus's true 10 m / 1.28 km; the eval protocol is byte-identical, same
seed, so the draw is the SAME 60 tiles and masks as the champion's 0.824 run):

| model (bo8 rerank) | elev | psd | vario10 | vario80 |
|---|---|---|---|---|
| champion `residual.pt` (single-scale) | 1.154 | 0.824 | 0.838 | 0.964 |
| **parity-eps (multi-scale)** | **1.130** | **0.937** | **0.842** | 0.959 |
| parity-v (multi-scale) | 1.011 | 0.806 | 0.814 | 0.961 |
| harmonic | 2.028 | 0.175 | 0.376 | 0.529 |

**The multi-scale model beats the single-scale champion at the fine scale on
the champion's own protocol** — psd 0.937 vs 0.824, variograms tied — while
additionally serving 12.8 km and 51.2 km from the same scale-conditioned
weights. Multi-scale training cost nothing at the fine scale and plausibly
helped (cross-scale exposure as augmentation). parity-v confirms once more
that v-prediction is not needed.

Every earlier "still lacking" reading was measurement, not model: my hole
distribution (worth ~0.2), my n-eval subset (worth ~0.2), and my mask port.
The same checkpoint that scores 0.636 in my harness scores 0.937 in the
original's. **The recipe restoration is complete; the open item is my
harness's external-checkpoint decode bug, which no longer gates anything.**
