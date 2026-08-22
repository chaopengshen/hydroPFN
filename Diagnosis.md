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

- **16-day patch aggregates, not daily.** Not comparable to published daily-NSE
  PUB results. Never quote alongside them without saying so.
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
ends day 9000, evaluate day 9600, 608-day gap), 16-day patches.

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
