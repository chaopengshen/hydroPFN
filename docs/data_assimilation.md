# How hydroPFN assimilates observations

A handover note on the time-series arm: how it uses neighbouring gauges, how
many of them it can take and from how far away, and how a station's own recent
observations get in. Written 2026-09-17.

The numbers here are copied from
[camels531_protocol.md](camels531_protocol.md), which is the source of truth.
If the two ever disagree, that page wins. All NSE values are median per-basin
NSE on raw mm/day, on the CAMELS-531 protocol, from a single seed unless noted.

---

## The short version

One set of weights serves every mode. What the model assimilates is decided at
inference time by which observations are put in the task, not by retraining,
a Kalman filter, or an adjoint. "Assimilation" here means conditioning: the
model is trained on tasks with a hole in them and learns to fill the hole from
whatever is visible.

| what goes in the task | result | best alternative |
|---|---|---|
| nothing, ungauged basin (PUB) | 0.682 (0.707 mode-B-trained) | LSTM+dHBV1.1p 0.700 |
| ~4 nearby gauges, ungauged basin (PUB) | **0.751** | IDW 0.645, StefaLand 0.721 |
| own gauge, 1–16 days stale, gauged basin (temporal) | 0.707 | LSTM 0.687, dHBV1.1p 0.743 |
| own gauge + nearby gauges, gauged basin (temporal) | **0.804** | dHBV1.1p 0.743 |

PUB = the basin was held out of training. Temporal = every basin was in
training for 1980–95 and is scored on 1995–2010. **The own-gauge stream has
only been tested on the temporal split**; see open question 6.

Neither LSTM nor dHBV can take these observation streams at inference. Where
any observation stream exists, the model wins without finetuning. The one
regime it loses is a gauged basin simulated with no observations at all (0.640
vs LSTM 0.687). A same-architecture specialist also scores 0.639 there, so
that gap is architectural, not a cost of multi-task training.

---

## 1. What a task is

A task is one **query** basin plus **K context** basins, all over the same
512-day window. Time is cut into 32 patches of 16 days. Each site carries 6
series (5 Daymet forcings plus `QObs`) and the 26 CAMELS statics.

A visibility mask of shape `(sites, patches, variables)` says what the model
may see. The query's discharge is hidden; the context sites' discharge is
visible; forcings are visible everywhere. The model reconstructs the hidden
patches, and the loss is scored only there.

Built by `build_task` in `src/hydropfn/train/train_pub.py`. Everything below
is a choice of which sites go in and which cells are hidden.

---

## 2. Nearby stations (mode A)

### Which neighbours

`--retrieval geo` (the default) takes the K nearest gauges by lat/lon, with
longitude scaled by 0.766 (cos 40°). The choice is deterministic: nearest
first, no sampling. The ranking table holds the 64 nearest per basin.

- In training, context comes from **training basins only**. Ranking over all
  basins once let training queries pull held-out basins in as context, which
  leaked (recorded in `Diagnosis.md`).
- At evaluation, `--context-pool all` (the default) lets a held-out basin use
  any gauge as context, including other held-out basins. Those gauges were
  never trained on, so this is not a leak. It does mean the setting is
  "ungauged basin among real gauges", not "no gauges anywhere".

### How the neighbours reach the query

`PUBModel` in `src/hydropfn/models/connector.py` has three stages:

1. **Site encoder**, weights shared across sites. Each site becomes 1 static
   token, 3 learned summary tokens and 32 × 6 = 192 series tokens. A series
   token is its value projection, or a `[MASK]` token where hidden, plus
   variable id, patch position and day of year.
2. **Connector: basin character.** A 4-layer transformer over every site's
   summary tokens, with a role embedding (query or context) and a Fourier
   encoding of each site's *displacement from the query*. Displacement is
   relative, so the model cannot memorise absolute regions. The query's
   series tokens then cross-attend to all the context-aware summaries.
3. **Time-aligned cross-attention: today's flow.** Each query patch attends
   to the context sites' tokens *at the same patch index*, all 6 variables of
   each. Context tokens carry the displacement encoding, and a learned
   per-head function of distance is added to the attention logits, which has
   the form of a kriging weight.

Stage 3 is the one that matters. It took the model from 0.556 to 0.853 in the
671-era tests (log space, so internal comparison only). Summaries pooled over
512 days can carry what kind of basin a neighbour is, but not what it did last
week.

### What was trained

The number of context sites is drawn per training step from
`{0,0,0,1,2,4,8,16}`: no context on 3/8 of steps, up to 16 sites. The K=0 draws
keep the no-observation arm working. Raising their share from 1/6 to 3/8 was
worth +0.055.

---

## 3. How many context points, and how far

### Compute scales roughly linearly in K

- The site encoder runs once per site, about 196 tokens each, so its cost is
  linear in the number of sites.
- The connector attends over 4 summary tokens per site: 68 tokens at K=16.
  That part is quadratic in K but negligible.
- The time-aligned path is 32 independent attentions, each over (K × 6) keys.
  Linear in K.

At the converged budget (800 epochs × 150 steps × 8 tasks), one fold trains in
roughly 2–3 hours on a single 11 GB GPU. The stride-1 evaluation used for
own-gauge data costs more than training.

### Measured behaviour against K

| setting | K=0 | K=2 | K=4 | K=8 |
|---|---|---|---|---|
| PUB spatial, neighbours only (10 folds) | 0.682 | – | 0.751 | – |
| temporal, own gauge always visible, plus K neighbours | 0.707 | 0.798 | 0.802 | 0.804 |

Most of the gain arrives by K=2–4 and is flat after that. The only test beyond
K=16 is from the 671 era (log space, internal comparison only). There, K=32
fell below the peak by 0.04 at the old small budget (0.845 → 0.805) and by 0.02
at 5× the budget (0.864 → 0.843). More training shrank the decline, so part of
it was undertraining.

Limits that are about the code, not the architecture:

- **K > 16 is outside the training distribution.** Using it well needs larger
  K in the training draws.
- **The ranking table stops at 64 neighbours.** It is one constant in
  `camels531_pub.py`.

### How far away a neighbour still helps

PUB spatial, mode A, split by distance to the nearest other gauge. IDW here
interpolates the 8 nearest gauges' concurrent flow, with no model:

| nearest gauge | basins | ours, K=0 | ours, K=4 | IDW | margin over IDW |
|---|---|---|---|---|---|
| 0–11 km | 58 | 0.765 | **0.881** | 0.838 | +0.043 |
| 11–22 km | 102 | 0.738 | 0.821 | 0.773 | +0.048 |
| 22–39 km | 152 | 0.716 | 0.780 | 0.714 | +0.066 |
| 39–67 km | 123 | 0.638 | 0.718 | 0.547 | +0.171 |
| >67 km | 96 | 0.465 | 0.477 | 0.187 | +0.290 |

Where gauges are dense the model is interpolation plus a little; the 0.88 in
the densest bin is the same regime where the literature reports kriging near
0.9. As gauges thin out, interpolation fails and the model falls back to its
no-context arm (0.477 vs 0.465 beyond 67 km). The context benefit fades
somewhere around 50–100 km. In the 671-era protocol, context forced to about
190 km away was worth nothing.

### What has not been tested

- **Distance is Euclidean.** Two gauges on the same river and two on
  different rivers at the same separation look identical to the model.
  Flow-network distance is the obvious fix and is not implemented.
- **Nested gauges.** CAMELS is curated for independence: 0.011% of basin
  pairs are nested. Upstream/downstream gauges on the same stem, the case
  where assimilation should be strongest, are essentially untested.
- **Context outside CONUS density.** Every test is at CAMELS gauge spacing.

---

## 4. The station's own near-real-time data

### Two ways to feed it in

| flag | where the own record goes | status |
|---|---|---|
| `--self-ctx-p p` (train), `--recent-obs P` (eval) | the query basin is **added as an extra context site**: its discharge is visible except the last P patches | **the one to use** |
| `--self-da` | the query's **own token stream**: discharge visible except the tail | loses at this budget |

With self-context, the site's own history goes through the same time-aligned
cross-attention as the neighbours, at displacement 0. It adds one site, so a
task with own gauge plus K neighbours has K+2 sites.

### Training draw

On each training step, with probability `--self-ctx-p`, the query is added as
self-context and the last `t` patches of its discharge are hidden, with `t`
drawn uniformly from `1..--self-ctx-max-tail`.

**The draw must make the evaluated case common.** Evaluation hides only the
last patch. With the old defaults (probability 0.4, tail uniform over 1–15
patches), that case was about 3% of training steps. Setting
`--self-ctx-p 0.8 --self-ctx-max-tail 2` raises it to 40%, and moved own-gauge
NSE from 0.655 to 0.707 and own gauge plus neighbours from 0.786 to 0.804.
Nothing in the code was wrong; the training distribution disagreed with the
evaluation.

### Evaluation: every day beyond the cutoff

`--recent-obs P` changes how the scored span is tiled. Windows step by one
patch instead of 32. The own record is hidden for the last P patches of each
window, and **only the final P patches of each window are written to the
scored series**. Every scored day is therefore predicted from a window where
it lies beyond the observation cutoff. With `patch 16, P=1`, the lag between
the last visible observation and the predicted day is 1 to 16 days. Windows
may start before the scored period and read pre-period observations, as an
operational system would.

### Results

Temporal split: the station's 1980–95 record was in training, and scoring is
1995–2010. The LSTM, dHBV1.1p and DI-LSTM references are gauged as well.

| | NSE |
|---|---|
| persistence (yesterday's flow as today's) | 0.444 |
| forcings-only LSTM (3 seeds) | 0.687 |
| dHBV1.1p, exact protocol | 0.743 |
| **ours, own gauge only** | **0.707** |
| **ours, own gauge + 8 neighbours** | **0.804** |
| DI-LSTM (Q at t−1 as input; Feng et al. 2020, literature) | ~0.86 at lead 1 |

### Known limits

- **No lead decay.** Scored by lead day, lead 1 and lead 16 are the same.
  With 16-day patches the model cannot see how recent the last observation is.
  Per-lead numbers sit about 0.03 above the all-days number at every lead;
  that is a subsampling artifact, not recency skill.
- **Finer patches have not helped yet.** Patch-1 variants at the same budget:
  self-context 0.626; self-da with a uniform tail 0.303, which learned to echo
  the flow from two days earlier because the scored position rarely had a
  recent observation in training; self-da with the tail fixed at 1 day 0.668.
- **The DI-LSTM gap (0.707 vs ~0.86) is the main open problem on this task.**

---

## 5. Historical context (mode B): useful for training, inert at inference

`--context-period train --ctx-align` gives the neighbours from a window
exactly 137 patches (6.001 years) earlier, same day of year. The neighbours'
long-term behaviour is visible, never today's flow.

- **At inference it adds nothing:** K=8 0.7084 vs K=0 0.7071. Same-season
  flow from six years earlier is climatology, which the forcings already give.
- **As a training draw it helps:** the no-context arm goes from 0.682 to
  0.707, the best no-observation number so far.
- IDW on historical flow is worse than predicting the mean (−0.44 to −0.51).

Do not combine `--context-period train` with `--extent temporal`. The fixed
137-patch offset reaches into the evaluation period there; a comment in the
code blocks it.

---

## 6. Rules that each cost a result once

- **Donor baselines must read the same window the model's context read.** In
  mode B that is the historical window. Reading the evaluation window hands
  the baselines concurrent flow the model was denied, and once made a trivial
  baseline look like it crushed the model. The tell was baseline columns
  identical between mode A and mode B runs.
- **In recent-obs mode the self-context slot is not a donor.** Baselines skip
  site index 1 there.
- **Score NSE on raw mm/day.** Standardised or log-space NSE is 0.05–0.15
  higher and cannot sit next to published values.
- **Use the converged budget:** `--epochs 800 --steps 150 --tasks 8`. The old
  40/150/4 smoke defaults left the no-context arm at 0.265.
- **Make the evaluated configuration common in training.** This has now
  happened three times: K=0 share, smoke defaults, own-gauge tail length.

---

## 7. How to run

On suntzu, from `/nfs/data/cxs1024/hydroPFN`:

```bash
export HYDROPFN_CAMELS_ROOT=/nfs/data/cxs1024/hydroPFN/data
export PYTHONPATH=/nfs/data/cxs1024/hydroPFN/src
PY=/data/cxs1024/tools/anaconda3/envs/pytorch_gpu/bin/python

# nearby stations, ungauged basins (PUB spatial, 10 folds)
$PY experiments/camels531_pub.py --extent PUB --k-eval 0,4 --tag pub531_e800

# own gauge + nearby stations, gauged basins (temporal)
$PY experiments/camels531_pub.py --extent temporal \
    --self-ctx-p 0.8 --self-ctx-max-tail 2 --recent-obs 1 \
    --k-eval 0,2,4,8 --tag pub531_drawshare_e800

# historical context (spatial extents only)
$PY experiments/camels531_pub.py --extent PUB \
    --context-period train --ctx-align --k-eval 0,2,4,8 --tag pub531_modeB_e800
```

Add `--save-ckpt` to keep per-fold weights and `--init-ckpt` to warm-start
from them. Outputs go to `logs/camels531/<tag>/`: `pred_K*.npy`, `targ.npy`,
`gage.npy`, `fold.npy`, `run.json`. Always join on `gage.npy`, never on row
order.

---

## 8. Code map

| file | what is there |
|---|---|
| `src/hydropfn/train/train_pub.py` | `build_task`: site selection, the visibility mask, mode B windows, self-context, self-da |
| `src/hydropfn/models/connector.py` | `PUBModel`, `CrossSiteConnector`, `GeoEncoding`, the distance bias |
| `src/hydropfn/models/site_encoder.py` | tokenisation, `[MASK]`, summary tokens |
| `experiments/camels531_pub.py` | training loop, draws, `predict_basins` (tiling, recent-obs scoring, donor baselines) |
| `src/hydropfn/data/protocol.py` | CAMELS-531 folds, periods, warmup |
| `docs/camels531_protocol.md` | all current results; the source of truth |
| `Diagnosis.md` | the 671-era history, including every correction |

---

## 9. Open questions

1. **The DI-LSTM gap on own-gauge data.** Candidates: a daily recurrent head
   conditioned on the transformer, or 4-day patches (same 512-day memory, 4×
   the time resolution).
2. **Flow-network distance** in place of Euclidean distance in retrieval and
   in the attention bias.
3. **Nested gauges and denser networks**, which CAMELS cannot test.
4. **K beyond 16** in training, to see whether the flat K=4–8 curve is a
   ceiling of the information or of the training draw.
5. **Statics masking plus mode-B draws together.** Each lifts the no-context
   arm by about 0.025 on its own; they have not been combined.
6. **Own-gauge data on an ungauged basin** (PUB extent with `--recent-obs`):
   a station that was never in training but has recent observations, e.g. a
   newly installed gauge. Not run. An LSTM cannot use this without retraining.
