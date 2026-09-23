# hydroPFN / StefaNP — handover

What this project is, where the code is, what each arm does, how the arms
connect, what we are trying to beat, and where we are stuck. Written
2026-09-22, at the end of a long session on the time-series arm.

Read this first, then follow the links. Two rules about the links:

- **[docs/camels531_protocol.md](camels531_protocol.md) is the source of truth
  for every CAMELS number.** If anything here disagrees with it, that page wins.
- **[Diagnosis.md](../Diagnosis.md) and [docs/all_results.md](all_results.md)
  are history.** Their numbers are log-space on a 671-basin protocol of our own
  invention; they carry warning banners. Read them for the reasoning and the
  corrections, never for a number to quote.
- `README.md` is out of date (it still reports the 671-era R² 0.853 and an open
  U3 gate). Fix or delete it before showing this repo to anyone.

---

## 1. What the model is

A masked autoencoder over **sites**, whose prediction for one site is
conditioned on a **retrieved set of other sites** supplied at inference. In
neural-process terms: the context set is data, not weights. Nothing is
retrained when new observations arrive.

The claim being tested is not "better streamflow model". It is: *one set of
weights that accepts whatever observations exist at run time — none, a
neighbour's gauge, the site's own recent record, or all of them — and beats
the specialised model for each of those situations.*

Where that claim currently stands (median per-basin NSE, raw mm/day,
CAMELS-531):

| what the model is given | ours | best alternative | verdict |
|---|---|---|---|
| nothing, basin held out of training | 0.682 (0.707 mode-B-trained) | LSTM+dHBV1.1p 0.700 | ties |
| ~4 nearby gauges, basin held out | **0.751** | StefaLand 0.721, IDW 0.645 | **wins** |
| nothing, basin in training | 0.640 | LSTM 0.687, dHBV1.1p 0.743 | **loses** |
| own gauge 1–16 d stale, basin in training | 0.707 | LSTM 0.687, dHBV1.1p 0.743 | mixed |
| own gauge + neighbours, basin in training | **0.804** | dHBV1.1p 0.743 | **wins** |

So: the in-context machinery works and wins wherever observations exist. The
plain forward model on a *gauged* basin is the hole, and §5 is about that.

---

## 2. Where the code is

Repo `github.com/chaopengshen/hydroPFN`, local `G:\Work\Claude\hydroPFN`.
**Working branch is `experiment`; `main` is 11 commits behind** as of writing.

| path | what it is |
|---|---|
| `src/hydropfn/models/site_encoder.py` | arm A: tokenises one site (statics, forcings, discharge, optional DEM token), `[MASK]`, summary tokens |
| `src/hydropfn/models/connector.py` | `PUBModel` + `CrossSiteConnector`: how sites talk to each other. The whole cross-site story is here |
| `src/hydropfn/models/diffusion.py` | arm B: conditional DDPM U-Net for terrain |
| `src/hydropfn/models/measurement_pfn.py` | arm D: irregular point measurements as tokens |
| `src/hydropfn/train/train_pub.py` | `build_task` — **the most important function in the repo**: which sites, which cells hidden, which window |
| `src/hydropfn/train/train_dem_multiscale.py` | arm B training, multi-scale + scale conditioning |
| `src/hydropfn/data/protocol.py` | CAMELS-531 folds, periods, warmup |
| `src/hydropfn/metrics/dmg_metrics.py` | dmg_dev's `Metrics`, vendored verbatim — the only scorer we quote |
| `experiments/camels531_pub.py` | the benchmark driver: training loop, evaluation tiling, donor baselines |
| `experiments/camels531_lstm.py` | our LSTM baseline on the same protocol |
| `experiments/kriging_density.py` | model-free IDW, stratified by gauge density |
| `experiments/dem_imputer.py`, `dem_attr_eval.py` | the DEM↔statics interaction tests |

Compute is on **suntzu** at `/nfs/data/cxs1024/hydroPFN` (never locally).
Python `/data/cxs1024/tools/anaconda3/envs/pytorch_gpu/bin/python`, with
`HYDROPFN_CAMELS_ROOT=/nfs/data/cxs1024/hydroPFN/data` and `PYTHONPATH=src`.
Run outputs land in `logs/camels531/<tag>/` as `pred_K*.npy`, `targ.npy`,
`gage.npy`, `fold.npy`, `run.json`. **Join on `gage.npy`, never on row order.**

---

## 3. The arms

### A — site / time series (built, benchmarked)

Per-site encoder over 16-day patches: 1 static token, 3 summary tokens, and
32 × 6 series tokens (5 Daymet forcings + discharge). Hidden cells become a
`[MASK]` token. 7.2M parameters with the connector.

This is the arm the whole CAMELS suite measures. Everything in §1 is arm A
plus the connector.

### B — terrain / DEM (built, and its streamflow value is now a closed negative)

Conditional diffusion sampler that inpaints terrain, trained multi-scale
(1.28 / 12.8 / 51.2 km footprints) with continuous scale conditioning so an
unseen resolution interpolates rather than falling off a table. It beats the
single-scale champion on the champion's own protocol (psd 0.937 vs 0.824).

Its hidden activations were then tested as features for arm A, four ways, and
all four came back null or negative. See §4.

Its remaining justification: **geology as a target** (lithology 0.59→0.64,
bedrock age 0.37→0.44 R², where no curated attribute table exists anywhere),
and the future 2D-field arm. History in
`G:\Work\Claude\River_Transport\dem_foundation\docs\dev_dem.md`.

### D — measurement (built, not on the current protocol)

In-context model for irregular point measurements; cross-variable gate
replicated 9/9. Not part of the CAMELS-531 suite.

### Connector — how sites talk (built, this is what works)

Three paths, in `PUBModel.forward`:

1. **Site encoder**, shared weights, per site.
2. **Pooled connector**: a transformer over each site's summary tokens, with a
   query/context role embedding and a Fourier encoding of each context site's
   *displacement from the query* (relative, so regions cannot be memorised).
   Carries basin character.
3. **Time-aligned cross-attention**: each query patch attends to the context
   sites' tokens *at the same patch index*, with a learned per-head distance
   term added to the attention logits (the functional form of a kriging
   weight). Carries today's weather.

Path 3 is the one that matters: it took the model from 0.556 to 0.853 in the
671-era tests. Detail, including how far away context still helps, is in
[docs/data_assimilation.md](data_assimilation.md).

---

## 4. How the arms connect — measured, mostly null

The DEM↔time-series connector was the big open design question. It was tested
in four directions before building anything permanent, and the honest answer is
that the coupling through basin-scalar quantities is thin everywhere:

| direction | test | result |
|---|---|---|
| terrain → statics | imputation ladder, PUB / new terrain | +0.002 / +0.010 |
| terrain → streamflow | in-model, converged budget, 5 compression arms | best arm +0.004, raw features −0.016 |
| statics → terrain (linear) | reconstruct terrain PCs | 0.069 — weak but above location (0.008) and climate (0.036) |
| statics → terrain (generative) | attribute-conditioned sampler, paired eval | null |

Two findings worth carrying:

- **Compression matters, and a learned bottleneck is not compression.** Raw 768
  terrain dims hurt (>1 dim per training basin is identification capacity);
  fixed PCA to 8–32 dims removes the damage; a *learned* 768→8 squeeze is the
  worst arm of all, because eight trainable numbers can still encode which
  basin this is. If terrain is ever fed to arm A again, feed it through a
  fixed, task-blind basis.
- **Attributes krige, terrain does not.** Attributes interpolate across space
  at R² 0.83 within regions; terrain features do not transfer across regions at
  all (−0.06). They are complementary information types, which is exactly why
  terrain should matter where no attribute table exists — a case CAMELS cannot
  test.

**Side result, more useful than the DEM result itself:** masking 50% of the
statics during training lifts the no-context arm from 0.675 to 0.701 on matched
folds. Mode-B training draws lift it to 0.711. Both are regularisers; they have
never been combined.

---

## 5. The hole: we cannot beat a local temporal projection

The one regime we lose: a basin that *is* in training, simulated forward with
no observations. Ours 0.640, our LSTM 0.687, dHBV1.1p 0.743.

This is odd, because the same architecture **wins** on held-out basins
(0.682–0.707 vs 0.700). So it is not general weakness. It is specifically the
case where the model has seen this basin's own history and should be able to
fit it closely.

### Ruled out

- **Multi-task dilution.** A from-scratch K=0-only specialist of our
  architecture scores **0.639** — identical to the generalist's 0.640.
  Specialising buys nothing, so there is no mixture tax and nothing for a
  finetune to recover. (Good news for the one-model claim; bad news for the
  easy explanation.)
- **Training budget.** 800 × 150 × 8 is converged; the curve is flat.
- **Metric or protocol mismatch.** Verified against a completed dmg_dev run at
  r = 1.0000000000.

### Live hypotheses, cheapest first

1. **Loss/metric mismatch — untested, and my first suspect.** We train a plain
   masked MSE on discharge standardised by a *single global* mean and sd
   (`experiments/camels531_pub.py`, the `loss = ((rec - series)**2 * w).sum()`
   line). The metric is *median per-basin NSE*, which divides each basin's
   error by that basin's own variance and then weights every basin equally.
   Under a global MSE, a low-variance basin contributes almost nothing to the
   gradient while counting fully in the score. dmg trains dHBV1.1p with
   `NseBatchLoss` — the metric itself. Our own LSTM uses global RMSE and also
   lands below dHBV1.1p (0.687 vs 0.743), which is consistent with part of that
   gap being the objective rather than the model. **Test:** divide each basin's
   squared error by its own variance in the loss. One expression; one run.
2. **Time resolution.** A 16-day patch is one token, and daily peak timing
   inside it has to be reconstructed from that token by a shared
   `Linear(d → 16)` head. On a known basin most of the remaining error is peak
   timing; on an unknown basin most of it is basin character, where we win.
   **Test:** 4-day patches in 128-token windows (same 512-day memory, 4× the
   time resolution).
3. **No carried state.** Each 512-day window starts cold; the LSTM runs its
   state through the whole record. **Test (bigger):** a small daily recurrent
   head conditioned on the transformer's patch outputs — transformer for
   cross-site context, recurrence for daily dynamics.
4. **Supervision volume.** Our forward run scores ~491M day-values
   (8 tasks × 512 days × 120k steps) against the LSTM's ~1.24B
   (128 × 365 × 26,600). A factor of 2.5, not a factor of 100 — worth a
   longer run but unlikely to be the whole story.
5. **Physics priors.** dHBV1.1p carries a conceptual model with per-basin
   parameters. Some of its 0.743 is not available to us by any training change.

### The related hole: own-record data (§4 of data_assimilation.md)

0.707 with the station's own recent record, against ~0.86 in the literature for
a DI-LSTM with yesterday's flow as an input. Diagnosis:

- Our evaluation predicts each day from an observation 1–16 days old. On these
  basins persistence is 0.444 at 1 day and −0.04 at 2 days, so an observation
  goes stale almost immediately, and the naive baseline for our exact setup is
  −0.31.
- **The model does not exploit freshness at all**: 0.722 at lead 1 vs 0.725 at
  lead 16. Two causes: the own record enters as a "neighbour", and the
  time-aligned path links same-index blocks — for the own record the same block
  is the hidden one; and the model gets ~770k one-day-ahead examples against an
  LSTM's ~1.2 billion.
- **Fix, which is a formulation change rather than a new architecture:**
  yesterday's flow as its own input channel on every day, today's flow hidden
  on every day, 1-day patches, causal attention (otherwise day *t* reads the
  lagged channel at *t+1*, which is the answer). Every day then becomes a
  supervised one-day-ahead example.
- **Before building it:** run our LSTM with yesterday's flow added as an input,
  to get the real bar on *this* protocol. ~30 minutes.

---

## 6. What we are trying to beat

All on CAMELS-531, raw mm/day, dmg `Metrics`, from
[camels531_protocol.md](camels531_protocol.md):

| reference | PUB (ungauged) | temporal (gauged) | provenance |
|---|---|---|---|
| our LSTM | 0.666 | 0.687 (3 seeds) | `experiments/camels531_lstm.py` |
| LSTM + dHBV1.1p | 0.700 | – | dmg, 3 seeds, same protocol |
| StefaLand | 0.706–0.721 | – | dmg, same protocol |
| dHBV1.1p | – | 0.743 | released dMG config, exact protocol |
| DI-LSTM (own gauge, lead 1) | – | ~0.86 | literature, different period/forcing |
| IDW (model-free) | 0.645 | 0.634 | `experiments/kriging_density.py` |
| persistence (lead 1) | – | 0.444 | own-gauge floor |

Everything of ours is **single seed** except where noted. Before any of this is
quoted in a paper: seed sweeps, and a check that the 0.666/0.545 PUB LSTM
values (from a 50-epoch run) still hold at a converged budget.

---

## 7. Process rules that were each paid for once

These are not style preferences; each one silently produced a wrong number.

1. **The shape and frequency of the training draw is part of the experiment.**
   Three instances: K=0 drawn 1/6 of steps (+0.055 when raised to 3/8);
   smoke-test defaults escaping into a benchmark (K=0 0.265 → 0.682 when
   converged); the own-record tail length, where the evaluated case got 3% of
   training steps (0.655 → 0.707 when raised to 40%). None of these was a code
   bug; the training distribution disagreed with the evaluation.
2. **Donor baselines must read the same window the model's context read.**
   Reading the evaluation window hands them concurrent flow the model was
   denied. Reversed a mode-B table once.
3. **Never compare across metrics or protocols.** Log-space NSE runs 0.05–0.15
   above raw NSE; the whole of `all_results.md` was quietly cross-metric.
4. **Diagnose by measuring the artifact, not by reading the code.** The
   lag-+2 echo (predictions best correlated with the observation two days
   earlier, 490/531 basins) and the byte-identical baseline columns were both
   found this way; neither was visible by inspection.
5. **Verify the ported copy.** `md5sum` local and suntzu before launching. A
   lost patch once produced plausible numbers from the wrong code path.

---

## 8. Where to go next, in the order I would do it

1. **Per-basin NSE loss** (§5, hypothesis 1). Cheapest test with the largest
   potential effect, and it applies to our LSTM baseline too.
2. **DI-LSTM baseline on this protocol**, to find out what ~0.86 really is here.
3. **Statics masking + mode-B draws combined** — each worth ~+0.025 alone on
   the forward arm, never combined; could close the gap to StefaLand on PUB.
4. **4-day patches** (§5, hypothesis 2).
5. **Own-record reformulation** (§5) once 2 has set the target.
6. **Seed sweeps** before anything is written up.

Open questions that need data we do not have: flow-network distance instead of
Euclidean in retrieval; nested upstream/downstream gauges (CAMELS has 0.011%
nested pairs); terrain value where no attribute table exists (needs a region
CAMELS cannot supply).

---

## 9. Document map

| document | what it holds |
|---|---|
| [docs/camels531_protocol.md](camels531_protocol.md) | **source of truth**: protocol, every current number, the DEM tests, caveats |
| [docs/data_assimilation.md](data_assimilation.md) | how context and own-record data work mechanically; scale and reach |
| [docs/architecture.md](architecture.md) | the three-level design |
| [Diagnosis.md](../Diagnosis.md) | 671-era history and every correction; log-space, do not quote numbers |
| [docs/all_results.md](all_results.md), [docs/benchmarks.md](benchmarks.md) | deprecated results, kept for provenance |
| `dem_foundation/docs/dev_dem.md` (other repo) | the DEM arm's full lab notebook |
| `docs/environments.md` | suntzu / ICDS environments |
