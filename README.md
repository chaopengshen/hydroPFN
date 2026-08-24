# hydroPFN / StefaNP

A multimodal masked autoencoder whose prediction is conditioned on a **retrieved
context set of sites** — a Transformer Neural Process for hydrology.

Sibling to [StefaLand](https://arxiv.org/abs/2509.17942), extending it with the
three things it does not have: **cross-site in-context inference**, **DEM pixels
as a modality**, and **irregular point measurements as tokens**.

![architecture](figs/fig_architecture.png)

---

## Status

Two components are built and measured. Two are templates. Nothing here is a
promise — every number below came from a leave-region-out run, and the failures
are on the page next to the passes.

| component | state |
|---|---|
| **B · terrain (DEM)** | built — conditional diffusion sampler, beats interpolation on every metric |
| **D · measurement** | built — in-context model, cross-variable gate replicates 9/9 |
| **A · site (static + series, MERGED)** | trained on real CAMELS, leave-region-out — ablations pass, **U3 gate (beat a regional LSTM) not yet run** |
| **connector (cross-site)** | **built and TESTED — ungauged basins predicted at R² 0.853 by conditioning on nearby gauges (+0.38 over no context), beating donor-averaging at every K** |

---

## Result 1 — the DEM sampler beats interpolation on every metric

A deterministic net is capped at ~0.31 of true fine-scale spectral power, and
that cap is *structural*: an L1/L2 loss returns the conditional mean, and the
mean of plausible terrains is smooth. Only a sampler can commit to one sharp
realisation.

Final recipe = **residual parameterisation + best-of-K reranking**.
60 held-out 1° tiles, K = 8 draws. Success bands were fixed **before** running.

| | **StefaNP** | harmonic | best deterministic U-Net |
|---|---|---|---|
| in-hole elevation RMSE | **0.891 m** | 2.028 | 1.88 |
| slope RMSE | **0.0328** | 0.0425 | 0.047 |
| slope Wasserstein-1 | **0.0072** | 0.0229 | 0.008 |
| short-λ PSD ratio (1.0 = right) | **0.810** | 0.175 | 0.31 |
| semivariogram ratio, 10 m | **0.842** | 0.376 | 0.42 |
| semivariogram ratio, 80 m | **0.980** | 0.529 | 0.52–0.64 |

Beating harmonic on the *pointwise* metrics and the *distributional* ones at
once is the substantive result. Pointwise error structurally rewards the
flattest consistent surface — harmonic provably **is** that surface — so a
textured method normally pays for its texture in RMSE. Doing both means
reconstruction, not decoration.

![DEM sampler](figs/fig_sampler_best.png)

### What was tried, and what worked

Five ideas, each run as its own arm so the effects are attributable.

| change | verdict | rerank elev | rerank PSD |
|---|---|---|---|
| baseline (ε-pred, generate the surface) | — | 1.192 | 0.803 |
| **best-of-K reranking** | **works** (no training) | — | 0.662 → 0.803 |
| **residual over harmonic** | **works** | **0.891** | **0.810** |
| v-parameterisation | **hurts** | 1.374 | 0.723 |
| all three stacked | worse than residual alone | 1.210 | 0.802 |
| uniform-CONUS patches | premise refuted | — | — |

**Reranking**: draw K, keep the one whose in-hole variogram best matches the
*surrounding rim*. The rim is observed at inference, so this is method, not
oracle picking — and it is justified by the spread (single-draw PSD runs p10
0.14 / median 0.66 / p90 1.95, so a good draw usually *exists*).

**Residual**: generate the *departure* from the harmonic surface. Harmonic
already solves the low frequencies exactly, so making the net re-derive them
wastes capacity on the part that was never the problem. This removes the
texture-vs-fidelity trade-off entirely: on the baseline, reranking **cost**
elevation accuracy (1.058 → 1.192); with the residual it **improves both**
(1.045 → 0.891).

> **A single stacked "all improvements" run would have shown a modest gain over
> baseline, and all three changes would have shipped — when two of them are
> actively harmful on top of the third.** One change per arm, three GPU-hours
> in parallel, correct answer.

---

## Result 2 — the model learns from a site's own measurements, better than a hand-built feature can

Concretely. Take a river site where somebody measured **width** on several
visits, and you want to know the **velocity**. Two ways to use those width
measurements:

**Hand-engineering.** Invent a feature: *"the median amount this site's width
deviates from what its attributes predict, across its other visits."* Feed that
one number to a random forest. Gain: **+0.041 R²** on velocity.

**In-context.** Hand the model the raw measurements as tokens —
`(width, 47 m, at discharge 3.2)`, `(width, 61 m, at discharge 8.1)` — and let
it work out what they imply. Gain: **+0.127 R²**.

Same raw data, **2.3× more extracted**. That is the argument for the
architecture: you do not have to invent a clever feature for each variable
pair and each measurement pattern — the model learns them. And those
measurements are supplied **at prediction time, with no retraining**, which is
what "in-context" means here.

Why does knowing width tell you about velocity at all? Because `W·d·v = Q`:
at a given discharge, a site that runs anomalously wide must be anomalously
shallow or slow. The data supply *how that trade-off splits* at this particular
site — and that split is real channel-shape information no attribute table
contains. (Same-visit measurements are excluded from context throughout, so
this is cross-visit inference, not arithmetic on one row.)

The measurement unit (D) trained on 64,797 HYDRoSWOT visits at 5,057 sites.
Acceptance gates were fixed before any run, then **replicated over 3 holdout
regions × 3 seeds**:

| gate | median | min | max | pass |
|---|---|---|---|---|
| cross-variable (predict velocity from a site's *width* history) | **+0.1265** | +0.0965 | +0.3029 | **9/9** |
| at-a-station vs a per-site power law | +0.0967 | −0.0072 | +0.2727 | 24/27 |
| zero-context vs an attributes-only RF | +0.0043 | −0.0601 | +0.0451 | 21/27 |

The cross-variable effect is **2.3× what a hand-built feature achieved** on the
same data, even at its worst run. Zero-context sits at *parity* with a tuned
random forest — it matches, it does not beat.

![context scaling](figs/fig_context_scaling.png)

Adding observations at inference, with **no retraining anywhere**, raises
accuracy monotonically (velocity R² 0.575 → 0.906 over 0 → 8 own-site visits).
Neighbouring sites give a **step**, not a curve (−0.145 → 0.582 on the first
neighbour, then flat to 16) — the model calibrates from one neighbour but does
**not** aggregate across many. Fixed context size during training is the likely
cause; randomising it is the untested fix.

---

## Results 3 and 4 — SUPERSEDED, see docs/all_results.md

These two sections previously carried a mask-mode table and a ceiling
comparison (ceiling 0.8606 / ours 0.8447). **Both predate the 2026-08-24
scoring correction and are wrong**: they compared a 16-day-mean LSTM number
against our daily number, which is worth +0.023 on its own.

Corrected, on a matched protocol: the gauged ceiling is **0.8477**, reading a
gauge's own record as context gives **0.8026**, and adding neighbours gives
**0.8816**.

**All current results live in [docs/all_results.md](docs/all_results.md)**,
with a protocol column and a highlights section. Nothing is restated here, so
there is one source of truth rather than two.

## Layout

```
src/hydropfn/
  data/     dem.py          3DEP patch acquisition (gage-centred or uniform CONUS)
            folds.py        leave-region-out and grouped splits
            forcing.py      CAMELS loader + synthetic generator + mask sampler
  models/   diffusion.py    DEM sampler: U-Net, cosine DDPM, conditional DDIM
            inpaint.py      masks, harmonic / IDW baselines, PConv U-Net
            measurement_pfn.py   the built in-context model (unit D + connector)
            site_encoder.py the merged unit A: static + series, one trunk
            encoders.py     superseded notes on the A/C merge decision
            connector.py    TEMPLATE — general cross-site transformer
            decoders.py     TEMPLATE — per-modality decoders
            stefanp.py      TEMPLATE — the assembled model
  metrics/  terrain.py      slope, hillshade, PSD, semivariogram, slope-W1
  train/    train_dem_sampler.py       built
            train_measurement_pfn.py   built
            train_site_encoder.py      built (unit A, masked reconstruction)
            train_stefanp.py           TEMPLATE — the four-stage plan
scripts/    figure and deck generation
experiments/  the evidence base (see below)
tests/      test_smoke.py   invariant checks; run after any lib edit
docs/       architecture.md, environments.md, stefaland_reuse.md,
            proposal_seed.md, dev_log.md, code review
```

The templates are not placeholders — each carries the design decision, the
tensor contract, and the traps found the hard way. Read them before implementing.

## Reproduce

```bash
source gpuenv.sh                      # suntzu: see the traps documented inside
export PYTHONPATH=$PWD/src
PATCHES=/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz

python tests/test_smoke.py                                    # ~10 s
python -m hydropfn.train.train_dem_sampler --patches $PATCHES \
    --residual --epochs 300 --n-eval 60 --k 8 --tag residual   # ~2.3 h
python -m hydropfn.train.train_measurement_pfn \
    --table <train_table_dem.csv> --holdout 03 --seed 0        # ~15 min
python scripts/fig_dem_sampler.py --patches $PATCHES --ckpt logs/residual.pt
```

DEM patches are ~300 MB and live on suntzu; regenerate with
`python -m hydropfn.data.dem --uniform --min-relief 1.0 --max-per-tile 12`.

## The evidence base

`experiments/` holds the measurements the design rests on — including the ones
that killed ideas:

| script | what it settled |
|---|---|
| `cross_variable_context.py` | a site's cross-visit width anomaly is worth +0.041 R² on velocity |
| `atastation_icl.py` | in-context beats train-once on identical information |
| `residual_learnability.py` | geometry residuals are **not** learnable from terrain (all cells negative, two fold structures) |
| `lithology_premise.py` | terrain → lithology fails once the topographically-defined class is removed |
| `sgmc_noise_floor.py` | the geology label ceiling is 0.91, so 0.593 was not noise-limited |
| `dem_hedging_diagnostic.py` | an L1 net *deletes* sharp channels rather than blurring them |
| `repair_site_no.py` | recovers 15% more usable rows; refuses a looser join its own guard rejects |

## Standing rules

Every number here was produced under these, and each exists because breaking it
once produced a confident wrong answer:

- **Split by site, never by COMID.** A COMID split silently duplicated 1,360
  sites across both arms.
- **Identical row/site population across compared models.** A forgotten control
  turned a real +0.024 into an apparent +0.090.
- **Report seeds, not single runs.** Same-config spread reached 0.11 in
  `psd_ratio` — larger than the effect being claimed at the time.
- **Exercise a metric on a known input before believing a table built from it.**
  Three separate metric bugs produced confident, wrong tables first.
- **When variables are algebraically linked, exclude the measurement occasion,
  not the measurement.** `W·d·v = Q` inflated the cross-variable gain from
  +0.108 to +0.141 until whole occasions were excluded.

## Known weaknesses

- **Sampler quality tracks terrain**: PSD 0.841 rough / 0.652 moderate / **0.452
  flat**. It does not continue anthropogenic linear features (roads, canals) —
  a natural-terrain prior gives no reason to.
- **The connector does not aggregate context sites** (see Result 2).
- **The existing DEM training set is latitude-biased** — tiles were walked in
  sorted order, so its 404 tiles skew south. Fixed in `data/dem.py`, but the
  shipped checkpoints predate the fix.
- **Thin channel-following water masks — the actual target — are untested.**
- Single seed per sampler arm; the ordering is likely solid, the exact numbers
  are not.

---

## How this model is trained: the draw structure

**Read this before interpreting any number in this repo.** What the model can
do at inference is decided by what was *asked of it* during pretraining, and
nothing else.

### "Mixture" means a mixture of MASKING TASKS

Not mixture-of-experts, not a data mixture. The `--mask-mix` flag samples
**which kind of hole** is punched in the query's data on each training task,
instead of always punching the same one.

Every capability here is created by a hole. The model is only ever asked to
fill in what was hidden — so **the shape of the hole is the task**, and the set
of shapes seen during pretraining is exactly the set of capabilities that
exists at inference.

| hole shape | what it enforces |
|---|---|
| `whole_site` — all query discharge hidden | predict a basin with **no** record (ungauged / PUB) |
| `causal_tail` — hide everything after *t* | forecasting and data assimilation |
| `random_span` — hide a contiguous chunk | gap filling (a sensor was down for three weeks) |
| `whole_variable` — hide one variable | cross-variable inference (infer precipitation from discharge) |
| self-as-context | use a site's **own** lagged record through the same interface as a neighbour's |
| K = 0 vs K > 0 | one set of weights must be both a standalone forward model and a context-conditioned one |
| attribute masking | recover static properties from dynamics |

### What is actually drawn

Headline configuration (`--k-train 0,0,0,1,2,4,8,16 --self-ctx-p 0.4`):

| draw | granularity | distribution |
|---|---|---|
| K (context size) | per **step** | K=0 **37.5%**; K ∈ {1,2,4,8,16} **12.5%** each |
| self-as-context | per **step** | present **40%**, tail ~ U{1…15} patches |
| mask kind | per task | `whole_site` **100%** — the mixture is **off** in every headline run |
| window start, query basin | per task | uniform |
| context sites | deterministic | K geographic nearest |

Four effective task types:

| | K=0 | K>0 |
|---|---|---|
| no self-context | forward model **22.5%** | neighbour assimilation **37.5%** |
| self-as-context | self assimilation **15%** | self + neighbours **25%** |

### The finding that governs all of this

**Nothing here is zero-shot.** The identical architecture evaluated on
self-as-context *without* that task in training scores **0.2490**; with it as a
training draw, **0.7878**. A capability absent from the pretraining
distribution does not exist at inference — verified mechanically as well: in a
K=0-only checkpoint the cross-site attention weights sit at their random
initialisation to five decimals, because that code path never executed.

Two consequences readers should carry:

- **The distribution is thin** — four task types, one hole shape. That is
  defensible for the present results, but it is *not* a task distribution in
  the TabPFN sense. Anything outside those four cells will not work.
- **Presence is not enough; share matters.** With K=0 at only 1/6 of tasks the
  forward pathway scored 0.658; upweighted it reaches 0.7164 and ties a
  regional LSTM. The same default cost the 1-day-lead assimilation task 0.17.

Draws still missing for the intended global, multivariate, DEM-integrated
model — forcing-variable dropout, context with a different variable set than
the query, cross-site cross-variable transfer, modality dropout, context from a
different period, window-length draws, realistic gap patterns, and K up to
20–50 with mixed entry types — are enumerated with rationale in
`dem_foundation/docs/dev_dem.md`.

---

## Results

**[docs/all_results.md](docs/all_results.md)** — every number with its
protocol, plus a highlights section of the conclusions worth
remembering. **[docs/benchmarks.md](docs/benchmarks.md)** — the
reference results we compare against, the command behind each number,
and how to verify the split.

Verify the split before trusting any number:
`python scripts/verify_split.py --nc data/CAMELS_Frederik.nc`
