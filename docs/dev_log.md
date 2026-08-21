# dev_dem.md — DEM foundation model: master log

Forked from Stage-I channel geometry 2026-08-15. Goal: pretrain on masked-DEM
prediction so a model learns how terrain is organised; use the representation
downstream, ultimately for subsurface materials; eventually narrate an inference.

Design: [`DESIGN_DEM_FOUNDATION_MODEL.md`](DESIGN_DEM_FOUNDATION_MODEL.md)

---

## Current versions (non-dominated)

| # | date | version | status |
|---|---|---|---|
| v0.1 | 08-15 | project scaffold, design doc, SGMC acquired | done |
| v0.2 | 08-16 | patch sampler + L1 inpainting probe | **superseded by v0.4** — L1 cannot make texture |
| v0.3 | 08-16 | hedging diagnostic: L1 is a low-pass filter | done, explains v0.2 |
| v0.4 | 08-16 | texture loss + hole-restricted distributional metrics | superseded by v0.5 numbers |
| v0.5 | 08-17 | code review: 4 bugs fixed, 3-seed re-measurement | done |
| v0.6 | 08-20 | conditional diffusion sampler — works | superseded by v0.7 |
| v0.7 | 08-21 | **rerank + residual: beats harmonic on every metric** | **current** |

---

## Objective A — is masked DEM prediction learnable beyond interpolation?

| # | lib | test | problem | result |
|---|---|---|---|---|
| v0.2 | `lib/inpaint.py` | `tests/sample_patches.py` | sample 3DEP patches, whole-tile split | 6,000 × 128² @10 m, 404 tiles; relief median 33.8 m |
| v0.2 | `lib/inpaint.py` | `tests/test_inpaint_probe.py` | can a partial-conv U-Net beat interpolation? | **no** — harmonic wins on slope in every terrain class |
| v0.3 | `lib/terrain.py` | `tests/test_hedging_diagnostic.py` | is the feature deleted or hedged? | **hedged at long λ, deleted at short λ** |
| v0.4 | `lib/inpaint.py:texture_loss`, `lib/terrain.py` | `tests/test_inpaint_probe.py` | does a distributional loss restore texture? | **partly: 0.26 → 0.47 of true roughness, still ≪ 1** |

### Results

**v0.2 — L1 inpainting, held-out 1° tiles** (`logs/inpaint_probe.csv`,
`figs/fig_inpaint_examples.png`)

| terrain | metric | harmonic | IDW | U-Net |
|---|---|---|---|---|
| flat | slope RMSE | **0.012** | 0.013 | 0.012 |
| moderate | slope RMSE | **0.024** | 0.035 | 0.027 |
| rough | slope RMSE | **0.051** | 0.076 | 0.056 |
| all | rim jump | **0.014** | 0.015 | 0.031 |

**v0.3 — hedging diagnostic** (`figs/fig_hedging.png`). In-hole elevation
anomaly amplitude: truth 1.18 m, harmonic 1.33, **U-Net 0.88**, IDW 11.55.
The U-Net reproduces the broad valley (75% of true amplitude) and deletes the
two sharp channel incisions entirely.

**v0.4 — texture loss, metrics restricted to the hole**
(`logs/inpaint_texture.csv`, `figs/fig_inpaint_texture.png`)

| method | **psd_ratio** | γ 10 m | γ 80 m | slope RMSE | rim jump |
|---|---|---|---|---|---|
| harmonic | 0.255 | 0.295 | 0.513 | **0.0235** | **0.0120** |
| IDW | 2.069 | 1.099 | 1.347 | 0.0364 | 0.0134 |
| U-Net L1 | 0.437 | 0.323 | 0.500 | 0.0278 | 0.0309 |
| **U-Net +texture** | **0.469** | 0.319 | 0.541 | 0.0272 | 0.0292 |

- The U-Net **does** carry more texture than harmonic (0.44–0.47 vs 0.26). An
  earlier "harmonic wins" conclusion came from pointwise metrics plus a diluted
  distributional one, and is withdrawn.
- The texture loss buys **+7%** roughness for free — pointwise elevation and
  slope both improved slightly (1.069→1.035, 0.0278→0.0272).
- **Nobody exceeds 0.47.** Over half the true roughness is still missing, so a
  deterministic net plus a distributional loss is *not sufficient*. A sampler is
  required — the same conclusion Li et al. and Zhao et al. reached, now
  reproduced on our data with our metrics.
- Variogram ratio rises with lag (0.32 at 10 m → 0.54 at 80 m), quantifying the
  low-pass behaviour that v0.3 showed qualitatively.
- IDW at 2.07 is *over*-rough: right roughness magnitude, wrong places.

**Three metric bugs preceded these numbers**, each producing a confident and
wrong table first: (1) distributional metrics computed on the whole patch, where
75–90% is identical to truth by construction, so every method scored ~1.0;
(2) missing `net.eval()`, so BatchNorm used single-sample statistics at
inference (elev RMSE 5.55 vs 1.11, variogram ratio 3.05); (3) `radial_psd`
assumes a square array and crashed on a 50×91 irregular-mask bounding box.
Lesson: exercise a metric on a real input before trusting a table from it.

**v0.5 — code review, four bugs, and re-measurement**
(`docs/code_review_2026-08-17.md`, `tests/test_lib_smoke.py`)

| bug | effect on the numbers |
|---|---|
| `fill_harmonic` pinned border-touching hole pixels to the patch mean every sweep, and stopped at a fixed 400 Jacobi sweeps (unconverged) | harmonic scored as flatter than it is |
| eval subset was `flatnonzero(te)[:250]`; the patch array sorts by tile name, i.e. **by latitude** | every aggregate was southern-CONUS-biased — the old "rough" class topped out at 42 m relief, the unbiased one reaches 157 m |
| torch never seeded | same-config runs differed by 0.11 in psd_ratio — larger than the effect being reported |
| `psd_ratio` diluted on stroke-mask bounding boxes (mostly valid pixels, where pred == truth) | all texture numbers flattered toward 1.0 |

Re-measured, 3 seeds, unbiased eval, undiluted metric: **harmonic psd 0.128,
U-Net L1 0.305, U-Net+texture 0.314** (previously reported 0.26/0.44/0.47).
Everyone is smoother than the v0.4 table showed. The texture-loss effect is
small but replicated in sign across both runs; its magnitude is not reliable.
`test_lib_smoke.py` now pins each fixed invariant.

---

## Objective B — can a sampler produce terrain texture the mean cannot?

| # | lib | test | result |
|---|---|---|---|
| v0.6a | `lib/diffusion.py` | unconditional DDPM + RePaint forcing | **FAIL** — the hole free-runs |
| v0.6b | `lib/diffusion.py` | mask-conditioned DDIM + resampling | **PASS on substance** |

**v0.6a failure** (kept because it is diagnostic): psd_ratio 2.90 against a
pre-registered 0.7–1.3 band, elev RMSE 11.3 m vs harmonic 2.03, in-hole
spread 11.9 m, and visible black seams at every rim. Forcing the known region
*between* denoising steps is too weak a signal without RePaint's resampling
loop — the hole generates almost unconditionally.

**v0.6b — three fixes, all genuine bugs rather than tuning:**
1. condition on `(masked input, mask)` as **input channels**, so boundary
   agreement is trained rather than patched on afterwards;
2. score the loss **only inside the hole** — outside, the answer is handed to
   the net as input, so scoring there rewards copying;
3. normalise with **valid-region** statistics at train time to match
   inference (v0.6a used whole-patch at train, valid-region at inference — a
   scale mismatch worth metres on a 214 m-relief patch).

**Results** (60 held-out-tile patches, K=8 draws, `logs/diffusion_eval_s0.csv`,
`figs/fig_diffusion_samples.png`, `figs/fig_sampler_truth_vs_generated.png`)

| | diffusion v0.6b | harmonic | deterministic U-Net (v0.5) |
|---|---|---|---|
| elev RMSE | **1.049** | 2.028 | 1.88 |
| best-of-8 elev | **0.795** | — | — |
| slope RMSE | **0.0333** | 0.0425 | 0.047 |
| slope W1 | **0.0093** | 0.0229 | 0.008 |
| psd_ratio | **0.662** | 0.175 | 0.31 |
| vario 10 m | **0.803** | 0.376 | 0.42 |
| vario 80 m | **0.930** | 0.529 | 0.52–0.64 |
| in-hole spread | 0.43 m | 0 | 0 |

- Both variogram ratios land **inside** the pre-registered 0.7–1.3 band;
  psd_ratio 0.662 is a marginal miss but **2.1x the deterministic ceiling**.
- Not collapsed (spread 0.43 m), and best-of-8 beats the deterministic net.
- **The decisive result: it beats harmonic on the pointwise metrics AND the
  distributional ones at once.** Core insight #2 says that should be hard —
  pointwise error structurally rewards the flattest surface, which harmonic
  provably is — so a textured method normally pays for texture in RMSE. Doing
  both means reconstruction, not decoration.
- Individual draws **continue linear features through the hole**, the exact
  behaviour v0.3 diagnosed as structurally impossible for an L1 objective.
- Per-patch honesty (`fig_sampler_truth_vs_generated.png`): 3 of 5 examples
  are near-indistinguishable (in-hole RMSE 0.20–0.30 m); the other 2 draw a
  *plausible alternative* valley rather than the true one (RMSE 1.9–2.0 m,
  psd 0.53 and 2.46). That is correct sampler behaviour and precisely what
  pointwise metrics punish.

**v0.7 — five proposed improvements, ablated one arm at a time**
(`logs/diffusion_eval_{rerank,residual,vparam,allfix}_s0.csv`)

| config | single psd | single elev | rerank psd | **rerank elev** | rerank vario80 |
|---|---|---|---|---|---|
| baseline (eps, surface) | 0.662 | 1.058 | 0.803 | 1.192 | 0.954 |
| **+ residual over harmonic** | **0.706** | 1.045 | **0.810** | **0.891** | **0.980** |
| + v-parameterisation | 0.516 | 1.335 | 0.723 | 1.374 | 0.971 |
| + residual + v + oversample | 0.620 | 1.003 | 0.802 | 1.210 | 0.963 |

**#1 best-of-K reranking — WORKS, no training.** Pick the draw whose in-hole
variogram matches the surrounding RIM (observed at inference, so this is
method not oracle picking). On the baseline it lifted psd 0.662 -> 0.803 and
put all three texture metrics inside the pre-registered 0.7-1.3 band.

**#2 residual over the harmonic fill — WORKS, and removes the trade-off.**
Generating the departure from the harmonic surface rather than the surface
itself. The structural change matters more than the numbers: on the baseline,
reranking COST elevation accuracy (1.058 -> 1.192) because texture was bought
with fidelity; with the residual it IMPROVES both (1.045 -> 0.891). Once the
net is not re-deriving low frequencies harmonic already solves exactly,
selecting for rim-matched roughness also selects for a better surface.

**#5 v-parameterisation — HURTS, clearly.** Worse on every metric (psd
0.662 -> 0.516, rerank elev 1.192 -> 1.374). Hypothesis, untested: v-prediction
re-weights the loss across timesteps and de-emphasises the low-noise steps
where fine texture is actually resolved; on small mean-removed normalised
patches with a cosine schedule, eps-prediction is already well conditioned.

**#4 texture oversampling — not separable, probably harmful.** Only appears in
the stacked arm. If the arms were additive, allfix would land at ~1.07 rerank
elev; it lands at 1.210, and the 0.14 excess is the best evidence available
that oversampling is not helping either. Single seed, so this is weak.

**#3 uniform-CONUS patches — built, rationale REFUTED.** 8,000 patches from
677 tiles spanning lat 25.2-49.4 (the first attempt gave 107 tiles because
`sorted(by_tile.items())` walks 3DEP names south-to-north and the loop stops
at the target -- the same latitude-ordering bug as review B2, and it means the
EXISTING 6,000-patch training set is drawn from 404 southern-biased tiles).
But uniform sampling has LESS of the failing regime, not more: 37.4% of
patches in the high texture/relief class vs 47.0% for the gage-centred set,
because uniform CONUS adds mountains (relief p90 245 vs 167) which are the
EASY case. Gages sit in valleys and agricultural lowlands. Still worth having
for the lithology probe's spatial coverage -- but not for the reason claimed.

**Best recipe: residual + best-of-K rerank.** elev RMSE **0.891 m** vs
harmonic 2.028 (56% better), psd 0.810, vario 10 m 0.842, vario 80 m 0.980,
slope RMSE 0.0328 vs 0.0425. The first configuration to beat harmonic on
EVERY metric, pointwise and distributional, with no trade.

**Lesson worth keeping: ablate one arm at a time.** A single stacked
"all improvements" run would have shown a modest gain over baseline and all
three changes would have shipped -- when two of them are actively harmful on
top of the third.

Also killed cheaply this round: the 0.5 m `STD_FLOOR` (binds on 2 of 60
patches) and more DDIM steps (50/100/250 flat at psd 0.56/0.51/0.54). The
inconsistency was never under-denoising.

Status: **left here deliberately.** The goal was to show the component
pretrains; refinement (more steps, larger net, stroke/water masks, uniform-
CONUS patches) is deferred.

---

## Core insights

**1. L1/L2 is a low-pass filter whose cutoff is set by positional
predictability.** A pixelwise loss yields the conditional mean/median. Long
wavelengths survive because plausible continuations agree on where the valley
is; a 30 m incision does not, because the continuations disagree by more than
its own width, so at any pixel it is absent from most of them and the median
erases it. **Not** a capacity or epoch problem — the objective has to change.
This is why the DEM literature is generative: Li et al. RSE 2021 (TKCGAN) add
valley/ridge loss terms; Zhao et al. RSE 2024 (TFDM) use diffusion.

**2. Pointwise slope RMSE structurally favours smooth fills, so it cannot test
what we care about.** Harmonic minimises ∫|∇z|², i.e. it is the provably
flattest consistent surface — it had the *largest* flattening bias (−0.0080)
and the *best* pointwise score, while IDW had nearly the right roughness
(−0.0029) and the worst score, because its roughness sat in the wrong places.
A pointwise metric penalises misplaced texture twice. Judge texture
distributionally: **semivariogram (autocorrelation), radial PSD, slope
Wasserstein**.

**3. Architecture was never the issue.** Diffusion denoisers *are* U-Nets; so is
the TKCGAN generator. Only the objective differs. **Confirmed by v0.6**: the
same U-Net shape that plateaued at psd 0.31 under L1 reaches 0.66 under a
diffusion objective. The objective was the whole story.

**3b. Conditioning must be an INPUT, not a post-hoc correction.** v0.6a
supplied the known region by overwriting it between denoising steps and
failed completely (psd 2.90, rim seams). v0.6b fed `(masked input, mask)` as
channels and passed. Same net, same data, same schedule — only where the
context enters. Generalises: **if a model must respect a constraint, train it
with the constraint visible rather than enforcing it at inference.**

**3c. A metric that can go wrong in BOTH directions is worth more than one
that cannot.** psd_ratio caught the deterministic nets being too smooth
(0.31) and the unconditional sampler being too rough (2.90). A one-sided
"more texture is better" score would have called v0.6a a triumph.

**4. MAE is not gated by any of this.** MAE trains with MSE and its
reconstructions are famously blurry — He et al. discard the decoder. Its gate is
downstream transfer (frozen-encoder probes), never reconstruction fidelity. An
earlier draft of this plan gated it wrongly.

**5. Carried from Stage I, established by measurement, do not re-litigate:**
absolute elevation is a location fingerprint and hurts (dropping `MEANELEVSMO`
*gained* +0.003 R²; StefaLand embeddings reconstruct elevation at R² 0.936 —
**corrected 08-20: that is not learned memorisation, `GMTED_elevation` is one of
its static INPUT variables, so de-locating is a one-line ablation**) — so
patch-mean removal is mandatory. Leave-region-out always (Al Mehedi's NSE 0.71
→ 0.058 once gages were held out). A learned embedding encoding real physics can
still add nothing if simpler columns already carry it.

---

## Data

| dataset | location | status |
|---|---|---|
| 3DEP 1/3″ (~10 m) | `prd-tnm` S3, streamed via `/vsicurl/` | working |
| DEM patches | suntzu `dem_foundation/logs/dem_patches.npz` | 6,000 patches |
| **USGS SGMC** | suntzu `/nfs/data/cxs1024/sgmc/` | downloaded |
| StefaLand embeddings | ICDS `.../shared_model_code/EmbeddingDatasets/` | 5,057 gages extracted |
| **StefaLand code** | ICDS `/storage/group/cxs1024/default/shared_model_code/MFFormer` | inspected 08-20 |
| **StefaLand weights** | ICDS `.../shared_model_code/FoundationModels/StefalandOriginalGlobal20.pt` (+ `.yaml`) | 57 MB, self-contained |

**StefaLand, as inspected** (`ssh icds`, needs an interactive MFA login first).
Config: `hidden_size 256, 4 heads, 4 enc layers, d_ffd 512` — a small model.
Inputs: 5 time-series vars (`P, RelHum, SWd, Tmax, Tmin`) + ~49 statics
(GMTED terrain, HWSD/SoilGrid2 soils, MSWX climate, land-cover fractions,
`catchsize`, porosity, soil_depth). **No channel geometry of any kind.**

Already in that repo, reuse rather than rebuild:
- `eval_probe.py` + `probe_configs/camels_spatial.yaml` — RankMe + linear probe,
  **leave-region-out by construction** (80 train / 20 held-out basins), already
  probing `geol_1st_class` (GLiM geology). This is the U3 gate, prebuilt.
- an image pathway: `models/StefaLand_withImage_PatchTokens.py`,
  `models/image_encoders/`, `data_provider/zarr_image_loader.py`,
  `Image_guide.txt`. DEM patches should plug in there.

Nicholas (08-20): the yaml's `pretrained_model:` path is a stale naming
artefact — ignore it; load directly from the checkpoint via the
resume-checkpoint path. Run dirs go to scratch; indexing files are not saved
but regenerate from the repo script.
| gNATSGO, Pelletier depth-to-bedrock | — | to obtain; Pelletier is a *modelled* product, prefer field observations |

**SGMC** (Horton 2017 v1.1, 48 states, 1:50k–1:1M; a 2026 GeMS re-release exists
at doi:10.5066/P1A3DQZK). Joins on `UNIT_LINK`:

| table | rows | use |
|---|---|---|
| `SGMC_Units.csv` | 6,644 | `UNITDESC` free text → narration corpus |
| `SGMC_Lithology.csv` | 18,426 | `LITH1..LITH5` hierarchy → classification probe |
| `SGMC_Age.csv` | 6,624 | `MIN_MA`/`MAX_MA` → regression probe |

Field-mapped, so it avoids the modelled-label circularity. Caveats: geologists
partly map units *using* topographic expression, and scale varies 20× across
states — carry as a per-polygon confidence weight.

---

## Environment

suntzu `/nfs/data/cxs1024/dem_foundation/`. Torch work uses conda `pytorch_gpu`
and **requires** the `nvjitlink` `LD_LIBRARY_PATH` fix plus
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (without it `CUDA_VISIBLE_DEVICES=2` lands on an
11 GB 2080 Ti, not the 24 GB 3090 Ti). Rasterio work uses the `demenv` venv,
which has no torch.
