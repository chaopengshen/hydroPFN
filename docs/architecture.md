# HydroPFN — incremental build plan and unit-onboarding template

Companion to [`PROPOSAL_SEED_hydrology_PFN.md`](PROPOSAL_SEED_hydrology_PFN.md).
That doc argues *what* and *why*; this one is *how*, in the order things get
built, with a pass/fail gate on every piece so a dead unit is discovered in
days rather than after the connection phase.

The organising rule, learned the expensive way this month: **every claim needs
a baseline it must beat and a noise ceiling it cannot exceed, fixed BEFORE the
run.** Three times a promising number turned out to be a metric artefact
(whole-patch dilution, southern-biased eval subset, unseeded torch). Units are
gated the same way.

## Naming (decided 2026-08-20): it is not a PFN

"PFN" means *prior-data fitted* — pretrained on samples from a **synthetic
prior** to amortise Bayesian inference. That is TabPFN's defining property.
Pretraining on real masked data makes this something else, and the accurate
lineage is:

| our component | closest established work |
|---|---|
| context set → predict at query | **Neural Process** family, esp. **Transformer Neural Process (TNP)**; TabPFN is a TNP with a synthetic prior |
| masked multimodal pretraining | **MAE**; in geoscience **StefaLand**, Prithvi, Clay |
| long series, many variates, missing channels | **Moirai** (any-variate, closest on missingness), **Chronos**, **TimesFM**, **PatchTST / Ti-MAE** for patch masking |

Honest one-liner: **a multimodal masked autoencoder whose prediction is
conditioned on a retrieved context set of sites — a Transformer Neural
Process over sites, with a Moirai/PatchTST-style temporal unit.** "PFN"
becomes defensible only if the simulator-generated task prior (proposal §3)
is actually built, and even then it is a hybrid.

Working name: **StefaNP** — sibling to StefaLand (which it extends with
cross-site conditioning and DEM pixels), and honest about the mechanism.
`HydroTNP` if mechanism-accuracy is preferred over the family link. Code
still says `hydropfn` pending a rename.

## Known data defect (2026-08-20)

`train_table_dem.csv`: `site_no` is NaN on **39,141 of 64,797 rows** (HUC2 is
complete). All 4,279 sites survive but ~60% of visits cannot be attributed,
so Phase 1 trains on 25,656 visits. A join defect worth fixing upstream —
it does not invalidate T2 (which grouped by `site_no` and so used the same
attributed subset) but it costs more than half the training signal.

---

## The unit template (also: how to add any future dataset)

Every modality — DEM, streamflow series, soil moisture, remote sensing,
water quality, anything — is onboarded through the same five steps. Steps
U0–U3 are runnable **without touching the connected model**, which is the
whole point: a dataset proves itself in isolation and cheaply.

| step | question | artefact | gate |
|---|---|---|---|
| **U0 adapter** | can the raw data become `(site, time, var, value, covariates)` records? | `adapters/<name>.py` | round-trips; missingness explicit, never imputed silently |
| **U1 encoder** | can it become a small set of tokens? | `lib/units/<name>.py` | fixed token count; handles absent modality; permutation-invariant where the data are exchangeable |
| **U2 objective** | can it be self-supervised standalone? | masked-value / masked-reconstruction loss | loss decreases on held-out **regions**, not just rows |
| **U3 standalone probe** | does the representation beat a simple baseline on ITS OWN task? | frozen-encoder linear/RF probe | **beat a named baseline under leave-region-out, with the label-noise ceiling measured first** |
| **U4 connection + ablation** | does it add anything the other units do not already carry? | plug into the core; then drop it | must move a downstream metric; if dropping it changes nothing, it does not ship |

U4 exists because of the redundancy principle we measured three separate
times (StefaLand embeddings −0.004, network structure ±0.001, cross-section
CNN ~0): **a unit can encode real physics and still add nothing** if simpler
inputs already carry it. Passing U3 is not permission to ship; U4 is.

---

## Build order

Each phase runs on one 3090-class GPU. The 8×A100 machine is engaged only
after Phase 3 passes (see the compute-gating note in the proposal).

### Phase 0 — evidence, no model (done / in flight)
- **T2 cross-variable** — DONE, positive: +0.041 R² velocity, +0.017 depth
  from a site's cross-visit width anomaly, leave-HUC2-out. This is the
  effect the core must reproduce *without* the hand-built feature.
- **T1 at-a-station ICL** (stock TabPFN) — DONE, positive. `tabpfn_ctx`
  (own rows + 30 attribute-nearest sites) wins on every metric: median-site
  R² 0.795 vs pooled-RF 0.708 (depth) and 0.825 vs 0.769 (velocity).
  `tabpfn_nbr` (neighbours only) is far worse (0.19 / 0.30), so the gain is
  partial pooling from a site's OWN measurements, not neighbour smoothing.
- **Sampler** — DONE. v0 (unconditional + RePaint forcing) FAILED its bands:
  psd 2.90 against 0.7–1.3, elev RMSE 11.3 vs harmonic 2.03, visible rim
  seams — the hole free-ran. **v1 (mask-conditioned, hole-only loss,
  resampling, valid-region normalisation at train time) PASSES the
  substance:**

  | | diffusion v1 | harmonic | deterministic U-Net |
  |---|---|---|---|
  | elev RMSE | **1.049** | 2.028 | 1.88 |
  | best-of-8 elev | **0.795** | — | — |
  | slope RMSE | **0.0333** | 0.0425 | 0.047 |
  | slope W1 | **0.0093** | 0.0229 | 0.008 |
  | psd_ratio | **0.662** | 0.175 | 0.31 |
  | vario 10 m | **0.803** | 0.376 | 0.42 |
  | vario 80 m | **0.930** | 0.529 | 0.52–0.64 |
  | in-hole spread | 0.429 m | 0 | 0 |

  Two of three texture metrics land INSIDE the 0.7–1.3 band; psd_ratio 0.662
  is a marginal miss but 2.1x the deterministic ceiling. It is not collapsed
  (spread 0.43 m) and best-of-8 beats the deterministic net outright.
  **The decisive point: it beats harmonic on the POINTWISE metrics and the
  distributional ones simultaneously** — which core insight #2 says should be
  hard, because pointwise error structurally rewards the flattest surface.
  Doing both means it is reconstructing, not decorating. Individual draws
  continue linear features through the hole (see
  `figs/fig_diffusion_samples.png`, flat row), the exact failure v0.3
  diagnosed as structural for an L1 objective.

  This retires the standing "a sampler is required" conclusion: it was
  required, and it works.

### Phase 1 — measurement unit + cross-site core (mini-HydroPFN v1) ← DECISIVE
Smallest thing that is genuinely a PFN: site attributes + per-visit
measurement tokens, two-level attention, masked-value pretraining, bar
(discretised) distributional head. Data: HYDRoSWOT via
`train_table_dem.csv` — 64,797 visits, 5,057 sites, three variables
(`log_W`, `log_d`, `log_v`) with `log_Q` as covariate.

**Acceptance (fixed in advance):**
1. **Reproduce T2**: predicting `log_v` at a site whose context holds only
   `log_W` visits must beat the attributes-only model by ≥ +0.03 R²
   (hand-built feature got +0.041) — with no anomaly feature anywhere.
2. **Beat the power law on T1**: median per-site R² above the per-site
   power-law fit on held-out visits.
3. **Degrade gracefully**: with zero context tokens it must still match an
   attributes-only RF (this is the "only x" mode).

Fail any of these and the architecture does not earn its complexity over an
afternoon of feature engineering — that is the honest exit.

**RESULT 2026-08-20** (1.2M params, 60 ep x 300 steps, holdout HUC2 03,
`logs/hydropfn_v1_s0.csv`). Numbers below are AFTER fixing a contamination
found in the first run — see the note beneath the table.

| var | own | crossvar | none | RF | powerlaw |
|---|---|---|---|---|---|
| log_W | 0.8124 | 0.7852 | 0.5968 | 0.6714 | 0.7309 |
| log_d | 0.7493 | 0.7331 | 0.6637 | 0.6507 | 0.5821 |
| log_v | 0.8651 | 0.8425 | 0.7350 | 0.7767 | 0.7842 |

- **A1 PASS, +0.1075** on log_v — 3.6x the gate and **2.6x the hand-built RF
  feature (+0.041)**. The learned model extracts more from the same
  information than the engineered anomaly did, which is the architecture's
  central justification.
- **A2 PASS x3**: +0.081 / +0.167 / +0.081 over the per-site power law.
- **A3 FAIL on 2 of 3**: with zero own-site context the model is worse than
  an attributes-only RF on width (−0.075) and velocity (−0.042). This mode
  still has 16 neighbour sites, so it is not even a pure-attribute handicap.
  **The zero-context floor is Phase 1's real weakness**, and it matters
  precisely for the global/ungauged deployment mode. Fixes in expected-value
  order: recover the ~60% of visits lost to the `site_no` join defect, add
  capacity/epochs, weight the curriculum harder toward the no-context regime.

**REPLICATED 2026-08-21 — 3 holdouts (03/07/17) x 3 seeds, on the
site_no-repaired table.** This supersedes the single-run numbers above.

| gate | median | min | max | pass |
|---|---|---|---|---|
| A1 cross-variable (log_v) | **+0.1265** | +0.0965 | +0.3029 | **9/9** |
| A2 at-a-station vs power law | +0.0967 | −0.0072 | +0.2727 | 24/27 |
| A3 zero-context vs attributes RF | +0.0043 | −0.0601 | +0.0451 | 21/27 |

- **A1 is robust**: 9/9, and even the WORST run (+0.0965) is 3.2x the gate and
  2.3x the hand-built RF feature. Region-dependent but always positive
  (holdout 03 +0.100, 07 +0.127, 17 +0.270).
- **A3 is no longer a clear failure.** The single-run "FAIL on 2 of 3" was
  substantially seed noise plus the missing data: with repaired labels and 9
  runs it sits at parity with the RF (median +0.004, 21/27). Honest status:
  the zero-context mode *matches* a tuned RF, it does not beat it.
- A2 fails 3 of 27, all on `log_W`, where the power law is strongest
  (median A2 by variable: log_W +0.029, log_d +0.167, log_v +0.094).

**Context-scaling curve** (`figs/fig_context_scaling.png`,
`logs/context_scaling.csv`, holdout 17, no retraining):

- **own-site visits 0 -> 8: monotone rise, then saturation.** log_v
  0.575 -> 0.906, log_d 0.718 -> 0.897, log_W 0.671 -> 0.877. This is the
  in-context claim demonstrated directly rather than inferred from a gate.
- **neighbour sites: a STEP, not a curve.** log_v jumps -0.145 -> 0.582 on the
  FIRST neighbour then is flat to 16. One neighbour supplies calibration; the
  model does **not** aggregate across many. Likely because training always
  used a fixed `n_ctx=16`, so variable-sized context was never learned —
  randomising context size during training is the obvious fix and is untested.

**Contamination found and fixed.** The first run reported A1 +0.1407 and
A2 +0.168/+0.264/+0.131 — all inflated. Each dataset row yields up to three
visits (`W`, `d`, `v`) sharing one `log_Q`, and `W*d*v = Q` is an identity
here, so leaving a target's same-occasion siblings in context let the model
do arithmetic. The tell was that **`crossvar` beat `own` on all three
variables**, impossible for honest inference since `own`'s context is a
superset. Excluding the whole occasion (as T2 already did with same-row
exclusion) restored `own > crossvar` everywhere and dropped every number.
Lesson for every future unit: **when variables are algebraically linked,
exclude the measurement occasion, not the measurement** — and watch for
ordering violations as the cheap tell.

### Phase 2 — terrain unit (E0/E1/E2)
Already scaffolded. Needs the uniform-CONUS patch pull and a probe with
headroom (age regression, R² 0.367; lithology at 0.593-vs-0.516 majority
cannot resolve anything). U3 gate: encoder beats hand-engineered terrain
descriptors. U4 gate: `t_dem` moves a Phase-1 metric.

### Phase 3 — forcing-response (temporal) unit

**Pretraining is masked-patch reconstruction with a four-way mask curriculum,
and each mask type IS a downstream capability:**

| mask | hide | learn from | capability |
|---|---|---|---|
| random span | contiguous chunks of an observation series | forcings + surrounding obs | gap-filling |
| causal tail | the future | past only (**the only causal mask** — the others must stay non-causal or interpolation skill is lost) | forecasting |
| whole-variable | ALL of one series (e.g. soil moisture) | forcings + the site's *other* series | **cross-variable inference — the T2 effect in time-series form** |
| whole-site | every observation at a site | forcings + attributes + *neighbouring sites'* obs | **PUB; the only mask that requires cross-site attention** |

Mechanics: daily series → ~16-day patches → tokens, channel-independent with
a variable-ID embedding per channel plus day-of-year features; bar-
distribution loss on masked patches. The unit emits **both** reconstructions
(its own objective) and a few pooled summary tokens — the core consumes only
the summaries, never raw timesteps.

Long records: 40 y daily ≈ 900 patches/variable, too many to pass upward.
Sample multi-year windows in training; summarise the full record with
FDC/seasonal-statistic tokens beside pooled window tokens. (This is where FDC
quantiles belong — as the cheap always-available summary, not as the encoder.)

**U3 gate:** masked-span streamflow reconstruction must beat a regional LSTM
on the same basins under leave-region-out. Only after this does the A100
connection phase start.

### Phase 4 — connection at scale
All units, task-sampled batches, missing-modality dropout, residual adapters.
StefaLand (arXiv:2509.17942) enters here as the pretrained static unit, with
the de-location caveats recorded in the proposal.

---

## Standing rules for every phase

- Splits by **site**, never by COMID; leave-region-out is the headline number.
- Every comparison on an **identical row/site population** — the v3-vs-v4
  confound (+0.090 apparent, +0.024 real) came from forgetting this.
- Report **seeds**, not single runs: same-config spread reached 0.11 in
  psd_ratio, larger than the effect being claimed.
- A metric gets exercised on a known input before any table built from it is
  believed.
