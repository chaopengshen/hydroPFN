# Proposal seed — an in-context foundation model for hydrologic prediction
## ("a TabPFN of hydrology")

Drafted 2026-08-19 from working discussion; numbers cite measurements from the
Stage-I channel-geometry and DEM-foundation sessions (repo: `channel_geometry/`,
`dem_foundation/`). This is a seed, not a proposal — the validation ladder at
the end is deliberately ordered so the idea can die cheaply.

---

## 1. Vision

One pretrained model that, given a **context set** of measured sites and a
**query** site, emits a predictive distribution for any requested hydrologic
variable at the query — no per-task, per-region, or per-variable retraining.

```
context:  (x_1, var_1, y_1) ... (x_n, var_n, y_n)     gauged/measured sites
query:    (x_q, var_q)                                 attributes + requested variable
output:   p(y_q | query, context)                      full distribution, not a point
```

`x` mixes modalities: tabular attributes, a DEM patch (via a terrain encoder),
network position. `var` is a variable-ID token, so context rows can carry
*different* observables (width here, depth there, a calibrated parameter
elsewhere). Prediction in ungauged basins (PUB) is the native mode: the query
never has a y — the context supplies the mapping, the prior supplies how to
use it, and zero context degrades gracefully to climatology-given-x.

The product is **amortization**: regionalization — currently one trained model
per variable per region — becomes a forward pass. TabPFN (Hollmann et al.)
demonstrates the recipe works for generic tabular tasks; the claim here is that
hydrology is unusually well suited to a domain version, for reasons in §3.

## 2. What in-context learning buys that a train-once model cannot

Ranked by how specifically each requires conditioning on fresh data at
inference time (1–4 are impossible, not merely harder, for a per-variable RF):

1. **At-a-station ratings as amortized partial pooling.** Context = a site's
   own few (Q, y) measurements; query = new Q at that site. No functional form
   imposed (per-site power-law fits are noise-limited: held-out per-site R²
   0.121, and forcing the exponential form costs ~0.06 R² vs a free model). A
   3-measurement site shrinks toward the prior; a 40-measurement site follows
   its own data. PFNs approximate the posterior predictive — this is
   hierarchical Bayes without the MCMC.
2. **Few-shot site updating.** Two field measurements at an otherwise ungauged
   reach enter the context and locally correct the prediction — no
   recalibration. Stage-II concrete case: one geometry measurement at tracer
   flow should sharply narrow W(Q), d(Q) at other flows.
3. **Cross-variable inference from mismatched measurements.** Context mixes
   variable types; query asks for depth at a site where only width was ever
   measured. This is the real structure of HYDRoSWOT (width 89% populated,
   mean depth 26%) and of essentially every field dataset. Cheap feasibility
   test defined in §5 (T2). Note the trap: at the *same* visit, W and d are
   algebraically linked through Q (continuity is an identity in these data),
   so the honest benefit is cross-visit, site-level shape information.
4. **Monitoring-network design as a by-product.** Conditioning is
   retraining-free, so "where does the next gauge go" = tentatively add each
   candidate to the context and measure predictive-variance reduction over the
   region. Value-of-information analysis as a loop over forward passes.
5. **Parameter regionalization with honest uncertainty.** Context = (basin
   attributes → calibrated dHBV parameters); query = uncalibrated basin.
   Parameters are unobservable, so no tabular shortcut exists by definition,
   and the output distribution propagates into ensemble simulation.
6. **Change detection (speculative).** Condition on a site's pre-period
   rating; systematic post-period residuals flag geomorphic change against a
   null that already accounts for regional behaviour.

Mechanics clarifications settled in discussion:
- Context sites can be spatial neighbours; coordinates/network position in x
  let attention learn distance decay (kriging-like behaviour, learned).
- Missing x-features in context or query are handled by per-cell masking
  (TabPFN v2 already accepts NaNs).
- A context row with **no y at all is dead weight** in the standard PFN
  formulation — unlabeled rows help only in semi-supervised extensions, which
  we should not promise.

## 3. Why hydrology specifically (the proposal's differentiators)

1. **The synthetic prior writes itself.** TabPFN's engine is millions of
   synthetic tasks from random structural causal models. We own *physical*
   task generators: dHBV ensembles, the dMC routing model, 59-zone global
   simulations, hydraulic-geometry relations with realistic noise and
   measurement-operator corruption. Sampling pretraining tasks from perturbed
   simulators is the hydrologic analogue — better grounded than random SCMs.
2. **Modalities TabPFN cannot ingest.** A DEM-patch encoder (the
   `dem_foundation` line of work) becomes the terrain tokenizer; network
   topology and coordinates enter the attention. Off-CONUS there is no
   NHDPlus attribute table, so a DEM-native x is what makes the model global.
3. **Spatial transfer is the field's actual failure mode, and ICL targets
   it.** Our leave-HUC2-out penalty is measured (width −0.039, depth −0.012
   vs random CV; Al Mehedi's NSE 0.71 collapsed to 0.058 under honest
   splits). A per-query context of local sites is a mechanism, not a hope.

## 4. Evidence base already in hand (and the discipline it imposes)

- **TabPFN ties RF** on channel geometry at 5,057 sites out of the box — so
  raw accuracy on attribute-rich tasks is NOT the pitch; amortization,
  uncertainty, and use cases 1–4 are.
- **The redundancy principle** (measured three ways: StefaLand embeddings
  −0.004, network structure ±0.001, cross-section CNN ~0): a representation
  pays only where the task has no tabular shortcut and labels are sparse.
  Target applications accordingly: global/ungauged geometry, SWOT inversion
  priors, bathymetry, Manning's n and transient storage (hundreds of labels),
  dHBV parameters.
- **Geometry residuals are not learnable from terrain** (leave-HUC2-out, all
  predictor sets negative, confirmed under two stage-1 fold structures). So
  "the encoder injects substrate knowledge" is NOT a claim we can make; "the
  encoder infers hydrologic context from bare terrain" is (raw log_d/log_v
  learnable at R² 0.68–0.72). Whether residuals are predictable *from each
  other* across variables is exactly test T2.
- Evaluation discipline from the SGMC work: measure the label-noise ceiling
  before interpreting any score (state-boundary agreement 0.91 at 1.28 km);
  use probes with headroom.

## 5. Validation ladder (each rung cheap, each can kill the next)

- **T1 — at-a-station ICL with stock TabPFN.** Context = site's own
  measurements + attribute-matched neighbours; query = held-out measurements
  at that site. Beat both the per-site power law (0.121) and the pooled RF?
  One script, no new model.
- **T2 — cross-variable benefit (use case 3), stock RF.** Does knowing a
  site's width anomaly (from *other* visits, same-row excluded to dodge the
  continuity identity) improve depth/velocity prediction under leave-HUC2-out?
  Implemented as `channel_geometry/src/test_cross_variable_context.py`.
  **RUN 2026-08-19 — POSITIVE.** On 23,740 rows / 2,925 sites:
  corr(w_anom, residual) = −0.250 (d), −0.368 (v); R² gain +0.017 (d),
  **+0.041 (v)** — the largest additive gain measured in this project (the
  whole DEM feature campaign gave +0.015; embeddings and network structure 0).
  The velocity gain survives (+0.037) on the subset whose depth rows have NO
  same-visit width (n=2,298), so it is not the continuity identity leaking;
  the depth gain does not survive there (+0.003). Honest claim: width context
  improves velocity strongly and depth weakly; a site anomalously wide for its
  attributes is first anomalously slow, second shallow. Closes the loop with
  the residual gate: site residuals are unlearnable from any x but predictable
  from each other — information that only in-context conditioning can use.
- **T3 — encoder probes (E0/E1/E2).** Does auxiliary geometry supervision
  produce a terrain encoder that transfers better downstream? Requires the
  fixed downstream probe (age regression, uniform CONUS sampling) first.
- **T4 — prototype hydro-PFN** on synthetic tasks from one simulator family,
  evaluated on real PUB splits. Only if T1–T3 justify it.

## 6. Proposed architecture (HydroPFN), 2026-08-19

**Design principle: a context point is a SITE, not a row** — a bundle of
tokens from weight-shared per-modality encoders; a PFN-style transformer then
attends across sites.

```
        per-site encoder (weights shared by all sites, context and query)
  +-------------------------------------------------------------------+
  | site i                                                            |
  |  tabular attrs ---- MLP -----------------------------> t_attr     |
  |  DEM patch -------- terrain encoder (E0/E1 line) ----> t_dem      |
  |  measurements ----- each visit j = [var-ID emb (+) value emb (+)  |
  |   {(var,val,Q,t)_j}   covariate emb] -- within-site  -> t_meas1..k|
  |                       transformer, attention-pooled               |
  |  time series ------ FDC quantiles / temporal patches -> t_ts      |
  |  lat,lon (+network) - Fourier geo-encoding added to every token   |
  +-------------------------------------------------------------------+
                     |  4-8 tokens per site; absent modality = absent token
                     v
     cross-site transformer (PFN core)
       - no positional order across sites (permutation-invariant set)
       - query tokens attend to context; context never attends to query
                     v
     [TASK] token: var-ID + covariates (e.g. "log_d at Q=3.2"), value masked
                     v
     distributional head (discretized Riemann / quantile bins -> full p(y))
```

**Unifying pretraining objective: masked-measurement modeling.** Sample a set
of sites; mask random measurement values (keep var-ID + covariates); predict
them from everything else. This single objective yields all modes: ICL with
context (other sites' values visible), zero-context "only x" prediction (no
context sites -> t_attr + t_dem carry everything; identical to a supervised
head), cross-variable transfer (a site's width visits visible while its depth
is queried — the T2 effect, learned not engineered), and at-a-station ratings
(the site's own visits are the context).

**Time series in context.** Short irregular records (the 6–40 (Q,W,d) visits)
stay as individual tokens — a summary would destroy exactly the per-visit
information T1/T2 exploit. Long regular series (daily Q) are compressed first:
v1 uses flow-duration-curve quantiles + seasonal statistics as a token set
(exchangeable, magnitude-complete, free); a learned temporal patch encoder
(PatchTST-style) is the v2 upgrade if FDC summaries prove lossy.

**DEM linkage, two routes.** Hand-extracted profile features (`dem_own_*`,
already computed CONUS-wide) enter t_attr as plain columns. The learned route
plugs the dem_foundation encoder in as t_dem — making E0/E1/E2 the component
qualification test for this slot. Off-CONUS, t_dem + forcings ARE the x.

**Deliberate omissions.** No GNN backbone: network structure measured at
±0.001 twice; topology enters only as relative features (upstream flags,
network distance) in the geo-encoding. No adversarial objectives.

**Sizing.** 200–500 retrieved context sites x ~5 tokens = 1–2.5k tokens; a
~10–30M-parameter transformer beyond the DEM encoder; single 3090 Ti class.
Context retrieval (top-k by attribute/geo similarity) is part of the method.

**v1 acceptance tests, fixed in advance:** reproduce the T2 gain (>= +0.04
velocity from width context) with NO hand-built anomaly feature, and beat the
per-site power law (0.121) on T1 — else the architecture is not earning its
complexity over an afternoon of feature engineering.

## 6b. Expanded design: pretrained units, connected (2026-08-20)

The fuller ambition: context sites carry **long forcing sequences plus several
observation series** (streamflow, soil moisture, ...; any subset missing), and
pretraining is jointly (i) predicting masked observation series from forcings
and context, (ii) completing masked static attributes, (iii) reconstructing
masked DEM, (iv) predicting masked point measurements (the §6 objective).
Units are pretrained individually, then connected. This makes sense and has
strong precedent — it is StefaLand's per-site recipe (location-aware masked
autoencoder fusing static + time-series inputs, arXiv:2509.17942) extended
with the two things it does not have: **cross-site in-context inference** and
**DEM pixels as a modality**.

Units and their pretraining status:

| unit | input | pretraining | source |
|---|---|---|---|
| static-landscape | attribute vectors | masked-attribute MAE | **StefaLand, reuse weights + residual adapters** |
| terrain | DEM patch | masked reconstruction + geometry aux head | this project (E0/E1 line) |
| forcing-response | weather series + obs series (var-ID per channel) | masked obs-series prediction | new; the learned temporal encoder that supersedes the FDC-only v1 when series are TARGETS, not just context (FDC tokens stay as cheap summaries) |
| measurement | (var, value, Q, t) visits | masked-value prediction | §6 |
| connector | all unit tokens across sites | all objectives, task-sampled | §6 cross-site core |

**StefaLand adoption — value and required modifications.** Measured facts
from our sessions: its embeddings added −0.004 to channel geometry on top of
20 tabular attributes (redundancy, expected) and reconstruct absolute
elevation at R² 0.936 (a location fingerprint). Its "location-aware" design
is a feature for its within-domain tasks and a hazard for spatial transfer —
so adopt the unit but (a) evaluate under leave-region-out before trusting it,
(b) add a de-location variant (adversarially or by conditioning out
coordinates) for transfer-critical tasks, (c) pair it with the DEM-pixel
encoder, which covers the global case where StefaLand's input attribute
table does not exist. Its residual-adapter mechanism is exactly the right
tool for the connection phase; its downstream suite (streamflow, soil
moisture, soil composition, landslides) doubles as our benchmark set.

**Cost, order-of-magnitude (single 3090-class GPU unless noted):**

| phase | estimate |
|---|---|
| terrain unit pretrain (100k patches) | 1–2 GPU-days |
| forcing-response unit (~10k site-decades, masked-series) | 2–5 GPU-days |
| static unit | ~0 (reuse StefaLand) |
| connection phase (30–100M params, task-sampled batches) | 1–2 GPU-weeks; days on 4×A100 |
| data engineering (forcing/obs alignment, missingness handling) | the real cost — weeks of person-time |

Total compute is academic-scale (~2–4 GPU-weeks single-GPU equivalent). The
scaling guardrail: **cross-site attention operates on site-summary tokens
only** (4–10 per site), never raw timesteps — daily sequences are consumed
inside the forcing-response unit. Missing-modality dropout during training is
mandatory so any subset of inputs works at inference. The mandatory bar:
beat regional LSTM / dHBV baselines on their own turf before claiming the
connected model earns its complexity.

**Compute gating (agreed 2026-08-20).** An 8×A100 machine is available but is
engaged ONLY after small-scale value is confirmed on single-3090-class runs:
T2 (done, positive), the sampler verdict against its pre-registered bands,
T1, the E0/E1/E2 encoder probes, a **mini-HydroPFN v1** (measurement unit +
cross-site core on masked HYDRoSWOT, ~10M params) that must reproduce the T2
gain with no hand-engineered feature and beat the per-site power law, and a
streamflow-only forcing-response unit vs a regional LSTM. The A100s fund the
connection phase at scale, not the search for signal.

## 7. Risks, stated plainly

- At attribute-rich sites ICL only ties trained models; the value is
  concentrated in transfer and in use cases no baseline can attempt — the
  proposal must be written around those, not around benchmark wins.
- Context-window economics: thousands of context sites × mixed modalities
  strains attention budgets; retrieval (choose the right 200 context sites)
  becomes part of the method.
- Simulator-prior mismatch: a PFN pretrained on dHBV-world may inherit its
  biases; the corpus needs deliberate diversity (multiple simulators,
  perturbed physics, real-data fine-tuning).
- The continuity identity and its relatives can make cross-variable skill
  look better than it transfers (§2.3 trap) — every test must state which
  algebraic links are broken by construction.
