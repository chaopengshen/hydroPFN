# The CAMELS-531 protocol — what we now run, and why the old numbers went

Every CAMELS number this project produced before 2026-08-28 used a protocol of
its own invention and was compared against published values that used a
different one. This page defines the protocol we now run, states exactly what
changed, and records which old numbers are dead.

Verify before quoting anything:

```bash
python scripts/verify_camels531.py
```

That script does not reason about the protocol. It reads the observation
series from a **completed dmg_dev run** and checks ours against it — basins,
order, days and scale at once. Current status: **PROTOCOL VERIFIED**, 1461-day
window identical, basin match a bijection at r = 1.0000000000, target values
identical to dmg_dev's (offset 0.000).

---

## The protocol

Read off dmg_dev's configs, not reconstructed from a paper:
`conf/Extended1980_1999_531/LSTMPUB.yaml`, `LSTMPUR.yaml`, and `conf/lstm.yaml`.

| | |
|---|---|
| basins | the 531-basin Newman/Addor subset (`531sub_id.txt`) |
| PUB | 10 random groups, leave-one-group-out (`PUB_ID` 1–10) |
| PUR | 7 contiguous regions, leave-one-region-out |
| spatial period | train 1980-10-01 … 1999-09-30, score 1995-10-01 … 1999-09-30 |
| temporal period | train 1980-10-01 … 1995-09-30, score 1995-10-01 … 2010-09-30 |
| warmup | 365 days of spin-up before the scored period, never scored |
| sequence | rho 365 + warmup 365 = 730-day training sequences |
| metric | NSE on **raw mm/day**, per basin, median over basins |
| forcings | Daymet — prcp, srad, tmax, tmin, vp |
| statics | the same 26 attributes dmg_dev's embedding configs use |

Fold sizes, printed by every run and checked to be disjoint and to cover all
531:

```
PUB  10 folds  [54, 48, 45, 46, 57, 60, 44, 60, 61, 56]
PUR   7 folds  [92, 95, 93, 51, 68, 60, 72]
```

**The scored period overlaps the training period in PUB and PUR.** That is not
a leak: the held-out basins are disjoint from the training basins, which is
the whole point of a spatial holdout, and it is what Feng et al. (2021, 2023)
and Li et al. (2025) do. The temporal protocol is the one that separates
periods.

**The PUR region labels are wrong and we did not fix them.** The `huc` column
of `gages_list_with_pub.csv` disagrees with the USGS part number of the
station beside it for 467 of the 531 basins, so `test.huc_regions` names
groups that are not the groups it builds. The blocks are still disjoint, still
cover all 531, and each still occupies a coherent longitude band, so this is a
valid regional holdout — only the labels lie. Region 5 is Colorado / Great
Basin / Oregon and region 6 is Pacific NW + California. Changing the CSV would
invalidate every trained split that already depends on it.

**Join on gage id, never on position.** Aggregated arrays are holdout blocks
concatenated, so column *j* is not basin *j* of the subset file — verified
here: dmg_dev's PUB and PUR target arrays are different permutations of the
same 531 basins, and neither is in subset order. Every run in this repository
writes `gage.npy` and `fold.npy` beside its predictions for exactly this
reason.

**Basin alignment checked (2026-09-22, `scripts/check_basin_alignment.py`).**
A port of ADW's `check_basin_alignment.py`, prompted by their finding that
arrays from a sorted subset joined to a CSV from an unsorted basin list
mislabelled 439 of 531 basins. The test is physical and comparative: mean
precipitation per basin must track that basin's `p_mean`, and the failure
shows up as some *other* ordering scoring better than the stored one. Ours
passes at all three places it could drift:

| where | stored | sorted-ID | reading |
|---|---|---|---|
| raw netCDF, 671 basins | 0.997 | 0.997 | ids already sorted, nothing to distinguish |
| after `load_531()` | **0.997** | 0.253 | subset is NOT sorted (4 ids out of place), so this is a real test: stored order is right and sorting would destroy it |
| `frac_snow` vs sub-zero precipitation | 0.994 | 0.362 | second, independent signal, same verdict |

The side file (`gages_list_with_pub.csv`, the one thing assembled from a basin
list rather than from the arrays) was checked separately: its rows correspond
to subset rows 531/531 by gage id, its `LAT`/`LONG` match the netCDF
coordinates to 0.0000°, and the PUR folds it defines are spatially coherent
(longitude spread 4.33° vs 15.99° shuffled) while the PUB folds are not, which
is correct — PUB is random by design.

We are safe largely by construction: attributes and forcings are variables on
the same basin dimension of one netCDF, and every join in `protocol.py` is by
gage id (`set_index("gage_int").reindex(want)`), never by position.

---

## What changed, item by item

Each row is a defect in `experiments/lstm_baseline.py` that made its number
incomparable to the published values it sat beside.

| | old | now |
|---|---|---|
| scored variable | `log1p(QObs)`, z-scored | **raw mm/day** |
| warmup at scoring | excluded from the loss, **scored anyway** with a cold state | excluded from both |
| basins | 671 | **531**, the quality-filtered subset |
| split | one hand-picked HUC2 triple (`01,11,17`) | **PUB 10-fold and PUR 7-fold** leave-one-out |
| period | a 512-day window at day 9600 | **the published periods**, 1461 or 5479 scored days |
| target length | 16-day tails, NSE denominator barely varying | the **continuous** test series |
| metric code | reimplemented locally | **dmg_dev's `Metrics`, vendored verbatim** |
| seeds | one | one (3-seed sweep not yet run) |

The first is the big one. Log-space NSE weights low flows differently from
NSE on discharge; the two are different numbers and neither bounds the other.
Every previous comparison against Kratzert (0.74) or Jamaat (0.74 → 0.82) was
comparing across that gap without saying so.

The second ran *against* us — it penalised the baseline in exactly the
full-window protocol where the baseline was being beaten — which is why it
survived so long. A bias that flatters gets challenged; a bias that penalises
looks like an honest result.

---

## What is dead

**Every CAMELS number in `docs/all_results.md` and `docs/benchmarks.md`.** Not
wrong arithmetic — they measure something else. They are kept as a record of
the protocols they were computed under and of the mistakes that produced them.
Nothing from those pages may be placed in a table with a published CAMELS
value or with anything produced by the scripts below.

The architectural findings those runs established are **not** invalidated,
because they are internal comparisons on a shared protocol: the time-aligned
attention being the component that matters, mode B (historical context) adding
nothing while mode A (concurrent context) adds a great deal, the model being
bidirectional rather than causal. Those stand and are worth re-testing here.

---

## Running it

```bash
python scripts/verify_camels531.py          # must print PROTOCOL VERIFIED
bash   scripts/submit_camels531.sh 0        # 6 jobs: 2 models x 3 protocols
python scripts/collect_camels531.py         # one table, protocol shown
```

| file | what it is |
|---|---|
| `src/hydropfn/data/forcing.py` | the loader, **restored** — it was missing entirely, so nothing in the repo ran |
| `src/hydropfn/data/protocol.py` | folds, periods, warmup, scoring |
| `src/hydropfn/metrics/dmg_metrics.py` | dmg_dev's `Metrics`, verbatim but for one import line |
| `experiments/camels531_lstm.py` | the regional LSTM baseline |
| `experiments/camels531_pub.py` | PUBModel, same folds and same metric |
| `scripts/verify_camels531.py` | the check against dmg_dev's own output |

`experiments/lstm_baseline.py` is kept for provenance only. Do not run it.

---

## Reading the results

The reference values, on the identical protocol, 3-seed means, from
`scratch/Extended1980_1999/extended_1980_1999_seed_averaged.csv`:

| | PUB | PUR |
|---|---|---|
| LSTM + HBV | 0.700 | 0.609 |
| Embedding adapter (Stefaland, Annually) | 0.706 | 0.627 |
| Condensed embedding (Stefaland, daily) | 0.721 | 0.638 |

**Temporal reference, exact-protocol (added 2026-08-30):** dHBV1.1p
(Song et al. 2026, WRR), run from the released dMG config
(`config_dhbv_1_1p.yaml`, seed 111111, ep50, 16-multiplier HBV + LSTM
parameterization): 531 basins, train 1980-10-01…1995-09-30, test
1995-10-01…2010-09-30, 365-day warmup, mm/day, dmg `Metrics` — the same
period, basins, metric and code path as this page's temporal extent.
**Median NSE 0.7431**, KGE 0.767, Corr 0.880, FHV −4.2 % (metrics summary
supplied by CS, 2026-08-30). This is a *gauged* temporal benchmark: every
test basin is in training.

Published values on the same 531 basins, for wider context — **different
forcing, so not same-protocol**: Feng et al. (2023, HESS), Maurer forcing:
δHBV 0.64 PUB / 0.59 PUR, LSTM 0.65 PUB / 0.55 PUR; temporal (NLDAS, Feng
et al. 2022) δHBV 0.711, LSTM 0.719. Jamaat et al. (2025), Daymet, temporal
1989–99: δHBV1.1p 0.75, LSTM 0.74 (no DA) — consistent with the 0.743
exact-protocol row above. Only the dmg rows share this page's exact
protocol; the published rows differ in forcing and/or period and are
context, not a leaderboard.

For `camels531_pub.py`, **K=0 is the row comparable to the LSTM.** K>0
additionally reads neighbouring gauges' concurrent discharge at inference,
which an LSTM structurally cannot do, and must be read against the `nn`,
`cm` and `idw` donor baselines printed beside it — a model that only matches
inverse-distance weighting of its neighbours is an expensive kriging.

**PUB and PUR ask materially different questions of K>0**, and the per-fold
distance diagnostic says which you are running. In PUB the held-out basins are
scattered, so a query's geographic neighbours are mostly *training* basins
(fold 0: nearest training basin 0.48°, nearest any 0.41°) — "an ungauged basin
among gauged ones". In PUR the whole region is held out, so the neighbours are
mostly basins the model never trained on.

Two asymmetries favour PUBModel wherever the two are compared, and both must
be stated:

* the LSTM is **causal**; PUBModel is a bidirectional smoother unless run with
  `--causal`.
* PUBModel reads context basins' **concurrent discharge**; the LSTM cannot.


---

## The full suite (closed 2026-09-01)

Four questions were posed on 2026-08-29 after the protocol changed twice
(raw-mm/day metric; converged budget — the old defaults were smoke-test-sized,
`--smoke` restores them). All four are answered. This section is the ONE
current-state record; run tags are given so every number can be traced to its
log under `logs/camels531/`.

### The scoreboard, by information regime

Median per-basin NSE, raw mm/day, e800 unless noted. "Gauged" = test basin in
training (temporal split); "ungauged" = held-out basin (PUB spatial).

| the model is given | extent | ours | references | verdict |
|---|---|---|---|---|
| nothing (forward) | ungauged | 0.682 · **0.707** mode-B-trained (`pub531_e800`, `pub531_modeB_e800`) | LSTM+dHBV1.1p **0.700** (3-seed) · StefaLand 0.721 · LSTM 0.666¹ | **ties the hybrid**, trails StefaLand by 0.014 |
| + concurrent neighbours | ungauged | K=4 **0.751** (`pub531_e800`) | StefaLand 0.721 · IDW 0.645 | **beats everything** (+0.030 / +0.106) |
| nothing (forward) | gauged | K=0 **0.640** (`pub531_temporal_fwd_e800`) | LSTM **0.687** (3-seed e100) · dHBV1.1p **0.743** | **behind both** (−0.047 / −0.10) — the specialists lead on their home turf |
| + concurrent neighbours | gauged | K=4 **0.773** (`pub531_temporal_fwd_e800`) | dHBV1.1p 0.743 · IDW 0.634 | beats both (+0.030 / +0.14) |
| + own gauge, 1–16 d stale | gauged | **0.707** (`pub531_drawshare_e800`) | persistence 0.444 · LSTM 0.687 · dHBV1.1p 0.743 · DI-LSTM ~0.86² | beats the LSTM (+0.020, seed spread 0.012); short of dHBV1.1p by 0.036; **far from DI-LSTM — open** |
| + both streams | gauged | K=8 **0.804** (`pub531_drawshare_e800`) | dHBV1.1p 0.743 · LSTM 0.687 · IDW 0.634 | **beats everything** (+0.061 / +0.117 / +0.17) |

¹ single-seed e50; the temporal 3-seed e100 sweep confirmed its e50
counterpart within noise (0.687 vs 0.692), so 0.666 is probably sound but has
not been re-run. ² Feng et al. 2020, lead-1, literature value on these
basins — not our run.

### The four questions, answered

- **(i) beat LSTM / dHBV1.1p forward or with mode-B context** — YES on
  ungauged basins (0.707 vs 0.700), NO on gauged temporal (0.640 vs 0.743):
  the forward arm matches the specialists only where nobody has the gauge.
- **(ii) beat IDW with mode B** — trivially yes (historical IDW is −0.44 to
  −0.51; interpolating wrong-year discharge is worse than the mean), and the
  real finding is below.
- **(iii) equal/surpass the LSTM with the recent-obs stream** — YES
  (0.707 vs 0.687) after the draw-share fix; dHBV1.1p (0.743) and DI-LSTM
  (~0.86) remain ahead. Open.
- **(iv) gains over LSTM and dHBV with context + recent-obs** — YES,
  decisively: 0.804, +0.061 over dHBV1.1p on its own exact benchmark.

### The findings under the numbers

**Mode B's training draw is the prize; its eval-time context is inert**
(`pub531_modeB_e800`, all 10 folds). Training on DOY-aligned historical
context lifts the no-context arm 0.682 → **0.707** (+0.025) — the best
forward number in the suite — while at inference the same context adds
+0.001 (K=8 0.7084 vs K=0 0.7071): same-DOY flows from a 6-year offset carry
climatology the forcings already imply. So mode B stays in the training
mixture and out of the deployment story.

**The draw distribution is part of the experiment** — the suite's recurring
lesson, three instances: K=0 drawn 1/6 of steps (+0.055 when raised to 3/8);
smoke-sized defaults escaping into benchmarks (K=0 0.265 at e40 → 0.682 at
e800, context gain shrinking from +0.14 to +0.06 as the forward arm
converged); and the recent-obs tail draw — the evaluated configuration
(shortest tail) received ~3% of training steps, and raising it to 40%
(`--self-ctx-p 0.8 --self-ctx-max-tail 2`) moved (iii) 0.655 → 0.707 and
(iv) 0.786 → 0.804. None of these was an implementation bug; the identical
eval code scored every configuration correctly.

**Recent-obs mechanics, what three configurations established.** The
persistence floor is **0.444** (median lag-1 autocorrelation 0.722; NSE =
2ρ−1 checks exactly). The patch-16 model shows **no lead decay** (lead 1 =
lead 16), so per-lead readouts may not be quoted against all-days baselines —
the ~+0.03 per-lead lift is a subsampling artifact. Routing own history
through **cross-attention beats the own-token-stream channel** at this
budget: self-ctx patch-16 0.707 > self-da patch-1 fixed-tail 0.668 >
self-ctx patch-1 0.626 (`pub531_p1_selfda_t1_e800`, `pub531_p1_recobs_e800`)
— reversing the 671-era ranking. A uniform-tail self-da run scored 0.303 by
learning a lag-+2 echo (best-lag +2 in 490/531 basins): the scored position
is the window's last, and a uniform 1..31-day tail trains it with its
nearest observation ~16 days away on average. Diagnosed by fingerprint, not
inspection. The remaining gap to DI-LSTM (~0.86 at lead 1) is real,
unexplained, and the suite's main open problem.

**The gauged-forward gap is architecture, not the training mixture
(2026-09-08, `pub531_temporal_k0spec_e800`).** A from-scratch K=0-only
specialist of our own architecture scores **0.639** on gauged temporal —
identical to the generalist's 0.640. Specializing buys zero, so there is no
mixture tax and nothing for a finetune to recover; the one-model-many-tasks
property holds in the strong sense. The 0.05 gap to the LSTM (0.687) is
architectural — same data, same globally-standardized squared-error loss
family, same budget; the LSTM's daily recurrence with 365-day spin-up state
is the remaining difference (candidates within it: 16-day patch granularity
and windowed scoring vs persistent state). dHBV1.1p's further margin to
0.743 is partly OBJECTIVE — dmg trains it with per-basin NSE loss, i.e. the
reported metric — plus physics priors. Replication run with `--save-ckpt`
(`pub531_temporal_fwd_ck`): K=0 0.642 / K=4 0.766, fold checkpoint saved.

**Baseline provenance.** LSTM temporal: 3-seed e100 median **0.687**
(0.676/0.688/0.699) — confirms the e50 single-seed 0.692 was converged, so
the e50 PUB values (0.666/0.545) are probably sound but unrefreshed.
dHBV1.1p **0.7431**: released dMG config, exact protocol, single seed (see
"Reading the results"). Feng 2022's LSTM 0.719 on this split is NLDAS
forcing — not comparable to our Daymet runs.

**Donor-baseline discipline.** All donor baselines (nn / ctx_mean / idw)
read the SAME window the model's context read — for mode B the historical
window, never the eval slice (reading the eval slice hands them concurrent
discharge the model was denied; it reversed a mode-B table once). In
recent-obs mode the query's self-slot is excluded from the donor set.

### Caveats before anything is quoted externally

Single seed everywhere except the temporal LSTM (3 seeds). Specifically
single-seed: every PUBModel row, the dHBV1.1p reference, the mode-B
regularizer claim (0.707 vs 0.682 — one comparison; needs a second seed
before it is load-bearing), and the PUB LSTM references.

### Open

1. The DI-LSTM gap on the own-gauge stream (0.707 vs ~0.86).
2. Second seed for the mode-B regularizer claim.
3. DEM statics value: stage 2 (imputed statics through the frozen model;
   `pub531_ckpt3_e800` checkpoints in progress). Stage 1 is DONE — see
   "DEM as attribute imputer" below.


### DEM as attribute imputer — stage 1 (2026-09-01, `experiments/dem_imputer.py`)

Can each input tier reconstruct the 26 curated statics on held-out PUB
groups? Out-of-fold ridge (per-tier alpha CV), leave-one-PUB-group-out.
CLIM = 11 climate indices computed from the basin's own forcings; KRIG =
statics IDW-interpolated from training basins; DEM = 768-dim multi-scale
diffusion features (parity checkpoint, 12.8 km, PCA-32 per fold — raw dims
cripple the shared ridge alpha and read falsely negative, −0.05).

| tier | all | climate | topo | veg | soil | geol |
|---|---|---|---|---|---|---|
| CLIM | 0.695 | 0.925 | 0.768 | 0.702 | 0.233 | 0.092 |
| KRIG | 0.833 | 0.881 | 0.845 | 0.857 | 0.663 | 0.512 |
| CLIM+KRIG | 0.834 | 0.888 | 0.846 | 0.856 | 0.663 | 0.512 |
| +DEM | 0.836 | 0.891 | 0.854 | 0.852 | 0.661 | 0.512 |
| DEM alone | 0.238 | 0.247 | 0.404 | 0.284 | 0.174 | 0.082 |

Findings, honest on both sides:

1. **On CAMELS, location interpolation saturates the imputation problem**
   (KRIG 0.833): attributes are spatially smooth at CONUS gauge density, so
   DEM is almost entirely redundant. The typical attribute gains +0.002.
2. **DEM's far-from-training gains are OROGRAPHIC, not subsurface** —
   p_mean 0.80→0.88, frac_snow 0.87→0.92, aridity +0.03 in the far half;
   soils +0.003, geology −0.001. Terrain makes weather; the registered
   prediction (gains concentrated in soils/geology) was half wrong.
3. **The donor-exclusion curve** (kriging donors forbidden within R km —
   the global-sparse regime manufactured on CONUS): DEM delta ≈ 0 at
   R ≤ 200 km, and at **R = 400 km soils flip to +0.038** while geology
   never gains at any radius. The statics-imputation claim survives only in
   the extreme-sparse regime, only for soils, and thinly.
4. DEM alone at 0.238 (topo 0.404) confirms real signal — just redundant
   with location + climate wherever a table exists nearby.

**PUR rerun (new terrain, median 269 km to the nearest table entry — the
honest venue, per CS's objection that PUB measures interpolation's home
advantage):** everything collapses — KRIG 0.833→0.196, CLIM+KRIG 0.249 —
and DEM's honest increment is **+0.010 overall, +0.030 on topography**.
The deepest fact: NO tier predicts soils/geology in new terrain (all ~0 or
negative; DEM-alone's +0.011 is the best single soil predictor, at a level
that is still nothing). Subsurface attributes in unseen regions are an open
problem for every method — the strongest motivation for the
geology-as-target work. DEM-alone goes negative overall under PUR (−0.05):
the location-fingerprint failure mode, measured cleanly.

**Reverse direction (CS's question: do statics support reconstructing the
DEM?)** Weakly yes, above the right controls: predicting DEM PCs under PUR,
variance-weighted R² — lat/lon 0.008, climate 0.036, **26 statics 0.069**
(top-2 PCs ≈ 0.22/0.21), DEM-kriged-from-neighbours **−0.058**. Two
readings: there is real cross-modal latent structure for a connector
(statics beat location and climate), and **terrain features do not krige
across regions at all** while attributes krige at 0.83 within them —
terrain is local texture, attributes are smooth fields. That asymmetry is
both why DEM could matter where tables are missing and why its linear
payoff is small.

**The generative direction is also null (2026-09-03,
`experiments/dem_attr_eval.py`).** An attribute-conditioned sampler
(statics + presence bit as a third additive term on the time embedding,
30% attr-dropout, HUC2 01/11/17 gauges excluded from training; commit
`34e4b6d`) was sampled on the held-out gauges' tiles twice under identical
masks and noise — true attrs vs null token. Result: a tie. 12.8 km psd
0.401 vs 0.384, elev RMSE 0.3751 vs 0.3749, paired win counts 27–43/64
(coin-flip except a whisper on slope-W1 at one scale, absent at the other).
Two caveats keep this from over-claiming: additive global conditioning is
the weakest coupling (a bias vector cannot paint spatial structure —
cross-attention or per-layer FiLM could carry more), and basin-integrated
scalar statics are spatially unlocalized, a poor match for specifying local
texture. But at this budget the reverse probe's ~0.2 R² does not cash out
generatively. Note for the 2D-field arm: its conditioning will be
*localized point measurements* (wells), a categorically stronger signal
than basin scalars — this null does not transfer to that design.

**Stage 2 (imputed statics through the frozen model) is closed by
redundancy**: with the forward probe (+0.002), the new-terrain probe
(+0.010) and the generative pair all reading null, the NSE-through-statics
question is answered three ways; the `pub531_ckpt3_e800` fold checkpoints
(0.727/0.75 heldout-fold NSE) are saved if anyone wants the fourth.

**In-model retest with compression (2026-09-10, commit `1823d4a`).** The
question CS posed: do 768 pooled diffusion dims overfit (>1 dim per
training basin, i.e. identification capacity), and would a hard squeeze
before the time-series arm help? Five arms, PUB spatial folds 0–2, e800,
seed 0, all trained with 50% per-attribute statics masking so there are
draws where DEM must stand in for the table; each scored with statics
visible and fully withheld (the global-case readout). Median NSE, pooled
over the 147 held-out basins:

| arm | DEM input | statics visible | statics withheld |
|---|---|---|---|
| A control | none | **0.709** | 0.639 |
| B raw | 768 dims, linear | 0.654 | 0.623 |
| C PCA-32 | per-fold basis, task-blind | 0.666 | **0.644** |
| D PCA-8 | per-fold basis, task-blind | 0.667 | 0.642 |
| E learned MLP→8 | 768→64→8, trained | 0.634 | 0.596 |

1. **Compression does what the counting argument predicted.** With statics
   withheld, raw features cost 0.016 against the control; PCA to 32 or 8
   removes that cost (0.644 / 0.642 vs 0.639). The overfitting was real.
2. **But compressed DEM still adds nothing.** +0.003 to +0.005 in the one
   regime it was supposed to win is noise at three folds and one seed.
3. **A learned bottleneck is worse than no compression at all** (0.596
   withheld, and unstable: fold 2 at 0.567). Eight trainable floats can
   still encode which basin this is, and they are pushed toward exactly
   that by the loss. A fixed, task-blind basis is what constrains the
   pathway; width alone does not. If DEM is fed to this arm again, feed it
   through PCA.
4. **With statics visible, every DEM arm hurts**: −0.042 even for PCA,
   and the loss appears in all three folds. DEM in the input is not free,
   compressed or not.

The DEM-for-streamflow question is closed on CAMELS: the raw arm's damage
was capacity, compression fixes the damage, and there is no gain
underneath it.

**Side result worth more than the DEM result: statics masking lifts the
forward arm.** Same folds, same budget, K=0 fold means — mode-A training
0.675, mode-A with 50% statics masking **0.701** (+0.026, positive in all
three folds), mode-B training 0.711. This replicates the 671-era
attribute-dropout gain (+0.032) on the verified protocol. Both statics
masking and mode-B draws are regularizers for the forward arm; they have
not been combined yet.

Consequence: the DEM arm's defensible pitches remain **geology as a
target** (lithology 0.59→0.64, age 0.37→0.44 — no curated table exists) and
the future 2D-field arm — NOT statics imputation on any landscape with a
mapped attribute table. Stage 2 (imputed statics through the frozen 531
model) will quantify how little the model's NSE cares, and the expectation
is now: barely.

---

## Kriging and gauge density (2026-08-30)

"Kriging reaches ~0.9 NSE" and "IDW scores 0.66 here" are the same
phenomenon at different gauge densities (`experiments/kriging_density.py`,
model-free, raw mm/day, spatial scored period):

| nearest gauge | n | median NSE (IDW K=8) |
|---|---|---|
| 0-11 km | 58 | **0.838** |
| 11-22 km | 102 | 0.773 |
| 22-39 km | 152 | 0.714 |
| 39-67 km | 123 | 0.547 |
| >67 km | 96 | 0.187 |

CAMELS' densest bin reaches 0.84 with cross-divide neighbours alone (the set
has 0.011% nested pairs); the literature's ~0.9 comes from dense NESTED
networks where downstream flow contains the upstream gauge. Geometry is the
variable; the interpolation method is second-order. Consequence for reading
the suite: the model's margin over IDW should be judged per density bin — the
operationally interesting claim is holding skill where gauges are FAR, where
interpolation collapses (0.19 beyond 67 km).

### The suite judged per density bin (mode A e800, PUB spatial)

Model columns from the saved e800 predictions; IDW recomputed identically to
the table above on each basin's scored days:

| nearest gauge | n | ours K=0 | ours K=4 | IDW | K=4 − IDW |
|---|---|---|---|---|---|
| 0–11 km | 58 | 0.765 | 0.881 | 0.838 | **+0.043** |
| 11–22 km | 102 | 0.738 | 0.821 | 0.773 | +0.048 |
| 22–39 km | 152 | 0.716 | 0.780 | 0.714 | +0.066 |
| 39–67 km | 123 | 0.638 | 0.718 | 0.547 | **+0.171** |
| >67 km | 96 | 0.465 | 0.477 | 0.187 | **+0.290** |
| ALL | 531 | 0.682 | 0.751 | 0.657 | +0.094 |

The margin over interpolation **grows monotonically with sparsity**: where
gauges are dense the model matches IDW plus a little (+0.04 — and that is
where the literature's ~0.9 lives), and where interpolation collapses the
model degrades gracefully to its forward arm (K=4 0.477 ≈ K=0 0.465 vs IDW
0.187). That is the operational claim in one table: context is exploited
where it is informative and ignored where it is not. It is also the honest
deflation of the aggregate +0.106-over-IDW headline — most of that margin is
earned in the sparse half of the network.

---

## Recovery pilots completed 2026-09-23

These are new **seed-0 pilots**, not replacements for the earlier benchmark
claims. Temporal experiments use the temporal dates above, all 531 basins,
5479 scored days, raw mm/day and the vendored dmg metric. Saved gauge IDs
and target arrays were checked before paired comparisons. Training-only basin
variances, with a 0.1 mm/day standard-deviation floor, define the new normalized
MSE loss. This approximates mean per-basin NSE, not median NSE; it is not a
verified line-for-line reproduction of dHBV's `NseBatchLoss`.

### Objective and forward diagnostics

| model | objective / ablation | temporal median NSE |
|---|---|---:|
| original PFN, K=0 | global MSE | 0.640079 |
| original PFN, K=0 | basin-normalized MSE | 0.635120 |
| original PFN, K=0 | global MSE, pooled path disabled | 0.640281 |
| regional LSTM | global RMSE | 0.682272 |
| regional LSTM | basin-normalized MSE | 0.691068 |

Neither changing the objective nor removing the pooled path closes the PFN
forward gap in this pilot. The fresh LSTM RMSE control does not reproduce the
historical seed-0 value exactly; the paired target/protocol checks pass, but
the historical runtime is incompletely recorded. Use the fresh control for
this objective comparison. Diagnostics: `logs/recovery_setup/stage1_diagnostics/`.

### Recurrent observation interface and paired training

These models use the LSTM observation interface in
`experiments/camels531_daily_di.py`; they are **not the original PFN**. The same
frozen mixed/paired checkpoint is evaluated under each information condition.
Daily means yesterday's Q is available; none means no observed Q is supplied.

| temporal model | epochs | no observed Q | daily Q | weekly Q |
|---|---:|---:|---:|---:|
| forward specialist | 100 | 0.692403 | — | — |
| daily specialist | 100 | — | 0.830728 | — |
| original mixed sampler | 100 | 0.704258 | 0.800624 | 0.743795 |
| paired daily + mixed | 50 | 0.714143 | 0.832320 | 0.753548 |
| paired daily + mixed | 100 | 0.709465 | 0.816062 | 0.744264 |
| paired daily + daily control | 100 | — | 0.826290 | — |

Each paired update averages two losses on the same sampled basin/window,
with independent dropout, one gradient clip and one optimizer step. Paired
50 epochs has 13,300 updates and 26,600 window views: the original specialist's
total view count, with half its optimizer updates. Paired 100 epochs and the
two-pass daily control each have 26,600 updates and 53,200 views, on GPU 2.
The paired 50-epoch model and original daily specialist ran on GPU 4.
Matching views is a compute proxy, not identical GPU time or optimizer history.

The 50-epoch pilot reaches daily headline parity. Its daily median-NSE
difference from the original specialist is +0.001591; the median of individual
basin differences is -0.000631, with 48.0% of basins improved. A paired basin
bootstrap gives a 95% interval [-0.007574, +0.012250] for the headline
difference. This interval excludes neither a small deficit nor a small gain;
it does not model seed uncertainty or spatial dependence. **Parity is not
certified.** The original future-period test has already been inspected.
Do not select further hyperparameters on it or call it newly untouched.

The 100-epoch paired result trails its same-GPU, same-budget daily+daily
control by 0.010228. The two paired durations also used different GPUs, so
their difference alone cannot establish overtraining. Confirm with replicated
contrasts and checkpoint selection using validation years inside the training
era. Do not infer uniform basin improvement from a difference of medians.

Full scenarios and paired basin differences:
`logs/recovery_setup/paired_parity_diagnostics/{regime_scores.csv,paired_basins.csv,audit.json}`.
The completed paired batch is a positive feasibility result for recurrent DI;
it does not resolve original-PFN forward performance or certify a unified
foundation model across tasks and regions.

### Unseen basins AND future years

All ten PUB folds completed, with training/test basin disjointness checked.
These runs train through 1995 and score 1995–2010, unlike the spatial-only
benchmark above, which trains through 1999. They use the original forward
and mixed recurrent models, **not the new paired models**.

| information | forward specialist | mixed recurrent model |
|---|---:|---:|
| none | 0.572102 | 0.590445 |
| daily Q | — | 0.719283 |
| weekly Q | — | 0.647390 |
| 7-day reporting delay | — | 0.621103 |
| 16-day reporting delay | — | 0.609814 |
| historical 30-day campaign | — | 0.590706 |
| historical 90-day campaign | — | 0.590645 |
| historical 365-day campaign | — | 0.590518 |

No unseen-basin daily specialist was trained in this batch, so its specialist
gap is unmeasured. Historical campaigns show negligible lasting benefit
relative to the **same mixed checkpoint** with no observations. Scoring only
1995–1999 still gives 0.583101 forward and 0.599376 mixed/no-Q; the long score
period alone does not explain the weak result. The training dates still
differ from the standard spatial benchmark.
Artifacts: `logs/recovery_setup/stage2_diagnostics/`.

### Basin versus outlet terrain information probe

All 531 paired footprints passed the 98% coverage requirement; none excluded.
This is a **training-era hydrologic-signature probe, not daily discharge NSE**.
Seven regional outer folds test attributes alone against an additive ridge
correction using 15 DEM descriptors, with inner regional cross-fitting for
the baseline residuals and a prespecified ridge alpha of 100.

| predictor | mean training-scaled squared error (lower is better) |
|---|---:|
| attributes | 0.509618 |
| attributes + outlet terrain | 0.607537 |
| attributes + basin terrain | 0.594372 |

Correcting coverage and basin footprint did not produce added skill in this
test. This is evidence against this descriptor/residual formulation, not a
proof that DEM has no hydraulic information or that every DEM encoder is
overfitted. It does not test fine local terrain for channel geometry.
Artifacts: `logs/recovery_setup/terrain_probe/`.

### Provenance

Remote snapshots under `/nfs/data/cxs1024/hydroPFN/`:

| work | immutable source snapshot | archive SHA256 |
|---|---|---|
| objective contrasts | `experiments_20260923_v2` | `6bd713c9623a715436cf74a33e91eed8d6eb59b2cd0cf179d9c510c98f8221f5` |
| recurrent DI / terrain probe | `experiments_20260923_v5` | `4ff5fb703106d564d14f641daab02815e6c9c63a0e3feb477953d32a5ec1e97e` |
| paired parity | `experiments_20260923_v6` | `0b9add8d3819cd1700c3737d6f2112aac6b534bc6b08f824ada48b8611c8348e` |

The paired supervisor completed at 2026-09-23 14:21:52 UTC. At the subsequent
status check, all batches from this program were finished and GPUs 2/4 idle.
Terrain extraction has separate input hashes and recovery provenance in
`logs/recovery_setup/terrain_probe/summary.json` and the remote
`logs/basin_terrain_20260923/recovery_launch.json`.

Implementation/run history: [recovery log](recovery_experiments_20260923.md).
Interpretation and next design:
[specialist parity](specialist_parity_formulation_20260923.md) and
[measurement assimilation / DEM](measurement_assimilation_dem_20260923.md).

## Frozen-PFN decoder and duration pilots (2026-09-23)

All ten 100-epoch suntzu jobs and three 300-epoch ICDS jobs completed. These
are temporal K=0 hindcasts over the same 531 basins and 5,479 scored days,
with raw-discharge per-basin NSE. They use a frozen source PFN and common
512-day windows with 256 days of warmup; these are not the canonical
365+365-day LSTM protocol. Daily readouts receive raw forcings/attributes as
well as frozen PFN features in the hybrid condition. The original patch-head
control receives only the frozen features, as the original architecture did.
Implementation and input/history caveats: [decoder comparison](decoder_comparison_20260923.md).

### 100-epoch matrix on suntzu

| readout | seed 0 median NSE | seed 1 median NSE |
|---|---:|---:|
| raw LSTM | 0.679044 | 0.651564 |
| raw TCN | 0.698380 | 0.671866 |
| PFN + LSTM | 0.663437 | 0.648272 |
| PFN + TCN | 0.678128 | 0.671927 |
| PFN + daily pointwise MLP | 0.622734 | 0.620248 |

Each seed's entire matrix used one GPU: seed 0 RTX 3090 Ti, seed 1 RTX 2080
Ti. Thus cross-seed variability also includes device differences. TCN improves
on LSTM for both raw and hybrid inputs in both seeds. The temporal readouts
also improve on the daily pointwise hybrid control, but adding frozen PFN
features does not consistently improve on the corresponding raw readout:
seed-0 hybrid TCN is 0.020252 below raw TCN, while seed 1 is essentially tied.
Paired basin differences and bootstrap intervals are in the artifact below;
these are not spatially independent or replicated-seed confidence intervals.

The unchanged source PFN's original head scores **0.651827** on the common
cache/evaluation inputs, before refitting. This differs from the historical
0.640079 because inputs/query-valid handling and evaluation tiling differ;
do not attribute the whole difference to history or to decoder architecture.

Artifacts: `logs/recovery_setup/decoder_comparison_diagnostics/` and
`logs/recovery_setup/decoder_cache_manifest.json`. Remote immutable snapshot:
`/nfs/data/cxs1024/hydroPFN/experiments_20260923_v7`; its supervisor completed
2026-09-23 18:50:54 UTC.

### 100 / 200 / 300 checkpoints within uninterrupted ICDS runs

| readout (seed 0) | 100 epochs | 200 epochs | 300 epochs | 300 minus 100 |
|---|---:|---:|---:|---:|
| PFN + LSTM | 0.654442 | 0.648534 | 0.645362 | -0.009080 |
| PFN + TCN | 0.677255 | 0.671351 | 0.672403 | -0.004852 |
| original PFN patch head, refitted | 0.577272 | 0.555639 | 0.534908 | -0.042364 |

All three use the same cached encoder features, random readout initialization,
seed-0 basin/window sample stream, Adadelta lr 1, batch 128 and loss. The
encoder stays frozen: the last row is **not** the original end-to-end model
retrained for 300 epochs. It has no added raw daily branch and no head dropout,
matching the original patch readout. Every checkpoint is prespecified; test
scores were computed after training, without affecting stopping or updates.

The training loss (last ten epochs' mean at each milestone) continues falling:

| readout | epochs 91-100 | epochs 191-200 | epochs 291-300 |
|---|---:|---:|---:|
| PFN + LSTM | 0.063091 | 0.042890 | 0.037043 |
| PFN + TCN | 0.115273 | 0.086160 | 0.076780 |
| refitted patch head | 0.169882 | 0.148089 | 0.142865 |

Thus this pilot does not support extending the same frozen-readout training
as a remedy for the temporal gap. Falling training loss with worse test scores
is consistent with overfitting. This is a single-seed duration experiment,
not proof that every longer-training schedule or end-to-end adaptation fails.
For TCN, the 300-versus-100 median paired basin delta is only -0.000312 and
49.7% of basins improve; the headline median drop does not mean uniform harm.
LSTM's paired median delta is -0.010752 (37.1% improved), and the refitted
patch head's is -0.034918 (19.4% improved).

ICDS Slurm array 55769402 tasks 0/1/2 all report COMPLETED with exit 0:0.
Elapsed times: LSTM 1:21:07, TCN 1:30:10, patch head 0:13:18. Each completed
116,400 optimizer updates. All ran on Tesla V100S GPUs with PyTorch 2.7.1
and CUDA 11.8; duration comparisons are within one job, not across machines.
The earlier suntzu scores are separate runs, not the 100-epoch checkpoints
of these ICDS trajectories.

Remote root:
`/storage/group/cxs1024/default/cxs1024/hydroPFN/decoder_long_20260923`.
Archive/source/cache hashes and job mapping are in
`logs/recovery_setup/decoder_long_submission.json`. Result records, training
curves, per-basin checkpoint NSE and Slurm logs are archived in
`logs/recovery_setup/decoder_long_results/`.
All nine checkpoint NSE arrays and medians were independently recomputed
from saved predictions using the protocol metric; targets and gauge order
matched the shared cache exactly, and every prediction was finite. Audit:
`logs/recovery_setup/decoder_long_results/results_audit.json`.

## Window-sampling ablation completed (checked 2026-09-27)

All 20 runs in ICDS array 55837226 and analysis job 55837227 completed with
exit 0:0. Fixed and daily sampling were paired within one GPU allocation per
model/seed, with identical initial decoder weights, basin/random draw streams,
100 epochs (38,800 updates), objective, dropout and history. Both treatments
used online frozen encoding, removing the historical-cache backend difference.
Only the start distribution changed: 79 fixed starts versus uniform sampling
from 4,967 valid daily starts. The daily runs visited 4,966 starts at seed 0
and all 4,967 at seed 1. Evaluation was identical across all runs.

| model | seed 0 fixed | seed 0 daily | seed 1 fixed | seed 1 daily |
|---|---:|---:|---:|---:|
| raw LSTM | 0.692159 | 0.678157 | 0.686893 | 0.690912 |
| raw TCN | 0.678270 | 0.691136 | 0.670252 | 0.689742 |
| PFN + LSTM | 0.669655 | 0.718254 | 0.681419 | 0.724341 |
| PFN + TCN | 0.670832 | 0.717008 | 0.672411 | 0.715528 |
| original patch head, refitted | 0.576393 | 0.677449 | 0.577164 | 0.678636 |

Daily starts improve hybrid LSTM headline NSE by +0.048599 / +0.042922
(seeds 0/1) and hybrid TCN by +0.046176 / +0.043116. Individual basin NSE
improves for 77.6% / 76.6% of hybrid LSTM basins and 79.5% / 80.4% of hybrid
TCN basins. The refitted original head improves by +0.101056 / +0.101472,
with approximately 93% of basins improved in each seed. These head-only runs
remain distinct from end-to-end PFN retraining.

The daily hybrid beats its corresponding daily raw control in both seeds:
LSTM by +0.040097 / +0.033429, TCN by +0.025873 / +0.025786. Accounting for
the raw control's sampling change, the raw-minus-hybrid gap shrinks by
0.062601 / 0.038903 for LSTM and 0.033310 / 0.023626 for TCN. This is stronger
evidence than a hybrid improvement alone. Raw LSTM's sampling effect is
inconsistent (-0.014002 / +0.004019); raw TCN improves (+0.012866 / +0.019490).

The fixed-window restriction was a substantial contributor to this frozen-
readout pilot's deficit. The prior conclusion of no consistent frozen-PFN
benefit is superseded for this corrected sampling setup. The experiment does
not isolate context diversity, patch alignment and target-day weighting from
each other. Two seeds and a previously inspected temporal benchmark do not
establish universal architecture superiority or parity with external models.
Hybrids retain bidirectional forcing context while raw controls are causal;
the hybrid gain cannot be assigned exclusively to representation quality at
identical information. No new spatial or daily-DI result is implied.

Analysis recomputed every NSE from predictions, verified matching targets,
gauge order, cache provenance, paired initialization/RNG streams, GPU job,
equal budgets and unchanged encoder weights. Local artifacts:
`logs/recovery_setup/window_sampling_diagnostics/` (scores, sampling effects,
gap changes, per-basin scores and audit). Source hashes, job mapping and
deployment details: [sampling experiment](window_sampling_ablation_20260926.md).

## Single-model benchmark follow-up (2026-09-27)

The user requested single-model comparisons only. Six fresh corrected-daily
sampling runs (LSTM and TCN hybrids, seeds 0/1/2) train for 300 epochs and
report each model independently at 100/200/300 epochs. No predictions are
combined across seeds or architectures. The earlier exploratory ensemble
calculation is excluded from this comparison.

Current 100-epoch single-model medians are LSTM 0.718254/0.724341 and
TCN 0.717008/0.715528. These remain below the recorded dHBV1.1p reference
of 0.7431. See [benchmark attempt](benchmark_attempt_20260927.md) for
comparison scope, launch provenance and limitations.

### 300-epoch follow-up completed (2026-09-28)

All six single-model runs completed. At epoch 300, LSTM seeds 0/1/2
score 0.700680/0.704532/0.693499, and TCN 0.711041/0.706535/0.711012.
None matches the recorded 0.7431 reference. LSTM training loss roughly
halved between epochs 100 and 300 while test NSE declined in every seed,
consistent with overfitting despite unrestricted window sampling. TCN
training loss also fell, without a consistent test improvement.
The full 100/200/300 table and TCN repeat-variability caveat are in
[the benchmark record](benchmark_attempt_20260927.md); machine-readable
records are in `results/benchmark_daily_20260927/`. No ensembles are used.

### Matched raw recipe and frozen residual follow-up (2026-09-28)

The next experiment restores the local RegionalLSTM recipe (365+365 days,
output dropout 0.5, continuous test state, original iteration rule), then
freezes its epoch-100 model. Separate 50-epoch raw-only and PFN-informed
correction heads start at exact baseline parity. Three seeds are queued as
ICDS array 55872463 after successful GPU preflight 55872461. No encoder joint
training, ensembles or test-based checkpoint selection are used. This matches
the local LSTM recipe, not an independently reproduced external HBV model.
See [full design and audit](specialist_residual_20260928.md).

### Frozen correction results checked 2026-10-02

Array 55872463 completed all three seeds. Raw LSTM e50 NSE was
0.708602/0.708294/0.714377; the prespecified e100 correction source scored
0.690394/0.699520/0.700812. PFN correction after 50 epochs scored
0.690701/0.691691/0.689384, failing to consistently improve that baseline.
Exact initial parity and frozen-weight checks passed in every seed.
See [complete experiment results](specialist_residual_20260928.md#completed-results-checked-2026-10-02).
The experiment has not reached the external specialist reference.
