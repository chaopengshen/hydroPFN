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
3. DEM-as-attribute-imputer experiment (checkpointed 3-fold PUB run
   `pub531_ckpt3_e800` in progress; controls: mean / climate-derived /
   lat-lon-kriged / +DEM statics, stratified by distance to training).

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
