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

## The full suite (2026-08-29, in progress)

Training protocol changed twice (raw-mm/day metric; converged budget — the
old defaults were smoke-test-sized and are now raised, `--smoke` restores
them), so every claim is being re-earned here. Four questions, each with its
reference row on THIS protocol:

| # | question | ours | reference | status |
|---|---|---|---|---|
| i | forward (K=0) and mode-A context vs LSTM / dHBV1.1p | **K=0 0.682 · K=4 0.751** (all 10 folds) | LSTM 0.666 · LSTM+dHBV1.1p 0.700 · StefaLand 0.721 · IDW 0.645 | **DONE — K=0 beats the LSTM (+0.016); K=4 beats every reference** (+0.051 over dHBV1.1p+LSTM, +0.106 over IDW); mode-B-trained K=0 goes higher still (0.707, row ii) |
| ii | mode B (historical, DOY-aligned) vs IDW-on-history | **K=0 0.7071 · K=8 0.7084** (all 10 folds) | historical IDW **−0.44 to −0.51** (worthless) · concurrent IDW 0.645 | **DONE — the training draw is the prize**: mode-B-trained K=0 **beats LSTM+dHBV1.1p** (0.707 vs 0.700); eval-time historical context adds only +0.001 |
| iii | recent-obs (own gauge, 1–16 d lag) vs LSTM | **0.707** (draw-share fix, 2026-09-01; was 0.655) | LSTM temporal 0.692 · **dHBV1.1p 0.743** (exact protocol) | **BEATS the LSTM** (+0.015); short of dHBV1.1p by 0.036 |
| iv | context + recent-obs combined vs LSTM / dHBV | K=2 0.798 · K=4 0.802 · K=8 **0.804** (draw-share fix; was 0.786) | LSTM 0.692 · **dHBV1.1p 0.743 (exact protocol)** · δHBV1.1p 0.75 (Jamaat, diff. period) · IDW 0.634 | **DONE — beats all** (+0.112 LSTM, **+0.061 dHBV1.1p**, +0.17 IDW) |

**(iii)/(iv) detail** (temporal, e800, `--self-ctx-p 0.4 --recent-obs 1`,
stride-1 final-patch scoring — every day predicted from beyond the
observation cutoff): own gauge alone at 1–16-day lag does NOT reach the
gauged LSTM (0.655 vs 0.692) nor the gauged dHBV1.1p (0.743). Adding
concurrent neighbours flips it decisively: **0.786 vs LSTM 0.692 and
dHBV1.1p 0.743** — the latter on its own exact benchmark period, basins and
metric — and +0.027 over the concurrent-only e200 run. Donor baselines:
nn 0.537, ctx_mean 0.53–0.58, IDW 0.61–0.63.

**Lead-1 readout (2026-08-30): the −0.037 gap is real, not a lag artifact.**
By-lead NSE from the saved predictions (buffer day *d* was predicted at lead
`(d mod 16)+1`; the extraction reproduces the recorded all-lead medians
exactly, n=531):

| | lead 1 | lead 2 | lead 4 | lead 8 | lead 16 | all leads |
|---|---|---|---|---|---|---|
| K=0 | 0.686 | 0.690 | 0.671 | 0.697 | 0.684 | 0.655 |
| K=8 | 0.804 | 0.806 | 0.798 | 0.817 | 0.818 | 0.786 |

There is **no lead decay**: lead 16 scores the same as lead 1 (K=0 0.684 vs
0.686). The per-lead values sit ~0.03 above the pooled number at EVERY lead —
a subsampling artifact (1/16 of the days per basin), not recency skill, so
none of them may be quoted against the LSTM's all-days 0.692. The patch-16
model predicts its block uniformly and never exploits how recent the last
own-gauge observation is — the same conclusion the 671 protocol reached,
where a dedicated `patch=1` specialist was worth +0.06 on this task.

**The persistence floor for this task is 0.444** (2026-09-01): yesterday's
observed flow as today's prediction, median per-basin NSE on the identical
eval buffer. That is the number any own-gauge method must beat to justify
itself. Our recent-obs stream clears it by +0.21 (0.655), so the stream does
real work -- but note what the comparison against the LSTM actually says:
the LSTM's 0.692 uses forcings and NO own-gauge observations, so at 0.655 we
are losing while holding STRICTLY MORE information than the model that beats
us. That, not the raw gap, is the indictment.

**Draw-share fix (2026-09-01, `pub531_drawshare_e800`): the gap was mostly
undertraining of the evaluated configuration.** The self-context draw fired
on 40% of steps with a hidden tail uniform over 1..16 patches, so the
configuration eval actually scores (shortest tail) received ~3% of training
steps. `--self-ctx-p 0.8 --self-ctx-max-tail 2` raises that to 40% — and
K=0 goes **0.655 → 0.707** (now beats the LSTM's 0.692), K=8 goes
**0.786 → 0.804** (+0.061 over dHBV1.1p). Third instance of the draw-share
lever: K=0 at 1/6 share (+0.055 when fixed), and the self-da tail below.

**Self-da diagnosis (2026-09-01): a lag-+2 echo, not a broken port.** The
first self-da run (uniform tail) scored 0.303 at K=0 — *below* the
persistence floor — and its predictions correlate best with the observation
from two days earlier in 490/531 basins (K=4, with concurrent neighbours,
stays at lag 0 and 0.764). Mechanism: eval scores only the window's LAST
position, and a uniform 1..31-day tail trains that position with its nearest
observation ~16 days away on average, so the model learns a stale smoothed
echo there. Not an implementation bug — the identical eval code scores the
self-ctx checkpoints correctly. `--self-da-max-tail` applies the same fix;
the patch-1 fixed-tail run (the DI-LSTM analog, Feng et al. 2020: Q at t−1
as input, ~0.86 on these basins) is `pub531_p1_selfda_t1_e800`.

**Patch-1 rerun (2026-08-31, `pub531_p1_recobs_e800`): finer patches alone
do NOT close it.** `--patch 1 --win 64 --self-ctx-p 0.4 --recent-obs 1`
(own gauge visible through yesterday, every day scored at lead 1), same
800×150×8 budget: K=0 **0.626** (below patch-16's 0.655), K=4 **0.775**
(vs 0.785). Two confounds before "the gap is real at any granularity" can
be claimed: (a) 64-day windows mean 8× fewer scored days per task view at
the same view count — and the K=0 arm shows the undertraining fingerprint
(FHV −21.9 %); (b) the own gauge entered as a CONTEXT SITE (`self_ctx`),
whereas the 671-era 0.8765 specialist used the `--self-da` channel — own
history in the query's own token stream — and self-as-context losing to
self-da is itself an established 671-era result. The self-da port is the
next test; until then (iii) stands: own-gauge assimilation has not matched
the gauged baselines (LSTM 0.692, dHBV1.1p 0.743) in any configuration.

**Mode B final (2026-08-30, all 10 folds).** Three findings, in decreasing
order of importance:

1. **The training draw is the prize, not the eval-time context.** Mode-B
   training lifts the no-context arm from 0.6820 (mode-A training) to
   **0.7071** — past LSTM+dHBV1.1p (0.700). Aligned-historical draws act as a
   forward-arm regularizer; this was sighted per-fold twice and now holds at
   the aggregate (+0.025). The best no-context number in the suite comes
   from the mode-B checkpoint.
2. **Eval-time historical context is nearly inert**: K=8 0.7084 vs K=0
   0.7071 (+0.001). Same-DOY flows from a 6-year offset carry climatology,
   which the forcings already imply.
3. **IDW-on-history is worthless** (−0.44 to −0.51 median NSE; nn −0.56):
   interpolating wrong-year discharge is far worse than predicting the mean.
   So "beat IDW with mode B" is trivially true; the meaningful bar is
   CONCURRENT IDW (0.645), which even the mode-B K=0 arm clears by +0.06 —
   an operator with no live gauges at all beating one who interpolates them.

Caveat on (1): mode-A and mode-B checkpoints differ only in the context
period of training draws, same seed/budget — but it is a single seed, and
0.7071 vs 0.6820 is one comparison. Rerun with a second seed before making
the regularizer claim load-bearing.

Provisional trajectory that motivated the budget change (PUB spatial, K=0
median NSE): e40 **0.265** → e200 **~0.50** (5 folds) → e800 fold-0 **0.630**
— still climbing at 32× the original default budget. The context gain shrinks
correspondingly (fold-0: +0.14 at e200 → +0.06 at e800): concurrent
neighbours were partly compensating for an undertrained forward arm.

Mode B's port carries the baseline discipline from train_pub: donor baselines
read the HISTORICAL context window, never the eval slice — reading the eval
slice hands them concurrent discharge the model was denied, which reversed a
mode-B table once already.


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
