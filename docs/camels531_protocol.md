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

Published values on the same 531 basins, for wider context — **different
forcing, so not same-protocol**: Feng et al. (2023, HESS), Maurer forcing:
δHBV 0.64 PUB / 0.59 PUR, LSTM 0.65 PUB / 0.55 PUR; temporal (NLDAS, Feng
et al. 2022) δHBV 0.711, LSTM 0.719. Jamaat et al. (2025), Daymet, temporal
1989–99: δHBV1.1p 0.75, LSTM 0.74 (no DA). Only the dmg rows above share
this page's exact protocol; the published rows differ in forcing and/or
period and are context, not a leaderboard.

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
| i | forward (K=0) and mode-A context vs LSTM / dHBV1.1p | folds 0–3: K=0 **0.675** avg (0.630/0.706/0.690/0.674) · K=4 **0.74** avg | LSTM 0.666 · LSTM+dHBV1.1p 0.700 | 4/10 folds; K=0 ≈ LSTM, K=4 beats all |
| ii | mode B (historical, DOY-aligned) vs IDW-on-history | fold 0: K=0 0.659 → K=8 **0.673** (+0.014) | historical IDW at aggregate | 1/10 folds |
| iii | recent-obs (own gauge, 1–16 d lag) vs LSTM | **0.655** (K=0 + self-context) | LSTM temporal **0.692** | **DONE — short by 0.037** at patch-16 lag |
| iv | context + recent-obs combined vs LSTM / dHBV | K=2 0.776 · K=4 0.785 · K=8 **0.786** | LSTM 0.692 · δHBV1.1p 0.75 (Jamaat, diff. period) · IDW 0.634 | **DONE — beats all** (+0.094 LSTM, +0.15 IDW) |

**(iii)/(iv) detail** (temporal, e800, `--self-ctx-p 0.4 --recent-obs 1`,
stride-1 final-patch scoring — every day predicted from beyond the
observation cutoff): own gauge alone at 1–16-day lag does NOT reach the
gauged LSTM (0.655 vs 0.692; the score averages leads 1–16, so a lead-1
readout is the fair short-lag number and remains to be extracted). Adding
concurrent neighbours flips it decisively: **0.786 vs LSTM 0.692**, and
+0.027 over the concurrent-only e200 run. Donor baselines: nn 0.537,
ctx_mean 0.53–0.58, IDW 0.61–0.63.

**Mode B early signal, fold 0**: historical aligned context is worth +0.014
(the first clearly positive mode-B reading anywhere), and mode-B *training*
lifts the K=0 arm +0.03 over mode-A training on the identical fold — second
independent sighting of aligned-historical draws acting as a forward-arm
regularizer.

Provisional trajectory that motivated the budget change (PUB spatial, K=0
median NSE): e40 **0.265** → e200 **~0.50** (5 folds) → e800 fold-0 **0.630**
— still climbing at 32× the original default budget. The context gain shrinks
correspondingly (fold-0: +0.14 at e200 → +0.06 at e800): concurrent
neighbours were partly compensating for an undertrained forward arm.

Mode B's port carries the baseline discipline from train_pub: donor baselines
read the HISTORICAL context window, never the eval slice — reading the eval
slice hands them concurrent discharge the model was denied, which reversed a
mode-B table once already.
