# Benchmarks — what we compare to, how to run it, how to verify the split

All numbers are **median per-basin NSE**. Pooled R² is also reported by the
code but inflates: it concatenates basins and subtracts one global mean, so
between-basin flow variance lands in the denominator.

---

## 1. Verify the split first

```bash
python scripts/verify_split.py --nc data/CAMELS_Frederik.nc
```

Expected output:

```
1. SPLIT TYPE
   held-out HUC2 regions : ['01', '11', '17']
   eval basins           : 124
   train basins          : 547
   regions in BOTH arms  : NONE -> this is PUR

2. SITE OVERLAP
   site_ids in both arms : 0  OK

3. TEMPORAL SPLIT
   train windows END by  : day 9000
   eval window           : day 9600 .. 10112
   gap                   : 600 days  OK

4. HOW FAR IS CONTEXT, REALLY
   eval -> nearest TRAINING basin : median 1.73 deg
   eval -> nearest ANY basin      : median 0.29 deg
```

What each check guards against:

- **PUR, not random holdout.** The split is by USGS **HUC2** region code
  (`station_id[:2]`); regions 01 (New England), 11 (Arkansas–White–Red) and 17
  (Pacific Northwest) are removed entirely. A random k-fold holdout would put a
  training basin ~0.3° from each eval basin; here it is **1.73°** (~190 km).
- **Split by site id, never by index or COMID.** A COMID-based split silently
  duplicated 1,360 sites across both arms in an earlier project.
- **Temporal split with a real gap.** Training windows end at day 9000,
  evaluation starts at day 9600. An earlier version bounded the window *start*
  rather than its *end*, giving our model 472 extra days — the ones nearest the
  evaluation period — in the very test meant to check recency.
- **Context distance is reported separately on purpose.** `--context-pool all`
  lets context come from other held-out basins ~0.29° away. Those gauges were
  never in training, so this is not a leak, but the setting is *"ungauged basin
  in an ungauged region, with nearby gauges that were never used for training"*
  — not *"no gauges exist"*. At K=0 it is pure PUR.

---

## 2. Reference results we compare to

Both references assimilate the **target gauge's own record** on **gauged**
basins (present in training), with a temporal split only. Ours is PUR. These
are context for interpretation, not a leaderboard we are winning.

### Jamaat et al. 2025 — variational DA for differentiable models

531 CAMELS basins, test 1989-10-01 to 1999-09-30, assimilation window 5 d,
median per-basin NSE.

| | NSE |
|---|---|
| δHBV, no DA | 0.75 |
| LSTM, no DA | 0.74 |
| δHBV-DA-PS (1-day lead) | **0.82** |
| LSTM-DA (1-day lead) | **0.82** |

Lead-time decay, their Table D1 (Appendix D):

| lead (d) | 1 | 3 | 5 | 7 | 9 |
|---|---|---|---|---|---|
| δHBV-DA-PS | 0.820 | 0.777 | 0.769 | 0.762 | 0.756 |
| LSTM-DA | 0.820 | 0.780 | 0.770 | 0.766 | 0.762 |

**−0.064 over nine days**, and monotone. Ours shows **no systematic decay**
over sixteen days (see "On flat lead curves" below — it is noisy and
non-monotonic, not literally flat), because we run with **known forcings**:
"lead" here is position inside a predicted block, not distance from an initial
condition. Comparable NSE at lead 1, a different problem beyond it.

### Yang et al. 2026 — h-Diffusion, hourly

516 CAMELS-US basins (Gauch hourly set), train 1990-10-01 to 2003-09-30, test
2003-10-01 to 2014-09-30. Median per-basin **hourly** NSE.

| | single forcing | multi forcing |
|---|---|---|
| MTS-LSTM | 0.763 | 0.812 |
| MF-LSTM | 0.756 | 0.805 |
| h-Diffusion | 0.780 | 0.800 |
| h-Diffusion-DA (RePaint inpainting) | **0.832** | **0.840** |

**No daily-scale metric exists in that paper** — daily discharge is an *input*
to h-Diffusion (observed or δHBV-simulated), never a scored output. Its DA is
training-free masked-token conditioning, the same mechanism family as ours.

---

## 3. Our results, with the command that produces each

Shared prefix:

```bash
COMMON="--nc data/CAMELS_Frederik.nc --holdout 01,11,17 --time-aligned \
        --retrieval geo --context-pool all --train-end 9000 --eval-start 600 \
        --seed 0 --causal --geo"
```

### Task 1 — forward run (no discharge anywhere)

512-day window, all patches scored.

```bash
# ours, one checkpoint             -> 0.7268
python -m hydropfn.train.train_pub $COMMON \
  --k-train 0,0,0,1,2,4,8,16 --epochs 160 --steps 300 --tag P_both160

# forward specialist, seeds 0/1/2  -> 0.7164 / 0.7208 / 0.7321
python -m hydropfn.train.train_pub $COMMON --k-train 0 --k-eval 0 \
  --epochs 160 --steps 300 --seed 0 --tag P_fwd160

# the LSTM bar, identical split    -> 0.7173
python experiments/lstm_baseline.py --nc data/CAMELS_Frederik.nc \
  --holdout 01,11,17 --train-end 9000 --eval-start 600 --epochs 60 --steps 200
```

### Task 2 — self t-1 (own gauge, no neighbours)

```bash
# THE PRETRAINED ARM, read at lead 1 -- SAME weights as Tasks 1/3/4
python -m hydropfn.train.train_pub $COMMON --k-eval 0,4 --score-tail 1 \
  --roll 100 --self-ctx 1 --by-lead --load logs/pub_Z_trained2.pt

# separate patch=1 specialist        -> 0.8765
python -m hydropfn.train.train_pub $COMMON --eval-start 9600 --patch 1 \
  --win 64 --self-da 1 --roll 200 --k-train 0 --k-eval 0 \
  --epochs 160 --steps 300 --tag J_da_k0
```

Shared-weight model, by lead day:

| lead | 1 d | 2 d | 4 d | 8 d | 16 d |
|---|---|---|---|---|---|
| K=0 (self only) | **0.8131** | 0.8117 | 0.7756 | 0.8184 | 0.8086 |
| K=4 (self + neighbours) | **0.8959** | 0.8906 | 0.8854 | 0.9005 | 0.8936 |

The separate `patch=1` model reaches **0.8765** at K=0, so unifying costs
~0.064 on this task: the patch=16 model is trained to predict a 16-day block
uniformly and never learns to exploit the *recency* of the last observation.

#### Why the by-lead numbers differ from the Task 3 numbers

This confuses people (it confused the author), so state it explicitly.

The model predicts a **16-day patch** and emits **all sixteen daily values**.
The by-lead table and the Task 3 table score **different subsets of the same
predictions**:

| number | scored over | values per basin |
|---|---|---|
| 0.8822 (Task 3) | **all 16 days** of the patch | ~640 (40 origins x 16) |
| 0.8816 | **all 16 days** of the patch | ~1600 (100 origins x 16) |
| 0.8959 (by-lead) | **day 1 only** | ~100 (100 origins x 1) |

The like-for-like pair is **0.8822 vs 0.8816** — the same quantity, differing
only in K (8 vs 4) and origin count (40 vs 100), i.e. identical within noise.
**0.8959 is a subset of that same run**: only the first day of each predicted
patch. Day 1 is the easiest of the sixteen because it sits closest to the last
observation, so scoring it alone gives a higher number.

Nothing improved between the two; a narrower target was scored. The lead-1
advantage over the 16-day average is about **+0.014** (0.8959 vs 0.8816 at K=4;
0.8131 vs 0.8026 at K=0).

#### On "flat" lead curves

The K=0 curve reads 0.8131 / 0.8117 / 0.7756 / 0.8184 / 0.8086 across leads
1/2/4/8/16 — **non-monotonic** (lead 8 beats lead 4), so this is noise around a
flat trend, not a clean decay and not perfectly flat either. The honest
statement is *no systematic decay, with lead 1 marginally best*. Compare
Jamaat's monotone 0.820 -> 0.756 over nine days.

The reason for the difference in shape: we run with **known forcings**, so a
later day is not harder in the way it is for a model propagating an initial
condition forward. Their decay measures state-adjustment relevance fading;
ours measures position inside a simulated block.

### Task 3 — neighbours up to t (concurrent)

16-day tail, 40 rolling origins.

```bash
python -m hydropfn.train.train_pub $COMMON \
  --k-train 0,0,0,1,2,4,8,16 --k-eval 0,1,2,4,8 --self-ctx-p 0.4 \
  --score-tail 1 --roll 40 --epochs 160 --steps 300 --tag Z_trained2
```

| | NSE |
|---|---|
| ours, K=8 + self-context | **0.8822** |
| ours, K=8 | 0.8724 |
| + mask mixture | 0.8689 |
| + variogram attention bias | 0.8697 |
| + drainage-area scaling | 0.8633 |
| **IDW kriging — the honest baseline** | **0.8390** |
| `ctx_mean` / `nn_donor` — weak baselines | 0.8306 / 0.7906 |
| gauged-ceiling LSTM (test gauge in training) | 0.8127 |

Margin over real spatial interpolation is **+0.033**, not the +0.08 against
`nn_donor`. Neither geometry refinement helped; both refine a **Euclidean**
metric, which is likely the binding constraint — two gauges 50 km apart on the
same river share water, two 50 km apart across a divide share only weather, and
lat/lon cannot tell them apart. Flow-network distance from the DEM arm is the
untested idea with headroom.

### Task 4 — other conditionals

```bash
python -m hydropfn.train.train_pub $COMMON --area-scale --k-eval 4 \
  --load logs/pub_W_areamix.pt --eval-conditionals
```

| conditional | on QObs | on precipitation |
|---|---|---|
| `random_span` (gap filling) | 0.7279 → **0.8112** | 0.0539 → **0.9372** |
| `whole_variable` (cross-variable) | 0.8512 → 0.8598 | 0.0503 → **0.9308** |
| `causal_tail` (forecasting) | 0.5512 → **0.7491** | −0.0779 → **0.8684** |

"before → after" is a checkpoint trained *without* the mask mixture versus one
trained with it. Cost on Task 3: **0.004**.

**Caveat:** context sites are fully visible, so their precipitation is in view.
Much of the precipitation result is likely spatial interpolation of neighbouring
rain rather than inference from the query's own discharge. The K=0 version of
that test has not been run.

---

## 4. Known caveats

- **Nothing here is zero-shot.** Self-as-context scores 0.2490 untrained and
  0.7878 as a training draw. A capability absent from the pretraining
  distribution does not exist at inference.
- **Untrained input configurations actively hurt.** Hiding all discharge scores
  0.851; hiding only a span 0.728; hiding only the tail 0.551 — strictly more
  visible data, monotonically worse, because partial visibility of the query's
  own discharge was never trained.
- **Single seed** for most rows. The LSTM tie is 3 seeds (spread ~0.015); treat
  differences under ~0.015 as noise.
- **`patch=16` and `patch=1` are different models** — though only in 1% of
  parameters (`value_proj`, `head`). Task 2's 0.8765 does not come from the
  shared weights; its 0.8131 does.
- **Protocols differ between task blocks** (full-window vs tail-scored vs
  patch=1) and are not cross-comparable. Compare only within a block.
