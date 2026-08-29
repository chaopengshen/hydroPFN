> **DEPRECATED for external comparison (2026-08-29).** Every NSE on this page is computed on `log1p(QObs)`, z-scored — NOT raw mm/day — on a 671-basin protocol of this project's own invention. None of it may sit in a table with a published CAMELS value. The verified protocol and metric live in [camels531_protocol.md](camels531_protocol.md). Internal comparisons on this page (ours-vs-ours, shared protocol) remain valid.

# Every number, one table, with its protocol

**This page is the single source of truth for results.** The README links here
and does not restate numbers; anything quoted elsewhere that disagrees with
this page is stale.

---

## Highlights — the conclusions worth remembering

1. **Nothing is zero-shot.** A capability absent from the pretraining task
   distribution does not exist at inference. Self-as-context: **0.249**
   untrained, **0.788** once trained as a draw. Verified mechanically too — in
   a K=0-only checkpoint the cross-site attention weights sit at their random
   init to five decimals, because that branch never executed.

2. **Presence is not enough; SHARE matters.** K=0 at 1/6 of tasks gives 0.658;
   K=0 upweighted gives 0.7164 and ties an LSTM. The same default cost the
   1-day assimilation task **0.17**. A pathway that is always available in
   training gets relied on; one that is starved never forms.

3. **Untrained input configurations actively HURT.** Hiding all discharge
   scores 0.851, hiding only a span 0.728, hiding only the tail 0.551 —
   strictly *more* visible data, monotonically worse, because partial
   visibility of the query's own discharge was never trained.

4. **Route matters, and it splits by whose data it is.** For a site's **own**
   lagged record, feeding it through that site's own token stream (`--self-da`,
   **0.8765**) decisively beats routing it through cross-site attention
   (`--self-ctx`, **0.7202**, or **0.5955** at matched allocation) — a gap of
   0.18–0.28 at matched resolution. For **other sites'** data cross-site
   attention is the only route available, and it works, worth **+0.39**.
   **So: own data down the site's own stream; other sites through
   cross-attention.** Composability is NOT free — an earlier claim that the
   context route was +0.11 better came from a confounded `patch=16` pair and
   is retracted.

5. **In-context reading of a gauge does NOT beat training on that gauge.** It
   recovers about **64%** of the benefit (+0.080 vs +0.125 over having nothing).
   What beats training on the gauge is adding **neighbours** — concurrent
   observations elsewhere are information no amount of training on the target
   can supply.

6. **The margin over honest spatial interpolation is +0.033, not +0.08.**
   Inverse-distance weighting reaches 0.8390 against our 0.8724. `nn_donor`
   (single donor) and `ctx_mean` (uniform mean) are strawmen.

7. **Data latency degrades gracefully**: −0.028 over an eightfold increase in
   staleness. A record four months old still beats having none. Because
   staleness was a training draw.

8. **Causality is free** (0.837 → 0.846) — so the model can be a filter rather
   than a smoother at no cost. **Geo-encoding helps only on the path that
   actually mixes context**; on the near-dead connector it was worth 0.003.
   Neither a variogram attention bias nor drainage-area scaling helped, which
   points at **Euclidean distance being the wrong metric** — the case for
   flow-network distance from the DEM arm.

9. **Compare only within a protocol.** Five wrong conclusions in this project
   came from comparing across them; the list is at the bottom of this page.

---

Numbers in this project have been quoted across four different protocols, and
comparing across them has produced wrong conclusions more than once. This page
exists so that never happens again: **every result, with the protocol column
visible, and an explicit statement of which rows may be compared.**

All values are **median per-basin NSE**. Split is PUR throughout (HUC2 regions
`01,11,17` held out entirely; 124 eval / 547 train; train windows end day 9000,
evaluation after day 9600). Verify with
`python scripts/verify_split.py --nc data/CAMELS_Frederik.nc`.

## The four protocols

| id | protocol | what is scored |
|---|---|---|
| **A** | full window | all 32 patches × 16 d = 512 daily values per basin |
| **B** | tail, all days | last patch only, 16 d × N rolling origins |
| **C** | tail, lead 1 | **day 1 of the last patch only**, × N origins — a *subset of B* |
| **D** | patch=1 | 64-day window, last **day** hidden, × 200 origins |

**Compare only within a protocol.** A vs B differ in target; B vs C differ in
which days of that target are scored; D is a different model (`patch=1`).

---

## Protocol A — full window

| model | information available | NSE |
|---|---|---|
| regional LSTM | forcings + statics | 0.7173 |
| ours, one checkpoint, K=0 | forcings + statics | **0.7268** |
| ours, forward specialist (3 seeds) | forcings + statics | 0.7164 / 0.7208 / 0.7321 |
| ours, one checkpoint, K=8 | + 8 neighbours, concurrent | 0.8530 |
| gauged-ceiling LSTM | **test gauge in training** | 0.8127 |

Ours ties the LSTM as a pure forward model, and exceeds the gauged ceiling once
neighbours are supplied.

---

## Protocol B — tail, all 16 days, 100 rolling origins

**This is the ladder.** Every row here is directly comparable.

| model | information available | NSE |
|---|---|---|
| LSTM, ungauged | forcings + statics | **0.7222** |
| ours, self-as-context, K=0 | + the gauge's **own** record to t−1 | **0.8026** |
| LSTM, **trained on the test gauge** | the gauge's history, baked into weights | **0.8477** |
| ours, self-as-context + K=4 | own record **and** 4 concurrent neighbours | **0.8816** |

What this says, stated carefully:

- Reading a gauge's own record **as context at inference** is worth **+0.080**
  over having nothing (0.7222 → 0.8026).
- **Training on that gauge** is worth **+0.125** (0.7222 → 0.8477).
- So in-context assimilation recovers about **64%** of what training on the
  gauge delivers. **It does not beat it.** Any claim that in-context conditioning
  substitutes for training on a site is not supported.
- What *does* beat training on the gauge is adding **neighbours** (0.8816),
  because concurrent observations elsewhere are information no amount of
  training on the target can supply. That is the defensible claim.

### Protocol B, other configurations (40 origins, K sweep)

| configuration | K=8 |
|---|---|
| ours, K=8 + self-context | 0.8822 |
| ours, K=8 | 0.8724 |
| + mask mixture (4 conditionals) | 0.8689 |
| + variogram attention bias | 0.8697 |
| + drainage-area scaling | 0.8633 |
| **IDW kriging — the honest baseline** | **0.8390** |
| `ctx_mean` / `nn_donor` — weak baselines | 0.8306 / 0.7906 |

0.8822 (40 origins, K=8) and 0.8816 (100 origins, K=4) are the same quantity
measured twice — identical within noise.

---

## Protocol C — tail, lead day 1 only

**A subset of Protocol B**, not an improvement over it. Same predictions,
narrower target: the model emits all 16 daily values of the patch and this
scores only the first.

| model | K=0 | K=4 |
|---|---|---|
| ours, self-as-context, **lead 1** | **0.8131** | **0.8959** |
| ours, same run, **all 16 days** (= protocol B) | 0.8026 | 0.8816 |

Lead 1 is worth about **+0.014** over the 16-day average, because day 1 sits
closest to the last observation. Full lead curve:

| lead | 1 d | 2 d | 4 d | 8 d | 16 d |
|---|---|---|---|---|---|
| K=0 | 0.8131 | 0.8117 | 0.7756 | 0.8184 | 0.8086 |
| K=4 | 0.8959 | 0.8906 | 0.8854 | 0.9005 | 0.8936 |

Non-monotonic (lead 8 beats lead 4), so this is **no systematic decay**, not a
clean flat line either. Contrast Jamaat's monotone 0.820 → 0.756 over nine days:
they propagate an initial condition forward, we run with **known forcings**, so
"lead" here is position inside a simulated block.

---

## Protocol D — patch=1, separate model

| model | route for the site's own t−1 data | K=0 training | K=0 |
|---|---|---|---|
| `J_da_k0` | **`--self-da`** — own token stream, self-attention | 7/7 | **0.8765** |
| `X_p1ctx` | **`--self-ctx`** — own basin as a context entry | 3/7 | **0.7202** |
| `X_p1ctx_k0` | `--self-ctx`, matched allocation | 7/7 | **0.5955** |
| *Jamaat 2025, variational DA* — **gauged basins** | own gauge, optimisation per step | — | *0.82* |

`X_p1ctx` at K=4 reaches 0.7901.

### The confound this resolves

`J_da_k0` (0.8765, protocol D) has been compared against `Z_trained2` (0.8131,
protocol C) to argue the `--self-da` route is stronger than the `--self-ctx`
route. **That comparison is invalid**: those runs differ in *both* route and
patch size. `X_p1ctx` is the missing cell — `patch=1` with the context route —
and it separates the two explanations:

**Result: it is the ROUTE.** At matched `patch=1`, the self-da route scores
0.8765 and the context route 0.7202 — a gap of **0.156** in favour of feeding a
site's own lagged data through its own token stream rather than through
cross-site attention.

This **reverses an earlier claim**. On a `patch=16` pair the context route
looked +0.11 *better* (0.7878 vs 0.6764), and that was used to argue
composability came free. It does not: at matched resolution, composability
appears to cost accuracy.

**Allocation ruled out.** `X_p1ctx_k0` matches `J_da_k0`'s allocation exactly
(K=0 on 7/7, self-context on every task) and scores **0.5955** — *worse* than
the 3/7 run, not better. So the gap is not allocation; the true route effect is
larger than 0.156.

Why the matched run is worst of the three: with `--k-train 0 --self-ctx-p 1.0`
every task is "K=0 plus self-context", so cross-site attention only ever sees
the degenerate self case and never learns general context use. Diversity in the
draw distribution matters here too.

---

## Reference results (context, not a leaderboard)

Both assimilate the **target gauge's own record** on **gauged** basins present
in training, temporal split only. Ours is PUR.

| | NSE | basins |
|---|---|---|
| Jamaat 2025, δHBV no DA | 0.75 | 531 gauged |
| Jamaat 2025, LSTM no DA | 0.74 | 531 gauged |
| Jamaat 2025, δHBV-DA-PS / LSTM-DA (1-day lead) | 0.82 | 531 gauged |
| Yang 2026, h-Diffusion — **HOURLY NSE** | 0.780 | 516 gauged |
| Yang 2026, h-Diffusion-DA — **HOURLY NSE** | 0.832 | 516 gauged |

**The Yang rows are not comparable to anything else on this page and must not
be placed beside our numbers.** They are *hourly* NSE; every other number here
is daily. Hourly and daily NSE are different targets, and neither bounds the
other. Yang reports **no daily-scale metric at all** — daily discharge is an
*input* to h-Diffusion (observed or δHBV-simulated), never a scored output. The
paper is relevant for its *mechanism* (RePaint inpainting as training-free DA,
the same family as ours), not for its numbers.

---

## Data latency — what if the most recent observation is missing?

Operationally decisive, since real gauges have reporting lag and outages.
Protocol B, K=0 (no neighbours), varying how stale the site's own record is:

| last observation | NSE |
|---|---|
| 16 days stale | 0.7878 |
| 32 days stale | 0.7751 |
| 64 days stale | 0.7622 |
| 128 days stale | 0.7603 |
| *no record at all (LSTM)* | *0.7222* |

**−0.028 over an eightfold increase in staleness.** Even ~4 months stale, the
record is still worth +0.038 over having none. The degradation is graceful
because tail length was **drawn randomly during training** (U{1…15} patches),
so staleness was in the task distribution — the same principle that governs
everything else here: the capability exists because it was trained.

---

## Mistakes this page exists to prevent

Each of these produced a wrong conclusion that survived until someone checked:

1. **Daily vs 16-day means.** Our model scored daily, the LSTM scored 16-day
   means, under a comment claiming they matched. Aggregation is worth +0.023.
2. **Full window vs tail.** A full-window 0.8530 was compared against a
   tail-scored 0.8724 as though one were better.
3. **Lead-1 vs all-days.** 0.8959 was reported alongside 0.8822 as if something
   had improved; it was a narrower target on the same predictions.
4. **Route vs resolution.** 0.8765 vs 0.8131 was read as a route effect while
   patch size also differed.
5. **Weak baselines.** `nn_donor` (single donor) and `ctx_mean` (uniform mean)
   flattered us by ~0.03 relative to inverse-distance weighting.


---

## Other units (not the time-series arm)

These measure different things and live on their own protocols.

### DEM sampler (Objective B)

60 held-out 1° tiles, K=8 draws, success bands fixed before running. A
deterministic net is capped at ~0.31 of true fine-scale spectral power, and the
cap is *structural*: an L1/L2 loss returns the conditional mean and the mean of
plausible terrains is smooth. Final recipe = residual parameterisation +
best-of-K reranking. Detail in the README's Result 1.

### Measurement PFN (unit D)

A site's own past measurements, read in context, beat a hand-engineered
"deviation from attribute prediction" feature fed to a random forest:
**+0.127 R²** cross-variable against +0.041. 9/9 across regions and seeds.
Detail in the README's Result 2.

### SUPERSEDED

The README's **Result 3** (mask-mode table) and **Result 4** (ceiling 0.8606 /
ours 0.8447) predate the 2026-08-24 scoring correction and **should not be
quoted**. Result 4 in particular compared a 16-day-mean LSTM number against our
daily number; corrected, the gauged ceiling is 0.8477 and we reach 0.8816 with
neighbours (protocol B above).
