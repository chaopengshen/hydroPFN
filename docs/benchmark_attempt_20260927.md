# Corrected-sampling temporal benchmark attempt

User authorized trying to beat the benchmarks at https://mhpi.github.io/benchmarks/
and publishing the training-protocol impact on GitHub.

## Comparison scope

The website's 15-year table uses **671 basins**, whereas the current experiment
uses 531. Its 10-year table has different dates. Neither is an exact comparator.
The linked [Li et al. 2025 study](https://hess.copernicus.org/articles/29/6829/2025/)
reports the selected 531 basins, distinguishes individual-model metrics from
ensembled predictions, and includes Daymet-only and multi-forcing comparisons.
Its Section 3.1 gives approximate Daymet single-model NSE around 0.735 (LSTM)
and 0.74 (HBV), and approximately 0.79 for their cross-model ensemble. The
headline near 0.83 uses a broader multi-forcing/model ensemble. These must
not be presented as one interchangeable benchmark. HBV training basin counts
also differ from the LSTM's in that study (Table 2).

Our separate prior dHBV1.1p reference is a user-supplied summary with matched
basins and dates, documented in the protocol. It is not independently audited
here from its prediction arrays and is not a replacement for the site's
ensemble benchmark. A single-model result must not be labeled surpassed by
an ensemble without explicitly stating the different member counts.

## Authorized bounded experiment

Keep the current 531-basin temporal dates, Daymet inputs, K=0, frozen PFN,
512-day input / 256-day warmup, objective and decoder settings. Train LSTM
and TCN hybrids with **random daily starts**, seeds 0,1,2, for 300 epochs:
six new runs. Record fixed 100/200/300 checkpoints and evaluate only after
training. The earlier long runs used the defective fixed-window grid and
cannot settle whether longer training helps under corrected sampling.

At each checkpoint report all six individual scores, with seeds kept separate.
The user explicitly requested single-model comparisons only. Do not combine
predictions across seeds or architectures, or choose the best epoch on test.
The already-inspected test makes this an exploratory benchmark attempt;
it is not a new untouched validation claim. Encoder pretraining used the
full training era, so a clean inner temporal validation experiment would
require refitting it inside that split. No claim of such validation is made.

The combined PFN model has bidirectional forcing context; these are hindcast
results and are not interchangeable with causal forecasts. There are no
query or neighboring discharge inputs at inference. No claim to beat the
multi-forcing benchmark follows from matching a Daymet single-model number.

## Publication and reproducibility

The fixed/daily sampling evidence is published under
`results/window_sampling_20260926/`: scores, per-basin scores, paired effects,
gap changes, provenance and audit. The authoritative interpretation remains
in `docs/camels531_protocol.md`. Code and test instructions accompany it;
raw CAMELS data, checkpoints and unpublished manuscripts are not uploaded.

Each training job writes `checkpoint_scores.json` and per-basin checkpoint
NSE arrays after training. No separate ensemble scoring job is needed.

## Launch record

- Training array: **55859797**, six runs, maximum four concurrent GPU jobs.
- Queue/account: ICDS `mgc-mri` / `cxs1024_mri`, V100S, eight-hour limit.
- Source commit: `7e8c8c5` (training code unchanged by the subsequent scope update).
- Immutable source: `/storage/group/cxs1024/default/cxs1024/hydroPFN/benchmark_daily_20260927/source`.
- Archive SHA256: `5309d687183f91a3801f9ec42915103c10f19b4a6daf834642aec8384f6f5ea7`.
- Source manifest SHA256: `75871d9821b1733e4deac1b92fddd2fa4a3f75120ded844275eaeec7e7221b03`.
- Dependent ensemble report **55859798 was cancelled** on the user's scope update.
  The snapshot retains the unused historical scorer; no ensemble report will run.
- Single-model checkpoint evaluation is already part of every training job.

The previous two-seed 100-epoch single-model results are LSTM
0.718254/0.724341 and TCN 0.717008/0.715528. No benchmark victory is claimed.

## Completed single-model results (2026-09-28)

All six array tasks completed with exit code 0 in approximately 2h07–2h17.

| Model | Seed | Epoch 100 | Epoch 200 | Epoch 300 |
|---|---:|---:|---:|---:|
| HydroPFN+LSTM | 0 | 0.7183 | 0.7122 | 0.7007 |
| HydroPFN+LSTM | 1 | 0.7243 | 0.7075 | 0.7045 |
| HydroPFN+LSTM | 2 | 0.7113 | 0.7063 | 0.6935 |
| HydroPFN+TCN | 0 | 0.7098 | 0.7117 | 0.7110 |
| HydroPFN+TCN | 1 | 0.7131 | 0.7087 | 0.7065 |
| HydroPFN+TCN | 2 | 0.7129 | 0.7123 | 0.7110 |

No model reached the recorded 0.7431 specialist reference. All LSTM seeds
degraded from 100 to 300 epochs; TCN changes were small and mixed.
Corrected sampling helped substantially in the controlled earlier ablation,
but extending decoder training alone does not close the remaining gap.

These are individual models, not ensembles. All prescribed checkpoints are
reported; no best-test-checkpoint selection is used. The TCN epoch-100
scores differ from the previous 100-epoch experiment and must be kept
separate; the source of that repeat variability has not yet been established.

Scores, run configurations and training loss histories are preserved in
`results/benchmark_daily_20260927/`.
