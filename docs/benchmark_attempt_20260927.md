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

At each checkpoint report all six individual scores, three-seed equal-weight
LSTM and TCN ensembles, and their six-member equal-weight ensemble. Do not
choose the best epoch, subset of members or weights using test scores.
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

The ensemble scorer is `experiments/benchmark_decoder_ensembles.py`. It also
supports rescoring the existing two-seed 100-epoch runs via
`--seeds 0,1 --epochs 100 --tag-template window_{model}_hybrid_daily_s{seed}`.
Prediction arrays must be aligned and finite; scores average predictions,
not individual NSE values. All member combinations above are reported.

Compute uses ICDS `mgc-mri` / `cxs1024_mri`, V100S, at most four concurrent
GPU jobs, with a dependent CPU report on `basic` / `open`. No paid standard
queue is used. Launch records and immutable source hashes accompany the run.
