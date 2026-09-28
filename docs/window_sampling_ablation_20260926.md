# Fixed versus random daily window sampling

Authorized 2026-09-26 to remove the fixed 79-start restriction and quantify
its contribution to the temporal PFN-versus-specialist gap. This experiment
changes sampling only; it is not an end-to-end encoder training experiment.

## Treatments and implementation

`experiments/camels531_decoders.py train` now defaults to `--sampling daily`.
Each batch draws uniformly from every valid daily start in the training
interval (4,967 possible 512-day starts). The frozen encoder computes features
from that exact crop. Absolute calendar offsets are preserved, query Q stays
zero and masked, and the complete crop and scored suffix remain inside the
training interval. No cached features from a different crop are reused.

The explicit `--sampling fixed` treatment draws from the original 79 starts.
`--feature-mode online` is used in **both treatments**, so features are
recomputed by identical code and hardware. The original cache still provides
unchanged normalization, targets, input arrays and evaluation features.
`--feature-mode cached` remains available only for fixed-window diagnostics.
Old immutable snapshots are untouched; legacy launch scripts now explicitly
request fixed sampling instead of depending on the changed CLI default.

One NumPy uniform draw selects the start in both treatments, with identical
basin batches and RNG consumption. This paired uniform sampler differs from
the earlier integer sampler, so new fixed controls are trained instead of
using historical scores as exact controls. Decoder initialization happens
after the encoder is loaded and the torch seed is reset. Initial-weight and
basin/random-stream hashes are saved and checked by the analysis.

## Prespecified matrix

Five model/input combinations: raw LSTM, raw TCN, PFN+LSTM, PFN+TCN, and the
original PFN patch head (frozen query features only). For each, run fixed and
daily sampling at seeds 0 and 1: **20 training runs, organized into 10 pairs**.

Each fixed/daily pair runs sequentially in one Slurm GPU allocation. Seed 0
runs fixed then daily; seed 1 reverses that order. Every run has 100 epochs,
388 updates per epoch, batch 128, 512-day input and 256-day warmup/scored
suffix, Adadelta lr 1, gradient clip 1, the same basin-normalized MSE and
readout settings as the earlier decoder pilot. LSTM dropout remains 0.1;
TCN dropout remains 0.1 within its residual branches; the original patch head
has no dropout, matching its architecture. Encoders remain frozen and in eval
mode. Encoder chunk size is 32. No change to sequence length, checkpoint
initialization strategy, loss, regularization or evaluated dates is bundled
into the sampling treatment.

The endpoint is epoch 100, not the test-selected best checkpoint. Random
starts change both contextual diversity and the weighting of individual
training days; this experiment measures their combined sampling effect.
It does not independently isolate patch alignment versus temporal weighting.
Existing future-period results have already been inspected, so this is a
diagnostic experiment, not newly untouched confirmatory validation.

## Readout of the experiment

`experiments/window_sampling_diagnostics.py` runs after successful completion
of the full array. It checks gauge/target/cache alignment, finite predictions,
identical initialization and basin/random streams, identical paired job/GPU,
equal optimizer budgets and unchanged encoder weights.

For each model/seed, report daily-minus-fixed median NSE, the median paired
basin change and fraction of improved basins. For each LSTM/TCN pair:

    gap_fixed = median_NSE(raw, fixed) - median_NSE(hybrid, fixed)
    gap_daily = median_NSE(raw, daily) - median_NSE(hybrid, daily)
    gap_reduction = gap_fixed - gap_daily

Positive gap reduction means random sampling narrows the hybrid deficit.
Improvement in the hybrid alone is insufficient if the raw control improves
just as much. With two seeds, avoid treating basin counts as independent seed
replicates or claiming universal resolution of the architecture gap.

## Validation and launch

Nine regression tests cover the old decoder checks plus all-start support,
inclusive training bounds, paired sampling RNG, exact absolute-time crop
construction, online/cached same-crop agreement, frozen encoder/RNG checks,
and complete fixed/daily training and scoring paths with paired initialization.

The real-cache GPU smoke on suntzu RTX 2080 Ti passed at batch 128, length 512.
Its maximum same-crop feature difference from the old GPU cache was 1.34e-5,
within the declared floating-point tolerance. Encoding consumed no dropout
RNG, encoder weights stayed unchanged, and full updates were finite for all
three PFN readouts. This numerical cross-device difference is why both actual
treatments use online encoding on the same allocated GPU.

ICDS uses group `mgc-mri` / `cxs1024_mri`, V100S, 4 CPUs and 16 GB per pair,
8-hour limit and at most four concurrent pairs. A short GPU preflight must
succeed before the training array starts. An `afterok` CPU job on `basic` /
`open` aggregates the results. No paid standard-queue resources are requested.

Remote experiment root:
`/storage/group/cxs1024/default/cxs1024/hydroPFN/window_sampling_20260926`.
`encoder/` contains the original source checkpoint plus normalization/run
metadata; their hashes must match the cache provenance before training.
`cache` references the unchanged earlier cache. Code is archived under
`source/`, scheduler logs under `slurm/`, outputs under
`source/logs/camels531/window_*`, and the final report under `diagnostics/`.

Submitted jobs:

- GPU preflight `55837214`.
- Training array `55837216`, indices 0-9, at most four active pairs.
- CPU report `55837217`, dependent on successful completion of the array.

Each seed occupies five array indices in the model order listed above;
indices 0-4 are seed 0 and 5-9 seed 1. The immutable archive contains 121
files. Archive SHA256:
`2b6fa6816fc1383b17a18fc4b8c28afb0031ea3af1242fb237611d1c9aa20537`.
Source manifest SHA256:
`7410007b67e1640758c62d240d04d9833fff0bdcd4b4d4524595864cced33ca9`.
Cache manifest SHA256:
`03ae30cb55854c39a99fd898353672e20686792705a2b5313ecfbd78aaca0704`.
Local submission record: `logs/recovery_setup/window_sampling_submission.json`.

### Active revision and ICDS verification

The initial preflight `55837214` passed all nine regression tests but stopped
on the historical cross-GPU cache tolerance (64 repeated daily elements,
corresponding to four underlying patch-feature elements, exceeded the initial
mixed relative/absolute tolerance). Dependent array `55837216` and report
`55837217` were cancelled before running. No training results were produced.

Revision 2 separates exact same-device crop-construction equivalence from
cross-device numerical agreement. On ICDS V100S, the former is **exactly 0.0**;
the historical cache's maximum absolute difference is 8.01e-5, within the
revised cross-device 1e-4 absolute / 1e-5 relative tolerance. Both experimental
treatments recompute training features on the same device/backend, so this
historical numerical difference is not the fixed/daily treatment. Source and
normalization hashes remain unchanged. All nine tests passed on ICDS and all
three full-size online training smokes had finite losses, unchanged encoder
weights and unchanged dropout RNG during feature extraction.

**Active jobs:** preflight `55837225` COMPLETED (20 seconds, exit 0:0),
training array `55837226`, dependent report `55837227`.
Active code and training outputs are under `source_v2/`; `source/` preserves
the original failed-preflight snapshot. No model/training code changed
between these two deployment revisions; only the GPU preflight check and
local documentation changed.

Revision-2 archive SHA256:
`b6640d75b31d4d6f5c983b1bde6ff142a8cbb33cf68866a04f44e4a637710277`.
Revision-2 source manifest SHA256:
`d5cbd27311fe82b6ef761fd0254f085da27fec57726281326aafa25729cc27c5`.

Startup verified: array tasks 0-3 are RUNNING on V100S nodes p-mc-3470/3471;
each completed at least one epoch with finite loss and all 79 fixed starts
represented. These first four seed-0 pairs begin with their fixed treatment;
the corresponding daily treatment follows in the same allocation. Tasks 4-9
are queued under the four-pair concurrency limit. The report remains pending
on successful completion of the full array. No sampling-effect result is
available at this startup check.

## Completed results, checked 2026-09-27

All ten GPU pairs (20 training runs) and the CPU report completed successfully,
exit 0:0. The analysis verified all prescribed matching conditions and unchanged
encoder weights. Random daily starts substantially improve both PFN hybrids
in both seeds; each now exceeds its corresponding daily raw control. The
refitted original patch head also improves substantially. This supersedes
the earlier fixed-grid pilot's conclusion of no consistent hybrid benefit.

Authoritative numbers and interpretation are in
[the protocol](camels531_protocol.md#window-sampling-ablation-completed-checked-2026-09-27).
The complete analysis is copied locally to
`logs/recovery_setup/window_sampling_diagnostics/`. The effect includes changes
to contextual diversity, patch alignment and target-day weighting; it does
not identify which mechanism dominates. The frozen-PFN hindcast information
and two-seed limitations remain in force.
