# LSTM and gated TCN decoder comparison

Authorized 2026-09-23: implement and try LSTM as the reference and TCN as the
parallel alternative. This initial matrix tests forward temporal K=0 prediction;
it does not claim daily data integration or unseen-basin parity.

## Implementation

- `src/hydropfn/models/temporal_decoders.py`: daily LSTM, pointwise control and
  gated residual TCN with a shared feature interface. The raw-input LSTM can
  reproduce `RegionalLSTM` exactly with copied weights and matching dropout.
- TCN uses causal left padding, two GLU convolutions per residual block,
  channel-only LayerNorm, static FiLM conditioning, width 64, six blocks and
  kernel 3. Dilations 1,2,4,8,16,32 give a 253-day receptive field. There is no
  full-window temporal pooling or normalization that could leak the future.
- `PUBModel(return_hidden=True)` exposes the query representation after its
  existing connector paths, before its reconstruction head. The default
  reconstruction path is unchanged. Frozen features retain their patch phase
  when lifted to daily resolution through a learned projection/phase embedding.
- `experiments/camels531_decoders.py`: prepare verified features, train each
  decoder, save predictions and run a zero-latent sensitivity diagnostic.
- `scripts/run_decoder_comparison.py`: finite source-verified queue, then
  `experiments/decoder_diagnostics.py` performs paired basin comparisons.

The user-provided reference was found at
`G:/OneDrive - The Pennsylvania State University/Group/Working/SAO/revision/submission/manuscript_rev.pdf`
(the supplied extra directory separator before `_rev.pdf` was not present).
The manuscript is *Surrogate-aided Differentiable Hydrologic Model Optimization
with Sensitivity Constraints and Continual Training*. Sections 2.2 and 4.1,
Figure 1 and Table 1 motivate gated convolutions, static FiLM and measured
speed/accuracy comparisons. Its task emulates a physical solver; its very high
surrogate NSE is not an observed-discharge benchmark target for this experiment.
This implementation borrows selected architecture ideas, not its exact model:
the manuscript's 12-layer schedule, causal SE, stochastic depth, sensitivity
loss and online physical-solver training are not reproduced here. No source
code from that manuscript was supplied or assumed.

## Prespecified initial matrix

| inputs | pointwise control | LSTM | gated TCN |
|---|---|---|---|
| raw daily forcings + attributes | not needed | yes | yes |
| those inputs + frozen PFN query features | yes | yes | yes |

Five configurations, seeds 0 and 1, 100 epochs each. All configurations within
seed 0 run sequentially on GPU 2; those within seed 1 on GPU 4. Thus each
within-seed architecture or context contrast uses one device. Seed and device
are coupled across seeds, so variation between them is not pure seed variance.
This is a bounded ten-run pilot, not a hyperparameter sweep.

Common: Adadelta lr 1, gradient clip 1, dropout 0.1, batch 128, training-only
basin-normalized MSE with std floor 0.1 mm/day. LSTM hidden size is 256.
Parameter counts, update counts, training time, inference time and peak memory
are recorded. Architectures are not exactly parameter- or FLOP-matched.
Epochs use the existing sampling coverage formula adjusted for this window's
256 supervised days. RNGs sample the same basin/window sequence within each
seed, independently of network initialization/dropout. A 50-epoch checkpoint
is retained for later diagnostics, not selected on the test score.

## Information and history controls

Source PFN: the completed temporal MSE seed-0 checkpoint at
`experiments_20260923_v2/logs/camels531/stage1_pfn_mse_s0` on suntzu. It remains
frozen. Extract the full K=0 query stream with the existing pooled connector;
time-aligned neighbor attention is inactive because there are no neighbors.
Relative geography at K=0 is zero, but its learned projection is still applied.

No query discharge is passed into feature extraction: the encoder batch is
constructed from forcings, puts zero in the Q value, masks that channel, and
makes its query token valid independently of target availability. Missing
forcing values are tracked. Normalization is taken unchanged from the source
checkpoint after checking its training gauge IDs against all 531 basins.

The source encoder expects 32 patches of 16 days. All configurations therefore
get the same 512-day windows. Training supervises only the last 256 days;
candidate training windows advance by 64 days plus a final window ending at
the training cutoff. Targets never extend beyond 1995-09-30. This finite window
grid is shared by every model and is coarser than the original random-day
LSTM sampler.

Evaluation uses overlapping windows and scores a suffix with at least 256
preceding real days. The last input window ends at the test cutoff; there is
no right padding or use of dates beyond it. Every one of the 5479 scored dates
is written exactly once for every basin. Both recurrent and convolutional
models restart on these same windows: a continuously running LSTM is not being
compared with a repeatedly cold-started PFN. The TCN's entire 252-day direct
input history is available before the first scored day in each window.

This is **not** the canonical 365-warmup + 365-supervised training sequence.
The new raw-input LSTM is the matched reference for this experiment; compare
historical benchmark scores as context, not as an isolated decoder contrast.
The source's original patch head is also evaluated on exactly these inputs
and windows. Because query-valid handling and window starts differ from the
historical run, its score is not a pure history-only ablation.

PFN features retain the source encoder's bidirectional forcing attention.
Consequently PFN-augmented runs are retrospective hindcasts, and the causal
decoder does not make the combined model a causal forecaster. Raw-only models
are causal with respect to forcings. The additional PFN-versus-raw comparison
therefore also includes its within-window forcing information; interpret it
as an augmented-model feasibility comparison, not uniquely as representation
quality at identical causal information. Pointwise/LSTM/TCN hybrid comparisons
do have identical PFN features and information. Daily DI requires a separately
validated causal encoder/observation interface.

## Decision rules and limits

- Compare hybrid sequence decoders with the hybrid pointwise control for the
  value of temporal readout on identical available features.
- Compare each hybrid with its raw-input counterpart for net benefit of adding
  the frozen PFN path. Zeroing latent features at evaluation is an additional
  distribution-shift diagnostic, not a trained no-context fallback.
- Compare all scores by gauge and identical raw targets; report paired basin
  deltas as well as differences of medians. Bootstrap intervals do not account
  for spatial dependence or seed variance.
- Failure of frozen-feature decoders does not establish that an end-to-end
  adapted representation would fail. A raw branch reaching parity while PFN
  adds nothing does not establish value of the foundation representation.
- Future-period scores are already inspected benchmark results. No architecture
  or training-duration selection is made from these pilots; a later selected
  configuration requires training-era validation with a source encoder fitted
  inside that split. The present source encoder has seen all training years.

## Validation and status

Seven new tests passed on suntzu CPU: future-input/gradient causality for all
three readouts, exact raw-LSTM reference equivalence, TCN receptive field and
overlap equivalence, exactly-once scoring/bounds, daily patch alignment and Q
masking, unchanged original head outputs, and training/checkpoint/scoring smoke
for each decoder. Both existing connector geometry regression checks also
passed. The environment has no pytest; those two plain-assert test functions
were executed directly without adding dependencies.
The full-size GPU smoke (batch 128, 512 days, hybrid inputs) completed for all
three heads on RTX 3090 Ti. Peak allocated memory was about 1.05 GiB for LSTM,
1.16 GiB for TCN and 0.51 GiB for pointwise. Three timed steps after one warmup
averaged 0.0359, 0.0289 and 0.0061 seconds respectively. These synthetic-step
timings are not end-to-end training benchmarks or claimed overall speedups.
Artifact: `logs/recovery_setup/decoder_gpu_smoke.json`.

Immutable snapshot on suntzu:
`/nfs/data/cxs1024/hydroPFN/experiments_20260923_v7` (113 source files).
Archive SHA256:
`704a411778ee4a6eaf2e6c03efb53d3a3db3de1c7b18d6d226e97b652078ce80`.
Detached supervisor PID 937362; status and logs live under the snapshot's
`logs/decoder_comparison/`. It prepares the frozen feature cache first, then
executes the five-model matrix for seed 0 on GPU 2 and seed 1 on GPU 4.
Each job verifies the source manifest and checks its GPU is free before starting.
The supervisor runs paired diagnostics after all ten jobs finish.

Verified startup: preparation finished at 2026-09-23 16:26:11 UTC; both raw-LSTM
workers then started (seed 0 PID 939639, seed 1 PID 939638) and completed their
first epoch with finite losses. Each run has 388 updates per epoch. The completed
cache manifest is copied locally to `logs/recovery_setup/decoder_cache_manifest.json`.

## ICDS 300-epoch duration experiment

User authorized ICDS jobs and longer training on 2026-09-23, then requested
the default readout alongside LSTM/TCN. Three additional seed-0 runs use the
same byte-verified frozen feature cache, sampled windows, objective, batch,
optimizer and architecture settings as this pilot:

| readout | direct readout inputs | epochs |
|---|---|---|
| LSTM | daily raw forcings, attributes, frozen PFN query features | 300 |
| gated TCN | daily raw forcings, attributes, frozen PFN query features | 300 |
| original PFN patch head | frozen PFN query features | 300 |

The original patch head is exactly LayerNorm(256), Linear(256,256), GELU,
Linear(256,16), independently at each query patch. It has no extra raw-input
branch, phase embedding or added temporal model. Its encoder has already seen
the raw forcings and attributes. All three readouts start from random weights;
the source encoder remains frozen. Thus the default control is a refitted
original head, **not a 300-epoch end-to-end retraining of the PFN**. Its unequal
direct input interface is explicit; the earlier hybrid pointwise control
remains the cleaner isolation of a temporal decoder on common daily inputs.

Each run saves its weights at 100, 200 and 300 epochs (38,800, 77,600 and
116,400 updates). Only after all 300 epochs does it evaluate these fixed
checkpoints and write `checkpoint_scores.json`, `checkpoint_basin_nse.npz`
and intermediate predictions. These are test diagnostics, not model selection.
Duration effects are paired **within the same uninterrupted ICDS job**; a
100-epoch suntzu versus 300-epoch ICDS comparison alone would mix duration,
hardware and software effects. Only seed 0 is repeated at this longer budget.

The submission script is `scripts/decoder_long_icds.slurm`: one GPU per array
task, V100S, group `mgc-mri` / `cxs1024_mri`, four CPUs, 16 GB RAM and an
8-hour limit. It uses the existing `tdmpc2` environment (PyTorch 2.7.1+cu118),
verifies the source manifest, tests the readouts and requires CUDA before
training. Slurm owns GPU allocation. It does not use the paid standard queue.

Remote root:
`/storage/group/cxs1024/default/cxs1024/hydroPFN/decoder_long_20260923`.
Source, feature cache and scheduler logs have separate directories. Existing
suntzu snapshots and jobs are unchanged. Runtime/GPU/job IDs are also stored
in each result's configuration.

Pre-submission validation passed all seven decoder tests, now covering four
heads, exact original-head equivalence, checkpoint outputs and identical final
predictions with/without milestone saving/evaluation (including dropout).

Submitted as Slurm array **55769402**:

- `55769402_0`: `decoder_lstm_hybrid_s0_e300_icds`.
- `55769402_1`: `decoder_tcn_hybrid_s0_e300_icds`.
- `55769402_2`: `decoder_patch_latent_s0_e300_icds` (original readout).

Immutable source archive `decoder_long_source_v1.tar.gz` has 114 files and
SHA256 `2f2ccc65fe9224790d5a45a7c325079b83e8f4726b734851b872f19c0a255390`.
Source manifest SHA256:
`aa23d4e824b58570707f486b428b87891e4c4462c9237a46222cc7ec921cbf5e`.
Cache manifest SHA256:
`03ae30cb55854c39a99fd898353672e20686792705a2b5313ecfbd78aaca0704`.
All transferred source and cache files passed checksum verification before
submission. Local artifacts are under `logs/recovery_setup/decoder_long_*`;
remote predictions/checkpoints will be in `source/logs/camels531/<tag>/`.

Scheduler check at 2026-09-23 13:04 Eastern: all three tasks were pending for
resources; Slurm estimated 14:02:40 for task 0. This is a queue estimate, not
a guaranteed start or a training-completion claim. Jobs persist independently
of the SSH session.

The same seven tests also passed in ICDS's actual `tdmpc2` environment on CPU
(27.8 seconds, including first-use imports); GPU startup checks remain part of
each queued job.

## Completion and interpretation

All ten suntzu runs and all three ICDS 300-epoch tasks have now completed
successfully. The authoritative score tables and duration interpretation are
in [the protocol](camels531_protocol.md#frozen-pfn-decoder-and-duration-pilots-2026-09-23).
Longer training reduced training loss while worsening the headline temporal
test NSE in all three ICDS runs. TCN is the stronger readout in this pilot,
but adding frozen PFN features still provides no consistent benefit over its
raw-input control. This does not test end-to-end encoder adaptation.

Archived results: `logs/recovery_setup/decoder_comparison_diagnostics/` and
`logs/recovery_setup/decoder_long_results/`. Historical startup/queue notes
above describe submission-time state and are superseded by this completion.
