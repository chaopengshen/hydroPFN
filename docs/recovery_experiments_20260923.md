# hydroPFN recovery experiments - 2026-09-23

Authorized sequence: matched objective and temporal diagnostics; daily DI and
unseen-basin adaptation; one basin-aware or network-aware DEM experiment.
Compute runs on suntzu. Existing benchmarks and artifacts remain intact.

**Completion update:** all batches below, including all ten unseen-basin folds,
all 531 paired terrain footprints, and the subsequent paired-training pilots,
finished on 2026-09-23. Current authoritative results and provenance are in
[the protocol](camels531_protocol.md#recovery-pilots-completed-2026-09-23).
The pending/running descriptions below record earlier checkpoints in the run
history. The [measurement/DEM analysis](measurement_assimilation_dem_20260923.md)
contains the follow-on design; no new training batch was launched for it.

**Subsequent authorization:** the user then requested LSTM/TCN decoder tests.
Those implementations and the newly launched frozen-PFN comparison are tracked
separately in [decoder comparison](decoder_comparison_20260923.md). The earlier
completion statement concerns the recovery/parity batches, not this new batch.

## Stage 1: objective and temporal dynamics

Use the verified CAMELS-531 temporal protocol: training 1980-10-01 through
1995-09-30, scoring 1995-10-01 through 2010-09-30, Daymet, raw mm/day, vendored
dmg Metrics. Compare basins by gauge ID, and verify the saved targets.

Initial seed-0 matrix:

| model | objective | budget | purpose |
|---|---|---|---|
| PUBModel | existing global MSE | 800 x 150 x 8 | reproduced control |
| PUBModel | basin-normalized MSE | same | objective effect |
| PUBModel, no pooled path | existing global MSE | same | independent architecture ablation |
| RegionalLSTM | existing global RMSE | 100 epochs, existing sampling rule | reproduced control |
| RegionalLSTM | basin-normalized MSE | same | objective effect |

Keep each architecture's existing optimizer and training distribution. The
new objective averages squared error within each sampled basin before averaging
basins. Its fixed variance uses only that basin's training-period observations.
The standard-deviation floor is 0.1 raw mm/day; record this in every run.
For PUBModel retain the legacy patch-size multiplier. This is a mean-NSE
surrogate, not optimization of the median score. No evaluation-period variance
enters training, and held-out basins have no training loss statistics.

First analyze existing artifacts: NSE by training-variance quartile, paired
error decompositions, and normalized squared error by window and patch position.
Position curves compare models on identical days; do not compare NSE computed
on separate lead-day subsets. These are diagnostics, not causal ablations.

After the pilot, replicate relevant contrasts on seeds 1 and 2 before adopting
a change. Aim for <=0.02 median NSE deficit to a strong matched forward model;
this is an acceptance criterion, not an expected outcome. Keep the final test
period out of hyperparameter tuning; new choices require a training-era
validation period for confirmation.

## Stage 2: daily DI and new-basin adaptation

Establish a causal DI-LSTM using lagged Q, availability and observation age on
every supervised day. Compare the same model at inference with zero, daily,
stale, and sparse own observations. Use masks drawn during training, including
whole-record absence and outages. Every input must be available before its
prediction day. Evaluate adaptation on spatially held-out basins, including
historical campaigns ending before the scored period. Do not present temporal
same-basin DI as proof of unseen-basin adaptation.

Use the daily baseline to inform a recurrent head conditioned on hydroPFN
context. Causality must hold through encoder summaries and all cross-site paths,
not only the final recurrent head. Perturb future observations in a regression
test and require earlier predictions to stay unchanged.

## Stage 3: one targeted terrain experiment

Prefer basin-aware coverage if CAMELS nesting remains too sparse for a network
test. Compare a strong frozen/control temporal model with terrain features
constructed over its actual contributing basin, retaining physical relief and
coverage metadata. Start with an out-of-fold residual probe against existing
attributes and simple geomorphic descriptors before training a large adapter.
Report regional holdouts, physical-unit errors and multiple seeds for any gain.
Do not repeat the current outlet-square/PCA-width sweep.

## Papers informing the design (not instructions)

- Liu et al., 2026, doi:10.1029/2026GL122814: descriptor substitution and global
  random-PUB gains; temporal results use different dates and seed ensembling.
  Gating is diagnostic and does not maximize median NSE in that paper.
- StefaLand manuscript supplied by CS: frozen cross-domain representations,
  residual forcing adapter, daily LSTM. Table 13 separates pretraining and
  adapter effects. These are manuscript-reported results, not reproduced here.
- StefaLand-HBV manuscript supplied by CS: embeddings enter physical parameter
  learning through an adapter/LSTM. Matching precipitation reduces but does not
  eliminate the reported gain. That manuscript uses RMSE; do not conflate its
  loss with the separately supplied temporal dHBV configuration's NSE loss.

The papers support testing representation and temporal dynamics separately.
They do not establish that a DEM-only diffusion representation is informative,
nor that an arbitrary-mask, no-retraining observation interface already works.

## Execution record

- Stage-1 immutable source snapshot:
  `/nfs/data/cxs1024/hydroPFN/experiments_20260923_v2`.
  Archive SHA256: `6bd713c9623a715436cf74a33e91eed8d6eb59b2cd0cf179d9c510c98f8221f5`.
- Supervisor PID 505702, started 2026-09-23 04:08 UTC, GPUs 2 and 4.
  Status and stdout: `logs/recovery_stage1/` under that snapshot.
- Four scientific objective-invariance tests passed. Both NSE training drivers
  passed end-to-end smoke checks with finite predictions and checkpoint output.
  A 10-step PUB smoke hit an existing OneCycleLR zero-duration warmup; the
  20-step check passed. The full 120,000-step schedule is unchanged.
- Initial artifact diagnostics reside locally at
  `logs/recovery_setup/recovery_diagnostics/` and remotely under the v1 snapshot.
  Gauge IDs and target values were verified before comparing predictions.
  PFN vs seed-0 LSTM median paired NSE differences by increasing training-Q
  variance quartile: -0.0353, -0.0562, -0.0519, -0.0387. The deficit is not
  concentrated in low-variance basins. PFN has lower median normalized bias
  error, but higher amplitude and correlation error than the LSTM.

The daily-DI driver is `experiments/camels531_daily_di.py`. Its recurrent
baseline uses four explicit observation channels: most recent observed Q,
whether Q(t-1) is observed, observation age, and history availability. A
separate RNG for availability preserves the baseline's basin/window draws.
`daily` is a specialist; `mixed` trains whole-record absence, daily readings,
delayed reports, outages, sparse visits and pre-score campaigns. Only `mixed`
is intended to support all inference scenarios. Use `--extent PUB --protocol
temporal` for the basin-plus-time holdout; partial folds must remain labeled.

### Stage-2 batch definition

The pilot objective is prespecified as basin-normalized MSE, not chosen using
the future test scores. A 0.1 mm/day floor prevents unbounded dry-basin weights.
This is matched across our new PFN/LSTM arms; it is not yet a bit-for-bit
reproduction of the external dHBV `NseBatchLoss` implementation.

| extent | training observation mode | evaluation | budget |
|---|---|---|---|
| same basins, future years | forward | no Q observations | 100 epochs |
| same basins, future years | daily specialist | yesterday's Q | 100 epochs |
| same basins, future years | mixed availability | all 8 scenarios | 100 epochs |
| 10-fold PUB, future years | forward | no Q observations | 100 epochs/fold |
| 10-fold PUB, future years | mixed availability | all 8 scenarios | 100 epochs/fold |

All five jobs use seed 0 and the same recurrent capacity, optimizer, sampling
rule and four observation input channels. Per-fold seeds allow restart without
changing later folds. Checkpoints, normalization and completed-fold predictions
are retained. The supervisor waits for stage 1 to succeed and produces its
paired diagnostic report before starting stage 2; it never edits the v2 source.
It waits for a free selected GPU rather than sharing an occupied GPU.

`daily_di_diagnostics.py` verifies gauge/target alignment and all ten training
versus test basin splits. It compares no-Q and daily specialist/generalist
scores, and reports campaign benefits on days 1-30, 31-90, 91-365, year 2 and
years 3-15 using a fixed error denominator. This separates transient state
initialization from persistence. It does not prove learned basin parameters or
PFN integration. A recurrent PFN head and seed replication remain subsequent
model changes, conditional on the diagnostic evidence.

### Terrain information test

Read-only inventory found 51 nesting pairs among 671 basins, so the chosen
test is basin-aware coverage. Existing 13 km outlet DEMs have valid-fraction
quantiles (minimum, tenth percentile, median) of 0.361, 0.703, 1.000; for the
51 km files these are 0.278, 0.516, 0.805. These are inventory statistics over
671, not a 531-basin benchmark. Previously filled collars can obscure this
issue; they are not evidence that terrain cannot help.

`basin_dem_features.py` mosaics adjacent public 3DEP tiles over each actual
CAMELS basin polygon and a matched 12.8 km outlet square. Both use the same
15 physical descriptors, nominal 30 m resolution and >=98% coverage threshold.
Very large bounding boxes are coarsened to at most approximately five million
pixels; actual resolution, source URLs, failures and coverage are recorded.
Extraction runs in `demenv` on suntzu, independently of the GPU training.

Before a large new adapter, `basin_terrain_probe.py` measures regional held-out
information about historical mean Q, Q variability, high flows, persistence
and flashiness. It uses only the 1980-1995 training era. Within each outer PUR
fold, attribute-model residual targets are cross-fitted across the remaining
regions. The two identically sized terrain arms predict these residuals with
the same prespecified ridge penalty (100). All preprocessing uses training
basins only. All arms use the same complete-coverage basin intersection;
excluded basins and per-region counts are reported. Require >=425 paired
basins or stop without a headline.

This is a diagnostic of incremental terrain information, not a daily NSE
result. It avoids reusing out-of-fold daily residuals from incompatible split
schemes, which could leak outer-test basin labels through the baseline model.
A positive result justifies a matched downstream terrain adapter; a null only
bounds this descriptor/probe configuration, not all learned DEM embeddings.

### Validation completed before queuing

- Four objective tests: training-only statistics, affine-unit invariance,
  missing-day basin weights and invalid variance handling.
- Five daily-observation tests: lag causality, missing versus zero Q, expiry,
  campaigns, reproducible masks and the actual recurrent model's predictions.
- Four terrain-probe tests: outer/inner label separation, training-only
  normalization, gauge joins/failed coverage and physical signature scaling.
- Two GIS tests: physical slopes and basin-mask statistics.
- End-to-end CPU smoke: one held-out PUB fold, future-year scoring, mixed
  availability, all tested predictions finite, checkpoint and fold caches.
- Completed-fold resume reproduced all three smoke prediction arrays
  byte-for-byte.
- Two actual DEM acquisitions: upstream basin and outlet square both passed.

Smoke training budgets deliberately do not estimate predictive skill. Existing
benchmark tables have not been changed. Full model comparisons remain pilots
until replicated; no improvement is claimed from successful execution alone.

### Active execution locations

Stage 1 continues from the v2 source above. Stage 2 and the terrain probe use
`/nfs/data/cxs1024/hydroPFN/experiments_20260923_v5`, an immutable 87-file
snapshot. Its archive SHA256 was verified on both machines:
`4ff5fb703106d564d14f641daab02815e6c9c63a0e3feb477953d32a5ec1e97e`.

- Stage-2 supervisor PID 517246; `logs/recovery_stage2/`. Initially waiting
  for successful stage-1 completion. It writes paired stage-1 diagnostics
  before releasing its five jobs and writes DI diagnostics after they finish.
- Terrain extraction PID 512972; full data and per-basin coverage records:
  `/nfs/data/cxs1024/hydroPFN/logs/basin_terrain_20260923/`.
- Terrain-probe supervisor PID 517247; `logs/recovery_terrain/` under v5.
  Initially waiting for the full 531-basin extraction. It runs on CPU only.
- At the initial execution audit, 210/531 basins had completed extraction;
  all 210 basin polygons and outlet squares passed the coverage threshold.
  This is a dated progress count, not the eventual coverage result.

These are finite, detached experiment batches; SSH disconnection does not
stop them. Each supervisor records failures and refuses to overwrite existing
runs. Expected runtime is several hours. Run directories are authoritative
for status; this document is not a live monitor. The next scientific decision
is to inspect the completed comparisons before claiming gains or selecting
a downstream PFN adapter and seed replications.

### First completed control

The new seed-0 LSTM/RMSE control finished with median NSE
**0.6822723159**. The saved historical seed-0 run was **0.6875141763**.
`logs/recovery_setup/recovery_control_check/` contains the local paired audit:
all 531 gauge IDs, all 5479 score days and all raw target values matched.
Both runs record 100 epochs, 266 steps/epoch, hidden size 256, dropout 0.5
and 296,193 parameters. The historical metadata does not capture all runtime
settings, so the cause of the difference has not been established.

The difference is concentrated in the lowest training-variance quartile:
fresh versus historical medians 0.4565 versus 0.5666, with a median paired
difference of -0.0674. Other quartiles have median paired differences -0.0069,
0.0001 and -0.0006. The small headline difference conceals a material subgroup
change; use the fresh matched control for the objective contrast and require
seed replication before attributing a gain. Do not call this an exact numerical
reproduction or silently replace the published benchmark.

### Completed pilot results and recovery check

Stage 1 completed successfully at 2026-09-23 07:21 UTC. All results below
are seed-0 pilots, with aligned 531 basins and 5479 future score days.

| arm | median temporal NSE |
|---|---:|
| PFN global MSE, K=0 | 0.640079 |
| PFN basin-normalized MSE, K=0 | 0.635120 |
| PFN global MSE, pooled path disabled, K=0 | 0.640281 |
| LSTM global RMSE | 0.682272 |
| LSTM basin-normalized MSE | 0.691068 |

The objective hypothesis does not explain the PFN forward gap in this pilot.
Removing pooled summaries is also effectively null. PFN K=4 medians are
0.773588, 0.775291 and 0.775466 respectively; these use neighboring discharge
and must not be presented as forward-only performance. LSTM's low-variance
quartile improves from 0.4565 to 0.5840 with the normalized objective, despite
the smaller aggregate gain. Paired diagnostics were retrieved to
`logs/recovery_setup/stage1_diagnostics/`.

Completed stage-2 temporal recurrent baselines:

| arm / available observations | median temporal NSE |
|---|---:|
| forward specialist / none | 0.692403 |
| daily specialist / yesterday's Q | 0.830728 |
| mixed model / none | 0.704258 |
| mixed model / daily | 0.800624 |
| mixed model / weekly | 0.743795 |
| mixed model / 7-day delay | 0.727690 |
| mixed model / 16-day delay | 0.716047 |
| mixed model / historical 30-day campaign | 0.705005 |
| mixed model / historical 90-day campaign | 0.704717 |
| mixed model / historical 365-day campaign | 0.704538 |

These are LSTM observation-interface results, not repaired PFN results.
Historical campaign gains must be compared against the same mixed model's
no-observation score; the full-period benefit is negligible. Daily specialist
versus mixed daily shows a roughly 0.030 median-NSE gap. Both unseen-basin
future-period jobs had finished 9/10 folds at this check; no aggregate score
is quoted until all folds and the split audit finish.

Terrain extraction stalled after 335 completed basins with its process still
alive in a futex wait. Only that verified process (PID 512972) was terminated.
Recovery PID 778224 uses the unchanged descriptor extractor in an isolated
subprocess per basin, with a 180-second hard timeout, single-thread settings,
and all successful caches preserved. The formerly stalled basin 07142300
completed successfully after restart. Wrapper provenance is recorded in
`basin_terrain_20260923/recovery_launch.json`; source is
`scripts/recover_terrain_extraction.py`. The terrain probe remains pending
until the complete inventory has been rebuilt; no terrain skill result yet.

### Full unseen-basin result available during formulation analysis

All ten PUB folds have now completed, and the paired diagnostic verified
gauge/target alignment and training/test basin separation. This combines
held-out basins with the 1995-2010 future period; it is not the historical
spatial-only PUB protocol.

| recurrent arm / observations | median NSE |
|---|---:|
| forward specialist / none | 0.572102 |
| mixed / none | 0.590445 |
| mixed / daily | 0.719283 |
| mixed / weekly | 0.647390 |
| mixed / 7-day delay | 0.621103 |
| mixed / 16-day delay | 0.609814 |
| mixed / historical 30-day campaign | 0.590706 |
| mixed / historical 90-day campaign | 0.590645 |
| mixed / historical 365-day campaign | 0.590518 |

Local artifacts: `logs/recovery_setup/stage2_diagnostics/`.
No unseen-basin daily specialist was trained in this batch, so the 0.7193
result does not estimate a generalist-versus-specialist penalty for PUB.
See `specialist_parity_formulation_20260923.md` for the mask-exposure audit,
gradient probe and proposed formulation; these are diagnostic/design work,
not an additional training run.
