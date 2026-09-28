# Frozen-specialist residual experiment

## Question and scope

Can a matched raw LSTM recipe close the remaining temporal gap, and can
frozen PFN information improve that LSTM without changing its dynamics?
This follows the failed 300-epoch decoder extension. It is a single-model
comparison; predictions are never averaged across seeds or architectures.

This restores the earlier **local RegionalLSTM recipe**, not an independently
verified reproduction of external dHBV1.1p. The recorded 0.7431 dHBV reference
remains an external target. Matching a raw LSTM below that value only
establishes parity with that LSTM, not parity with dHBV.

## Prespecified experiment

Three seeds (0/1/2), each with:

1. Raw RegionalLSTM, hidden 256, output dropout 0.5, Adadelta lr 1,
   gradient clipping 1, batch 128, daily random starts, 365 warmup + 365
   supervised days. The existing protocol iteration rule sets updates per
   epoch. Train 100 epochs; report epochs 50 and 100 separately.
2. Freeze the epoch-100 raw LSTM. Train a small raw-input correction for
   50 epochs: forcing, static attributes and the frozen baseline prediction.
3. Independently start from the same frozen LSTM and train the corresponding
   correction with PFN latent features added. PFN stays frozen in eval mode.

Both corrections use width 32, GELU, dropout 0.5 and a zero-initialized final
linear layer. This implements baseline plus a learned correction; there is
no simultaneously zero gate that would block gradients. The heads have
different parameter counts because one receives PFN features; both counts
are recorded. They use identical sampled basin/window streams and training
budgets. Report correction epochs 10, 25 and 50 without selecting on test.

The objective is the existing training-basin variance normalized MSE, with
a 0.1 mm/day standard-deviation floor. All inputs and normalization use the
existing verified 531-basin cache; dates and raw-discharge targets are unchanged.
The new raw sampler includes the final valid start (an older raw LSTM driver
omitted it). Warmup remains differentiable when training the raw LSTM.

## Geometry and invariants

Raw evaluation is one continuous recurrent sequence with 365 preceding days,
not restarted pilot windows. Both correction arms reuse exactly these raw
predictions. PFN retains its trained 512-day/16-day-patch geometry. In a
730-day raw training crop, encode the final 512 days and use its last 365
daily lifted features for the supervised days. All Q inputs are masked and
zero; targets never enter the encoder. Evaluation tiles the score period
into up-to-365-day blocks, each using a 512-day PFN crop ending at that block's
end. The final shorter block uses the matching suffix. PFN remains a
noncausal forcing hindcast representation.

Before either correction trains, compare all 531 basins' complete test
prediction arrays with the frozen raw baseline using exact equality.
Abort on any mismatch. Hash the raw LSTM and PFN weights before and after
correction fitting and abort if either changes. Unit tests check exact
initial equality, a nonzero learning signal, and no gradient into baseline
or latent inputs. A GPU smoke run executes the full pipeline with two
updates per stage before production is released.

No early stopping or validation-based hyperparameter selection is claimed.
The supervised PFN source used the full training era, so a genuinely clean
inner validation split would require refitting it. This bounded experiment
uses fixed durations and reports all prescribed checkpoints on the already
inspected test set. A correction may worsen performance; initial parity
does not guarantee held-out improvement after fitting.

## Provenance

- Driver: `experiments/specialist_residual.py`.
- Model: `src/hydropfn/models/residual_correction.py`.
- Launcher: `scripts/specialist_residual_icds.slurm`.
- Remote snapshot: `/storage/group/cxs1024/default/cxs1024/hydroPFN/specialist_residual_20260928/source`.
- Source archive SHA256: `36b562c9ea3b1407b0496f02c07cb41cecf1d1da697d19b70d7dd38b2017f2d0`.
- GPU preflight job: `55872461`.
- Preflight completed successfully in 38 seconds: unit test passed; the full
  pipeline trained all three stages, both correction heads reproduced every
  baseline test prediction exactly, and frozen-weight checks passed.
- Production array: `55872463`, seeds 0/1/2, released only after successful
  preflight. Each task runs raw fitting and both correction arms sequentially.
- Outputs: sibling `runs/seed0`, `runs/seed1`, `runs/seed2`, each with
  configuration, raw/correction weights, prediction arrays, per-basin NSE,
  score records, exact-parity audit and completion/frozen-weight audit.
