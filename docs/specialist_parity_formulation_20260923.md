# Specialist parity and a conditional hydroPFN formulation

This note separates observed results, diagnostics and proposed mechanisms.
It does not change or start any training runs.

## There are two different gaps

1. **PFN forward dynamics:** global-MSE PFN K=0 scores 0.6401 versus the
   fresh LSTM/RMSE control's 0.6823. Basin-normalized loss gives PFN 0.6351
   versus LSTM 0.6911. Disabling pooled summaries gives PFN 0.6403. The older
   dedicated PFN specialist also failed to close its gap. Thus multi-task
   dilution and this loss change do not explain the PFN forward deficit.
2. **Daily observation specialization:** identical-capacity recurrent models
   score 0.8307 for the daily specialist and 0.8006 for mixed availability with
   daily observations. Here the architectures and information at evaluation
   match, while the training observation distributions differ. These are
   distinct experiments; the earlier null does not rule out a task-distribution
   effect in this new recurrent comparison.

The mixed model's no-observation result is 0.7043 versus the forward
specialist's 0.6924. Sharing is therefore helping one condition while failing
to match another in this seed-0 pilot. A universal generalization penalty has
not been established.

## What the present implementation actually trains

The sampler chooses 25% whole-record absence, 35% daily-with-possible-outages,
20% sparse observations and 20% pre-score campaigns. A quarter of the daily
group also receives reporting delays. With 4,000 synthetic 730-day sequences
and all discharge finite, seed 123 gives:

- Yesterday's Q available on 30.80% of the 365 supervised days.
- Yesterday's Q available on every supervised day in 20.08% of sequences.
- Uninterrupted daily observations from day 1 onward in 13.18% of sequences.
- Whole-record absence in 25.03% of sequences.

These are Monte Carlo mask-audit results, not measured fractions in the
completed training trace. Real gauge missingness reduces freshness further.
The daily specialist gets the daily regime on every sampled sequence, subject
to real missing observations. Equal epochs therefore do not match exposure to
daily DI. Inverse-probability loss weights change the objective but do not
create the missing independent daily training examples.

A second diagnostic evaluated the frozen mixed checkpoint on 12 training-era
batches of eight basins, with dropout disabled. All sampled gradient cosines
against the daily loss were positive:

| other regime | mean cosine with daily | median gradient norm / daily |
|---|---:|---:|
| none | 0.633 | 1.60 |
| weekly | 0.769 | 1.30 |
| 365-day historical campaign | 0.607 | 1.79 |

Artifact: `logs/recovery_setup/specialist_gradient_probe.json`.
This is a small local probe at one checkpoint, not a reconstruction of its
optimization history. It gives no direct evidence for gradient conflict here.
Larger gradients from harder observation regimes and lower daily exposure are
more immediate suspects. PCGrad should follow evidence of conflict, rather
than be the default response to a generalist gap.

The same recurrent state and gates currently have to support both a forcing-
driven simulator and a recent-Q-conditioned predictor. The possibility that
finite recurrent capacity or a shared update rule limits both is a hypothesis,
not a measured causal conclusion. Disabling a pooled transformer path does not
test recurrent state propagation versus independent 512-day windows.

## What can be guaranteed

There is no representational law requiring a conditional model to be worse
than the appropriate specialist when both receive the same information. A
model class containing all specialists and an availability-conditioned router
can reproduce them. An optimizer training a finite shared network on an
average loss is not guaranteed to find that solution.

Under any fixed expected loss, a richer information set cannot worsen the
optimal achievable risk when its predictor can ignore the extra information.
This is a statement about attainable expected risk, not a guarantee for every
basin, empirical median NSE, finite-sample training, or forecasts with less
information than the specialist.

An exact-parity control can retain three frozen functions:

    prediction(I) = daily_expert(I)   for the daily expert's information regime
                  forward_expert(I) for the forward regime
                  partial_model(I)  for the remaining regimes

The router uses availability, never unseen targets or the better model on the
test set. Exact equality requires the same normalization, initial state,
entire causal history, evaluation mode and output transformation, as well as
unchanged weights. Switching merely because yesterday's reading returned
after an outage does not recreate the daily specialist's hidden state.
Maintain separate recurrent states or replay only the legitimately available
history. The guarantee applies where the expert receives exactly its original
information; it cannot synthesize observations that were never made.

This establishes a parity floor for a modular system, at the cost of retaining
expert parameters. It does not demonstrate that the partial-observation branch
generalizes, or that one fully shared network has learned all regimes.
A soft gate, zero-initialized residual, distillation loss or validation-based
fallback alone does not guarantee future-test parity after training.

## Proposed scientific formulation

Use a strong local recurrent dynamics model as the prediction engine. Make
the contextual model infer basin properties and update hydrologic state,
rather than requiring a patch-reconstruction model to relearn the predictor.

There are two different latent quantities:

- **Slow basin representation z:** information about storage, recession,
  runoff response and other persistent properties. Infer it from available
  attributes, terrain and historical forcing-response sequences. It survives
  beyond the last observed Q and is supplied repeatedly to the dynamics.
- **Fast hydrologic state h:** current wetness, snow and storage conditions.
  Update it when a measurement arrives, then propagate it forward under the
  forcing. Old readings are not continually treated as fresh state.

A causal ordering for predicting Q(t) with observations available by t-1 is:

    z_b       = BasinContext(available historical sequences, statics, terrain)
    h+(t-1)   = Update(h-(t-1), available observation innovations, uncertainty, z_b)
    h-(t)     = Dynamics(h+(t-1), forcing(t), z_b)
    Q_hat(t)  = Readout(h-(t), z_b)

If no new observation arrives, the measurement update is identity. For delayed
observations, use their actual timestamps and a learned delayed-observation
operator or replay from the observation time; do not pretend they observe the
current state. Every context encoder must obey the prediction-time information
cutoff. Historical Q needs its associated forcing history, not just a mean Q
or basin ID. The static latent is a useful approximation; persistent changes
such as management changes may require a slower evolving component.

Probabilistically, the target is prediction marginalized over uncertain basin
properties and current state, conditioned on the available information. An
absent gauge gives a broader inferred state, not a zero-flow observation.
Terrain primarily informs the prior over basin response; fresh Q primarily
constrains state. Historical records can inform both.

Initialize the dynamics from the specialist and preserve a route that exactly
recovers it. Add context-driven state/parameter adapters with an identity or
zero correction as an available solution. Freezing a forward backbone protects
forward behavior only when the full recurrent path is unaffected; it does not
automatically protect daily-DI performance. Preserve the daily expert too until
a shared replacement meets its separate validation criterion.

The now-complete unseen-basin/future-period pilot supports this separation:
mixed no-Q NSE is 0.59045, daily-Q is 0.71928, and historical 365-day campaign
is 0.59052. These compare the same frozen model on all 531 held-out basins.
Recent observations clearly help; persistent historical adaptation is still
missing. This does not prove that the slow/fast decomposition will fix it.

## Experiments that distinguish explanations

1. **Match daily exposure first.** On each sampled basin/window, retain a full
   daily pass and add a masked pass, rather than replacing most daily passes
   with masks. Report both equal-total-compute and equal-daily-exposure controls.
   This separates a data-allocation effect from a capacity/parameter-sharing
   effect. A 50/50 sampler alone still halves the specialist's daily exposure.
2. **Warm start from the daily specialist with explicit per-regime criteria.**
   Track daily loss and every missingness regime separately. Do not select by
   one average loss that can hide regression on daily observations. Same-mask
   teacher anchoring can stabilize an approximate shared model; a full-data
   teacher cannot require a no-data student to make identical point forecasts.
3. **If parity still fails, isolate updates.** Compare separate recurrent
   adapters/update cells against more shared width at matched parameter and
   compute budgets. A new output head alone may not remove interference in
   the recurrent state. Use gradient surgery only if stronger probes identify
   conflicting gradients during training.
4. **Test persistent context separately.** Train on short observed campaigns
   followed by long unobserved targets. Pass the basin context throughout the
   target sequence, and score persistence against that same model with no
   campaign. Predicting full-period NSE only can conceal an early state benefit.

Choose new hyperparameters on a training-era validation design. For example,
fit all teachers and trunks on 1980-1990 and validate on 1990-1995 before the
final full 1980-1995 fit. The existing checkpoints have already seen 1990-1995
labels and cannot serve as clean validation teachers for that split. The
1995-2010 benchmark is already repeatedly inspected and should not be described
as a newly untouched test. Replicate key contrasts before claiming parity or
improvements; also test PUB/PUR, outages and delayed-report transitions.

## Relevant primary literature

- [Feng, Fang and Shen (2020), daily data integration](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2019WR026793):
  recent discharge changes the prediction problem toward forecasting flow
  evolution from an observed condition. This motivates an explicit observation
  update, but does not prove the proposed architecture will match this pilot.
- [Becker et al. (2019), Recurrent Kalman Networks](https://proceedings.mlr.press/v97/becker19a.html):
  precedent for learned recurrent state inference with explicit uncertainty.
- [Garnelo et al. (2018), Conditional Neural Processes](https://proceedings.mlr.press/v80/garnelo18a.html):
  precedent for conditioning prediction on variable-sized context sets without
  per-task retraining. Temporal causality and persistent basin inference require
  additional structure in the proposed hydrologic model.
- [Yu et al. (2020), Gradient Surgery for Multi-Task Learning](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html):
  supports treating gradient interference as a testable mechanism; our small
endpoint probe did not observe negative gradient cosines.

## Paired-training test launched after the parity-gate request

The user made specialist parity the immediate decision gate. Further model
extensions are deferred while this recurrent observation-formulation test runs.
This gate does not certify the separate original-PFN forward architecture.

Implementation: `training_di_views` in `src/hydropfn/data/observations.py` and
`--mode paired` in `experiments/camels531_daily_di.py`. Each optimizer update
draws one basin/window batch and evaluates both its original daily view and
one view from the existing mixed sampler. Both losses use the same targets,
training-only basin variances and equal basin weights. Their mean is
backpropagated sequentially; clipping and the optimizer step occur once after
both views. The two views have independent dropout draws. The basin/window RNG
remains independent of availability masks.

The preserved daily exposure is a statement about examples/passes, not an
identical daily-only optimizer trajectory: the extra loss changes the gradient.
Paired training can still exhibit a compromise. Its success is an empirical
question; the sampler ratio alone does not imply impossibility at any capacity
or compute budget.

| pilot | epochs | views/update | updates | relation to original 100-epoch specialist |
|---|---:|---:|---:|---|
| daily + mixed | 100 | 2 | 26,600 | same sampled daily-window exposure; twice the window views |
| daily + mixed | 50 | 2 | 13,300 | same total window-view count; half as many updates |
| daily + daily | 100 | 2 | 26,600 | control with matching passes, capacity and optimizer updates |

The two 100-epoch arms run sequentially on GPU 2, so their compute-control
contrast also uses the same device. The 50-epoch arm runs on GPU 4, which ran
the original daily specialist. All use seed 0, hidden size 256, the existing
optimizer and basin-normalized loss. GPU time is logged; matching view counts
does not claim exact wall-clock equality.

`paired_parity_diagnostics.py` reports each availability condition separately.
Daily predictions are compared with the original specialist and the new
two-pass daily control; partial-information conditions are compared with the
original mixed model under the same information. Gauge and raw target arrays
must match. The report includes paired basin differences and bootstrap
intervals, explicitly excluding seed uncertainty and spatial dependence from
those intervals. A single-seed pilot cannot certify general parity; relevant
contrasts require replication. No mean-over-regimes score can conceal a daily
regression in this decision gate.

Validation: four new tests cover preservation of the daily view, causal input
construction, equivalence of duplicate-daily mean gradients to a single pass
with dropout disabled, and nonzero contribution of the masked view. The nine
existing objective/observation tests also passed. A CPU end-to-end paired
smoke completed with finite predictions, checkpoint and training-budget output.

Immutable source: `/nfs/data/cxs1024/hydroPFN/experiments_20260923_v6`.
Archive SHA256:
`0b9add8d3819cd1700c3737d6f2112aac6b534bc6b08f824ada48b8611c8348e`.
Finite batch: `scripts/run_paired_parity.py`; status, training stdout and final
diagnostics are under the snapshot's `logs/paired_parity/`.

The low unseen-basin result is not dismissed as a longer-score-period effect:
scoring only 1995-1999 still gives 0.5994 for mixed/no-Q and 0.5831 for the
forward baseline. Those models trained only through 1995, so this is not the
standard spatial protocol that trains through 1999. Further spatial
generalization diagnostics remain unresolved while the parity test runs.

## Completed paired pilot and next design

The three paired jobs finished at 2026-09-23 14:21:52 UTC. Authoritative scores,
budget details, paired basin intervals and caveats are now in
[the protocol](camels531_protocol.md#recovery-pilots-completed-2026-09-23).
The 50-epoch daily+mixed model reaches daily headline parity in this seed while
retaining useful no-Q performance. The 100-epoch model still trails its matched
two-pass daily control. Duration and GPU differ between the two paired runs,
so do not call their difference proof of overtraining.

This is evidence for feasibility, not certification of parity across seeds,
regions or the original PFN architecture. Preserve the user's parity gate:
replicated contrasts and training-era validation precede adopting an extension.
The [measurement/DEM design](measurement_assimilation_dem_20260923.md) records
the proposed fast-state/persistent-property interface, current measurement
trainer audit findings, and one geometry-focused terrain experiment.

## Temporal decoder hypothesis after the objective null

User suggestion: use a decoder adapted to daily dynamics, as in StefaLand's
LSTM decoder, or a more parallel Conv1D alternative. Code inspection supports
testing this, but does not establish the decoder as the cause of the gap.

The active reconstruction head is `PUBModel.head` in `models/connector.py`,
not only `SiteEncoder.head`. It maps a contextualized 256-dimensional token
through LayerNorm/MLP to 16 daily values. Temporal attention is upstream, so
outputs are not temporally independent, and a 16-to-256 input projection is
not inherently lossy. However, the readout has no explicit daily recurrence
or convolution, and separate within-patch output coordinates do not impose
the same daily transition rule across patch boundaries.

Evaluation has another relevant difference. `camels531_pub.py` tiles separate
32-patch/512-day windows. Dropping one initial 365-day warmup does not provide
historical forcing before every later tile. `camels531_lstm.py` processes the
entire evaluation input continuously before dropping its warmup. Its training
also uses 365 warmup days plus 365 supervised days; PFN supervises its masked
patches throughout the sampled window. Matching dates/targets therefore did
not match effective history or warmup behavior. The position diagnostic has
larger PFN errors at both ends of its window; it is not evidence that a pure
cold-start effect explains the entire gap.

Proposed daily interface:

    PFN query/context tokens -> daily latent features z(t)
    decoder input u(t) = [daily forcings, visible statics, masks, z(t)]
    daily sequence decoder(u) -> Q(t)

Explicitly lift patch features to daily resolution, retaining day-within-patch
identity. A raw daily forcing path is a useful protection against relying on
learned patch features for storm timing, but changes more than the head; label
that experiment as a hybrid, not a pure decoder ablation. Keep other masked
variable/measurement heads where useful: a foundation model need not emit every
variable through one shared output head.

LSTM is the first diagnostic reference because the local specialist already
works and the inspected StefaLand checkpoint contains an LSTM decoder
([reuse audit](stefaland_reuse.md)). A residual, dilated causal TCN is the
appropriate Conv1D comparison. A shallow short-kernel CNN alone provides very
little daily memory. For two kernel-3 convolutions per block at dilations
1,2,4,...,128, the direct-input receptive field is
`1 + 2*(3-1)*(1+2+...+128) = 1021 days`. Seven such blocks give 509 days.
These are candidate scales, not selected settings. Receptive field must be
supported by real input history; zero padding is not additional observed history.
History present indirectly through the encoder must be accounted for separately.

TCN processes time positions in parallel during training and full-sequence
inference when inputs are available. Streaming prediction retains a finite
history/activation cache; LSTM retains a recurrent state. End-to-end speed
depends on the PFN encoder, memory use and implementation, so measure throughput
and peak memory rather than claiming an automatic overall speedup.
[Bai et al. (2018)](https://arxiv.org/abs/1803.01271).

Suggested controlled sequence, not launched:

1. Diagnose history independently: evaluate a saved PFN with trailing windows
   and a common scored suffix, compared on the same dates with the original
   tiling. This is an inference diagnostic, not a new benchmark selection rule.
2. Compare daily pointwise, LSTM and TCN readouts on identical frozen PFN
   features/daily lifting and identical history. This isolates readout capacity;
   failure cannot establish that end-to-end adaptation would also fail.
3. For the deployable hybrid, compare LSTM and TCN with raw forcing/statics alone
   against the same decoder plus PFN context. Use identical normalization,
   loss, target dates and warmup. Include context ablation at inference so a
   strong raw-input bypass cannot masquerade as useful foundation features.
4. Fine-tune promising combinations with controlled budgets, training-era
   validation and replicated seeds; test K=0 first, then K>0 and daily DI.

Use overlap/discard with adequate history for a finite-window model, or properly
maintained state for the recurrent model. A causal decoder cannot remove future
information already present in its inputs. The current encoder can read future
patches; even its patch-causal mask permits access within a 16-day patch. Daily
DI therefore needs a daily-safe observation path and encoder masking, plus
checks on summary/context paths. Historical hindcast diagnostics should be
labeled separately from claims of causal forecasting.

The intended division of work is shared context inference plus a strong daily
dynamics decoder. If the hybrid reaches specialist skill but removing PFN
features changes nothing, the dynamics are competitive while the foundation
representation has not yet earned a contribution.

Implementation follow-up: the user authorized trying the LSTM and TCN. The
implemented modules, manuscript-derived choices, controlled pilot matrix and
its limits are recorded in [decoder comparison](decoder_comparison_20260923.md).
