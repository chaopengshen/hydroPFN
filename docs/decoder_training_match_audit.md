# Audit: what the decoder training matched

The user's question was whether the PFN decoder experiments matched LSTM
training and whether their deficits imply that the architecture cannot work.
The audit used the actual saved stage-1 NSE LSTM and source-PFN `run.json`
files, the v7 decoder `config.json` files, and the training/evaluation code.

## Scope of the match

Within the v7 comparison, raw LSTM and PFN+LSTM share the sample stream for
each seed, objective, optimizer, batch, output dropout, history, updates and
scoring windows. However, neither follows the earlier specialist recipe in
full. The following comparison uses the earlier **NSE-loss** specialist;
the still-earlier RMSE specialist also differs in objective.

| setting | earlier NSE LSTM | new decoder pilot |
|---|---|---|
| trainable representation | entire LSTM | entire daily decoder; PFN encoder frozen |
| output dropout for LSTM | 0.5 | 0.1 |
| warmup + supervised days | 365 + 365 | 256 + 256 |
| window starts | sampled at daily resolution | 79 fixed starts, stride 64 plus final start |
| test state/history | continuous recurrent sequence | restarted overlapping 512-day windows |
| updates at 100 epochs | 26,600 | 38,800 |
| optimizer / lr / batch | Adadelta / 1 / 128 | same |
| objective | training-basin variance normalized MSE | same form and 0.1 mm/day std floor |

Both score identical temporal dates/basins in raw discharge units. Exposure
is not identical: the pilot has more updates, shorter supervised suffixes and
far fewer distinct window contexts. Dropout is also weaker. None of these
differences was independently ablated, so they are candidates, not established
causes of the gap. Merely extending this setup to 300 epochs does not repair
the differences. See the authoritative [score tables](camels531_protocol.md#frozen-pfn-decoder-and-duration-pilots-2026-09-23).

The source PFN had previously been trained end to end under global MSE with
mixed K contexts. The new NSE readout loss never updates that encoder. The
300-epoch original-head control reinitializes/refits the head, not the whole
PFN; its poor score is not the performance of a newly trained end-to-end PFN.

## A constructive capacity check

An additional CPU check loaded the actual v7 seed-0 trained raw-LSTM and
raw-TCN checkpoints. For each, its recurrent/convolutional and readout weights
were copied into the corresponding hybrid; the extra PFN-feature and phase
input weights were set to zero. The remaining feature projection was retained.

On three synthetic 512-day sequences with five forcings, 26 attributes and
256-dimensional daily latent features, both hybrids reproduced the trained
raw model with **maximum absolute prediction difference 0.0**. Scaling the
latent features by 20 also left predictions unchanged. This is an executable
check of the architecture's raw-model subcase, not a new hydrologic score.

Thus these hybrid architectures can represent their raw baselines exactly.
Their independently trained worse solutions cannot establish an unavoidable
representational deficit. Optimization, reliance on non-generalizing frozen
features and regularization remain plausible explanations. This containment
argument applies to the hybrids with the raw branch, not to the original
patch-only PFN architecture.

## Next decisive experiment, not launched by this audit

First reproduce the earlier specialist recipe and establish training-era
validation. For a clean validation split, the supervised PFN encoder must also
be trained only inside that split; the existing full-training-era checkpoint
would otherwise have seen validation targets.

Initialize the hybrid from that specialist and zero its additional input
weights so predictions match before training. Then compare leaving the PFN
frozen with fine-tuning it at a smaller learning rate, retaining the specialist
training recipe and fixed evaluation definition. A separately parameterized
residual correction to a frozen specialist is another way to preserve the
fallback. Exact parity at initialization is guaranteed by weight construction;
generalization after optimization is not. Added PFN information must earn a
benefit on validation rather than simply increasing training fit.

## Window diversity and spatial interpretation (2026-09-26)

The cache schedule has 79 training windows per basin versus 4,967 valid
daily-resolution starts for a 512-day window. Across 531 basins this is
41,949 basin/window combinations, sampled an expected 118.39 times at 100
epochs and 355.17 times at 300 epochs. These are repeated contextual views,
not independent observations. The overlapping windows still cover the scored
training dates; random cropping would add views, not new hydrologic records.

Because stride 64 is divisible by patch size 16, 78 of the 79 training starts
have the same patch-grid alignment (start index modulo 16 equals 2); the
appended final start has remainder 8. Of 22 evaluation windows, 21 have
remainder 8 and the final window has remainder 15. An event repeated in the
regular training grid therefore stays at the same within-patch position.
This removes augmentation over patch boundaries. The different evaluation
alignment is a diagnostic to test, not evidence by itself of a harmful shift.

Frozen bidirectional PFN features depend on the whole 512-day crop. Resampling
starts changes patch groupings, boundary locations and the surrounding
attention context; one must recompute those features. Merely slicing/shuffling
the existing cached latents does not implement this experiment. Online frozen
encoding or periodically regenerated random-window caches can avoid the
restriction; joint encoder training necessarily needs current online features.

There is an important counterexample to a broad window-overfitting argument:
the raw TCN has a 253-day receptive field with 256 real warmup days. For the
same scored basin/day, moving the crop while retaining sufficient history
does not change its deterministic effective input. Random starts primarily
change target weighting for this raw model, whereas they change contextual
PFN features and can change LSTM spin-up. A controlled fixed-versus-random
sampling contrast should keep history, dropout, loss and updates unchanged.

TCN dropout is elementwise `nn.Dropout(0.1)` after each of the two GLU
convolutions in each of six residual blocks. The identity residual path is
not dropped. This is different from the LSTM's output-only dropout; setting
TCN dropout to 0.5 does not make regularization equivalent. Feature/channel
dropout or dropping the PFN branch are additional hypotheses, not implemented
features of these runs.

Specialist-preserving initialization establishes a known baseline function;
it does not add information or guarantee subsequent test improvement. Spatial
transfer is a plausible larger opportunity for PFN context, especially with
neighbor gauges or limited local calibration observations under an explicit
protocol. A clean spatial test must exclude held-out basin discharge from
both specialist fitting and supervised PFN pretraining. The current source
PFN, fitted on all 531 basins' historical discharge, cannot serve as an unseen-
basin representation without this distinction. Temporal sampling diagnosis
and spatial information-gain tests answer different questions.

The user subsequently authorized the sampling fix and controlled experiment.
Implementation and submitted jobs are tracked in
[the window-sampling ablation](window_sampling_ablation_20260926.md).
