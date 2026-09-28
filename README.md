# hydroPFN

A hydrologic model combining site/time-series encoding and contextual inference,
with experimental terrain and measurement arms.

## Training-window sampling materially changes the conclusion

A fixed feature cache restricted decoder training to **79 window starts per
basin**. Random daily starts admit **4,967 starts**, with the frozen PFN
re-encoding each actual crop. A controlled two-seed temporal experiment finds:

| Readout | Seed 0: fixed → daily | Seed 1: fixed → daily |
|---|---:|---:|
| PFN + LSTM | 0.670 → **0.718** | 0.681 → **0.724** |
| PFN + TCN | 0.671 → **0.717** | 0.672 → **0.716** |
| Original patch head, refitted | 0.576 → **0.677** | 0.577 → **0.679** |
| Raw LSTM | 0.692 → 0.678 | 0.687 → 0.691 |
| Raw TCN | 0.678 → 0.691 | 0.670 → 0.690 |

Scores are median per-basin NSE on raw discharge, **531 CAMELS basins**,
training 1980-10-01 through 1995-09-30 and testing 1995-10-01 through 2010-09-30.
Each fixed/daily pair shares initial decoder weights, basin draws, GPU,
100-epoch budget, loss and dropout. Both treatments encode online. The
encoder stays frozen; the original patch head is refitted, not trained end
to end. The effect combines context diversity, patch alignment and target
weighting. Both daily hybrids exceed their corresponding daily raw controls.

These are K=0 hindcasts: no discharge is supplied at inference, but PFN
features retain bidirectional forcing context. Two seeds and an already
inspected temporal test do not establish universal superiority or an external
benchmark record. The 671-basin table and multi-forcing ensembles on the MHPI
benchmark page are different comparisons.

- [Authoritative protocol and results](docs/camels531_protocol.md)
- [Controlled sampling experiment and provenance](docs/window_sampling_ablation_20260926.md)
- [Machine-readable results and per-basin scores](results/window_sampling_20260926/)
- [Training-match audit](docs/decoder_training_match_audit.md)
- [Next benchmark attempt and comparison limits](docs/benchmark_attempt_20260927.md)

## Run and test

The training CLI in `experiments/camels531_decoders.py` defaults to random
daily starts. Use `--sampling fixed` for the fixed-grid control. For a
controlled contrast, set `--feature-mode online` in both treatments. Provide
the source PFN checkpoint/normalization and prepared CAMELS cache; raw data
and checkpoints are not bundled in this repo.

```bash
PYTHONPATH=src python tests/test_temporal_decoders.py
PYTHONPATH=src python tests/test_objectives.py
python experiments/camels531_decoders.py train --help
```

The paired launch is `scripts/window_sampling_icds.slurm`; the report is
`experiments/window_sampling_diagnostics.py`. Site-specific Slurm paths must
be adapted outside the original compute environment. The analysis verifies
initialization, sampling streams, cache provenance and target alignment,
then measures changes in the hybrid/raw gap.

[Architecture and earlier work](docs/HANDOVER.md) ·
[Historical README, not a current scorecard](docs/README_historical.md)
