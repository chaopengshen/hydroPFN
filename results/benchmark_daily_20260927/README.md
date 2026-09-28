# Single-model 300-epoch follow-up

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
