# Frozen-specialist correction results

## Completed results (checked 2026-10-02)

All three tasks completed with exit code 0 in approximately 37 minutes.
Scores are median per-basin NSE for individual models; no ensembles.

| Seed | Raw e50 | Raw e100 (frozen baseline) | Raw correction e50 | PFN correction e50 |
|---|---:|---:|---:|---:|
| 0 | 0.708602 | 0.690394 | 0.686922 | 0.690701 |
| 1 | 0.708294 | 0.699520 | 0.690682 | 0.691691 |
| 2 | 0.714377 | 0.700812 | 0.688586 | 0.689384 |

Both corrections start from the epoch-100 raw model, not epoch 50. All
initial full-test prediction differences were exactly zero, and both the
raw model and PFN remained unchanged during correction fitting. Complete
10/25/50 correction checkpoint scores and configurations are preserved in
`results/specialist_residual_20260928/`.

The raw model deteriorated from epoch 50 to 100 in all seeds. The PFN
correction at epoch 50 changed its frozen baseline by approximately
+0.0003, -0.0078 and -0.0114 NSE across seeds. It did slightly better than
the raw-only correction in each seed, but did not consistently improve
the baseline. No model reached the recorded external 0.7431 reference.
This demonstrates exact parity with the copied local model at initialization,
not parity with the external specialist. Because the prespecified correction
source was epoch 100, this also does not test correction of the stronger
epoch-50 raw models. No checkpoint has been selected retrospectively.
