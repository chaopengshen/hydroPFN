# Fixed versus random daily starts

Completed experiment: five model/input combinations, two samplers and two
seeds, each at 100 epochs / 38,800 updates. NSE is scored on raw discharge
for 531 basins and 5,479 test days.

- scores.csv: every run and unique starts visited.
- sampling_effects.csv: daily minus fixed, headline and paired-basin changes.
- gap_changes.csv: positive gap_reduction means the hybrid deficit narrows
  after accounting for the raw control's sampling response.
- basin_scores.csv: individual basin NSE (join by gauge ID).
- audit.json: matching conditions verified by the automatic analysis.
- provenance.json: immutable source/cache hashes and scheduler mapping.

See [the protocol](../../docs/camels531_protocol.md) and
[experiment details](../../docs/window_sampling_ablation_20260926.md).
No raw observations or model weights are included.
