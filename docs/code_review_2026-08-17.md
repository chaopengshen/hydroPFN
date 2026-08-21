# Code review — 2026-08-17

Scope: `src/lib/terrain.py`, `src/lib/inpaint.py`, `src/tests/{sample_patches,
test_inpaint_probe, test_hedging_diagnostic, test_lithology_baseline,
test_sgmc_noise_floor}.py`, plus `channel_geometry/src/test_residual_learnability.py`.

Verdict up front: **no finding overturns a logged conclusion**, but three findings
(B2, M1, D1) mean specific numbers are less precise or more conservative than the
logs present them, and should be re-measured before anything goes into
`dev_dem.md` or a paper.

---

## Bugs

### B1 — harmonic fill pins edge-touching hole pixels to the patch mean
`lib/inpaint.py:73-77`. The Jacobi sweep computes neighbour averages only for
`nb[1:-1, 1:-1]`; `nb` is zero-initialised, so any hole pixel on the patch
border is reassigned `0.0` (= the patch mean, since inputs are mean-removed)
**every iteration**. That is a spurious Dirichlet condition `z = mean` along
`border ∩ hole`, and it propagates into the interior solution.

Both mask kinds can touch the border: squares when `r0 == 0` or `c0 == 0`
(`rng.integers(0, size-h)` includes 0), and strokes wander to any edge (their
coordinates are clipped into the patch). Impact: harmonic scores are wrong for
the subset of masks touching an edge — plausibly a few percent of patches, all
biased toward extra flatness.

*Fix*: treat border hole pixels with one-sided neighbour averages, or pad the
array by 1 with edge replication before iterating.

### B2 — evaluation subset is geographically biased
`tests/test_inpaint_probe.py:161`: `ev = np.flatnonzero(te)[:n_eval]` takes the
**first** 250 test patches in array order. The patch array is ordered by tile
name (`sample_patches.py` iterates `sorted(by_tile.items())`), and 3DEP tile
names sort by latitude — `n25...` before `n49...`. So the 250-patch eval set is
drawn from the alphabetically-earliest (southernmost) test tiles, not from all
~80 of them. Every median in `inpaint_texture.csv` is computed on a
southern-CONUS-weighted sample.

*Fix*: `ev = rng.choice(np.flatnonzero(te), n_eval, replace=False)`. One line.
Re-run before quoting any aggregate number.

### B3 — torch RNG never seeded → the run-to-run variance we just measured
`tests/test_inpaint_probe.py` seeds numpy (`default_rng(seed)`) but never calls
`torch.manual_seed`. Network init and `torch.randperm` order are therefore
different every run. This is the direct cause of the two same-config runs
differing by Δpsd_ratio ≈ 0.11 and Δelev_rmse ≈ 0.15 — which is larger than the
L1-vs-texture *effect* being reported. Full determinism on GPU is not required;
seeded init plus 3–5 seeds with median ± IQR is.

*Fix*: `torch.manual_seed(seed)` at the top of `train()` (derive per-net seeds,
e.g. `seed` and `seed+1`, so the two nets don't share init), and add a
`--seed`-sweep wrapper.

### B4 — stale checkpoint default in the hedging diagnostic
`tests/test_hedging_diagnostic.py:154`: `--ckpt` defaults to
`/nfs/data/cxs1024/channel_geometry/results/inpaint_unet.pt`, a pre-reorg
artefact of a differently-trained net. The probe now writes
`dem_foundation/logs/unet_l1.pt`. Running the diagnostic with defaults silently
analyses the wrong (older) model.

*Fix*: point the default at `ROOT / "logs" / "unet_l1.pt"` and error if missing.

---

## Metric validity

### M1 — `psd_ratio` is diluted for stroke masks
`lib/terrain.py:134-155` (`hole_crop`) approximates the hole by the largest
centred square inside its **bounding box**. For square masks that box is
(almost) all hole. For stroke masks — half of all masks — the bbox of a
wandering stroke is mostly *valid* pixels, where `pred == truth` by
construction. Those pixels drag `psd_ratio` toward 1.0 regardless of method:
the same dilution failure mode already fixed once at the whole-patch level,
recurring at the bbox level. The docstring says "a few valid pixels"; for
strokes it can be the majority.

Consequences: aggregate `psd_ratio` medians mix clean (square-mask) and
diluted (stroke-mask) values, understating the true smoothness gap; and the
per-panel figure captions (stroke mask) are individually diluted.

*Fix options*, in order of preference: (a) record mask kind per row and report
`psd_ratio` on square masks only (variogram already handles irregular masks
correctly via pair masking, so nothing is lost); (b) compute hole fraction of
the crop and drop crops below ~70% hole; (c) both.

### M2 — the figure's three rows share one mask
`tests/test_inpaint_probe.py:196` builds `np.random.default_rng(seed + 7)`
**inside** the loop, so all three example rows get the identical stroke mask.
Fine if intentional (controls the mask across terrain classes), but it means
the figure shows one mask shape, not the mask distribution — and combined with
M1 its captions are all diluted the same way. If intentional, hoist the rng out
of the loop and say so in a comment; if not, vary it.

### M3 — harmonic fill may not be converged
`fill_harmonic` runs 400 Jacobi sweeps; a 40-px hole needs on the order of
1,000+ for convergence. The unconverged solution sits between the constant
initial fill and the true harmonic surface — still smooth, so no conclusion
flips, but "harmonic" in the tables is not exactly harmonic, and its
`elev_rmse` is not exactly the Dirichlet minimiser's. *Fix*: iterate to a
residual tolerance instead of a fixed count (or solve the 5-point Laplacian
directly with `scipy.sparse.linalg.spsolve`; holes here are ≤ ~1,700 unknowns).

---

## Design / statistics

### D1 — the residual gate is structurally conservative; say so before citing it
`channel_geometry/src/test_residual_learnability.py`. Stage-1 residuals are
out-of-fold under leave-HUC2-out, so **each region's residual comes from a
different stage-1 model and contains that region's stage-1 bias** — a regional
offset that is *by construction* invisible from the other regions' training
data. Stage 2 is then asked to predict, from other regions, a quantity that
includes an unpredictable per-region shift. That pushes stage-2 R² negative
even if a genuine within-region substrate signal exists.

The all-negative table (`log_d`: −0.082/−0.057/−0.045; `log_v`:
−0.126/−0.023/−0.034) is therefore a *lower bound*, and the qualitative
conclusion (no strong terrain-readable substrate signal in the residual)
stands — but the gate as run cannot distinguish "no signal" from "signal
smaller than the regional-offset noise".

*Cheap sharpening*: add a variant where stage-1 residuals come from grouped
site-level CV (one model, no per-region offsets) and stage-2 is still scored
leave-HUC2-out; also score stage 2 after per-region centring of the residual.
If both are still ≈ 0, the negative is airtight.

### D2 — lithology CV: 1° tiles allow adjacent-tile leakage
`test_lithology_baseline.py` groups folds by 1° tile. Geologic units and
physiographic provinces span many adjacent tiles, and adjacent tiles land in
different folds, so "leave-tile-out" is closer to grouped-random than to
leave-region-out. The honest 0.593 could be *optimistic*. When the uniform-CONUS
re-run happens, block by something larger (state, or 5°×5° blocks) — the
noise-floor script shows state is a natural unit here anyway.

### D3 — noise-floor cross-state estimate is starved (n = 7–263)
`test_sgmc_noise_floor.py` samples uniformly over CONUS, so almost no pairs
straddle a state line. The same-state curve (the number that mattered) is
solid at n ≈ 23k, but the cross-survey penalty (~0.10) rests on n ≤ 263.
*Fix if the number is ever quoted*: sample second points only from an annulus
around the first, keeping pairs whose segment crosses a state boundary — or
directly sample along state-line buffers. One extra hour of compute buys a
usable estimate.

### D4 — patch sampler draws gages with replacement
`sample_patches.py:78`: `rng.integers(0, len(gl), n*1.6)` can draw the same
gage repeatedly; two ±15 km jitters of one gage can overlap substantially.
Tile-level splitting prevents train/test leakage, but duplicated neighbourhoods
up-weight some landscapes within the training set and make the effective n
smaller than 6,000. *Fix*: sample gages without replacement per round, or
enforce a minimum centre-to-centre spacing (~2 patch widths).

---

## Minor / architectural notes (no action required now)

- `PConvUNet.u0` skips the **raw masked input** (zeros in the hole) into a
  plain `Conv2d` (`lib/inpaint.py:197,208`) — reintroducing at full resolution
  exactly the mask-encoding problem `PartialConv2d`'s docstring warns about.
  The original PConv paper uses partial convs in the decoder too. Candidate
  explanation for the rim-jump gap vs harmonic (0.026–0.031 vs 0.012); cheap to
  test by masking that skip.
- Decoder `BatchNorm` after concatenating hole-containing features is standard
  but is exactly what made the missing-`eval()` bug so violent; consider
  `GroupNorm`, which removes the train/eval statistics distinction entirely.
- `test_inpaint_probe.py` mask RNG is one shared stream (tile shuffle →
  training masks net 1 → training masks net 2 → eval masks), so changing
  `--epochs` changes the eval masks. Give evaluation its own
  `default_rng(seed + k)`.
- `df["terrain"]` (flat/moderate/rough) is computed but no longer used in any
  printed table — either restore the per-terrain breakdown (it was informative)
  or delete the dead code.
- `fill_idw` is O(hole × valid) brute force — fine at 128², will not scale to
  larger patches; `scipy.spatial.cKDTree` when that day comes.
- `test_lithology_baseline.py` imports `uniform_filter` inside the per-radius
  loop — hoist for clarity (trivial cost either way).

## Status — fixes applied 2026-08-17

| finding | action | verified by |
|---|---|---|
| B1 | `fill_harmonic` relaxes border hole pixels via edge-replicated pad | `test_lib_smoke.py` (ramp with border-touching hole) |
| M3 | iterate to tol 1e-3 m (cap 4000) instead of fixed 400 sweeps | smoke: interior hole reproduces exact plane |
| B2 | eval subset drawn randomly from ALL test tiles, own rng | inspection + 1-epoch end-to-end run |
| B3 | `torch.manual_seed` per net; seed-suffixed CSVs; 3-seed sweep | smoke: same seed → identical nets |
| B4 | hedging `--ckpt` defaults to `logs/unet_l1.pt`, errors if missing | inspection |
| M1 | `hole_crop` → `(crop, hole_frac)`; `score` NaNs crops < 60% hole; probe records `mask_kind` and prints the by-kind table | smoke (L-shape diluted → NaN; square → 1.0) + end-to-end |
| M2 | figure rng hoisted; square masks for examples; NaN-safe captions | end-to-end run |
| D1 | `--stage1 grouped` variant added to the residual gate | run in progress |
| minor | per-terrain table restored; eval masks decoupled from training rng; `uniform_filter` hoisted | end-to-end run |

New: `src/tests/test_lib_smoke.py` — pins each fixed invariant; run after any
lib edit. Its first version failed twice on the *harmonic border* check — both
times the physics was right and the tolerance wrong (a zero-flux boundary
*must* bend a ramp fill by ~0.4·L, decaying as exp(−πy/L)); the final
assertions test "boundary-layer sized, not patch-mean sized", which is the
actual bug signature.

Not addressed (by design): D2/D3 fold into the uniform-CONUS lithology re-run;
D4 would invalidate the existing patch set — revisit when patches are next
regenerated; PConv decoder notes are experiments, not fixes.

## Suggested order of work

1. B3 + B2 (seeding, unbiased eval) — then the 3–5-seed re-run that was already
   agreed, which also refreshes every number under B1/M1 fixes.
2. M1 (mask-kind split for `psd_ratio`) and B1 (border fill) — small, before
   the same re-run so it only happens once.
3. B4, M2 — one-line hygiene.
4. D1 variant of the residual gate — before the auxiliary-target design doc
   cites the gate as evidence.
5. D2/D3 fold into the already-planned uniform-CONUS lithology re-run.
