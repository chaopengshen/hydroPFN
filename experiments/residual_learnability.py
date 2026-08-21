"""Gate: is the channel-geometry RESIDUAL learnable from terrain at all?

Reproduces: data/processed/residual_learnability.csv

Context.  The proposal is to pretrain the DEM foundation model with an auxiliary
head that predicts channel geometry, so the encoder is forced to represent
substrate erodibility -- something masked-elevation reconstruction has no
incentive to encode.  At inference the head is discarded and the encoder runs
anywhere, so this is NOT feature fusion and the usual redundancy objection does
not apply.

But an auxiliary target only shapes an encoder if it carries signal the encoder
can reach.  This tests that, and it costs no pretraining.

Two stages, one fold structure (leave-HUC2-out) throughout:

  stage 1   tabular attributes -> log_d, log_v.  Out-of-fold predictions give
            residuals that are, by construction, the part drainage area, slope
            and discharge cannot explain.  That is precisely the substrate part.
  stage 2   can anything predict those residuals?

Stage 2 runs three predictor sets, and the contrast is the point:

  tabular   the same columns that MADE the residual.  Must score ~0.  This is
            the sanity check -- a non-zero score here means the machinery leaks.
  dem       terrain descriptors around the reach.  THIS IS THE GATE.
  both      whether the two interact.

Read it as: R2 ~ 0 on `dem` means the DEM patch holds nothing for an auxiliary
head to latch onto, and pretraining on geometry cannot shape the representation.
Clearly positive means the signal exists and a learned encoder should beat these
hand-engineered features.

Residual targets, not raw W and d, and no power law is fitted anywhere -- fitted
per-site AHG exponents need 6-20 measurements and ~25% of sites have too little
flow range to identify b, so forcing the data through an exponential form would
inject noise rather than remove it.

**Stage-1 fold structure matters more than it looks** (review D1).  With
`--stage1 region` (the original run), each region's residual comes from a
DIFFERENT leave-that-region-out model and therefore contains that region's
stage-1 bias -- an offset that is unpredictable from other regions by
construction.  That makes the gate conservative: it cannot distinguish "no
signal" from "signal smaller than the regional-offset noise".  `--stage1
grouped` defines residuals from a single site-grouped random-CV stage-1 (no
per-region offsets; still no site predicts itself), then scores stage 2 under
leave-HUC2-out as before.  If the gate is still ~0 under `grouped`, the
negative is airtight.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from hydropfn.data.folds import assign_folds, assign_region_folds, r2

# The attribute model.  Deliberately excludes every DEM-derived column so that
# stage-2 `dem` is a genuinely held-out predictor set, and excludes MEANELEVSMO
# (absolute elevation is a location fingerprint -- dropping it GAINED +0.003).
TABULAR = [
    "log_Q", "log_A_drain", "log_slope", "log_q_mm_day", "sinuosity",
    "log_lengthkm", "log_arbolatesu", "log_drain_density", "log_A_local",
    "NDVI", "AI", "Forest", "Agriculture", "Developed",
    "b0_sand", "b0_clay", "b10_clay", "b10_sand", "Silt_101",
]


def _dem_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("dem_")]


def _oof(X: np.ndarray, y: np.ndarray, fold: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold predictions under leave-HUC2-out."""
    pred = np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        ok = tr & np.isfinite(y)
        if ok.sum() < 200 or te.sum() == 0:
            continue
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed)
        rf.fit(X[ok], y[ok])
        pred[te] = rf.predict(X[te])
    return pred


def main(table: str, out_csv: str, seed: int, stage1: str) -> None:
    df = pd.read_csv(table, low_memory=False)
    dem = _dem_cols(df)
    print(f"{len(df):,} rows, {df.site_no.nunique():,} sites, "
          f"{len(TABULAR)} tabular + {len(dem)} DEM features")

    fold, regions = assign_region_folds(df)
    keep = fold >= 0
    df, fold = df[keep].reset_index(drop=True), fold[keep]
    print(f"  leave-HUC2-out over {len(regions)} regions, {len(df):,} rows kept")

    # Stage-1 folds: `region` reuses the leave-HUC2-out folds (conservative --
    # residuals then contain per-region stage-1 bias, see module docstring);
    # `grouped` uses site-grouped random CV, so the residual is purely
    # site-level.  Stage 2 is leave-HUC2-out in BOTH cases.
    f1 = assign_folds(df, seed=seed) if stage1 == "grouped" else fold
    print(f"  stage-1 folds: {stage1}")

    Xt = np.nan_to_num(df[TABULAR].to_numpy(float), nan=0.0)
    Xd = np.nan_to_num(df[dem].to_numpy(float), nan=0.0)
    Xb = np.hstack([Xt, Xd])

    rows = []
    for tgt in ("log_d", "log_v"):
        y = df[tgt].to_numpy(float)
        p1 = _oof(Xt, y, f1, seed)
        base = r2(y, p1)
        res = y - p1
        m = np.isfinite(res)
        print(f"\n=== {tgt} ===  stage-1 attribute model R2 = {base:.3f}")
        print(f"  residual sd = {np.nanstd(res):.4f} (target sd {np.nanstd(y):.4f})")

        for name, X in (("tabular", Xt), ("dem", Xd), ("both", Xb)):
            r = np.full(len(res), np.nan)
            r[m] = _oof(X[m], res[m], fold[m], seed)
            sc = r2(res[m], r[m])
            print(f"  residual <- {name:8s} R2 = {sc:+.4f}")
            rows.append({"target": tgt, "stage1": stage1, "stage1_r2": base,
                         "predictors": name, "residual_r2": sc,
                         "n": int(m.sum())})

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n  `tabular` near 0 = machinery is honest (those columns made the")
    print("  residual).  `dem` is the gate: if it is also near 0, a geometry")
    print("  auxiliary head has nothing in the DEM to learn from.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="residual_learnability.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1", choices=["region", "grouped"], default="region")
    a = ap.parse_args()
    main(a.table, a.out, a.seed, a.stage1)
