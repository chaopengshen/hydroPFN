"""Shared CV folds and metrics, so every model is scored identically.

Splits are grouped by COMID.  A random split at the *row* level would put
measurements from the same gage on both sides and inflate every model here --
median 6 measurements per site means row-level CV is close to memorisation.
Leave-region-out comes later; this is the random-but-honest baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

N_FOLDS = 5
SEED = 0


def assign_region_folds(df: pd.DataFrame, col: str = "HUC2") -> tuple[np.ndarray, list]:
    """One fold per HUC2 region -- leave-region-out.

    This is the split the structural prior exists for.  Grouped-random CV holds
    out gages but keeps them inside the training distribution; holding out a
    whole water-resources region forces genuine spatial transfer, which is what
    a tracer reach in an unsampled basin actually asks of the model.
    """
    regions = sorted(df[col].dropna().unique())
    lut = {r: i for i, r in enumerate(regions)}
    fold = df[col].map(lut).fillna(-1).astype(int).to_numpy()
    return fold, regions


def assign_folds(df: pd.DataFrame, n_folds: int = N_FOLDS, seed: int = SEED) -> np.ndarray:
    """Fold index per row, grouped so a COMID never spans folds."""
    rng = np.random.default_rng(seed)
    sites = df.COMID.unique()
    rng.shuffle(sites)
    # GroupKFold is deterministic given group order; shuffling the site list
    # first is what makes the split random rather than COMID-ordered.
    order = {c: i for i, c in enumerate(sites)}
    shuffled = df.COMID.map(order).to_numpy()
    fold = np.empty(len(df), dtype=int)
    gkf = GroupKFold(n_splits=n_folds)
    for k, (_, test) in enumerate(gkf.split(df, groups=shuffled)):
        fold[test] = k
    return fold


# ---------------------------------------------------------------- metrics


def r2(obs: np.ndarray, pred: np.ndarray) -> float:
    """Coefficient of determination, 1 - SSE/SST."""
    m = np.isfinite(obs) & np.isfinite(pred)
    if m.sum() < 2:
        return np.nan
    o, p = obs[m], pred[m]
    return 1.0 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


# NSE and R2 share this formula; the names differ by convention (NSE in
# hydrology, R2 in ML) and both are reported against the literature values
# using whichever name that paper used.
nse = r2


def pbias(obs: np.ndarray, pred: np.ndarray) -> float:
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    return 100.0 * (p - o).sum() / o.sum()


def score_block(obs_log: np.ndarray, pred_log: np.ndarray) -> dict:
    """Score a log10-space prediction in both log and linear space."""
    m = np.isfinite(obs_log) & np.isfinite(pred_log)
    o, p = obs_log[m], pred_log[m]
    return {
        "n": int(m.sum()),
        "R2_log": r2(o, p),
        "NSE_linear": nse(10.0 ** o, 10.0 ** p),
        "PBIAS_%": pbias(10.0 ** o, 10.0 ** p),
        "RMSE_log": float(np.sqrt(np.mean((o - p) ** 2))),
    }


def site_median_scores(df: pd.DataFrame, obs_col: str, pred_col: str) -> dict:
    """Collapse to one value per site, the way Chang et al. 2024 framed it.

    Chang trained on per-site medians, so their R2 is a *downstream* hydraulic
    geometry score.  Collapsing our per-measurement predictions the same way is
    the only like-for-like comparison available.
    """
    g = df.groupby("COMID")[[obs_col, pred_col]].median()
    return score_block(g[obs_col].to_numpy(), g[pred_col].to_numpy())
