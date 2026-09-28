"""Regional held-out terrain probe using training-era hydrologic signatures.

Tests whether basin-wide physical terrain explains residual information beyond
CAMELS attributes. This is an information diagnostic, NOT daily streamflow NSE.
Both terrain arms have exactly the same descriptors and ridge penalty. The
attribute residual targets are cross-fitted within each outer training region
set. No discharge from the future evaluation period is used anywhere.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels

TARGETS = ["log_mean_q", "log_std_q", "log_p95_q", "lag1_correlation",
           "log_flashiness"]


def signatures(q):
    out = np.full((len(q), len(TARGETS)), np.nan)
    for i, values in enumerate(q):
        finite = np.isfinite(values)
        if finite.mean() < .9:
            continue
        v = values[finite].astype(float)
        pairs = finite[1:] & finite[:-1]
        previous, current = values[:-1][pairs], values[1:][pairs]
        if len(v) < 365 or min(previous.std(), current.std()) <= 0:
            continue
        out[i] = [np.log(max(v.mean(), .001)), np.log(max(v.std(), .001)),
                  np.log(max(np.percentile(v, 95), .001)),
                  np.corrcoef(previous, current)[0, 1],
                  np.log(max(np.abs(current - previous).sum() /
                             max(current.sum(), .001), .001))]
    return out


def ridge_predict(x, y, test_x, alpha):
    """Fit normalization and ridge on training rows only; unpenalized intercept."""
    mean = np.nanmean(x, 0)
    mean = np.nan_to_num(mean)
    std = np.nanstd(x, 0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.)
    a = np.nan_to_num((x - mean) / std)
    b = np.nan_to_num((test_x - mean) / std)
    ym = y.mean(0)
    weights = np.linalg.solve(a.T @ a + alpha * np.eye(a.shape[1]),
                              a.T @ (y - ym))
    return b @ weights + ym


def crossfit_baseline(x, y, groups, train, alpha):
    """Residual targets cannot come from an attribute model fit to those labels."""
    pred = np.full_like(y, np.nan, dtype=float)
    for group in np.unique(groups[train]):
        val = train[groups[train] == group]
        inner = train[groups[train] != group]
        pred[val] = ridge_predict(x[inner], y[inner], x[val], alpha)
    if not np.isfinite(pred[train]).all():
        raise ValueError("cross-fitting failed to cover every training basin")
    return y[train] - pred[train]


def load_terrain(path, gages):
    d = np.load(path)
    ids = [str(int(v)).zfill(8) for v in d["site_id"]]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate terrain IDs: {path}")
    pos = {v: i for i, v in enumerate(ids)}
    x = np.full((len(gages), d["feats"].shape[1]), np.nan)
    for i, gage in enumerate(gages):
        j = pos.get(str(int(gage)).zfill(8))
        if j is not None and d["ok"][j]:
            x[i] = d["feats"][j]
    return x, d["feature_names"].tolist()


def main(a):
    if a.alpha <= 0:
        raise ValueError("positive prespecified ridge penalty required")
    out = Path(a.out)
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    sub, gage = P.load_531(load_camels(a.nc))
    tw = P.windows(sub["time"], "temporal")["train"]
    q = sub["x"][:, tw, -1].copy()
    q[sub["valid"][:, tw, -1] == 0] = np.nan
    y = signatures(q)
    attrs = sub["attrs"].astype(float)
    ids = gage["gage"].to_numpy().astype(str)
    basin, names = load_terrain(a.basin, ids)
    outlet, outlet_names = load_terrain(a.outlet, ids)
    if names != outlet_names:
        raise ValueError("terrain descriptors must match exactly across arms")
    groups = np.full(len(ids), -1)
    for k, rows in enumerate(P.folds("PUR", gage)):
        groups[rows] = k
    if (groups < 0).any():
        raise ValueError("regional folds do not cover all basins")
    keep = np.isfinite(y).all(1) & np.isfinite(basin).all(1) & np.isfinite(outlet).all(1)
    if keep.sum() < a.min_basins:
        raise ValueError(f"only {keep.sum()} complete paired basins; need {a.min_basins}")
    predictions = {label: np.full_like(y, np.nan) for label in
                   ("attributes", "attributes+outlet", "attributes+basin")}
    standardized_errors = {label: np.full_like(y, np.nan) for label in predictions}
    for group in np.unique(groups):
        train = np.flatnonzero(keep & (groups != group))
        test = np.flatnonzero(keep & (groups == group))
        if not len(test):
            raise ValueError(f"region {group} has no paired complete basins")
        baseline = ridge_predict(attrs[train], y[train], attrs[test], a.alpha)
        residual = crossfit_baseline(attrs, y, groups, train, a.alpha)
        predictions["attributes"][test] = baseline
        for label, terrain in (("outlet", outlet), ("basin", basin)):
            predictions[f"attributes+{label}"][test] = baseline + ridge_predict(
                terrain[train], residual, terrain[test], a.alpha)
        scale = np.maximum(y[train].std(0), 1e-6)
        for label in predictions:
            standardized_errors[label][test] = ((predictions[label][test] - y[test]) / scale) ** 2
    out.mkdir(parents=True)
    details = pd.DataFrame({"gage": ids, "region": groups, "included": keep})
    rows = []
    for j, target in enumerate(TARGETS):
        details[f"target_{target}"] = y[:, j]
        for label, pred in predictions.items():
            details[f"{label}_{target}"] = pred[:, j]
            for region in [-1, *np.unique(groups).tolist()]:
                mask = keep & ((groups == region) if region >= 0 else True)
                err = pred[mask, j] - y[mask, j]
                sst = ((y[mask, j] - y[mask, j].mean()) ** 2).sum()
                rows.append({"model": label, "target": target, "region": region,
                             "n": int(mask.sum()), "rmse": float(np.sqrt(np.mean(err ** 2))),
                             "r2": float(1 - (err ** 2).sum() / sst) if sst else None,
                             "train_scaled_mse": float(np.mean(standardized_errors[label][mask, j]))})
    details.to_csv(out / "paired_basin_predictions.csv", index=False)
    pd.DataFrame(rows).to_csv(out / "regional_scores.csv", index=False)
    summary = {"task": "training-era hydrologic signature residual probe",
               "not_a_streamflow_benchmark": True, "dates": P.PERIODS["temporal"][:2],
               "n_paired": int(keep.sum()), "n_excluded": int((~keep).sum()),
               "alpha_prespecified": a.alpha, "terrain_features": names,
               "files_sha256": {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest()
                                for p in (a.basin, a.outlet)},
               "mean_train_scaled_mse": {k: float(v[keep].mean()) for k, v in standardized_errors.items()},
               "args": vars(a)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=P.CAMELS_NC)
    p.add_argument("--basin", required=True)
    p.add_argument("--outlet", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--alpha", type=float, default=100.)
    p.add_argument("--min-basins", type=int, default=425)
    main(p.parse_args())
