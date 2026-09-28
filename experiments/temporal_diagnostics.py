"""Paired artifact diagnostics, using the CAMELS-531 scorer and gauge joins.

Window-position curves are normalized MSE on the SAME days for both models,
not NSE on small subsamples. Their seasonal sampling is therefore controlled
by the paired comparator, though they do not by themselves establish causality.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels


def gauge_ids(values):
    return np.asarray([str(int(x)).zfill(8) for x in values])


def load_run(path, filename, wanted, expected_target):
    meta = json.loads((path / "run.json").read_text())
    if meta["protocol"] != "temporal" or meta["n_days"] != expected_target.shape[1]:
        raise ValueError(f"not the temporal protocol: {path}")
    ids = gauge_ids(np.load(path / "gage.npy", allow_pickle=True))
    if len(np.unique(ids)) != len(ids) or set(ids) != set(wanted):
        raise ValueError(f"duplicate or mismatched gauges: {path}")
    index = {g: i for i, g in enumerate(ids)}
    rows = [index[g] for g in wanted]
    target = np.load(path / "targ.npy")[rows]
    if not np.allclose(target, expected_target, atol=1e-6, rtol=1e-6, equal_nan=True):
        raise ValueError(f"target/date/scale mismatch: {path}")
    pred = np.load(path / filename)[rows]
    if pred.shape != target.shape or not np.isfinite(pred).all():
        raise ValueError(f"invalid predictions: {path}")
    return pred, meta


def window_positions(time, patch, win):
    windows = P.windows(time, "temporal")
    day0 = windows["eval_in"].start // patch * patch
    span = -(-(windows["eval_in"].stop - day0) // patch) * patch
    n_patch = span // patch
    starts = list(range(0, n_patch - win + 1, win))
    if starts[-1] != n_patch - win:
        starts.append(n_patch - win)
    position = np.full(span, -1, dtype=int)
    for start in starts:
        position[start * patch:(start + win) * patch] = np.arange(win * patch)
    keep = slice(windows["score"].start - day0, windows["score"].stop - day0)
    if (position[keep] < 0).any():
        raise ValueError("incomplete position reconstruction")
    return position[keep]


def decomposition(pred, obs):
    rows = []
    for p, y in zip(pred, obs):
        valid = np.isfinite(y)
        p, y = p[valid].astype(float), y[valid].astype(float)
        sy = y.std()
        alpha = p.std() / sy
        beta = (p.mean() - y.mean()) / sy
        corr = np.corrcoef(p, y)[0, 1]
        rows.append(((alpha - 1) ** 2, beta ** 2, 2 * alpha * (1 - corr)))
    return np.asarray(rows)


def main(a):
    sub, gage = P.load_531(load_camels(a.nc))
    windows = P.windows(sub["time"], "temporal")
    q = sub["x"][..., -1].astype(float)
    q[sub["valid"][..., -1] == 0] = np.nan
    target = q[:, windows["score"]]
    ids = gauge_ids(gage["gage"])
    train_var = np.nanvar(q[:, windows["train"]], axis=1)
    variance_group = pd.qcut(train_var, 4, labels=False)
    position = window_positions(sub["time"], a.patch, a.win)
    by_basin = pd.DataFrame({"gage": ids, "train_variance": train_var,
                             "variance_quartile": variance_group + 1})
    results, predictions = {}, {}
    for spec in a.run:
        label, path, filename = spec.split("=", 2)
        pred, meta = load_run(Path(path), filename, ids, target)
        metrics = P.nse_table(pred, target)
        dec = decomposition(pred, target)
        np.testing.assert_allclose(dec.sum(1), 1 - metrics.nse, atol=1e-5, rtol=1e-4)
        by_basin[label + "_nse"] = metrics.nse
        by_basin[label + "_rmse"] = metrics.rmse
        for j, name in enumerate(("amplitude_error", "bias_error", "correlation_error")):
            by_basin[label + "_" + name] = dec[:, j]
        predictions[label] = pred
        results[label] = {"path": path, "prediction_file": filename,
                          "median_nse": float(np.nanmedian(metrics.nse)),
                          "median_error_components": dict(zip(
                              ("amplitude", "bias", "correlation"), np.nanmedian(dec, 0))),
                          "run": meta}
    baseline = a.reference or next(iter(predictions))
    summary, curves = [], []
    for label, pred in predictions.items():
        delta = by_basin[label + "_nse"] - by_basin[baseline + "_nse"]
        by_basin[label + "_delta_vs_" + baseline] = delta
        for group in range(4):
            rows = variance_group == group
            summary.append({"model": label, "variance_quartile": group + 1,
                            "n": int(rows.sum()),
                            "median_nse": float(np.nanmedian(by_basin.loc[rows, label + "_nse"])),
                            "median_paired_delta": float(np.nanmedian(delta[rows])),
                            "fraction_better": float((delta[rows] > 0).mean())})
        error = (pred - target) ** 2 / np.maximum(train_var[:, None], 0.01)
        for dimension, bins in [("window_64d", position // 64),
                                 ("patch_day", position % a.patch)]:
            for b in np.unique(bins):
                basin_mse = np.nanmean(error[:, bins == b], axis=1)
                curves.append({"model": label, "dimension": dimension, "bin": int(b),
                               "days": int((bins == b).sum()),
                               "median_basin_nmse": float(np.nanmedian(basin_mse)),
                               "mean_basin_nmse": float(np.nanmean(basin_mse))})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    by_basin.to_csv(out / "basin_diagnostics.csv", index=False)
    pd.DataFrame(summary).to_csv(out / "variance_quartiles.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "position_errors.csv", index=False)
    payload = {"reference": baseline, "target_alignment": "verified by gauge and raw targets",
               "patch": a.patch, "win": a.win, "models": results,
               "warning": "Position labels describe the forward PUBModel tiling only; other models use identical dates."}
    (out / "summary.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps({k: v["median_nse"] for k, v in results.items()}, indent=2))
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"Wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=P.CAMELS_NC)
    p.add_argument("--run", action="append", required=True,
                   help="LABEL=/absolute/run/directory=pred.npy (or pred_K0.npy)")
    p.add_argument("--reference")
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--win", type=int, default=32)
    p.add_argument("--out", required=True)
    main(p.parse_args())
