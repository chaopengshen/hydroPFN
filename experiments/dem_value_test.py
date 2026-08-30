"""Cheap yes/no: does DEM carry signal the existing attributes do not?

Motivation. Building the DEM connector and then checking whether PUB improves
CONFLATES two questions -- does DEM carry signal, and did we wire it correctly.
That confound has produced a wrong conclusion repeatedly in this project (the
geo-encoding was declared useless when it had merely been attached to a dead
path). So test the signal FIRST, with no neural network and no GPU.

Two tests:

  A. Does DEM add over the statics we already have?
     Predict flow signatures under leave-one-region-out from
       (i) CAMELS statics, (ii) statics + DEM descriptors, (iii) DEM alone.
     If (ii) is not better than (i), DEM adds nothing FOR THIS TARGET and no
     architecture will rescue that.

  B. Does DEM explain where the model already FAILS?
     Regress the trained model's per-basin NSE on DEM descriptors. This is the
     sharper test: it asks whether DEM is COMPLEMENTARY to what the model
     already extracts, rather than merely correlated with streamflow.

Caveat carried throughout: the patches are 128x128 @10 m -- a 1.28 km box at
the GAUGE. That is the right view for channel geometry and the wrong one for
basin runoff response, which depends on basin-wide hypsometry and drainage
texture. A negative result here therefore means "an outlet patch adds nothing",
NOT "DEM adds nothing".
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def dem_descriptors(z: np.ndarray) -> np.ndarray:
    """(N,H,W) elevation -> (N,F) terrain descriptors.

    Deliberately simple and interpretable. Anything a CNN could extract that
    these miss is a reason to build the connector; anything these capture that
    the statics miss is already a reason.
    """
    out = []
    for a in z:
        gy, gx = np.gradient(a, 10.0)                 # 10 m pixels
        slope = np.hypot(gx, gy)
        lap = (np.gradient(gx, 10.0, axis=1) +
               np.gradient(gy, 10.0, axis=0))
        rng = a.max() - a.min()
        # TPI: how much the centre sits above its surroundings
        c = a.shape[0] // 2
        tpi = a[c, c] - a.mean()
        out.append([
            a.mean(), a.std(), rng,
            (a.mean() - a.min()) / (rng + 1e-6),      # hypsometric integral
            slope.mean(), slope.std(), np.percentile(slope, 90),
            (slope < 0.02).mean(),                    # flat fraction
            np.abs(lap).mean(), lap.std(),            # curvature
            tpi,
            np.percentile(a, 90) - np.percentile(a, 10),
        ])
    return np.asarray(out, dtype=np.float64)


DEM_NAMES = ["elev_mean", "elev_std", "relief", "hypso_int", "slope_mean",
             "slope_std", "slope_p90", "flat_frac", "curv_abs", "curv_std",
             "tpi", "elev_p90_p10"]


def ridge_cv(X, y, groups, alpha=1.0):
    """Leave-one-group-out ridge; returns out-of-fold R^2."""
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 20 or te.sum() < 3:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        A = Xtr.T @ Xtr + alpha * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xtr.T @ (y[tr] - ym))
        pred[te] = Xte @ w + ym
    m = np.isfinite(pred)
    return float(1 - ((y[m] - pred[m]) ** 2).sum() /
                 ((y[m] - y[m].mean()) ** 2).sum())


def main(a):
    d = xr.open_dataset(a.nc)
    dm = np.load(a.dem, allow_pickle=True)
    ok = dm["ok"]
    print(f"DEM available for {ok.sum()} / {len(ok)} gauges")

    sid = np.array([str(s).zfill(8) for s in d.station_ids.values])
    region = np.array([s[:2] for s in sid])
    stat_names = [v for v in d.data_vars if "time" not in d[v].dims]
    S = np.stack([np.nan_to_num(np.asarray(d[v].values, dtype=np.float64))
                  for v in stat_names], -1)

    Q = np.asarray(d["QObs"].values, dtype=np.float64)
    P = np.asarray(d["prcp_daymet"].values, dtype=np.float64)
    Q = np.where(np.isfinite(Q) & (Q >= 0), Q, np.nan)

    # ---- flow signatures: what a terrain descriptor could plausibly explain
    with np.errstate(invalid="ignore", divide="ignore"):
        sig = {
            "runoff_ratio": np.nanmean(Q, 1) / (np.nanmean(P, 1) + 1e-9),
            "log_q_mean": np.log1p(np.nanmean(Q, 1)),
            "flashiness": (np.nanmean(np.abs(np.diff(Q, axis=1)), 1) /
                           (np.nanmean(Q, 1) + 1e-9)),
            "baseflow_idx": (np.nanpercentile(Q, 10, axis=1) /
                             (np.nanmean(Q, 1) + 1e-9)),
            "fdc_slope": (np.log1p(np.nanpercentile(Q, 66, axis=1)) -
                          np.log1p(np.nanpercentile(Q, 33, axis=1))),
        }

    D = np.full((len(sid), len(DEM_NAMES)), np.nan)
    D[ok] = dem_descriptors(dm["dem"][ok])

    use = ok & np.isfinite(D).all(1) & np.isfinite(S).all(1)
    print(f"usable basins: {use.sum()}\n")

    print(f"{'signature':>14} | {'statics':>8} {'stat+DEM':>9} "
          f"{'DEM only':>9} | {'DEM adds':>9}")
    print("-" * 62)
    for name, y in sig.items():
        m = use & np.isfinite(y)
        yy, g = y[m], region[m]
        r_s = ridge_cv(S[m], yy, g)
        r_sd = ridge_cv(np.hstack([S[m], D[m]]), yy, g)
        r_d = ridge_cv(D[m], yy, g)
        print(f"{name:>14} | {r_s:8.4f} {r_sd:9.4f} {r_d:9.4f} | "
              f"{r_sd - r_s:+9.4f}")

    # ---- Test B: does DEM explain the model's per-basin errors?
    if a.per_basin_nse:
        z = np.load(a.per_basin_nse, allow_pickle=True)
        nse_sid = np.asarray(z["site_id"]).astype(str)
        nse = np.asarray(z["nse"], dtype=np.float64)
        idx = {s: i for i, s in enumerate(sid)}
        rows = [(idx[s], v) for s, v in zip(nse_sid, nse)
                if s in idx and np.isfinite(v)]
        ii = np.array([r[0] for r in rows])
        yy = np.array([r[1] for r in rows])
        k = use[ii]
        ii, yy = ii[k], yy[k]
        print(f"\n=== Test B: predicting the model's per-basin NSE "
              f"({len(yy)} basins) ===")
        g = region[ii]
        print(f"  from statics only : R2 {ridge_cv(S[ii], yy, g):+.4f}")
        print(f"  from DEM only     : R2 {ridge_cv(D[ii], yy, g):+.4f}")
        print(f"  from both         : R2 "
              f"{ridge_cv(np.hstack([S[ii], D[ii]]), yy, g):+.4f}")
        print("  A positive DEM-only R2 means terrain predicts WHERE WE FAIL,")
        print("  i.e. there is signal the model is not currently using.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True)
    ap.add_argument("--dem", default="logs/camels_dem.npz")
    ap.add_argument("--per-basin-nse", default=None,
                    help="npz with site_id/nse from a trained model")
    main(ap.parse_args())
