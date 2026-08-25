"""Does LOCAL DEM carry information about LOCAL channel geometry?

Why this target instead of streamflow. The CAMELS probe asked whether a DEM
patch at a gauge improves prediction of basin-integrated discharge, and the
answer was no at every scale (1.3 / 12.8 / 51.2 km). But discharge is a
BASIN-scale quantity and the patch is LOCAL, so that test was poorly matched:
it could only ever detect DEM acting as a climate/position proxy, which the
curated attributes already supply.

Channel width and depth are different. They are set by valley form AT THE SITE
-- confinement, cross-sectional shape, local slope, terrace structure. If local
terrain carries geomorphic information that an attribute table does not, this
is where it must show up. A null result here is much stronger evidence against
the DEM pathway than a null result on discharge.

Protocol, following the channel_geometry project's standing rules:
  * leave-HUC2-out, never random CV
  * `max_depth` is MEAN HYDRAULIC depth (xsec_area / width) despite the name
  * width is partly DERIVED (W = Q/(d*v)); `log_W` here is the table's own
    column and its provenance is inherited, not re-derived
  * site medians, because the DEM patch is per-site -- stated openly rather
    than pretending this is per-measurement
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


BASE_FEATS = ["log_A_drain", "log_slope", "slope_at_floor", "sinuosity",
              "StreamOrde", "log_lengthkm", "log_arbolatesu",
              "log_drain_density", "log_A_local", "MEANELEVSMO"]

DEM_NAMES = ["elev_mean", "elev_std", "relief", "hypso_int", "slope_mean",
             "slope_std", "slope_p90", "flat_frac", "curv_abs", "curv_std",
             "tpi", "elev_p90_p10"]


def dem_descriptors(z):
    out = []
    for a in z:
        gy, gx = np.gradient(a, 10.0)
        slope = np.hypot(gx, gy)
        lap = np.gradient(gx, 10.0, axis=1) + np.gradient(gy, 10.0, axis=0)
        rng = a.max() - a.min()
        c = a.shape[0] // 2
        out.append([a.mean(), a.std(), rng,
                    (a.mean() - a.min()) / (rng + 1e-6),
                    slope.mean(), slope.std(), np.percentile(slope, 90),
                    (slope < 0.02).mean(),
                    np.abs(lap).mean(), lap.std(), a[c, c] - a.mean(),
                    np.percentile(a, 90) - np.percentile(a, 10)])
    return np.asarray(out, dtype=np.float64)


def ridge_cv(X, y, groups, alpha=1.0):
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 50 or te.sum() < 5:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(X.shape[1]),
                            Xtr.T @ (y[tr] - ym))
        pred[te] = Xte @ w + ym
    m = np.isfinite(pred)
    return float(1 - ((y[m] - pred[m]) ** 2).sum() /
                 ((y[m] - y[m].mean()) ** 2).sum())


def main(a):
    df = pd.read_csv(a.table, low_memory=False)
    print(f"{len(df):,} measurements")

    # site medians: the DEM patch is per-site, so the target must be too
    g = df.groupby("site_no")
    site = g[["nearest_x", "nearest_y", "log_W", "log_d", "log_Q"]].median()
    site["HUC2"] = g["HUC2"].first()
    for f in BASE_FEATS:
        if f in df.columns:
            site[f] = g[f].median()
    site = site.reset_index()
    print(f"{len(site):,} unique sites")

    if a.export_coords:
        m = site[["nearest_x", "nearest_y"]].notna().all(1)
        s = site[m]
        np.savez(a.export_coords,
                 lon=s["nearest_x"].to_numpy(dtype=np.float64),
                 lat=s["nearest_y"].to_numpy(dtype=np.float64),
                 site_id=s["site_no"].astype(str).to_numpy())
        print(f"wrote {a.export_coords}: {m.sum():,} sites with coordinates")
        return

    dm = np.load(a.dem, allow_pickle=True)
    dsid = np.asarray(dm["site_id"]).astype(str)
    ok = dm["ok"]
    D_all = np.full((len(dsid), len(DEM_NAMES)), np.nan)
    D_all[ok] = dem_descriptors(dm["dem"][ok])
    dmap = {s: i for i, s in enumerate(dsid)}

    site["_di"] = site["site_no"].astype(str).map(dmap)
    site = site[site["_di"].notna()]
    D = D_all[site["_di"].to_numpy(dtype=int)]

    feats = [f for f in BASE_FEATS if f in site.columns]
    B = site[feats].to_numpy(dtype=np.float64)
    huc = site["HUC2"].astype(str).to_numpy()

    print(f"\nusable sites with DEM + attributes: "
          f"{np.isfinite(B).all(1).sum():,}")
    print(f"baseline attributes ({len(feats)}): {', '.join(feats)}")
    print(f"\n{'target':>8} | {'attrs':>8} {'attr+DEM':>9} {'DEM only':>9} "
          f"| {'DEM adds':>9}  n")
    print("-" * 66)
    for tgt in ("log_W", "log_d", "log_Q"):
        y = site[tgt].to_numpy(dtype=np.float64)
        m = np.isfinite(y) & np.isfinite(B).all(1) & np.isfinite(D).all(1)
        if m.sum() < 200:
            print(f"{tgt:>8} | too few rows ({m.sum()})")
            continue
        r_b = ridge_cv(B[m], y[m], huc[m])
        r_bd = ridge_cv(np.hstack([B[m], D[m]]), y[m], huc[m])
        r_d = ridge_cv(D[m], y[m], huc[m])
        print(f"{tgt:>8} | {r_b:8.4f} {r_bd:9.4f} {r_d:9.4f} | "
              f"{r_bd - r_b:+9.4f}  {m.sum():,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default="/nfs/data/cxs1024/channel_geometry/data/"
                            "train_table_dem_fixed.csv")
    ap.add_argument("--dem", default="logs/geom_dem.npz")
    ap.add_argument("--export-coords", default=None,
                    help="write site coordinates and exit (step 1 of 2)")
    main(ap.parse_args())
