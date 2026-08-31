"""Does kriging-style interpolation reach ~0.9 NSE -- and under what geometry?

Claim to test (CS): median per-basin NSE ~0.9 should be reachable by kriging
concurrent neighbour discharge. The literature supports it FOR DENSE NESTED
NETWORKS (top-kriging in Austrian-type settings, gauges km apart on shared
rivers). CAMELS-531 is curated for independence: 0.011% nested pairs, nearest
gauges tens of km across divides. If geometry is the real variable, IDW skill
on our own data should climb steeply as nearest-gauge distance shrinks -- and
the dense tail of CAMELS should approach the literature number while the
median sits far below it.

Model-free: IDW of the K nearest other gauges' concurrent specific discharge
(mm/day; per-basin NSE is invariant to the per-basin area rescaling, so m3/s
gives identical numbers). All other gauges are eligible donors, as an operator
would use them. Scored on the spatial protocol's period, raw scale.
"""
from __future__ import annotations

import argparse

import numpy as np
from scipy.spatial import cKDTree

from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels


def main(a):
    d = load_camels(a.nc)
    sub, gage = P.load_531(d)          # sub IS the restricted dict
    X = sub["x"]
    valid = sub["valid"]
    ll = sub["latlon"]
    obs = X.shape[-1] - 1

    win = P.windows(sub["time"], "spatial")
    sc = win["score"]
    q = X[:, sc, obs].astype(np.float64)
    q[valid[:, sc, obs] == 0] = np.nan
    n = len(q)
    print(f"{n} basins, scored days {sc.start}..{sc.stop} ({q.shape[1]} d)")

    # neighbour geometry (deg, lon scaled to CONUS mid-latitude)
    pts = np.stack([ll[:, 0], ll[:, 1] * 0.766], -1)
    tree = cKDTree(pts)
    dist, idx = tree.query(pts, k=a.k + 1)
    dist, idx = dist[:, 1:], idx[:, 1:]          # drop self

    def nse(y, p):
        m = np.isfinite(y) & np.isfinite(p)
        if m.sum() < 100:
            return np.nan
        den = ((y[m] - y[m].mean()) ** 2).sum()
        return 1 - ((y[m] - p[m]) ** 2).sum() / den if den > 0 else np.nan

    rows = []
    for i in range(n):
        w = 1.0 / (dist[i] ** 2 + 1e-3)
        nb = q[idx[i]]
        wm = np.where(np.isfinite(nb), w[:, None], 0.0)
        pred = np.nansum(nb * wm, 0) / np.clip(wm.sum(0), 1e-9, None)
        rows.append((dist[i, 0], nse(q[i], pred)))
    dd = np.array([r[0] for r in rows])
    ns = np.array([r[1] for r in rows])
    ok = np.isfinite(ns)
    print(f"\nIDW K={a.k}, all-gauge donors: median NSE "
          f"{np.nanmedian(ns[ok]):+.3f}  (n={ok.sum()})")

    print(f"\n{'nearest gauge':>16} {'n':>5} {'median NSE':>11}")
    print("-" * 36)
    edges = [0, 0.10, 0.20, 0.35, 0.60, 10.0]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = ok & (dd >= lo) & (dd < hi)
        km = f"{lo*111:.0f}-{hi*111:.0f} km" if hi < 10 else f">{lo*111:.0f} km"
        if m.sum() >= 5:
            print(f"{km:>16} {m.sum():>5} {np.nanmedian(ns[m]):>+11.3f}")
    print("\nIf the dense bins approach ~0.9 while the median does not, the")
    print("literature number and ours are the same phenomenon at different")
    print("gauge densities -- geometry, not method.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=P.CAMELS_NC)
    ap.add_argument("--k", type=int, default=8)
    main(ap.parse_args())
