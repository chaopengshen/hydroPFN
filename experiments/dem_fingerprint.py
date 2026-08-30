"""The fingerprint check: how strongly do DEM features encode WHERE you are?

This is the interpretive control for every downstream feature comparison.
Terrain is spatially autocorrelated, so terrain features partly encode
location; under leave-region-out a location fingerprint is precisely the
wrong thing to carry (the substitution test's -0.075), while for geology --
whose label is itself a function of location -- fingerprinting is not
penalised and can masquerade as signal.

So before asking "do parity features help more downstream", measure: can a
linear probe recover the HUC2 region from each feature set? If the better
generative model's features are MORE region-identifying, its downstream
gains must be discounted accordingly.

Random 5-fold CV (not leave-region-out -- the question is whether region is
recoverable at all), multinomial ridge classifier, accuracy vs majority.
"""
from __future__ import annotations

import argparse

import numpy as np


def probe(X, y, folds=5, alpha=10.0, seed=0):
    """One-vs-rest ridge classifier accuracy under random k-fold."""
    rng = np.random.default_rng(seed)
    classes = np.unique(y)
    Y = (y[:, None] == classes[None, :]).astype(np.float64)
    idx = rng.permutation(len(y))
    correct = 0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        W = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(X.shape[1]),
                            Xtr.T @ Y[tr])
        correct += (classes[np.argmax(Xte @ W, 1)] == y[te]).sum()
    return correct / len(y)


def main(a):
    import xarray as xr
    d = xr.open_dataset(a.nc)
    sid = np.array([str(s).zfill(8) for s in d.station_ids.values])
    region = np.array([s[:2] for s in sid])
    maj = max(np.mean(region == r) for r in set(region))
    print(f"{len(sid)} sites, {len(set(region))} HUC2 regions, "
          f"majority baseline {maj:.3f}\n")
    print(f"{'feature set':>28} {'dims':>5} {'n':>5} | {'HUC2 acc':>9}")
    print("-" * 56)

    # hand descriptors from the fine DEM
    dm = np.load(a.dem_fine, allow_pickle=True)
    ok = dm["ok"]
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dem_value_test import dem_descriptors
    H = dem_descriptors(dm["dem"][ok])
    acc = probe(np.nan_to_num(H), region[ok])
    print(f"{'12 hand descriptors (fine)':>28} {H.shape[1]:>5} {ok.sum():>5} "
          f"| {acc:9.3f}")

    for label, path in [t.split("=", 1) for t in a.sets]:
        z = np.load(path, allow_pickle=True)
        fok = z["ok"]
        fsid = np.asarray(z["site_id"]).astype(str)
        pos = {s: i for i, s in enumerate(sid)}
        rows = [(i, pos[s]) for i, s in enumerate(fsid)
                if fok[i] and s in pos]
        F = np.nan_to_num(z["feats"][[r[0] for r in rows]].astype(np.float64))
        rg = region[[r[1] for r in rows]]
        acc = probe(F, rg)
        print(f"{label:>28} {F.shape[1]:>5} {len(rg):>5} | {acc:9.3f}")

    print("\nHigher = more of the feature vector is a location fingerprint.")
    print("A downstream gain from a MORE fingerprinting feature set must be")
    print("discounted; a gain from a LESS fingerprinting one is real signal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default="data/CAMELS_Frederik.nc")
    ap.add_argument("--dem-fine", default="logs/camels_dem.npz")
    ap.add_argument("--sets", nargs="+", required=True,
                    help="label=path/to/feats.npz ...")
    main(ap.parse_args())
