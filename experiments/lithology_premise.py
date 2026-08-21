"""Premise test: can hand-engineered terrain descriptors predict LITHOLOGY?

Reproduces: logs/lithology_baseline.csv

This is the project's go/no-go, and it is the one experiment the
channel-geometry results could never settle.

Adding DEM information to the channel-geometry model failed three times --
StefaLand embeddings (-0.004), upstream network structure (+-0.001), a
cross-section CNN (slight width gain only) -- each time because the information
was genuinely present but genuinely redundant.  That is expected there:
hydraulic geometry says W ~ Q^b and d ~ Q^f, so discharge, drainage area and
slope are the *theoretically correct* predictors and we measure all three.  A DEM
image is an indirect route to quantities we already hold directly.

Subsurface has no such table.  There is no "lithology attribute" for terrain to
be redundant with -- which is the whole premise, and why redundancy on one task
implies nothing about the other.

So: hand-engineered terrain descriptors -> SGMC lithology class and age, under
leave-region-out.  If this fails, terrain does not carry subsurface signal and
no amount of pretraining rescues it.  If it succeeds, that number is the bar any
learned representation has to beat.

**Circularity caveat, carried into the output.**  Geologists partly map contacts
*using* topographic expression, so terrain->lithology is not fully independent.
Accuracy concentrated in units whose boundaries follow topography (alluvium in
valleys, for instance) is weaker evidence than accuracy on units that do not.
The per-class breakdown is printed for exactly that reason.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hydropfn.metrics.terrain import slope_mag  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
GDB = ("/nfs/data/cxs1024/sgmc/USGS_SGMC_Geodatabase/"
       "USGS_StateGeologicMapCompilation_ver1.1.gdb")
TABLES = "/nfs/data/cxs1024/sgmc/tables/USGS_SGMC_Tables_CSV"


def terrain_features(z: np.ndarray, px: float = 10.0) -> dict:
    """Standard descriptors -- the bar a learned representation must clear.

    Deliberately the classical set, computed at several radii because geomorphic
    expression of lithology lives at different scales: fine dissection for weak
    rock, broad benches for resistant layers.
    """
    out: dict[str, float] = {}
    z = z - z.mean()
    s = slope_mag(z, px)
    out["slope_mean"] = float(s.mean())
    out["slope_sd"] = float(s.std())
    out["slope_p90"] = float(np.percentile(s, 90))
    out["relief"] = float(z.max() - z.min())
    out["elev_sd"] = float(z.std())
    out["elev_skew"] = float(((z - z.mean()) ** 3).mean() /
                             (z.std() ** 3 + 1e-9))
    # hypsometric integral: shape of the elevation distribution, a classical
    # discriminator of erosional stage
    out["hypso"] = float((z.mean() - z.min()) / (z.max() - z.min() + 1e-9))

    gy, gx = np.gradient(z, px)
    gyy, _ = np.gradient(gy, px)
    _, gxx = np.gradient(gx, px)
    curv = gxx + gyy
    out["curv_sd"] = float(curv.std())
    out["curv_p90"] = float(np.percentile(np.abs(curv), 90))

    from scipy.ndimage import uniform_filter
    for r in (2, 4, 8, 16):                     # TPI / roughness at nested radii
        sm = uniform_filter(z, size=2 * r + 1)
        out[f"tpi_sd_{r*10}m"] = float((z - sm).std())
        out[f"rough_{r*10}m"] = float(np.abs(z - sm).mean())
    return out


def main(patches: str, min_class: int, seed: int) -> None:
    from pyogrio import read_dataframe
    import shapely

    z = np.load(patches)
    P, tiles = z["patches"], z["tile"]
    lon, lat = z["lon"], z["lat"]
    print(f"{len(P):,} patches from {len(set(tiles)):,} tiles")

    # --- labels: point-in-polygon against SGMC geology
    pts = shapely.points(lon, lat)
    gdf = read_dataframe(GDB, layer="SGMC_Geology",
                         columns=["UNIT_LINK"], force_2d=True)
    gdf = gdf.to_crs("EPSG:4326") if gdf.crs and gdf.crs.to_epsg() != 4326 else gdf
    import geopandas as gpd
    q = gpd.GeoDataFrame(geometry=gpd.GeoSeries(pts, crs="EPSG:4326"))
    j = gpd.sjoin(q, gdf, how="left", predicate="within")
    j = j[~j.index.duplicated(keep="first")]
    unit = j.UNIT_LINK.reindex(range(len(P))).to_numpy()
    print(f"  patches with a geology unit: {pd.notna(unit).sum():,}")

    lith = pd.read_csv(f"{TABLES}/SGMC_Lithology.csv", low_memory=False)
    lith = lith[lith.LITH_RANK == "Major"].drop_duplicates("UNIT_LINK")
    lmap = dict(zip(lith.UNIT_LINK, lith.LITH1))
    age = pd.read_csv(f"{TABLES}/SGMC_Age.csv", low_memory=False)
    age = age.drop_duplicates("UNIT_LINK")
    amap = dict(zip(age.UNIT_LINK, (age.MIN_MA + age.MAX_MA) / 2.0))

    y_lith = np.array([lmap.get(u, None) for u in unit], dtype=object)
    y_age = np.array([amap.get(u, np.nan) for u in unit], dtype=float)
    ok = pd.notna(y_lith)
    print(f"  with a LITH1 class: {ok.sum():,}   with an age: "
          f"{np.isfinite(y_age).sum():,}")
    vc = pd.Series(y_lith[ok]).value_counts()
    print("\n  lithology classes (LITH1, major):")
    for k, v in vc.head(10).items():
        print(f"    {v:5d}  {k}")

    keep = ok & pd.Series(y_lith).isin(vc[vc >= min_class].index).to_numpy()
    print(f"\n  usable after dropping classes with <{min_class}: {keep.sum():,}")

    # --- features
    X = pd.DataFrame([terrain_features(P[i]) for i in np.flatnonzero(keep)])
    cols = list(X.columns)
    yl = y_lith[keep]
    ya = y_age[keep]
    tl = tiles[keep]
    print(f"  {len(cols)} terrain descriptors")

    # --- leave-region-out by 1-degree tile
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    gkf = GroupKFold(n_splits=5)
    Xv = np.nan_to_num(X.to_numpy(float), nan=0.0)
    pred = np.empty(len(yl), dtype=object)
    for tr, te in gkf.split(Xv, yl, groups=tl):
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                     n_jobs=-1, random_state=seed,
                                     class_weight="balanced_subsample")
        clf.fit(Xv[tr], yl[tr])
        pred[te] = clf.predict(Xv[te])

    base = pd.Series(yl).value_counts(normalize=True).max()
    acc = accuracy_score(yl, pred)
    bacc = balanced_accuracy_score(yl, pred)
    print(f"\n=== LITHOLOGY, leave-tile-out ===")
    print(f"  accuracy          {acc:.3f}")
    print(f"  balanced accuracy {bacc:.3f}")
    print(f"  majority baseline {base:.3f}   ({1/len(set(yl)):.3f} = uniform)")
    print(f"  -> {'ABOVE' if acc > base + 0.02 else 'NOT ABOVE'} the majority baseline")

    print("\n  per-class recall (circularity check: units whose contacts follow")
    print("  topography, e.g. alluvium in valleys, are the weak evidence):")
    df = pd.DataFrame({"true": yl, "pred": pred})
    for k in vc.index[:8]:
        sub = df[df.true == k]
        if len(sub) >= 10:
            print(f"    {len(sub):5d}  recall {(sub.pred == k).mean():.3f}  {k}")

    m = np.isfinite(ya)
    if m.sum() > 200:
        pa = np.full(m.sum(), np.nan)
        Xa, yaa, ta = Xv[m], ya[m], tl[m]
        for tr, te in GroupKFold(n_splits=5).split(Xa, yaa, groups=ta):
            rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                       n_jobs=-1, random_state=seed)
            rf.fit(Xa[tr], np.log10(yaa[tr] + 1))
            pa[te] = rf.predict(Xa[te])
        r2 = 1 - ((np.log10(yaa + 1) - pa) ** 2).sum() / \
            ((np.log10(yaa + 1) - np.log10(yaa + 1).mean()) ** 2).sum()
        print(f"\n=== AGE (log10 Ma), leave-tile-out ===  R2 = {r2:.3f}  "
              f"(n={m.sum():,})")

    out = pd.DataFrame({"tile": tl, "lith_true": yl, "lith_pred": pred,
                        "age_ma": ya})
    (ROOT / "logs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "logs" / "lithology_baseline.csv", index=False)
    print(f"\nwrote {ROOT/'logs'/'lithology_baseline.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(ROOT / "logs" / "dem_patches.npz"))
    ap.add_argument("--min-class", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.patches, a.min_class, a.seed)
