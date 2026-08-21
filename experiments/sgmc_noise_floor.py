"""How much of SGMC's label disagreement is real geology, and how much is the map?

Reproduces: logs/sgmc_noise_floor.csv

The lithology baseline reached accuracy 0.593 against a 0.516 majority, which is
uninterpretable without knowing what accuracy is even attainable.  SGMC is a
compilation of 48 SEPARATE state maps at scales from 1:50,000 to 1:1,000,000, so
label noise could be large enough to cap any model well below 1.0.

Two comparisons, and the second is the point:

  SAME state, distance d      two points, one survey.  Disagreement here is real
                              geological variation over distance d, plus that
                              survey's own positional error.
  DIFFERENT state, distance d two points, TWO INDEPENDENT SURVEYS mapped the
                              same rock.  Disagreement adds cross-survey
                              inconsistency on top.

The same-state curve is the practical ceiling for a model whose effective
positional resolution is d.  The gap between the curves isolates how much
disagreement is an artefact of compiling separate maps.

Without this, "0.593" could mean the premise is dead or nearly saturated, and we
have no way to tell which.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from hydropfn.paths import ROOT  # noqa: E402
GDB = ("/nfs/data/cxs1024/sgmc/USGS_SGMC_Geodatabase/"
       "USGS_StateGeologicMapCompilation_ver1.1.gdb")
TABLES = "/nfs/data/cxs1024/sgmc/tables/USGS_SGMC_Tables_CSV"

# CONUS bounding box; points are rejected if they miss the polygons.
LON = (-124.7, -67.0)
LAT = (25.1, 49.4)


def main(n_pairs: int, dists_km: tuple[float, ...], seed: int) -> None:
    import geopandas as gpd
    import shapely
    from pyogrio import read_dataframe

    rng = np.random.default_rng(seed)
    print(f"loading SGMC geology polygons ...", flush=True)
    gdf = read_dataframe(GDB, layer="SGMC_Geology", columns=["UNIT_LINK"],
                         force_2d=True)
    if gdf.crs is None or (gdf.crs.to_epsg() or 0) != 4326:
        gdf = gdf.to_crs("EPSG:4326")
    print(f"  {len(gdf):,} polygons")

    lith = pd.read_csv(f"{TABLES}/SGMC_Lithology.csv", low_memory=False)
    lith = lith[lith.LITH_RANK == "Major"].drop_duplicates("UNIT_LINK")
    lmap = dict(zip(lith.UNIT_LINK, lith.LITH1))

    rows = []
    for dkm in dists_km:
        # A random point, and a second at distance dkm in a random direction.
        lon0 = rng.uniform(*LON, n_pairs)
        lat0 = rng.uniform(*LAT, n_pairs)
        ang = rng.uniform(0, 2 * np.pi, n_pairs)
        dlat = (dkm / 110.54) * np.sin(ang)
        dlon = (dkm / (111.32 * np.cos(np.radians(lat0)))) * np.cos(ang)
        lon1, lat1 = lon0 + dlon, lat0 + dlat

        q = gpd.GeoDataFrame(geometry=gpd.GeoSeries(
            np.concatenate([shapely.points(lon0, lat0),
                            shapely.points(lon1, lat1)]), crs="EPSG:4326"))
        j = gpd.sjoin(q, gdf, how="left", predicate="within")
        j = j[~j.index.duplicated(keep="first")].sort_index()
        u = j.UNIT_LINK.to_numpy()
        ua, ub = u[:n_pairs], u[n_pairs:]

        ok = pd.notna(ua) & pd.notna(ub)
        # UNIT_LINK is prefixed with the state code, e.g. "ALat;7" -> AL.
        sa = np.array([s[:2] if isinstance(s, str) else "" for s in ua])
        sb = np.array([s[:2] if isinstance(s, str) else "" for s in ub])
        la = np.array([lmap.get(s) if isinstance(s, str) else None for s in ua],
                      dtype=object)
        lb = np.array([lmap.get(s) if isinstance(s, str) else None for s in ub],
                      dtype=object)
        have = ok & pd.notna(la) & pd.notna(lb)

        same = have & (sa == sb)
        diff = have & (sa != sb)
        agree = la == lb
        rows.append({
            "dist_km": dkm,
            "n_same": int(same.sum()), "n_diff": int(diff.sum()),
            "agree_same_state": float(agree[same].mean()) if same.sum() else np.nan,
            "agree_diff_state": float(agree[diff].mean()) if diff.sum() else np.nan,
        })
        print(f"  d={dkm:5.2f} km   same-state {agree[same].mean():.3f} "
              f"(n={same.sum():,})   diff-state "
              f"{agree[diff].mean() if diff.sum() else float('nan'):.3f} "
              f"(n={diff.sum():,})", flush=True)

    out = pd.DataFrame(rows)
    (ROOT / "logs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "logs" / "sgmc_noise_floor.csv", index=False)
    print(f"\nwrote {ROOT/'logs'/'sgmc_noise_floor.csv'}\n")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nHow to read this:")
    print("  same-state agreement at the scale of our patch (1.28 km) is the")
    print("  practical CEILING -- a model cannot beat the map's own consistency.")
    print("  Our lithology baseline scored 0.593 (majority 0.516); compare it to")
    print("  that ceiling rather than to 1.0.")
    print("  A large same-vs-different gap means compiling 48 separate surveys")
    print("  injects disagreement beyond real geological variation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=40000)
    ap.add_argument("--dists-km", type=float, nargs="+",
                    default=[0.1, 0.5, 1.28, 5.0])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.n_pairs, tuple(a.dists_km), a.seed)
