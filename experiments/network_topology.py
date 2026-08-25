"""Is context gauge j INSIDE query basin i? Exact nesting from the polygons.

The motivating failure: neither the variogram attention bias nor drainage-area
scaling helped (0.8697 / 0.8633 against 0.8724). Both refine a EUCLIDEAN
metric, and Euclidean distance cannot express the thing that actually matters
hydrologically -- two gauges 50 km apart on the same river share water, two
50 km apart across a divide share only weather.

Nesting is the exact version of that relation, and it needs no flow-direction
grid: if gauge j falls inside basin i's polygon, then j is UPSTREAM of i and
its discharge is literally part of i's. That is a hard topological fact, not a
proxy.

This script answers the prior question -- how often does it even happen? If
nested pairs are rare among the neighbours retrieval actually selects, the
feature cannot matter regardless of how well it is encoded, and that is worth
knowing before any model change.

Needs geopandas/shapely: the demenv venv, not pytorch_gpu.
"""
from __future__ import annotations

import argparse

import numpy as np


def main(a):
    import geopandas as gpd
    from shapely.geometry import Point

    g = gpd.read_file(a.shp)
    print(f"shapefile: {len(g)} polygons, crs {g.crs}")
    print(f"columns: {list(g.columns)[:12]}")

    idcol = next((c for c in g.columns
                  if c.lower() in ("hru_id", "gage_id", "site_no", "gauge_id",
                                   "id", "station_id")), None)
    if idcol is None:
        print("!! could not identify the id column; inspect the list above")
        return
    print(f"using id column: {idcol}")

    c = np.load(a.coords, allow_pickle=True)
    lat, lon = np.asarray(c["lat"]), np.asarray(c["lon"])
    sid = np.asarray(c["site_id"]).astype(str)

    gid = g[idcol].astype(str).str.zfill(8).to_numpy()
    keep = np.isin(gid, sid)
    g, gid = g[keep].reset_index(drop=True), gid[keep]
    print(f"{len(g)} polygons matched to {len(sid)} coordinates")

    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)

    pos = {s: i for i, s in enumerate(sid)}
    pts = [Point(lon[pos[s]], lat[pos[s]]) for s in gid]
    pgdf = gpd.GeoDataFrame({"j": np.arange(len(gid))}, geometry=pts,
                            crs=4326)

    # spatial join: which gauge points fall inside which basin polygons
    poly = gpd.GeoDataFrame({"i": np.arange(len(g))}, geometry=g.geometry,
                            crs=4326)
    hit = gpd.sjoin(pgdf, poly, predicate="within", how="inner")
    pairs = [(int(r.i), int(r.j)) for r in hit.itertuples()
             if int(r.i) != int(r.j)]

    n = len(gid)
    print(f"\n=== NESTING ===")
    print(f"  gauge pairs total          : {n*(n-1):,}")
    print(f"  NESTED pairs (j inside i)  : {len(pairs):,} "
          f"({100*len(pairs)/max(1, n*(n-1)):.3f}%)")

    has_up = len({i for i, _ in pairs})
    print(f"  basins containing >=1 gauge: {has_up} / {n} "
          f"({100*has_up/n:.1f}%)")

    # The decisive number: among the NEAREST neighbours retrieval actually
    # picks, how many are nested? A relation that never appears in the
    # selected context cannot influence the model however it is encoded.
    from scipy.spatial import cKDTree
    ll = np.stack([lat[[pos[s] for s in gid]],
                   lon[[pos[s] for s in gid]]], -1)
    tree = cKDTree(ll)
    _, nn = tree.query(ll, k=min(9, len(ll)))
    nested = set(pairs)
    for K in (1, 2, 4, 8):
        cnt = sum(1 for i in range(n) for j in nn[i, 1:K+1]
                  if (i, int(j)) in nested or (int(j), i) in nested)
        print(f"  of K={K:<2d} nearest neighbours: {cnt:,} nested "
              f"({100*cnt/(n*K):.2f}% of selected context)")

    np.savez(a.out, pairs=np.asarray(pairs, dtype=np.int32), site_id=gid)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp",
                    default="/nfs/data/cxs1024/data/camels/loc/camels671.shp")
    ap.add_argument("--coords", default="logs/camels_coords.npz")
    ap.add_argument("--out", default="logs/camels_nesting.npz")
    main(ap.parse_args())
