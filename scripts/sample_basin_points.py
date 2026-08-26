"""Sample interior points inside each basin, for the within-basin patch bag.

Everything DEM so far has been a single view centred on the GAUGE. That sees
the outlet and nothing else. A basin spans several lithologies, a range of
slopes, and usually more than one landform -- so an outlet-centred patch
answers "what is the terrain at the outlet", not "what terrain does this basin
drain".

This draws N points inside each basin polygon so a bag of small patches can be
fetched at them. Two things are recorded per point and both matter:

  * RELATIVE POSITION within the basin, normalised by basin extent. Pooling a
    bag of patches without position discards shape -- elongation, whether the
    steep ground is upstream or downstream, how the basin is oriented. With
    relative position the set keeps layout, exactly as context sites keep
    theirs through the geo-encoding.
  * the basin AREA, since a fixed patch size means very different coverage
    for a 20 km2 basin than a 3,000 km2 one, and the model should know which.

Points are drawn by rejection sampling inside the polygon rather than on a
grid, so that irregular basins are covered without a spurious lattice
structure appearing in the bag.

Needs geopandas/shapely: the demenv venv.
"""
from __future__ import annotations

import argparse

import numpy as np


def main(a):
    import geopandas as gpd

    g = gpd.read_file(a.shp)
    idcol = next((c for c in g.columns
                  if c.lower() in ("hru_id", "gage_id", "site_no", "id")), None)
    if idcol is None:
        raise SystemExit(f"no id column in {list(g.columns)}")
    if g.crs is not None and g.crs.to_epsg() != 4326:
        g = g.to_crs(4326)
    g["_sid"] = g[idcol].astype(str).str.zfill(8)
    print(f"{len(g)} basins from {a.shp}")

    # area in km2 from an equal-area projection, NOT from degrees
    area_km2 = g.to_crs(6933).area.to_numpy() / 1e6

    rng = np.random.default_rng(a.seed)
    lons, lats, sids, rel, areas, slot = [], [], [], [], [], []
    for i, (geom, sid) in enumerate(zip(g.geometry, g["_sid"])):
        if geom is None or geom.is_empty:
            continue
        x0, y0, x1, y1 = geom.bounds
        span = max(x1 - x0, y1 - y0) + 1e-9
        got, tries = 0, 0
        while got < a.n_points and tries < a.n_points * 200:
            tries += 1
            px = rng.uniform(x0, x1)
            py = rng.uniform(y0, y1)
            from shapely.geometry import Point
            if not geom.contains(Point(px, py)):
                continue
            lons.append(px); lats.append(py); sids.append(sid)
            # relative position, centred and scaled by the basin's own extent
            rel.append([(px - (x0 + x1) / 2) / span,
                        (py - (y0 + y1) / 2) / span])
            areas.append(area_km2[i])
            slot.append(got)
            got += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(g)} basins, {len(lons):,} points", flush=True)

    lons = np.asarray(lons); lats = np.asarray(lats)
    sids = np.asarray(sids); rel = np.asarray(rel, dtype=np.float32)
    areas = np.asarray(areas, dtype=np.float32)
    print(f"\n{len(lons):,} interior points across "
          f"{len(set(sids)):,} basins "
          f"({len(lons)/max(1,len(set(sids))):.1f} per basin)")
    print(f"  basin area km2: median {np.median(areas):.0f}  "
          f"5-95% {np.percentile(areas,5):.0f}-{np.percentile(areas,95):.0f}")

    # site_id must be UNIQUE per point for the fetcher, but carry the basin
    np.savez(a.out, lon=lons, lat=lats,
             site_id=np.array([f"{s}_{k}" for s, k in zip(sids, slot)]),
             basin_id=sids, rel_pos=rel, area_km2=areas)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--shp",
                    default="/nfs/data/cxs1024/data/camels/loc/camels671.shp")
    ap.add_argument("--n-points", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/basin_points.npz")
    main(ap.parse_args())
