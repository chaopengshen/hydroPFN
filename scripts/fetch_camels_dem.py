"""Fetch one 3DEP DEM patch per CAMELS gauge, for the DEM <-> time-series link.

The DEM arm and the time-series arm have been two disconnected tracks sharing
only the RePaint idea. Connecting them needs DEM at the SAME sites the
time-series model runs on, which is what this produces: one square patch
centred on each CAMELS gauge, keyed by station id so it joins to everything
else by site, never by index.

Source is USGS 3DEP 1/3 arc-second (~10 m), streamed from the public S3 bucket
with /vsicurl/ -- no auth, no local mirror. Patches are grouped by 1-degree
tile so each tile is opened once; opening per-patch would be ~600 separate
range-read sessions.

NOTE ON SCOPE: 3DEP is CONUS-only. Globally the equivalent is 30 m (Copernicus
GLO-30 / SRTM), so any DEM pathway trained on 10 m here must be shown to
survive at 30 m before it means anything for the global model. Use --downsample
to test that directly.

Usage (needs rasterio -- the demenv venv, not pytorch_gpu):
    python scripts/fetch_camels_dem.py --nc data/CAMELS_Frederik.nc \
        --size 128 --out logs/camels_dem.npz
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict

import numpy as np

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "512")

import rasterio                                              # noqa: E402
from rasterio.windows import Window                          # noqa: E402

TILE = ("/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
        "TIFF/current/n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif")


def tile_name(lon: float, lat: float) -> tuple[int, int]:
    """3DEP tiles are named by their NORTH-WEST corner."""
    return int(np.ceil(lat)), int(np.ceil(abs(lon)))


def main(a):
    # Coordinates come from a small npz, not the NetCDF directly: the rasterio
    # env (demenv) has no xarray and the torch env has no rasterio, so the two
    # steps cannot share an interpreter. scripts/export_coords.py makes it.
    c = np.load(a.coords, allow_pickle=True)
    lat = np.asarray(c["lat"], dtype=np.float64)
    lon = np.asarray(c["lon"], dtype=np.float64)
    sid = np.asarray(c["site_id"])
    print(f"{len(sid)} CAMELS gauges")

    by_tile = defaultdict(list)
    for i, (la, lo) in enumerate(zip(lat, lon)):
        by_tile[tile_name(lo, la)].append(i)
    print(f"{len(by_tile)} distinct 1-degree tiles to open")

    out = np.full((len(sid), a.size, a.size), np.nan, dtype=np.float32)
    ok = np.zeros(len(sid), dtype=bool)

    for t, (key, idxs) in enumerate(sorted(by_tile.items()), 1):
        n, w = key
        url = TILE.format(n=n, w=w)
        try:
            with rasterio.open(url) as src:
                H, W = src.height, src.width
                for i in idxs:
                    r, c = src.index(lon[i], lat[i])
                    r0, c0 = int(r) - a.size // 2, int(c) - a.size // 2
                    if r0 < 0 or c0 < 0 or r0 + a.size > H or c0 + a.size > W:
                        continue          # patch would straddle a tile edge
                    z = src.read(1, window=Window(c0, r0, a.size, a.size))
                    z = z.astype(np.float32)
                    # 3DEP nodata is a large negative sentinel
                    if not np.isfinite(z).all() or z.min() < -1e4:
                        continue
                    out[i], ok[i] = z, True
        except Exception as e:                                # noqa: BLE001
            print(f"  [{t}/{len(by_tile)}] n{n:02d}w{w:03d} FAILED: "
                  f"{type(e).__name__}", flush=True)
            continue
        if t % 20 == 0 or t == len(by_tile):
            print(f"  [{t}/{len(by_tile)}] tiles done, {ok.sum()} patches",
                  flush=True)

    print(f"\n{ok.sum()} / {len(sid)} gauges have a patch "
          f"({100*ok.mean():.1f}%)")
    if ok.sum():
        z = out[ok]
        print(f"  elevation range {np.nanmin(z):.0f} .. {np.nanmax(z):.0f} m")
        rel = z - z.mean(axis=(1, 2), keepdims=True)
        print(f"  within-patch relief: median {np.median(rel.max((1,2)) - rel.min((1,2))):.0f} m")
    np.savez_compressed(a.out, dem=out, ok=ok, site_id=sid,
                        lat=lat, lon=lon, size=a.size)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--coords", required=True,
                    help="npz with lat/lon/site_id (see "
                         "scripts/export_coords.py)")
    ap.add_argument("--size", type=int, default=128,
                    help="patch side in pixels (128 @10 m = 1.28 km)")
    ap.add_argument("--out", default="logs/camels_dem.npz")
    main(ap.parse_args())
