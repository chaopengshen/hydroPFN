"""Sample square DEM patches from 3DEP for the inpainting probe.

Windowed reads, not point sampling.  The rest of this project samples the DEM at
scattered points along flowlines, which suits profiles but would cost ~16k
`sample()` calls per 128x128 patch.  `rasterio` can read a grid-aligned window in
one go, so a patch costs roughly one read.

Patch centres are drawn near known gages with a random offset of up to
`--jitter-km`.  That is a pragmatic way to land on real CONUS terrain without
needing a land mask, while decorrelating the patch from the river itself.

Split unit is the **1-degree 3DEP tile**, not the patch.  Neighbouring patches
are near-duplicates, so a random patch split measures memorisation -- the same
mistake that made Al Mehedi's Manning's-n score collapse from 0.631 to 0.058
once whole gages were held out.
"""

from __future__ import annotations

import argparse
import math
import os

from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("VSI_CACHE", "TRUE")
os.environ.setdefault("GDAL_CACHEMAX", "256")

import rasterio  # noqa: E402
from rasterio.windows import Window  # noqa: E402

try:                                             # noqa: E402
    from paths import RESULTS                     # channel_geometry helper
except ModuleNotFoundError:
    # Standalone use (e.g. the uniform-CONUS pull inside dem_foundation):
    # fall back to this project's logs/ directory.
    RESULTS = Path(__file__).resolve().parents[2] / "logs"
    RESULTS.mkdir(parents=True, exist_ok=True)

TILE = ("/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
        "TIFF/current/{t}/USGS_13_{t}.tif")


def tile_name(lon: float, lat: float) -> str:
    return f"n{int(math.ceil(lat)):02d}w{int(math.ceil(-lon)):03d}"


def sample_tile(tile: str, centres: list[tuple[float, float]], size: int,
                min_relief: float = 0.0, max_per_tile: int = 0):
    """Read one `size`x`size` window per centre from a single tile."""
    out = []
    try:
        src = rasterio.open(TILE.format(t=tile))
    except Exception:                                # noqa: BLE001
        return out
    with src:
        nod = src.nodata
        H, W = src.height, src.width
        for lon, lat in centres:
            try:
                r, c = src.index(lon, lat)
            except Exception:                        # noqa: BLE001
                continue
            r0, c0 = int(r) - size // 2, int(c) - size // 2
            if r0 < 0 or c0 < 0 or r0 + size > H or c0 + size > W:
                continue                             # skip tile-edge patches
            a = src.read(1, window=Window(c0, r0, size, size)).astype(np.float32)
            if nod is not None:
                a[a == nod] = np.nan
            a[a < -1e30] = np.nan
            if not np.isfinite(a).all():
                continue                             # no partial patches
            if float(a.max() - a.min()) < min_relief:
                continue    # water bodies and dead-flat ground carry no
                            # terrain signal; 13% of a uniform CONUS pull
            out.append((tile, lon, lat, a))
            if max_per_tile and len(out) >= max_per_tile:
                break        # spread the budget over MORE tiles
    return out


# CONUS bounding box for uniform sampling.  3DEP tiles that do not exist
# (ocean, Canada, Mexico) simply fail to open and are skipped.
LON_RANGE = (-124.7, -67.0)
LAT_RANGE = (25.1, 49.4)


def main(n_patches: int, size: int, jitter_km: float, seed: int, out: str,
         uniform: bool = False, min_relief: float = 0.0,
         max_per_tile: int = 0) -> None:
    rng = np.random.default_rng(seed)

    if uniform:
        # Uniform over CONUS instead of gage-centred.
        #
        # Measured reason: gage-centred +-15 km patches are valley-biased, and
        # the sampler's WORST regime is flat, finely textured terrain
        # (farmland, roads, urban) -- psd_ratio 0.35 in the top texture/relief
        # tercile vs 0.79 in the bottom, Spearman -0.417.  That class is
        # exactly what a gage-centred sample under-represents.  Uniform
        # sampling is also what the lithology probe needs, so this pull serves
        # two open items.
        # Oversample HEAVILY.  Measured yield of the 3x version was ~1.2
        # patches per tile (ocean, nodata, and tile-edge rejections), which
        # would have taken all 1,450 tiles to reach ~1,700 patches.  Opening a
        # tile is a network read; extra windows from an already-open tile are
        # nearly free -- so more candidates PER TILE is the cheap axis, and
        # the run stops early once the target is met.
        over = int(n_patches * 15.0)
        lat = rng.uniform(*LAT_RANGE, over)
        lon = rng.uniform(*LON_RANGE, over)
        idx = np.arange(over)
    else:
        gl = pd.read_csv(RESULTS / "gage_list_for_embeddings.csv")
        # Draw centres, then GROUP BY TILE so each tile is opened once.
        idx = rng.integers(0, len(gl), size=int(n_patches * 1.6))
        lat0 = gl.lat.to_numpy()[idx]
        lon0 = gl.lon.to_numpy()[idx]
        dy = rng.uniform(-jitter_km, jitter_km, len(idx)) / 110.54
        dx = (rng.uniform(-jitter_km, jitter_km, len(idx)) /
              (111.32 * np.cos(np.radians(lat0))))
        lat, lon = lat0 + dy, lon0 + dx

    by_tile: dict[str, list[tuple[float, float]]] = {}
    for a, b in zip(lon, lat):
        by_tile.setdefault(tile_name(a, b), []).append((float(a), float(b)))
    print(f"{len(idx):,} candidate centres across {len(by_tile):,} tiles "
          f"(target {n_patches:,} patches of {size}x{size} @10 m "
          f"= {size*10/1000:.2f} km)")

    # SHUFFLE the tile order.  `sorted()` walks tile names alphabetically,
    # which for 3DEP means SOUTH to NORTH -- and because the loop stops as
    # soon as n_patches is reached, the result is a narrow southern slice, not
    # a CONUS sample.  Measured: a "uniform" 8,000-patch pull came from just
    # 106 tiles.  This is the same latitude-ordering bug as review finding B2
    # (the evaluation subset); it was fixed there and left here.
    items = list(by_tile.items())
    rng.shuffle(items)

    tiles, lons, lats, arrs = [], [], [], []
    for i, (t, cs) in enumerate(items, 1):
        for tt, lo, la, a in sample_tile(t, cs, size, min_relief,
                                         max_per_tile):
            tiles.append(tt); lons.append(lo); lats.append(la); arrs.append(a)
        if i % 25 == 0:
            print(f"  {i}/{len(by_tile)} tiles  {len(arrs):,} patches", flush=True)
        if len(arrs) >= n_patches:
            break

    arr = np.stack(arrs[:n_patches])
    tiles = np.asarray(tiles[:n_patches])
    dst = RESULTS / out
    np.savez_compressed(dst, patches=arr, tile=tiles,
                        lon=np.asarray(lons[:n_patches], dtype=np.float32),
                        lat=np.asarray(lats[:n_patches], dtype=np.float32))
    relief = arr.max((1, 2)) - arr.min((1, 2))
    print(f"\nwrote {dst}  ({dst.stat().st_size/1e6:.0f} MB)")
    print(f"  {len(arr):,} patches from {len(set(tiles)):,} tiles "
          f"(of {len(by_tile):,} candidate tiles, shuffled)")
    print(f"  relief per patch: median {np.median(relief):.1f} m, "
          f"p10 {np.percentile(relief,10):.1f}, p90 {np.percentile(relief,90):.1f}")
    print(f"  -> p10 relief this low means many patches are near-flat; the probe "
          f"must stratify by roughness or flat terrain will dominate the average")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-patches", type=int, default=6000)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--jitter-km", type=float, default=15.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dem_patches.npz")
    ap.add_argument("--uniform", action="store_true",
                    help="sample uniformly over CONUS instead of near gages")
    ap.add_argument("--max-per-tile", type=int, default=0,
                    help="cap patches taken from any one tile (0 = no cap). "
                         "Without it, heavy oversampling yields ~75 patches "
                         "from each successful tile, so 8,000 patches came "
                         "from only 107 tiles -- geographically concentrated "
                         "even after shuffling the tile order.")
    ap.add_argument("--min-relief", type=float, default=0.0,
                    help="reject patches flatter than this (metres); 1.0 "
                         "removes water bodies and dead-flat ground")
    a = ap.parse_args()
    main(a.n_patches, a.size, a.jitter_km, a.seed, a.out, a.uniform,
         a.min_relief, a.max_per_tile)
