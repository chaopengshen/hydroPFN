"""Matched outlet-square and contributing-basin terrain descriptors.

Runs in suntzu's demenv (rasterio/geopandas). Adjacent 3DEP tiles are mosaicked;
missing collars are never filled with mean elevation. Physical elevation and
relief are retained. This tests spatial support before another learned-embedding
experiment. Per-basin JSON caches permit restart without repeated acquisition.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_TIMEOUT", "60")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "2")

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.merge import merge
from shapely.geometry import box, mapping

TILE = ("/vsicurl/https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/"
        "TIFF/current/n{n:02d}w{w:03d}/USGS_13_n{n:02d}w{w:03d}.tif")
NAMES = ["elevation_mean_m", "elevation_std_m", "relief_p99_p01_m",
         "elevation_p10_m", "elevation_p25_m", "elevation_p50_m",
         "elevation_p75_m", "elevation_p90_m", "hypsometric_index",
         "slope_mean", "slope_std", "slope_p90", "flat_fraction",
         "elevation_skew", "elevation_iqr_m"]


def terrain_descriptors(z, mask, dx, dy):
    valid = mask & np.isfinite(z)
    v = z[valid].astype(float)
    if len(v) < 100:
        raise ValueError("fewer than 100 valid terrain cells")
    quantiles = np.percentile(v, [1, 10, 25, 50, 75, 90, 99])
    p01, p10, p25, p50, p75, p90, p99 = quantiles
    gy, gx = np.gradient(z.astype(float), dy, dx)
    slope = np.hypot(gx, gy)
    slopes = slope[valid & np.isfinite(slope)]
    if len(slopes) < 100:
        raise ValueError("insufficient cells for physical slope")
    sd = v.std()
    return [v.mean(), sd, p99 - p01, p10, p25, p50, p75, p90,
            (v.mean() - p01) / max(p99 - p01, 1e-6), slopes.mean(),
            slopes.std(), np.percentile(slopes, 90), np.mean(slopes < 0.02),
            np.mean(((v - v.mean()) / max(sd, 1e-6)) ** 3), p75 - p25]


def acquire(geometry, latitude, resolution_m, max_pixels, min_coverage):
    west, south, east, north = geometry.bounds
    metres_lon = 111320. * math.cos(math.radians(latitude))
    resx, resy = resolution_m / metres_lon, resolution_m / 111320.
    pixels = (east - west) * (north - south) / (resx * resy)
    factor = max(1., math.sqrt(pixels / max_pixels))
    resx, resy = resx * factor, resy * factor
    bounds = (west - resx, south - resy, east + resx, north + resy)
    sources, failures, urls = [], [], []
    with ExitStack() as stack:
        for lon in range(math.floor(bounds[0]), math.ceil(bounds[2])):
            for n in range(math.floor(bounds[1]) + 1, math.ceil(bounds[3]) + 1):
                url = TILE.format(n=n, w=abs(lon))
                try:
                    src = stack.enter_context(rasterio.open(url))
                    if src.crs.to_epsg() != 4269:
                        raise ValueError(f"unexpected source CRS {src.crs}")
                    sources.append(src)
                    urls.append(url)
                except Exception as exc:
                    failures.append({"url": url, "error": str(exc)})
        if not sources:
            raise ValueError(f"no DEM tiles available: {failures}")
        data, transform = merge(sources, bounds=bounds, res=(resx, resy),
                                nodata=np.nan, dtype="float32",
                                resampling=Resampling.bilinear)
    z = data[0]
    mask = geometry_mask([mapping(geometry)], out_shape=z.shape,
                         transform=transform, invert=True)
    valid = np.isfinite(z) & (z > -10000)
    z[~valid] = np.nan
    coverage = float((valid & mask).sum() / max(mask.sum(), 1))
    if coverage < min_coverage:
        raise ValueError(f"DEM coverage {coverage:.3f} below {min_coverage}; {failures}")
    dx, dy = float(transform.a * metres_lon), float(-transform.e * 111320.)
    features = terrain_descriptors(z, mask, dx, dy)
    return {"features": features, "coverage": coverage, "resolution_m": [dx, dy],
            "shape": list(z.shape), "tiles": urls, "tile_failures": failures,
            "bounds": list(bounds)}


def write_arrays(records, out):
    ids = np.asarray([r["gage"] for r in records])
    for name in ("basin", "outlet"):
        features = np.full((len(records), len(NAMES)), np.nan)
        coverage = np.full(len(records), np.nan)
        resolution = np.full((len(records), 2), np.nan)
        for i, r in enumerate(records):
            if name in r:
                features[i] = r[name]["features"]
                coverage[i] = r[name]["coverage"]
                resolution[i] = r[name]["resolution_m"]
        np.savez_compressed(out / f"{name}_terrain.npz", site_id=ids,
                            feats=features, ok=np.isfinite(features).all(1),
                            feature_names=np.asarray(NAMES), coverage=coverage,
                            resolution_m=resolution)


def main(a):
    out = Path(a.out)
    cache = out / "basins"
    cache.mkdir(parents=True, exist_ok=True)
    with open(a.subset) as f:
        wanted = {str(int(v)).zfill(8) for v in json.load(f)}
    basins = gpd.read_file(a.shp).to_crs(4269)
    basins["gage"] = basins.gage_id.map(lambda x: str(int(x)).zfill(8))
    basins = basins[basins.gage.isin(wanted)].sort_values("gage")
    if len(basins) != len(wanted):
        raise ValueError("polygon inventory does not cover the selected 531 basins")
    if a.limit:
        basins = basins.iloc[:a.limit]
    records = []
    for i, row in enumerate(basins.itertuples(), 1):
        path = cache / f"{row.gage}.json"
        config = {"resolution_m": a.resolution, "max_pixels": a.max_pixels,
                  "min_coverage": a.min_coverage, "outlet_width_km": a.outlet_width}
        if path.exists():
            record = json.loads(path.read_text())
            if record["config"] != config:
                raise ValueError(f"cached extraction settings differ: {path}")
            if "basin" in record and "outlet" in record:
                records.append(record)
                continue
        record = {"gage": row.gage, "config": config, "lat": row.lat, "lon": row.lon}
        t0 = time.time()
        half_lat = a.outlet_width * 1000 / (2 * 111320.)
        half_lon = half_lat / math.cos(math.radians(row.lat))
        shapes = {"basin": row.geometry, "outlet": box(row.lon - half_lon,
                  row.lat - half_lat, row.lon + half_lon, row.lat + half_lat)}
        for label, shape in shapes.items():
            try:
                record[label] = acquire(shape, row.lat, a.resolution,
                                        a.max_pixels, a.min_coverage)
            except Exception as exc:
                record[label + "_error"] = str(exc)
        record["elapsed_seconds"] = time.time() - t0
        path.write_text(json.dumps(record, indent=2))
        records.append(record)
        write_arrays(records, out)
        print(f"{i}/{len(basins)} {row.gage}: basin={'basin' in record} "
              f"outlet={'outlet' in record} {time.time() - t0:.1f}s", flush=True)
    write_arrays(records, out)
    print(f"Wrote {out}; descriptors retain physical units and verified coverage.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shp", default="/nfs/data/cxs1024/data/camels/loc/camels671.shp")
    p.add_argument("--subset", default="/nfs/data/cxs1024/hydroPFN/data/1-camels/531sub_id.txt")
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=float, default=30.)
    p.add_argument("--max-pixels", type=int, default=5000000)
    p.add_argument("--min-coverage", type=float, default=0.98)
    p.add_argument("--outlet-width", type=float, default=12.8)
    p.add_argument("--limit", type=int, default=0)
    main(p.parse_args())
