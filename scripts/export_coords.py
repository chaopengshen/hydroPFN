"""Export CAMELS gauge coordinates so the rasterio env can read them.

Trivial, but necessary: demenv (rasterio) has no xarray and pytorch_gpu
(xarray) has no rasterio, so the DEM fetch cannot open the NetCDF itself.
"""
from __future__ import annotations

import argparse

import numpy as np
import xarray as xr


def main(a):
    d = xr.open_dataset(a.nc)
    sid = np.array([str(s).zfill(8) for s in d.station_ids.values])
    np.savez(a.out,
             lat=np.asarray(d.lat.values, dtype=np.float64),
             lon=np.asarray(d.lon.values, dtype=np.float64),
             site_id=sid)
    print(f"wrote {a.out}: {len(sid)} gauges")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True)
    ap.add_argument("--out", default="logs/camels_coords.npz")
    main(ap.parse_args())
