"""CAMELS loading, patchification and mask sampling.

This module was missing from the repository -- every entry point imported it
(`train_pub`, `train_site_encoder`, `lstm_baseline`, `verify_split`) and none
of them could run. Restored 2026-08-28.

**The data file.** `CAMELS_Frederik.nc` is byte-for-byte the same underlying
dataset as dmg_dev's `Camels_Pretrain.nc`: 671 stations x 12,784 days
(1980-01-01 .. 2014-12-31), identical `QObs`, identical lat/lon, only the
variable NAMES differ. Verified by array comparison, not by reading configs --
that mistake ("StefaLand's input paths transfer") is already in Diagnosis.md.
The name map is `DMG_ALIASES` below, so either file loads with either
spelling.

**QObs is in mm/day and is NOT transformed here.** Earlier versions of this
project scored `log1p(QObs)` and compared the result against published CAMELS
NSE, which is computed on raw discharge. Log-space NSE weights low flows
differently and is simply a different number. Any transform is now the
caller's explicit choice (`log_target=True`), and `hydropfn.data.protocol`
always scores on the raw mm/day scale.
"""

from __future__ import annotations

import numpy as np

# The five Daymet forcings plus discharge, in the order the models expect:
# QObs is ALWAYS last, because `obs_col = Xp.shape[2] - 1` throughout.
DEFAULT_SERIES = ["prcp_daymet", "srad_daymet", "tmax_daymet", "tmin_daymet",
                  "vp_daymet", "QObs"]

# The 26 statics. This is exactly the attribute set dmg_dev's
# EmbeddingPUB531.yaml uses, under this file's spellings -- see DMG_ALIASES.
DEFAULT_STATICS = [
    "elev_mean", "slope_mean", "area_gages2", "frac_forest", "lai_max",
    "lai_diff", "gvf_max", "gvf_diff", "soil_depth_pelletier",
    "soil_depth_statsgo", "soil_porosity", "soil_conductivity",
    "max_water_content", "sand_frac", "silt_frac", "clay_frac",
    "carbonate_rocks_frac", "geol_permeability", "p_mean", "pet_mean",
    "aridity", "frac_snow", "high_prec_freq", "high_prec_dur",
    "low_prec_freq", "low_prec_dur",
]

# dmg_dev spelling -> this file's spelling. Both netCDFs hold the same arrays.
DMG_ALIASES = {
    "P": "prcp_daymet", "Tmax": "tmax_daymet", "Tmin": "tmin_daymet",
    "meanelevation": "elev_mean", "meanslope": "slope_mean",
    "forest_frac": "frac_forest", "soil_depth": "soil_depth_pelletier",
    "porosity": "soil_porosity", "HWSD_sand": "sand_frac",
    "HWSD_silt": "silt_frac", "HWSD_clay": "clay_frac",
    "carbonate_sedimentary_rocks_frac": "carbonate_rocks_frac",
    "permeability": "geol_permeability", "meanP": "p_mean",
    "snowfall_fraction": "frac_snow",
}

# Attributes derived from the forcing record itself rather than from the
# landscape. `train_site_encoder`'s `no_climate` ablation drops these to ask
# whether the encoder is using geology or merely re-reading its own weather.
CLIMATE_STATICS = {
    "p_mean", "pet_mean", "aridity", "frac_snow", "high_prec_freq",
    "high_prec_dur", "low_prec_freq", "low_prec_dur", "p_seasonality",
}

# The four conditionals the mask mixture is trained over. Each is a different
# question asked of the same weights; `whole_site` alone lets the time-aligned
# path copy a neighbour's concurrent flow and nothing else ever learns
# anything.
MASK_KINDS = ["whole_site", "whole_variable", "random_span", "causal_tail"]


def _resolve(ds, name):
    """Fetch a variable under either spelling, so both netCDFs load."""
    if name in ds.variables:
        return ds[name]
    for dmg, ours in DMG_ALIASES.items():
        if ours == name and dmg in ds.variables:
            return ds[dmg]
        if dmg == name and ours in ds.variables:
            return ds[ours]
    raise KeyError(f"{name!r} not in dataset (have {list(ds.data_vars)[:8]}...)")


def load_camels(nc_path, series=None, statics=None, log_target=False):
    """Load CAMELS into the dense arrays every entry point expects.

    Returns
    -------
    dict with
        x           (B, T, V)  float32, series in `series` order, QObs last
        valid       (B, T, V)  float32, 1.0 = real observation
        attrs       (B, A)     float32, NaN preserved (callers impute)
        region      (B,)       str, USGS part number (first two digits of id)
        site_id     (B,)       str, zero-padded 8-character gage id
        latlon      (B, 2)     float32
        area        (B,)       float32, `area_gages2` in km^2
        time        (T,)       numpy datetime64[D]
        series_vars / static_vars   the resolved name lists

    `log_target=False` is deliberate and is the fix for the mismatch described
    in the module docstring. Pass True only if you also score in log space.
    """
    import xarray as xr

    series = list(series or DEFAULT_SERIES)
    statics = list(statics or DEFAULT_STATICS)
    ds = xr.open_dataset(nc_path)

    cols = []
    for v in series:
        a = np.asarray(_resolve(ds, v).values, dtype=np.float32)
        cols.append(a)
    x = np.stack(cols, axis=-1)                                  # (B, T, V)

    if log_target and "QObs" in series:
        j = series.index("QObs")
        x[..., j] = np.log1p(np.clip(x[..., j], 0.0, None))

    valid = np.isfinite(x).astype(np.float32)
    x = np.nan_to_num(x, nan=0.0)

    attrs = np.stack(
        [np.asarray(_resolve(ds, s).values, dtype=np.float32) for s in statics],
        axis=-1)                                                 # (B, A)

    sid = np.asarray(ds["station_ids"].values).astype(str)
    sid = np.array([s.strip().zfill(8) for s in sid])
    region = np.array([s[:2] for s in sid])
    latlon = np.stack([np.asarray(ds["lat"].values, dtype=np.float32),
                       np.asarray(ds["lon"].values, dtype=np.float32)], -1)
    area = attrs[:, statics.index("area_gages2")].astype(np.float32)
    time = np.asarray(ds["time"].values).astype("datetime64[D]")
    ds.close()

    return {"x": x, "valid": valid, "attrs": attrs, "region": region,
            "site_id": sid, "latlon": latlon, "area": area, "time": time,
            "series_vars": series, "static_vars": statics}


def patchify(x, patch):
    """(B, T, V) -> (B, T//patch, V, patch). Trailing partial patch dropped."""
    x = np.asarray(x)
    B, T, V = x.shape
    N = T // patch
    return x[:, :N * patch].reshape(B, N, patch, V).transpose(0, 1, 3, 2)


def sample_mask(win, n_vars, rng, kind="whole_site", n_obs=1):
    """Visibility array (win, n_vars); 1.0 = visible to the model.

    `n_obs` is how many trailing columns are observation channels (QObs); the
    kinds that must target an observation pick from those, the ones that may
    target any variable pick from all -- hiding a FORCING and inferring it from
    discharge is the inverse-rainfall direction, and it is a real conditional,
    not a bug.
    """
    v = np.ones((win, n_vars), np.float32)
    obs_cols = list(range(n_vars - n_obs, n_vars))

    if kind == "whole_site":
        # every observation channel hidden for the whole window -- the PUB task
        for c in obs_cols:
            v[:, c] = 0.0
    elif kind == "whole_variable":
        # one variable, any variable, hidden throughout -- cross-variable
        v[:, int(rng.integers(0, n_vars))] = 0.0
    elif kind == "random_span":
        # a contiguous outage in one variable -- gap filling
        c = int(rng.integers(0, n_vars))
        ln = max(1, int(rng.integers(1, max(2, win // 2))))
        st = int(rng.integers(0, max(1, win - ln + 1)))
        v[st:st + ln, c] = 0.0
    elif kind == "causal_tail":
        # cut anywhere, predict the rest -- forecasting
        c = int(rng.choice(obs_cols))
        cut = int(rng.integers(1, max(2, win)))
        v[cut:, c] = 0.0
    else:
        raise ValueError(f"unknown mask kind {kind!r}; expected {MASK_KINDS}")
    return v


def synthetic(n_sites=64, n_days=4096, seed=0, n_vars=3):
    """A cheap stand-in with the same dict shape, for smoke tests only.

    Each site is a linear-reservoir response to its own noise forcing with a
    site-specific recession, so attributes genuinely predict behaviour and a
    working encoder must score well above zero.
    """
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.05, 0.6, n_sites).astype(np.float32)
    x = np.zeros((n_sites, n_days, n_vars), np.float32)
    for i in range(n_sites):
        p = np.clip(rng.gamma(0.4, 6.0, n_days) - 1.0, 0, None)
        t = 12 + 10 * np.sin(2 * np.pi * np.arange(n_days) / 365.25)
        q = np.zeros(n_days, np.float32)
        s = 0.0
        for j in range(n_days):
            s = s * (1 - k[i]) + p[j]
            q[j] = s * k[i]
        x[i, :, 0], x[i, :, 1], x[i, :, -1] = p, t, q
    attrs = np.stack([k, k ** 2, rng.normal(size=n_sites)], -1).astype(np.float32)
    return {"x": x, "valid": np.ones_like(x), "attrs": attrs,
            "region": np.array([f"{i % 6:02d}" for i in range(n_sites)]),
            "site_id": np.array([f"{i:08d}" for i in range(n_sites)]),
            "latlon": rng.uniform(-1, 1, (n_sites, 2)).astype(np.float32),
            "area": np.abs(attrs[:, 0]) * 100,
            "time": np.arange("1980-01-01", dtype="datetime64[D]") +
            np.arange(n_days),
            "series_vars": ["prcp", "tmean", "QObs"][:n_vars],
            "static_vars": ["k", "k2", "noise"]}
