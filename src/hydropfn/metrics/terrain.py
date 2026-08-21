"""Terrain derivatives, rendering, and TEXTURE metrics.

Why the texture metrics exist.  The first inpainting probe scored pointwise
slope RMSE and concluded harmonic interpolation was best.  It is -- at that
metric -- but for the wrong reason: harmonic minimises the Dirichlet energy
integral|grad z|^2, so it is the provably FLATTEST surface matching the rim, and
pointwise error rewards exactly that.  Measured on held-out tiles:

    method     slope RMSE   slope bias   share of RMSE from bias
    harmonic     0.0257      -0.0080            31%
    unet         0.0279      -0.0067            24%
    idw          0.0359      -0.0029             8%

IDW has nearly the RIGHT amount of roughness (smallest bias) and the WORST
pointwise score, because its roughness sits in the wrong places.  A pointwise
metric penalises misplaced texture twice -- once for missing the true texture,
once for adding false texture -- so a generative model that invents realistic
but not positionally exact terrain scores worse than a flat fill.

Hence: judge texture DISTRIBUTIONALLY.  These metrics ask "does the fill have the
right roughness, at the right scales" without demanding pointwise correspondence.
"""

from __future__ import annotations

import numpy as np

PX = 10.0          # 3DEP 1/3 arc-second, metres


# ------------------------------------------------------------- derivatives

def slope_mag(z: np.ndarray, px: float = PX) -> np.ndarray:
    gy, gx = np.gradient(z, px)
    return np.hypot(gx, gy)


def hillshade(z: np.ndarray, az: float = 315.0, alt: float = 45.0,
              px: float = PX) -> np.ndarray:
    """Standard hillshade.

    Used instead of an elevation colour ramp because the claim under test is
    about TEXTURE: a ramp over a 1.28 km patch shows a smooth blob and makes a
    harmonic fill look like real terrain.
    """
    gy, gx = np.gradient(z, px)
    slope = np.pi / 2.0 - np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    az_r, alt_r = np.radians(360.0 - az + 90.0), np.radians(alt)
    hs = (np.sin(alt_r) * np.sin(slope) +
          np.cos(alt_r) * np.cos(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0.0, 1.0)


# --------------------------------------------------------- texture metrics

def slope_wasserstein(truth: np.ndarray, pred: np.ndarray,
                      mask: np.ndarray) -> float:
    """W1 distance between slope DISTRIBUTIONS inside the hole.

    Position-free: a fill with the right roughness in the wrong places scores
    well here and badly on pointwise RMSE.  That is the intended difference.
    """
    from scipy.stats import wasserstein_distance
    hole = mask < 0.5
    if hole.sum() < 16:
        return np.nan
    return float(wasserstein_distance(slope_mag(truth)[hole],
                                      slope_mag(pred)[hole]))


def radial_psd(z: np.ndarray, px: float = PX):
    """Radially averaged power spectral density.  Returns (wavelength_m, power)."""
    z = z - z.mean()
    n = z.shape[0]
    w = np.hanning(n)[:, None] * np.hanning(n)[None, :]      # reduce edge leakage
    F = np.fft.fftshift(np.abs(np.fft.fft2(z * w)) ** 2)
    cy, cx = n // 2, n // 2
    y, x = np.indices(F.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    tbin = np.bincount(r.ravel(), F.ravel())
    nr = np.bincount(r.ravel())
    prof = tbin / np.maximum(nr, 1)
    k = np.arange(len(prof))
    with np.errstate(divide="ignore"):
        lam = np.where(k > 0, n * px / np.maximum(k, 1), np.inf)
    return lam[1:n // 2], prof[1:n // 2]


def psd_ratio(truth: np.ndarray, pred: np.ndarray,
              band_m: tuple[float, float] = (20.0, 160.0)) -> float:
    """Predicted / true power in a short-wavelength band.

    1.0 means the fill carries the right amount of fine roughness; << 1 means it
    is too smooth.  This is the single number that most directly says "did the
    model produce texture", and it is what a pointwise metric cannot see.
    """
    lam_t, p_t = radial_psd(truth)
    lam_p, p_p = radial_psd(pred)
    m = (lam_t >= band_m[0]) & (lam_t <= band_m[1])
    if m.sum() == 0 or p_t[m].sum() <= 0:
        return np.nan
    return float(p_p[m].sum() / p_t[m].sum())


def semivariogram(z: np.ndarray, lags: tuple[int, ...] = (1, 2, 4, 8, 16),
                  px: float = PX, mask: np.ndarray | None = None) -> dict:
    """gamma(h) = 0.5 * mean((z(x) - z(x+h))^2), the geostatistical standard.

    Directly comparable to the kriging baselines the DEM void-filling papers use,
    since kriging is fitted to exactly this function.

    **`mask` is not optional in practice.**  A prediction equals the truth
    outside the hole, so computing gamma over the whole patch makes 75-90% of the
    comparison self-identical and every method scores ~1.0.  Passing the mask
    restricts the pairs to those with BOTH endpoints inside the hole, which is
    the only part either method actually predicted.
    """
    out = {}
    inside = None if mask is None else (mask < 0.5)
    for h in lags:
        dv = z[h:, :] - z[:-h, :]
        dh = z[:, h:] - z[:, :-h]
        if inside is None:
            d = np.concatenate([dv.ravel(), dh.ravel()])
        else:
            kv = inside[h:, :] & inside[:-h, :]
            kh = inside[:, h:] & inside[:, :-h]
            d = np.concatenate([dv[kv].ravel(), dh[kh].ravel()])
        out[f"gamma_{int(h*px)}m"] = float(0.5 * np.mean(d ** 2)) if d.size else np.nan
    return out


def hole_crop(z: np.ndarray, mask: np.ndarray):
    """Largest centred square inside the hole's bounding box, plus how much of
    that square is actually hole.  Returns ``(crop, hole_frac)``; ``(None, 0.0)``
    if the hole is too small.

    The FFT needs a rectangle, so irregular masks are approximated by their
    bounding box.  For a SQUARE mask the box is all hole; for a stroke mask the
    box of a wandering stroke is mostly VALID pixels, where pred == truth by
    construction, and those pixels drag any spectral ratio toward 1.0 regardless
    of method -- the same dilution failure already fixed once at the whole-patch
    level (review M1).  Hence the fraction is returned so callers can DISCARD
    diluted crops instead of averaging them.
    """
    ys, xs = np.where(mask < 0.5)
    if len(ys) < 16:
        return None, 0.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    # radial_psd needs a SQUARE array (its Hanning window and radial binning
    # both assume it).  An irregular mask's bounding box is not square -- an
    # earlier version crashed on a 50x91 crop -- so take the largest centred
    # square that fits inside the box.
    side = min(y1 - y0, x1 - x0)
    if side < 16:
        return None, 0.0
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h = side // 2
    sl = np.s_[cy - h:cy - h + side, cx - h:cx - h + side]
    return z[sl], float((mask[sl] < 0.5).mean())


def score(truth: np.ndarray, pred: np.ndarray, mask: np.ndarray) -> dict:
    """Pointwise AND distributional scores, so the two can be compared directly."""
    from scipy.ndimage import binary_dilation
    hole = mask < 0.5
    e = pred[hole] - truth[hole]
    st, sp = slope_mag(truth), slope_mag(pred)
    ds = sp[hole] - st[hole]
    rim = binary_dilation(hole) & ~hole
    gy, gx = np.gradient(pred - truth, PX)

    out = {
        # pointwise -- structurally biased toward smooth fills, see module docstring
        "elev_rmse": float(np.sqrt(np.mean(e ** 2))),
        "slope_rmse": float(np.sqrt(np.mean(ds ** 2))),
        "slope_bias": float(np.mean(ds)),
        "rim_jump": float(np.mean(np.hypot(gx, gy)[rim])) if rim.any() else np.nan,
        # distributional -- these are the ones that can reward real texture
        "slope_w1": slope_wasserstein(truth, pred, mask),
    }
    # Spectral + variogram restricted to the hole, else the identical surround
    # dominates and every method scores ~1.0.
    ct, hf = hole_crop(truth, mask)
    cp, _ = hole_crop(pred, mask)
    # Diluted crops (stroke-mask bounding boxes that are mostly valid pixels)
    # are reported as NaN, not averaged in -- a <60%-hole crop scores every
    # method toward 1.0 (review M1).  The variogram needs no such guard: its
    # pair masking already restricts to hole-hole pairs exactly.
    ok = (ct is not None and ct.shape[0] >= 16 and ct.shape == cp.shape
          and hf >= 0.6)
    out["psd_ratio"] = psd_ratio(ct, cp) if ok else np.nan
    out["psd_hole_frac"] = hf
    gt = semivariogram(truth, mask=mask)
    gp = semivariogram(pred, mask=mask)
    for k in gt:
        out[f"vario_ratio_{k.split('_')[1]}"] = (gp[k] / gt[k]) if gt[k] > 0 else np.nan
    return out
