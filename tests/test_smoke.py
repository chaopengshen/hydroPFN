"""Fast regression checks for lib/{terrain,inpaint}.py -- run after any edit.

Born from the 2026-08-17 review (docs/code_review_2026-08-17.md): three of its
findings were silent-wrong-number bugs that a table cannot reveal.  Each check
here pins one invariant that a past bug violated:

  B1  fill_harmonic pinned border-touching hole pixels to the patch mean
  M1  psd_ratio was diluted on stroke-mask bounding boxes
  M3  fixed 400 Jacobi sweeps returned an unconverged surface
  B3  torch was never seeded, so nets differed every run
  --  score(truth, truth) must be exactly clean (RMSE 0, every ratio 1)

Pure asserts, ~10 s on CPU:  python test_lib_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from hydropfn.models.inpaint import (PConvUNet, fill_harmonic, fill_idw,  # noqa: E402
                     make_mask)
from hydropfn.metrics.terrain import hole_crop, score  # noqa: E402


def check_harmonic_border_and_convergence():
    """B1 + M3 on a planar ramp, where the right answer is known.

    z = row index is harmonic, so away from the top border the fill must
    reproduce the plane.  The hole touches row 0: the old code re-pinned those
    pixels to the patch mean (~31.5 here, true value 0) every sweep.  The
    zero-flux pad bends the solution slightly near row 0 -- tolerated -- but a
    patch-mean pin is 30+ m wrong and unmistakable.
    """
    n = 64
    z = np.tile(np.arange(n, dtype=np.float64), (n, 1)).T   # z = row index
    m = np.ones((n, n), np.float32)
    m[0:21, 20:41] = 0.0                                    # touches row 0
    out = fill_harmonic(z, m)
    mean = float(z[m >= 0.5].mean())
    # The zero-flux pad MUST bend the fill at the open border: it forces
    # dz/dy = 0 against the ramp's unit gradient, a Neumann boundary layer of
    # scale ~0.4 x hole width (~8 m here; measured 7.8).  So the check is not
    # "error is small" -- it is "error is boundary-layer sized, not
    # patch-mean sized" (the old pin put ~31.5 here, true value 0).
    err = abs(out[0, 30] - z[0, 30])
    assert err < 12.0, f"border hole pixel off by {err:.1f} m"
    assert abs(out[0, 30] - mean) > 15.0, \
        "border hole pixel sits near the patch mean -- B1 has regressed"
    # convergence: interior Laplacian residual ~ 4 * last Jacobi delta
    lap = (out[0:19, 21:40] + out[2:21, 21:40] +
           out[1:20, 20:39] + out[1:20, 22:41]) / 4.0 - out[1:20, 21:40]
    assert np.abs(lap).max() < 0.05, \
        f"unconverged: max Laplacian residual {np.abs(lap).max():.3f}"
    # The Neumann layer decays like exp(-pi*y/L): ~1.7 m at row 10, ~0.5 m by
    # row 18.  Check near the Dirichlet rim, where the plane must be recovered.
    assert np.abs(out[17:21, 25:36] - z[17:21, 25:36]).max() < 1.5

    # A hole NOT touching any border is fully Dirichlet, and the ramp is its
    # own harmonic solution -- the fill must reproduce the plane exactly (up to
    # the Jacobi tolerance).  This is the sharp convergence check (M3): the old
    # fixed 400 sweeps left visible error here.
    m2 = np.ones((n, n), np.float32)
    m2[20:41, 20:41] = 0.0
    out2 = fill_harmonic(z, m2)
    err2 = np.abs(out2 - z).max()
    assert err2 < 0.5, f"interior hole off by {err2:.2f} m -- unconverged?"
    print("  harmonic: border + convergence ok")


def check_hole_crop_dilution():
    """M1: the crop must report its hole fraction, and score must refuse
    diluted crops instead of averaging them toward 1.0."""
    n = 64
    rng = np.random.default_rng(0)
    z = rng.normal(size=(n, n)).cumsum(0).cumsum(1) / 50.0  # correlated field

    m_sq = np.ones((n, n), np.float32); m_sq[10:40, 10:40] = 0.0
    crop, hf = hole_crop(z, m_sq)
    assert crop.shape == (30, 30) and hf == 1.0

    # thin 3xN stroke: centred square side 3 < 16 -> unusable by design
    m_thin = np.ones((n, n), np.float32); m_thin[30:33, 5:60] = 0.0
    crop, hf = hole_crop(z, m_thin)
    assert crop is None and hf == 0.0

    # L-shaped hole: bbox 30x30 but only ~2/5 hole -> diluted, psd must be NaN
    m_L = np.ones((n, n), np.float32)
    m_L[10:40, 10:16] = 0.0
    m_L[34:40, 10:40] = 0.0
    _, hf = hole_crop(z, m_L)
    assert hf < 0.6, f"L-shape hole_frac {hf:.2f}, expected < 0.6"
    s = score(z, fill_harmonic(z, m_L), m_L)
    assert not np.isfinite(s["psd_ratio"]), \
        "diluted crop produced a psd_ratio -- M1 guard has regressed"
    assert s["psd_hole_frac"] == hf

    # square mask passes the guard and scores
    s = score(z, fill_harmonic(z, m_sq), m_sq)
    assert np.isfinite(s["psd_ratio"]) and s["psd_hole_frac"] == 1.0
    print("  hole_crop / psd guard: ok")


def check_score_identity():
    """score(truth, truth) must be exactly clean under any mask kind."""
    rng = np.random.default_rng(1)
    z = rng.normal(size=(64, 64)).cumsum(0).cumsum(1) / 50.0
    for kind in ("square", "stroke"):
        m = make_mask(1, 64, np.random.default_rng(2), kind=kind)[0]
        s = score(z, z.copy(), m)
        assert s["elev_rmse"] == 0.0 and s["slope_rmse"] == 0.0
        assert s["slope_w1"] == 0.0
        if np.isfinite(s["psd_ratio"]):
            assert abs(s["psd_ratio"] - 1.0) < 1e-6
        for k, v in s.items():
            if k.startswith("vario_ratio") and np.isfinite(v):
                assert abs(v - 1.0) < 1e-6
    print("  score identity: ok")


def check_idw_and_masks():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(64, 64)).cumsum(0) / 10.0
    m = np.ones((64, 64), np.float32); m[20:30, 20:30] = 0.0
    out = fill_idw(z, m)
    assert np.isfinite(out).all()
    assert (out[m >= 0.5] == z[m >= 0.5]).all(), "IDW touched valid pixels"
    for kind in ("square", "stroke", "mixed"):
        mk = make_mask(2, 64, np.random.default_rng(4), kind=kind)
        assert ((mk < 0.5).sum(axis=(1, 2)) > 0).all(), f"no hole for {kind}"
    print("  idw / make_mask: ok")


def check_torch_seeding():
    """B3: same seed -> identical net; forward pass has the right shape."""
    torch.manual_seed(11); n1 = PConvUNet(8)
    torch.manual_seed(11); n2 = PConvUNet(8)
    for p1, p2 in zip(n1.parameters(), n2.parameters()):
        assert torch.equal(p1, p2), "same torch seed gave different nets"
    n1.eval()
    x = torch.zeros(2, 1, 64, 64)
    m = torch.ones(2, 1, 64, 64); m[:, :, 20:40, 20:40] = 0.0
    with torch.no_grad():
        y = n1(x * m, m)
    assert y.shape == (2, 1, 64, 64) and torch.isfinite(y).all()
    print("  torch seeding / forward: ok")


if __name__ == "__main__":
    check_harmonic_border_and_convergence()
    check_hole_crop_dilution()
    check_score_identity()
    check_idw_and_masks()
    check_torch_seeding()
    print("all smoke tests passed")


def test_connector_geo_translation_invariant():
    """The geo encoding must depend on DISPLACEMENT, never absolute position.

    An encoding that varies with absolute lat/lon lets the connector identify
    which region a task came from, which would quietly break leave-region-out.
    The first version of this failed here: it scaled longitude by
    cos(query_latitude).
    """
    import torch
    from hydropfn.models.connector import CrossSiteConnector

    torch.manual_seed(0)
    c = CrossSiteConnector(64, 2, 4, geo=True).eval()
    tok, sv = torch.randn(2, 5, 4, 64), torch.ones(2, 5)
    ll = torch.randn(2, 5, 2) * 3
    # Tolerance is 1e-3, not 1e-5. The encoding is EXACTLY translation-
    # invariant in real arithmetic; in float32 the subtraction
    # `latlon - latlon[:, :1]` loses ~2e-6 on CONUS-magnitude coordinates and
    # the top Fourier frequency (64 rad/deg) amplifies that into ~1e-4 on the
    # output. Measured, not assumed: a genuine absolute-position leak moves
    # the output by O(1), three orders above this.
    with torch.no_grad():
        a = c(tok, sv, ll)
        for shift in ([7.0, 0.0], [0.0, -11.0], [20.0, 35.0], [100.0, 100.0]):
            b = c(tok, sv, ll + torch.tensor(shift))
            assert torch.allclose(a, b, atol=1e-3), f"leaks position: {shift}"


def test_connector_geo_responds_to_distance():
    """...but it must still DISTINGUISH near context from far context."""
    import torch
    from hydropfn.models.connector import CrossSiteConnector

    torch.manual_seed(0)
    c = CrossSiteConnector(64, 2, 4, geo=True).eval()
    tok, sv = torch.randn(2, 5, 4, 64), torch.ones(2, 5)
    near = torch.randn(2, 5, 2) * 0.1
    far = near * 50
    near[:, 0] = far[:, 0] = 0.0                 # query at the origin in both
    with torch.no_grad():
        assert not torch.allclose(c(tok, sv, near), c(tok, sv, far), atol=1e-4)


def test_extractor_scale_conditioning_is_live():
    """The wrong-extractor episode, pinned as a test.

    Three behavioral facts that must hold or the extraction is silently wrong
    again: (1) for a scale-conditioned net, changing the scale vector changes
    the features; (2) omitting it changes them too (the 10%-dropout regime is
    a DIFFERENT regime, which is exactly what the buggy extractor fed on);
    (3) for a non-scale net the argument is inert, so passing it can never
    corrupt an old checkpoint's features.
    """
    import numpy as np
    import torch
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                           / "experiments"))
    from dem_diffusion_features import diffusion_features
    from hydropfn.models.diffusion import DenoiseUNet, Diffusion

    torch.manual_seed(0)
    Z = np.random.default_rng(0).normal(size=(4, 128, 128)).astype("float32")
    diff = Diffusion(device="cpu")

    net = DenoiseUNet(w=32, in_ch=3, scale_cond=True).eval()
    s1 = torch.tensor([[2.0, 1.107]])          # 100 m/px, 12.8 km
    s2 = torch.tensor([[1.0, 0.107]])          # 10 m/px, 1.28 km
    torch.manual_seed(0); f1 = diffusion_features(net, diff, Z, 50, scale=s1)
    torch.manual_seed(0); f2 = diffusion_features(net, diff, Z, 50, scale=s2)
    torch.manual_seed(0); f0 = diffusion_features(net, diff, Z, 50, scale=None)
    assert not np.allclose(f1, f2), "scale vector ignored"
    assert not np.allclose(f1, f0), "scale=None indistinguishable from scaled"

    plain = DenoiseUNet(w=32, in_ch=3, scale_cond=False).eval()
    torch.manual_seed(0); g1 = diffusion_features(plain, diff, Z, 50, scale=s1)
    torch.manual_seed(0); g0 = diffusion_features(plain, diff, Z, 50, scale=None)
    assert np.allclose(g1, g0), "scale corrupted a non-scale checkpoint"
