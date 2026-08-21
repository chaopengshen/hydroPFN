"""Inpainting models, masks, baselines, and TEXTURE losses.

The first probe trained on plain L1 and produced smooth fills.  That is not a
budget problem: an L1/L2 loss yields the CONDITIONAL MEAN, and the conditional
mean of terrain given its surroundings is smooth.  No amount of capacity or
epochs changes it.  This is exactly why the DEM void-filling literature is
generative -- Li et al. (RSE 2021) add valley/ridge loss terms to a CGAN, and
Zhao et al. (RSE 2024) use diffusion.

Note the architecture was never the issue: diffusion denoisers ARE U-Nets, and
TKCGAN's generator is a U-Net too.  What differs is the objective.

`texture_loss` here is the cheap middle rung: keep the deterministic net, but add
terms that penalise wrong roughness STATISTICS rather than wrong pixels.  It can
produce plausible roughness but still emits one answer, so it suits
representation learning; a genuine sampler (diffusion/GAN) is still required
where a distribution is wanted, e.g. bathymetry under the water mask.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ masks

def make_mask(n: int, size: int, rng: np.random.Generator,
              kind: str = "mixed") -> np.ndarray:
    """1 = valid, 0 = hole.  `kind`: "square", "stroke", or "mixed" (half each).

    Irregular masks matter because the real target is the WATER surface: a thin,
    elongated, channel-following region.  Zhao et al. report GAN methods degrade
    specifically on awkward void shapes.
    """
    m = np.ones((n, size, size), dtype=np.float32)
    for i in range(n):
        if kind == "square" or (kind == "mixed" and rng.random() < 0.5):
            h = int(rng.integers(size // 8, size // 3))
            r0, c0 = rng.integers(0, size - h), rng.integers(0, size - h)
            m[i, r0:r0 + h, c0:c0 + h] = 0.0
        else:
            x, y = rng.integers(0, size, 2)
            w = int(rng.integers(3, 9))
            for _ in range(int(rng.integers(12, 30))):
                ang = rng.uniform(0, 2 * np.pi)
                ln = int(rng.integers(5, 20))
                for t in range(ln):
                    xx = int(np.clip(x + t * np.cos(ang), 0, size - 1))
                    yy = int(np.clip(y + t * np.sin(ang), 0, size - 1))
                    m[i, max(0, yy - w):yy + w, max(0, xx - w):xx + w] = 0.0
                x = int(np.clip(x + ln * np.cos(ang), 0, size - 1))
                y = int(np.clip(y + ln * np.sin(ang), 0, size - 1))
    return m


# -------------------------------------------------------------- baselines

def fill_harmonic(z: np.ndarray, m: np.ndarray, max_iters: int = 4000,
                  tol: float = 1e-3) -> np.ndarray:
    """Laplace fill: solve grad^2 z = 0 in the hole with the rim as Dirichlet BC.

    This is the MINIMISER of integral|grad z|^2 subject to the boundary, i.e. the
    provably flattest consistent surface -- which is why it wins any pointwise
    slope metric and why such metrics are the wrong test for texture.

    Two review fixes (docs/code_review_2026-08-17.md, B1 and M3):

      * Neighbour averages come from an edge-replicated pad, so a hole pixel on
        the patch border relaxes under a zero-flux boundary.  The previous
        version computed averages only for the interior and left `nb` zero on
        the border, silently re-pinning border hole pixels to the patch mean
        (0 in mean-removed coordinates) on EVERY sweep -- a spurious Dirichlet
        condition that propagated inward.
      * Iteration runs to a tolerance, not a fixed 400 sweeps.  Jacobi needs
        on the order of the hole area in sweeps, so a 40 px hole was returned
        unconverged -- i.e. extra-smooth, partway back to the constant initial
        fill.
    """
    out = z.copy()
    hole = m < 0.5
    if not hole.any():
        return out
    out[hole] = float(np.nanmean(z[~hole]))
    for _ in range(max_iters):
        p = np.pad(out, 1, mode="edge")
        nb = (p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:]) / 4.0
        delta = nb[hole] - out[hole]
        out[hole] = nb[hole]
        if np.abs(delta).max() < tol:
            break
    return out


def fill_idw(z: np.ndarray, m: np.ndarray, power: float = 2.0,
             k: int = 16) -> np.ndarray:
    """Inverse-distance weighting from the k nearest valid pixels."""
    out = z.copy()
    hole = np.argwhere(m < 0.5)
    if len(hole) == 0:
        return out
    valid = np.argwhere(m >= 0.5)
    vz = z[m >= 0.5]
    for r, c in hole:
        d2 = (valid[:, 0] - r) ** 2 + (valid[:, 1] - c) ** 2
        idx = np.argpartition(d2, k)[:k]
        w = 1.0 / np.power(d2[idx], power / 2.0)
        out[r, c] = float(np.sum(w * vz[idx]) / np.sum(w))
    return out


# ------------------------------------------------------------ texture loss

def _slope_t(z: torch.Tensor, px: float = 10.0) -> torch.Tensor:
    gy = (z[:, :, 2:, 1:-1] - z[:, :, :-2, 1:-1]) / (2 * px)
    gx = (z[:, :, 1:-1, 2:] - z[:, :, 1:-1, :-2]) / (2 * px)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)


def texture_loss(pred: torch.Tensor, truth: torch.Tensor,
                 mask: torch.Tensor) -> torch.Tensor:
    """Penalise wrong roughness STATISTICS, not wrong pixels.

    Two terms, both position-free by design:

      moment matching  -- mean and s.d. of |grad z| inside the hole.  A smooth
                          fill has too little of both; L1 alone never notices.
      spectral         -- log power spectrum of the composited patch, so the
                          fill must carry the right energy at each wavelength.

    Deliberately NOT a pointwise gradient loss: that is still mean-seeking and
    would reproduce the smoothing it is meant to cure.
    """
    hole = (1.0 - mask)[:, :, 1:-1, 1:-1]
    sp, st = _slope_t(pred), _slope_t(truth)
    n = hole.sum() + 1e-6
    mp = (sp * hole).sum() / n
    mt = (st * hole).sum() / n
    vp = ((sp - mp) ** 2 * hole).sum() / n
    vt = ((st - mt) ** 2 * hole).sum() / n
    moment = (mp - mt).abs() + (vp.sqrt() - vt.sqrt()).abs()

    comp = pred * (1 - mask) + truth * mask          # composite, as deployed
    Fp = torch.fft.fft2(comp - comp.mean(dim=(-2, -1), keepdim=True)).abs() ** 2
    Ft = torch.fft.fft2(truth - truth.mean(dim=(-2, -1), keepdim=True)).abs() ** 2
    spec = (torch.log1p(Fp) - torch.log1p(Ft)).abs().mean()
    return moment, spec


# ------------------------------------------------------- partial-conv U-Net

class PartialConv2d(nn.Conv2d):
    """Conv renormalised by the fraction of valid pixels in its window.

    A plain conv cannot distinguish real elevation from the hole's fill value, so
    it encodes the mask itself and the decoder learns hole-shaped artefacts.
    Here invalid pixels contribute exactly zero and the mask is updated per
    layer, so the hole SHRINKS with depth -- which is what lets deep skips carry
    real context across a void while shallow skips only do so near the rim.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.register_buffer("ones", torch.ones(1, 1, self.kernel_size[0],
                                                self.kernel_size[1]))
        self.win = float(self.kernel_size[0] * self.kernel_size[1])

    def forward(self, x, mask):
        with torch.no_grad():
            upd = F.conv2d(mask, self.ones, stride=self.stride,
                           padding=self.padding)
            new_mask = torch.clamp(upd, 0, 1)
            ratio = (self.win / (upd + 1e-8)) * new_mask
        out = super().forward(x * mask)
        if self.bias is not None:
            b = self.bias.view(1, -1, 1, 1)
            out = (out - b) * ratio + b
        else:
            out = out * ratio
        return out * new_mask, new_mask


class _Down(nn.Module):
    def __init__(self, ci, co, k=3, stride=2):
        super().__init__()
        self.pc = PartialConv2d(ci, co, k, stride, k // 2, bias=True)
        self.bn = nn.BatchNorm2d(co)

    def forward(self, x, m):
        x, m = self.pc(x, m)
        return F.leaky_relu(self.bn(x), 0.2), m


class _Up(nn.Module):
    def __init__(self, ci, cs, co):
        super().__init__()
        self.conv = nn.Conv2d(ci + cs, co, 3, 1, 1)
        self.bn = nn.BatchNorm2d(co)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        return F.leaky_relu(self.bn(self.conv(torch.cat([x, skip], 1))), 0.2)


class PConvUNet(nn.Module):
    def __init__(self, w: int = 32):
        super().__init__()
        self.d1, self.d2 = _Down(1, w, 7, 2), _Down(w, w * 2, 5, 2)
        self.d3, self.d4 = _Down(w * 2, w * 4, 3, 2), _Down(w * 4, w * 4, 3, 2)
        self.u3, self.u2 = _Up(w * 4, w * 4, w * 4), _Up(w * 4, w * 2, w * 2)
        self.u1, self.u0 = _Up(w * 2, w, w), _Up(w, 1, w)
        self.out = nn.Conv2d(w, 1, 3, 1, 1)

    def forward(self, x, m):
        x1, m1 = self.d1(x, m)
        x2, m2 = self.d2(x1, m1)
        x3, m3 = self.d3(x2, m2)
        x4, _ = self.d4(x3, m3)
        y = self.u3(x4, x3)
        y = self.u2(y, x2)
        y = self.u1(y, x1)
        return self.out(self.u0(y, x))
