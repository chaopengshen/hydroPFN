"""Denoising diffusion for DEM patches -- the sampler every probe demanded.

Why a sampler at all (measured, not assumed): a deterministic net is capped at
~0.31 of true fine-scale spectral power (3-seed median, unbiased eval), because
an L1/L2 objective returns the conditional mean and the mean of plausible
terrains is smooth.  The hedging diagnostic showed the information is present
but spread over plausible positions; only sampling can commit to ONE sharp
realisation.  Li et al. (RSE 2021) reached this with a GAN, Zhao et al. (RSE
2024) with diffusion; diffusion is chosen here because the denoiser is a plain
U-Net (no adversarial instability on 6k patches) and masking composes cleanly
at sampling time.

Design choices, deliberate:
  * GroupNorm everywhere -- no train/eval statistics distinction, so the
    BatchNorm single-sample failure mode (elev RMSE 5.55 vs 1.11) cannot recur.
  * Unconditional training + known-region forcing at sampling (RePaint-style,
    DDIM schedule).  One model serves ANY mask shape -- squares, strokes, and
    the real target, channel-following water masks -- without retraining.
  * Per-patch normalisation by std with a 0.5 m floor (a near-flat patch must
    not be amplified into fake relief).  At inference the std comes from the
    VALID region only, which is what is actually known.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)


def harmonic_torch(z: torch.Tensor, mask: torch.Tensor,
                   iters: int = 300) -> torch.Tensor:
    """Batched Jacobi Laplace fill on GPU, for the residual parameterisation.

    The numpy `fill_harmonic` iterates to a tolerance and is far too slow to
    call inside a training loop; this is the same relaxation, vectorised over
    the batch and run for a fixed budget.  Edge-replicated padding, so a hole
    touching the patch border relaxes under zero flux rather than being pinned
    to the mean (the B1 bug from the code review).
    """
    out = z * mask
    hole = mask < 0.5
    # seed the hole with the valid-region mean of each patch
    v = mask.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
    mu = (z * mask).sum(dim=(1, 2, 3), keepdim=True) / v
    out = torch.where(hole, mu.expand_as(z), out)
    for _ in range(iters):
        p = F.pad(out, (1, 1, 1, 1), mode="replicate")
        nb = (p[:, :, :-2, 1:-1] + p[:, :, 2:, 1:-1] +
              p[:, :, 1:-1, :-2] + p[:, :, 1:-1, 2:]) * 0.25
        out = torch.where(hole, nb, out)
    return out


class TimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Linear(dim * 4, dim * 4))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000.0) *
                          torch.arange(half, device=t.device) / (half - 1))
        a = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([a.sin(), a.cos()], dim=1))


class Block(nn.Module):
    """Two convs with FiLM-lite time injection and a residual skip."""

    def __init__(self, ci: int, co: int, temb: int):
        super().__init__()
        self.n1, self.c1 = nn.GroupNorm(8, ci), nn.Conv2d(ci, co, 3, 1, 1)
        self.n2, self.c2 = nn.GroupNorm(8, co), nn.Conv2d(co, co, 3, 1, 1)
        self.t = nn.Linear(temb, co)
        self.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(e)[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class DenoiseUNet(nn.Module):
    """Small time-conditioned U-Net (128 -> 16 -> 128), ~8M params at w=64."""

    def __init__(self, w: int = 64, temb: int = 64, in_ch: int = 1,
                 scale_cond: bool = False):
        """in_ch=1 unconditional; in_ch=3 conditional inpainting, where the
        extra channels are (known * mask, mask).

        v0 was unconditional + RePaint-style forcing at sampling time and it
        FAILED hard: psd_ratio 2.90 against a 0.7-1.3 band, elev RMSE 11.3 vs
        harmonic 2.03, and visible black seams at every hole rim.  Forcing the
        known region between denoising steps is too weak a signal without
        RePaint's resampling loop -- the hole free-runs.  Giving the net the
        context as INPUT makes boundary agreement part of what it is trained
        to do, rather than something patched on afterwards.
        """
        super().__init__()
        td = temb * 4
        self.temb = TimeEmb(temb)
        # SCALE CONDITIONING. One network for every footprint, told which one
        # it is looking at. Without this a 10 m patch and a 400 m patch are
        # the same tensor with wildly different statistics, and the net must
        # infer the scale from texture -- which it can only do where texture
        # survives, i.e. not at the coarse end.
        #
        # This is what makes mixed-resolution training possible, and mixed
        # resolution is not optional: 3DEP 10 m is CONUS-only and the global
        # tier is 30 m (Copernicus GLO-30) or 90 m (MERIT). A model that has
        # only ever seen 10 m is out-of-distribution over most of the planet.
        # Inputs are log10(metres per pixel) and log10(footprint km), both
        # continuous, so an unseen resolution INTERPOLATES rather than falling
        # off a lookup table.
        self.scale_cond = scale_cond
        if scale_cond:
            self.smlp = nn.Sequential(nn.Linear(2, td), nn.SiLU(),
                                      nn.Linear(td, td))
        self.inp = nn.Conv2d(in_ch, w, 3, 1, 1)
        self.d1 = Block(w, w, td)
        self.d2 = Block(w, w * 2, td)
        self.d3 = Block(w * 2, w * 2, td)
        self.mid = Block(w * 2, w * 2, td)
        self.u3 = Block(w * 4, w * 2, td)
        self.u2 = Block(w * 4, w, td)
        self.u1 = Block(w * 2, w, td)
        self.out = nn.Sequential(nn.GroupNorm(8, w), nn.SiLU(),
                                 nn.Conv2d(w, 1, 3, 1, 1))

    def forward(self, x, t, scale=None):
        e = self.temb(t)
        if self.scale_cond and scale is not None:
            e = e + self.smlp(scale.to(e.dtype))
        h1 = self.d1(self.inp(x), e)                      # 128, w
        h2 = self.d2(F.avg_pool2d(h1, 2), e)              # 64,  2w
        h3 = self.d3(F.avg_pool2d(h2, 2), e)              # 32,  2w
        m = self.mid(F.avg_pool2d(h3, 2), e)              # 16,  2w
        y = F.interpolate(m, scale_factor=2, mode="nearest")
        y = self.u3(torch.cat([y, h3], 1), e)             # 32,  2w
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        y = self.u2(torch.cat([y, h2], 1), e)             # 64,  w
        y = F.interpolate(y, scale_factor=2, mode="nearest")
        y = self.u1(torch.cat([y, h1], 1), e)             # 128, w
        return self.out(y)


class Diffusion:
    def __init__(self, T: int = 1000, device: str = "cuda",
                 param: str = "eps"):
        """param: 'eps' predicts the noise (standard DDPM); 'v' predicts the
        velocity v = sqrt(ab)*eps - sqrt(1-ab)*x0.

        v-prediction is the standard remedy when the signal of interest lives
        at low SNR -- which is our measured failure mode: the sampler is worst
        where fine texture is a LARGE fraction of total relief (Spearman -0.417
        between texture/relief ratio and psd_ratio). eps-prediction weights the
        high-noise steps heavily, where fine structure is indistinguishable
        from noise; v-prediction balances the objective across the schedule.
        """
        self.T = T
        self.param = param
        self.ab = torch.cumprod(1 - cosine_betas(T).to(device), 0)

    def _to_eps_x0(self, pred, x, t):
        """Convert whatever the net predicted into (eps, x0)."""
        ab = self.ab[t][:, None, None, None] if pred.dim() == 4 else self.ab[t]
        if self.param == "v":
            x0 = ab.sqrt() * x - (1 - ab).sqrt() * pred
            eps = (1 - ab).sqrt() * x + ab.sqrt() * pred
        else:
            eps = pred
            x0 = (x - (1 - ab).sqrt() * eps) / ab.sqrt()
        return eps, x0

    def _target(self, x0, noise, t):
        if self.param == "v":
            ab = self.ab[t][:, None, None, None]
            return ab.sqrt() * noise - (1 - ab).sqrt() * x0
        return noise

    def q_sample(self, x0, t, noise):
        ab = self.ab[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    def loss(self, net, x0, scale=None):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        n = torch.randn_like(x0)
        return F.mse_loss(net(self.q_sample(x0, t, n), t, scale), n)

    def loss_cond(self, net, x0, mask, cond=None, scale=None):
        """Conditional inpainting loss.  Noise is predicted everywhere but
        scored ONLY in the hole: outside it the answer is handed to the net as
        input, so scoring there would reward copying and drown the part we
        actually care about."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        n = torch.randn_like(x0)
        xt = self.q_sample(x0, t, n)
        inp = torch.cat([xt, cond if cond is not None else x0 * mask, mask],
                        dim=1)
        pred = net(inp, t, scale)
        tgt = self._target(x0, n, t)
        hole = 1.0 - mask
        return ((pred - tgt) ** 2 * hole).sum() / hole.sum().clamp(min=1.0)

    @torch.no_grad()
    def ddim_cond(self, net, known, mask, steps: int = 50,
                  resample: int = 2, ctx_override=None):
        """DDIM with the context supplied as input channels.

        `resample` implements RePaint's jump-back: after each step, re-noise
        and redo it, which lets the hole reconcile with the rim instead of
        committing on the first pass.  Even with conditioning this measurably
        cleans up seams; 2 is cheap insurance.
        """
        B = known.shape[0]
        ctx = (ctx_override if ctx_override is not None
               else torch.cat([known * mask, mask], dim=1))
        ts = torch.linspace(self.T - 1, 0, steps,
                            device=known.device).round().long()
        x = torch.randn_like(known)
        for i, t in enumerate(ts):
            for u in range(resample):
                tb = torch.full((B,), int(t), device=known.device,
                                dtype=torch.long)
                pred = net(torch.cat([x, ctx], dim=1), tb)
                eps, x0 = self._to_eps_x0(pred, x, tb)
                x0 = x0.clamp(-6, 6)
                x0 = mask * known + (1 - mask) * x0
                if i == len(ts) - 1:
                    x = x0
                    break
                ab_n = self.ab[ts[i + 1]]
                x_next = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
                if u < resample - 1:            # jump back to t and retry
                    x = self.q_sample(x0, tb, torch.randn_like(x0))
                else:
                    x = x_next
        return mask * known + (1 - mask) * x

    @torch.no_grad()
    def ddim_inpaint(self, net, known, mask, steps: int = 50):
        """DDIM (eta=0) with the valid region forced to its correct noise
        level at every step.  Diversity across calls comes from the initial
        noise; the valid region is returned exactly as given."""
        B = known.shape[0]
        ts = torch.linspace(self.T - 1, 0, steps,
                            device=known.device).round().long()
        x = torch.randn_like(known)
        for i, t in enumerate(ts):
            tb = torch.full((B,), int(t), device=known.device,
                            dtype=torch.long)
            x_known = self.q_sample(known, tb, torch.randn_like(known))
            x = mask * x_known + (1 - mask) * x
            eps = net(x, tb)
            ab_t = self.ab[t]
            x0 = ((x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()).clamp(-6, 6)
            if i == len(ts) - 1:
                x = x0
            else:
                ab_n = self.ab[ts[i + 1]]
                x = ab_n.sqrt() * x0 + (1 - ab_n).sqrt() * eps
        return mask * known + (1 - mask) * x
