"""Scale-conditioned DEM pretraining: one model, every footprint.

Why multi-scale, and why the conditioning is not optional.

A 1.28 km patch at 10 m and a 51.2 km patch at 400 m are the same tensor shape
with entirely different statistics -- relief, texture, the spatial frequency
where signal lives. Trained together WITHOUT being told which is which, the
model must infer scale from texture, which it can only do where texture
survives. It therefore learns the fine end and treats the coarse end as noise,
or blurs both.

Told the scale, one network covers all of them, and the conditioning is
CONTINUOUS -- log10(metres per pixel) and log10(footprint km) -- so a
resolution never seen in training INTERPOLATES rather than falling off the end
of a lookup table. That property is the point: 3DEP 10 m is CONUS-only, and
the global tier is 30 m (Copernicus GLO-30) or 90 m (MERIT). A model that has
only seen 10 m is out-of-distribution over most of the planet; a model
conditioned on continuous resolution can be asked for 30 m directly.

The masking follows the same principle as the time-series arm: the shape of the
hole is the task. Holes are drawn, not fixed, so one checkpoint does
inpainting, super-resolution-style refinement, and unconditional generation
depending on what is hidden at inference.

Evaluation is deliberately NOT reconstruction error -- a model can win that by
blurring. It is the downstream probe that already discriminates: lithology and
bedrock age from the frozen features (experiments/lithology_premise.py),
where hand descriptors reach 0.593/0.367 and single-scale diffusion features
reach 0.622/0.333.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from hydropfn.models.diffusion import DenoiseUNet, Diffusion
from hydropfn.paths import LOGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_levels(specs):
    """[(path, metres_per_pixel, footprint_km)] -> stacked patches + scale."""
    P, S, src = [], [], []
    for path, mpp, km in specs:
        z = np.load(path, allow_pickle=True)
        a = z["patches"] if "patches" in z else z["dem"]
        ok = z["ok"] if "ok" in z else np.ones(len(a), bool)
        a = a[ok].astype(np.float32)
        good = np.isfinite(a).all((1, 2))
        a = a[good]
        P.append(a)
        S.append(np.tile([np.log10(mpp), np.log10(km)],
                         (len(a), 1)).astype(np.float32))
        src.append(np.full(len(a), f"{km:g}km", dtype=object))
        print(f"  {path.split('/')[-1]:26s} {len(a):6,d} patches  "
              f"{mpp:>5.0f} m/px  {km:>5.1f} km")
    return np.concatenate(P), np.concatenate(S), np.concatenate(src)


def sample_holes(n, size, rng, device):
    """1 = KNOWN, 0 = hidden. Mixture of hole shapes, drawn per sample."""
    m = torch.ones(n, 1, size, size, device=device)
    for i in range(n):
        r = rng.random()
        if r < 0.25:                                   # centre block
            h = int(size * rng.uniform(0.2, 0.5))
            o = (size - h) // 2
            m[i, :, o:o + h, o:o + h] = 0
        elif r < 0.55:                                 # random block
            h = int(size * rng.uniform(0.15, 0.45))
            y, x = (int(rng.integers(0, size - h)) for _ in range(2))
            m[i, :, y:y + h, x:x + h] = 0
        elif r < 0.8:                                  # scattered swath
            k = int(size * rng.uniform(0.1, 0.3))
            y = int(rng.integers(0, size - k))
            m[i, :, y:y + k, :] = 0
        else:                                          # almost everything
            m[i] = 0
            keep = int(size * 0.1)
            m[i, :, :keep, :keep] = 1
    return m


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    specs = []
    for tok in a.levels.split(";"):
        path, mpp, km = tok.split(",")
        specs.append((path, float(mpp), float(km)))
    print("levels:")
    P, S, src = load_levels(specs)
    print(f"  TOTAL {len(P):,} patches\n")

    # PER-PATCH normalisation. Absolute elevation differs by kilometres across
    # CONUS and would dominate the loss; the model should learn FORM. The
    # scale token carries the physical extent that normalisation removes.
    mu = P.mean((1, 2), keepdims=True)
    sd = P.std((1, 2), keepdims=True) + 1e-6
    Z = (P - mu) / sd

    n = len(Z)
    perm = rng.permutation(n)
    n_val = max(256, int(0.05 * n))
    va, tr = perm[:n_val], perm[n_val:]
    print(f"train {len(tr):,}  val {len(va):,}")

    net = DenoiseUNet(w=a.width, in_ch=3, scale_cond=True).to(DEVICE)
    diff = Diffusion(device=DEVICE)
    print(f"DenoiseUNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"params, scale-conditioned", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)

    Zt = torch.tensor(Z)
    St = torch.tensor(S)
    t0 = time.time()
    for step in range(1, a.steps + 1):
        b = rng.choice(tr, size=a.batch, replace=False)
        x0 = Zt[b][:, None].to(DEVICE)
        sc = St[b].to(DEVICE)
        if a.scale_dropout > 0:
            # Sometimes hide the scale tag, so the model does not become
            # DEPENDENT on metadata that may be missing or wrong in the field.
            # Same reasoning as modality dropout elsewhere in this project.
            k = (torch.rand(len(sc), 1, device=DEVICE) > a.scale_dropout)
            sc = sc * k
        m = sample_holes(len(b), Z.shape[-1], rng, DEVICE)
        loss = diff.loss_cond(net, x0, m, scale=sc)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % a.log_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                vb = va[:a.batch]
                vm = sample_holes(len(vb), Z.shape[-1], rng, DEVICE)
                vl = diff.loss_cond(net, Zt[vb][:, None].to(DEVICE), vm,
                                    scale=St[vb].to(DEVICE)).item()
            net.train()
            print(f"  step {step:6d}/{a.steps}  train {loss.item():.4f}  "
                  f"val {vl:.4f}  [{(time.time()-t0)/60:.1f} min]", flush=True)

    out = LOGS / f"dem_ms_{a.tag}.pt"
    torch.save({"net": net.state_dict(), "levels": a.levels,
                "scale_cond": True}, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True,
                    help="path,metres_per_pixel,footprint_km separated by ';'")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--scale-dropout", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    main(ap.parse_args())
