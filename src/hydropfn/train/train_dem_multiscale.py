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

from hydropfn.models.diffusion import (DenoiseUNet, Diffusion,
                                       harmonic_torch)
from hydropfn.paths import LOGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_levels(specs):
    """[(path, metres_per_pixel, footprint_km)] -> patches, scale, src, ids.

    `ids` carries each patch's site_id where the level file has one (the
    CAMELS gauge corpora), else "" -- the attribute-conditioning path keys
    on it, and everything without an id gets the null attr token.
    """
    P, S, src, ids = [], [], [], []
    for path, mpp, km in specs:
        z = np.load(path, allow_pickle=True)
        a = z["patches"] if "patches" in z else z["dem"]
        ok = z["ok"] if "ok" in z else np.ones(len(a), bool)
        sid = (np.asarray(z["site_id"]).astype(str) if "site_id" in z
               else np.full(len(a), "", dtype=object))
        a, sid = a[ok].astype(np.float32), sid[ok]
        good = np.isfinite(a).all((1, 2))
        a, sid = a[good], sid[good]
        P.append(a)
        S.append(np.tile([np.log10(mpp), np.log10(km)],
                         (len(a), 1)).astype(np.float32))
        src.append(np.full(len(a), f"{km:g}km", dtype=object))
        ids.append(sid)
        print(f"  {path.split('/')[-1]:26s} {len(a):6,d} patches  "
              f"{mpp:>5.0f} m/px  {km:>5.1f} km")
    return (np.concatenate(P), np.concatenate(S), np.concatenate(src),
            np.concatenate(ids))


def sample_holes_orig(n, size, rng, device):
    """The hole distribution the SUCCESSFUL sampler trained on: modest
    squares (size/8..size/3, i.e. ~1.5-11% of the patch) and stroke masks.
    Ported from make_mask in dem_foundation/src/lib/inpaint.py.

    My replacement distribution hid up to 25% of the area in one block and
    hid ~90% of the patch on a fifth of draws. For holes that large the
    conditional mean -- a blur -- IS the loss-optimal answer, so the model was
    being trained toward the exact failure mode the sampler exists to avoid.
    Hole size is task design, not a nuisance parameter.
    """
    m = np.ones((n, size, size), dtype=np.float32)
    for i in range(n):
        if rng.random() < 0.5:                              # square
            h = int(rng.integers(size // 8, size // 3))
            r0 = int(rng.integers(0, size - h))
            c0 = int(rng.integers(0, size - h))
            m[i, r0:r0 + h, c0:c0 + h] = 0.0
        else:                                               # strokes
            x, y = (int(v) for v in rng.integers(0, size, 2))
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
    return torch.tensor(m, device=device).unsqueeze(1)


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
    P, S, src, ids = load_levels(specs)

    # ---- attribute conditioning table (site_id -> z-scored statics + bit)
    Att = np.zeros((len(P), 1), np.float32)      # placeholder when unused
    n_attr = 0
    if a.attr_npz:
        za = np.load(a.attr_npz, allow_pickle=True)
        amat = za["attrs"].astype(np.float64)
        amu = np.nanmean(amat, 0)
        asd = np.nanstd(amat, 0) + 1e-9
        amat = (np.where(np.isfinite(amat), amat, amu) - amu) / asd
        table = {str(sid_): amat[i]
                 for i, sid_ in enumerate(np.asarray(za["site_id"]).astype(str))}
        n_attr = amat.shape[1] + 1               # + presence bit
        Att = np.zeros((len(P), n_attr), np.float32)
        hit = 0
        for i, sid_ in enumerate(ids):
            v = table.get(str(sid_))
            if v is not None:
                Att[i, :-1] = v
                Att[i, -1] = 1.0
                hit += 1
        print(f"  attrs attached to {hit:,} of {len(P):,} patches "
              f"({n_attr - 1} statics + presence bit)")
        if a.attr_holdout:
            pref = tuple(a.attr_holdout.split(","))
            drop = np.array([str(i_).startswith(pref) and len(str(i_)) == 8
                             for i_ in ids])
            print(f"  HOLDOUT: excluding {drop.sum():,} gauge patches "
                  f"(HUC2 {a.attr_holdout}) from training entirely")
            keep = ~drop
            P, S, src, ids, Att = P[keep], S[keep], src[keep], ids[keep],                 Att[keep]
        np.savez(LOGS / f"attr_norm_{a.tag}.npz", mu=amu, sd=asd)
    print(f"  TOTAL {len(P):,} patches\n")

    if a.valid_norm:
        # Mean-removed RAW metres; the per-sample normalisation happens after
        # the mask is drawn, from VALID pixels only -- exactly what inference
        # can see. Whole-patch normalisation leaks the hole's own statistics
        # into the input (and at eval time leaks truth), which the original
        # sampler's v0->v1 fix documented as costing metres on high-relief
        # patches. STD_FLOOR stops a near-flat patch being amplified.
        Z = P - P.mean((1, 2), keepdims=True)
    else:
        # PER-PATCH normalisation. Absolute elevation differs by kilometres
        # across CONUS and would dominate the loss.
        mu = P.mean((1, 2), keepdims=True)
        sd = P.std((1, 2), keepdims=True) + 1e-6
        Z = (P - mu) / sd

    n = len(Z)
    perm = rng.permutation(n)
    n_val = max(256, int(0.05 * n))
    va, tr = perm[:n_val], perm[n_val:]
    print(f"train {len(tr):,}  val {len(va):,}")

    net = DenoiseUNet(w=a.width, in_ch=3, scale_cond=True,
                      attr_cond=n_attr).to(DEVICE)
    # v-prediction was part of the winning recipe (allfix.pt stores
    # param='v'), and for a texture-motivated sampler it is arguably the
    # core of it: eps-prediction underweights the low-SNR steps where fine
    # texture lives, which is the blur mechanism itself. Every earlier
    # multi-scale run silently used the eps default -- the SIXTH element
    # lost in the port.
    diff = Diffusion(device=DEVICE, param=a.param)
    print(f"DenoiseUNet {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"params, scale-conditioned", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.steps, pct_start=0.05)

    # EMA: the original sampler kept an exponential moving average of the
    # weights and EVALUATED the EMA, not the raw weights. Standard for
    # diffusion sample quality; its absence was one of the ported-recipe gaps.
    ema = ({k_: v_.detach().clone() for k_, v_ in net.state_dict().items()}
           if a.ema > 0 else None)
    Zt = torch.tensor(Z)
    St = torch.tensor(S)
    At = torch.tensor(Att) if n_attr else None
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
        at = None
        if n_attr:
            at = At[b].to(DEVICE)
            if a.attr_dropout > 0:
                # null out attrs on a fraction of bearing rows, so the model
                # keeps an unconditional mode and the paired attrs-vs-null
                # eval reads one checkpoint (classifier-free style)
                k = (torch.rand(len(at), 1, device=DEVICE)
                     > a.attr_dropout).float()
                at = at * k
        holes = sample_holes_orig if a.orig_masks else sample_holes
        m = holes(len(b), Z.shape[-1], rng, DEVICE)
        if a.valid_norm:
            v = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            mu_ = (x0 * m).sum(dim=(1, 2, 3), keepdim=True) / v
            sd_ = ((((x0 - mu_) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                   / v).sqrt().clamp(min=a.std_floor)
            # Clamp AFTER normalising. When the known pixels are flat but the
            # hole holds relief (a valley window onto mountains -- common at
            # the 51 km level, absent from the original's single-scale
            # corpus), valid-region sd is tiny and the hole normalises to
            # hundreds. Under v-prediction the TARGET contains
            # -(1-ab).sqrt()*x0, so the loss explodes (val 299-570 at the
            # start of the first parity-v run). The sampler already clamps x0
            # to +-6 at every DDIM step, so targets beyond that range are
            # unreachable at inference and training on them is pure waste.
            x0 = ((x0 - mu_) / sd_).clamp(-8.0, 8.0)
        if a.residual:
            # RESIDUAL PARAMETERISATION. Predict the field MINUS its
            # harmonic fill, with the harmonic handed over as the
            # conditioning channel. Harmonic is provably the flattest
            # surface matching the boundary, so it already carries the
            # smooth part; the net then only has to produce TEXTURE, and
            # cannot score well by predicting zero the way it can when
            # predicting the field directly. Without this the plain
            # multi-scale model reached psd_ratio 0.141 against a target
            # of 1.0 -- a blur that still beat harmonic (0.017) because
            # harmonic is maximally smooth.
            h = harmonic_torch(x0 * m, m)
            loss = diff.loss_cond(net, x0 - h, m, cond=h, scale=sc,
                                  attrs=at)
        else:
            loss = diff.loss_cond(net, x0, m, scale=sc, attrs=at)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if a.ema > 0:
            with torch.no_grad():
                for k_, v_ in net.state_dict().items():
                    ema[k_].mul_(a.ema).add_(v_, alpha=1 - a.ema)
        if step % a.log_every == 0 or step == 1:
            net.eval()
            with torch.no_grad():
                vb = va[:a.batch]
                vm = sample_holes(len(vb), Z.shape[-1], rng, DEVICE)
                vx = Zt[vb][:, None].to(DEVICE)
                vsc = St[vb].to(DEVICE)
                if a.valid_norm:
                    vv = vm.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
                    vmu = (vx * vm).sum(dim=(1, 2, 3), keepdim=True) / vv
                    vsd = ((((vx - vmu) * vm) ** 2)
                           .sum(dim=(1, 2, 3), keepdim=True)
                           / vv).sqrt().clamp(min=a.std_floor)
                    vx = ((vx - vmu) / vsd).clamp(-8.0, 8.0)
                if a.residual:
                    vh = harmonic_torch(vx * vm, vm)
                    vl = diff.loss_cond(net, vx - vh, vm, cond=vh,
                                        scale=vsc).item()
                else:
                    vl = diff.loss_cond(net, vx, vm, scale=vsc).item()
            net.train()
            print(f"  step {step:6d}/{a.steps}  train {loss.item():.4f}  "
                  f"val {vl:.4f}  [{(time.time()-t0)/60:.1f} min]", flush=True)

    out = LOGS / f"dem_ms_{a.tag}.pt"
    save = {"net": net.state_dict(), "levels": a.levels,
            "scale_cond": True, "residual": a.residual,
            "valid_norm": a.valid_norm, "param": a.param,
            "attr_cond": n_attr, "attr_npz": a.attr_npz,
            "attr_holdout": a.attr_holdout}
    if ema is not None:
        save["ema"] = ema
    torch.save(save, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True,
                    help="path,metres_per_pixel,footprint_km separated by ';'")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--param", choices=["eps", "v"], default="eps",
                    help="the winning single-scale recipe used v")
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay; 0 disables. The original evaluated EMA")
    ap.add_argument("--orig-masks", action="store_true",
                    help="the successful sampler's hole distribution")
    ap.add_argument("--valid-norm", action="store_true",
                    help="normalise from VALID pixels only, after masking")
    ap.add_argument("--std-floor", type=float, default=0.5,
                    help="metres; a near-flat patch must not be amplified")
    ap.add_argument("--residual", action="store_true",
                    help="predict the residual over a harmonic fill")
    ap.add_argument("--scale-dropout", type=float, default=0.1)
    ap.add_argument("--attr-npz", default=None,
                    help="npz with site_id + attrs; patches whose level file "
                         "carries a matching site_id are attribute-"
                         "conditioned, everything else gets the null token")
    ap.add_argument("--attr-dropout", type=float, default=0.3)
    ap.add_argument("--attr-holdout", default=None,
                    help="comma HUC2 prefixes; gauge patches matching are "
                         "EXCLUDED from training for the paired eval")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="v1")
    main(ap.parse_args())
