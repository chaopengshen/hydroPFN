"""Did multi-scale DEM pretraining actually work, and at EVERY scale?

Two distinct questions that a single training loss cannot separate:

  1. did it learn to inpaint?   -> masked-region error vs trivial baselines
  2. did it learn ALL THREE scales, or collapse onto the easy one?

(2) is the one that matters here. Trained jointly, a model can minimise the
average loss by fitting whichever scale is easiest and treating the others as
noise. Averaged over the corpus that looks like success. Broken out per scale
it does not.

Baselines, both trivial and both necessary:
  * MEAN fill -- predict the patch mean inside the hole. In per-patch
    normalised units its RMSE is roughly the hole's own std, so ~1.0. Anything
    not beating this has learned nothing.
  * EDGE fill -- propagate the hole's boundary mean inward. Much stronger,
    because terrain is smooth: it is the "just interpolate" competitor, and it
    is the one a sampler has to beat to justify existing.

Reported on HELD-OUT patches with the same hole distribution used in training.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hydropfn.models.diffusion import DenoiseUNet, Diffusion
from hydropfn.train.train_dem_multiscale import load_levels, sample_holes

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def edge_fill(x0, m):
    """Fill each hole with the mean of the KNOWN pixels adjacent to it."""
    out = x0.clone()
    known = m > 0.5
    for i in range(len(x0)):
        k = known[i, 0]
        if k.all() or (~k).all():
            continue
        vals = x0[i, 0][k]
        # dilate the hole by one pixel and take known pixels in the rim
        hole = (~k).float()[None, None]
        dil = torch.nn.functional.max_pool2d(hole, 3, 1, 1)[0, 0] > 0.5
        rim = dil & k
        fill = x0[i, 0][rim].mean() if rim.any() else vals.mean()
        out[i, 0][~k] = fill
    return out


def main(a):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    specs = [(t.split(",")[0], float(t.split(",")[1]), float(t.split(",")[2]))
             for t in a.levels.split(";")]
    P, S, src = load_levels(specs)
    mu = P.mean((1, 2), keepdims=True)
    sd = P.std((1, 2), keepdims=True) + 1e-6
    Z = (P - mu) / sd

    # SAME split as training: same seed, same permutation, first 5% is val
    perm = np.random.default_rng(a.seed).permutation(len(Z))
    n_val = max(256, int(0.05 * len(Z)))
    va = perm[:n_val]

    ck = torch.load(a.ckpt, map_location=DEVICE)
    sd_ = ck["net"]
    net = DenoiseUNet(w=sd_["inp.weight"].shape[0],
                      in_ch=sd_["inp.weight"].shape[1],
                      scale_cond=any(k.startswith("smlp.") for k in sd_)
                      ).to(DEVICE)
    net.load_state_dict(sd_)
    net.eval()
    diff = Diffusion(device=DEVICE)
    print(f"\nloaded {a.ckpt}\n")

    print(f"{'scale':>10} {'n':>6} | {'MEAN fill':>10} {'EDGE fill':>10} "
          f"{'MODEL':>10} | {'vs edge':>9}")
    print("-" * 66)
    rows = []
    for path, mpp, km in specs:
        tag = f"{km:g}km"
        sel = va[src[va] == tag]
        if len(sel) < 32:
            print(f"{tag:>10} {len(sel):>6} | too few held-out patches")
            continue
        sel = sel[:a.max_per_scale]
        # BATCHED: 256 patches at once is ~7 GB and OOMs on a shared card.
        # Errors accumulate as sums of squares, so batching changes nothing.
        se_mean = se_edge = se_mod = 0.0
        n_hole = 0
        for i in range(0, len(sel), a.batch):
            b = sel[i:i + a.batch]
            x0 = torch.tensor(Z[b])[:, None].to(DEVICE)
            sc = torch.tensor(S[b]).to(DEVICE)
            m = sample_holes(len(b), Z.shape[-1], rng, DEVICE)
            hole = (m < 0.5)
            with torch.no_grad():
                # one-shot x0 estimate at a fixed noise level: far cheaper
                # than full sampling and enough to RANK the scales
                t = torch.full((len(b),), a.t_eval, device=DEVICE,
                               dtype=torch.long)
                xt = diff.q_sample(x0, t, torch.randn_like(x0))
                eps = net(torch.cat([xt, x0 * m, m], 1), t, sc)
                pred = diff._to_eps_x0(eps, xt, t)[1]
                ef = edge_fill(x0, m)
            se_mean += float((x0[hole] ** 2).sum())
            se_edge += float(((ef - x0)[hole] ** 2).sum())
            se_mod += float(((pred - x0)[hole] ** 2).sum())
            n_hole += int(hole.sum())
        r_mean = (se_mean / n_hole) ** 0.5
        r_edge = (se_edge / n_hole) ** 0.5
        r_mod = (se_mod / n_hole) ** 0.5
        rows.append((tag, len(sel), r_mean, r_edge, r_mod))
        print(f"{tag:>10} {len(sel):>6} | {r_mean:10.4f} {r_edge:10.4f} "
              f"{r_mod:10.4f} | {100*(r_edge-r_mod)/r_edge:+8.1f}%")

    print()
    if rows and all(r[4] < r[3] for r in rows):
        print("  BEATS edge-fill at EVERY scale -- multi-scale training did")
        print("  not collapse onto one level.")
    elif rows:
        bad = [r[0] for r in rows if r[4] >= r[3]]
        print(f"  FAILS to beat edge-fill at: {', '.join(bad)}")
        print("  -> the model collapsed onto the easier scale(s); the joint")
        print("     average hid it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--t-eval", type=int, default=200)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--sample", action="store_true",
                    help="full DDIM sampling instead of a one-shot estimate")
    ap.add_argument("--max-per-scale", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
