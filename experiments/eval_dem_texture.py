"""Distributional terrain metrics per scale -- the ones a blur cannot win.

The reconstruction table in eval_dem_multiscale.py reports masked-region RMSE,
and RMSE structurally rewards smoothing: an L1/L2-optimal fill is the
conditional MEAN, and the mean of plausible terrains is flat. The original DEM
sampler failed exactly there -- pointwise numbers looked fine while psd_ratio
sat at 2.90 with visible seams at every hole rim.

So this uses the metrics that arm was always judged on, where **closer to 1.0
is better** because they compare the STATISTICS of the fill against the
statistics of the truth:

  psd_ratio      spectral power inside the hole, pred / truth.
                 < 1 = too smooth (the blur failure). > 1 = too rough.
  vario_ratio_L  semivariogram at lag L, pred / truth. Same reading.
  slope_w1       Wasserstein distance between slope DISTRIBUTIONS. Lower is
                 better; 0 means the fill has the right roughness.

Against HARMONIC interpolation, which is the honest competitor: it is
provably the flattest surface matching the boundary, so it is what "just
interpolate sensibly" achieves. A sampler that cannot beat harmonic on
distributional metrics has no reason to exist.

Uses full DDIM sampling, not a one-shot x0 estimate -- the one-shot estimate
IS the conditional mean and would be guaranteed to look smooth.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hydropfn.metrics.terrain import score
from hydropfn.models.diffusion import DenoiseUNet, Diffusion, harmonic_torch
from hydropfn.train.train_dem_multiscale import (load_levels, sample_holes,
                                                 sample_holes_orig)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
def rerank(cands, known, mask):
    """Pick the draw whose hole texture best matches the SURROUNDING terrain.

    The criterion uses only the KNOWN region, never the truth inside the hole,
    so this is a legitimate inference-time choice and not a peek at the answer.
    Terrain roughness is locally stationary, so the observed rim is a fair
    estimate of what the hole should look like -- and the failure mode we are
    fighting is systematic SMOOTHNESS, which this directly penalises.
    """
    from hydropfn.metrics.terrain import slope_mag
    best, best_d = None, None
    k = mask[0, 0].cpu().numpy() > 0.5
    for c in cands:
        z = c[0, 0].cpu().numpy()
        s_out = slope_mag(z)[k]
        s_in = slope_mag(z)[~k]
        if s_out.size == 0 or s_in.size == 0:
            return cands[0]
        # match the MEDIAN slope inside the hole to that outside it
        d = abs(float(np.median(s_in)) - float(np.median(s_out)))
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best


# NOTE the key names. semivariogram() labels lags by lag*PX where PX is a
# module constant of 10 m, so the keys are gamma_10m / gamma_80m regardless of
# the patch's ACTUAL resolution -- at the 100 m and 400 m levels the label
# understates the true lag by 10x and 40x. The RATIO is still valid (the same
# pixel lags are used for truth and prediction); only the printed name is
# misleading. Asking for "vario_ratio_1"/"vario_ratio_8" returned NaN at every
# cell, which is how the mistake surfaced.
KEYS = ["psd_ratio", "vario_ratio_10m", "vario_ratio_80m", "slope_w1",
        "elev_rmse"]


def summarise(rows):
    out = {}
    for k in KEYS:
        v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
        out[k] = float(np.median(v)) if len(v) else np.nan
    return out


def main(a):
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    specs = [(t.split(",")[0], float(t.split(",")[1]), float(t.split(",")[2]))
             for t in a.levels.split(";")]
    P, S, src = load_levels(specs)
    ck0 = torch.load(a.ckpt, map_location="cpu")
    vnorm = bool(ck0.get("valid_norm", False)) or a.force_valid_norm
    if vnorm:
        # match the checkpoint's training: mean-removed raw metres here, then
        # per-sample normalisation from VALID pixels after the mask is drawn.
        # Whole-patch normalisation at eval leaks the truth inside the hole.
        Z = P - P.mean((1, 2), keepdims=True)
        print("checkpoint uses valid-region normalisation; matching it")
    else:
        Z = ((P - P.mean((1, 2), keepdims=True))
             / (P.std((1, 2), keepdims=True) + 1e-6))
    perm = np.random.default_rng(a.seed).permutation(len(Z))
    va = perm[:max(256, int(0.05 * len(Z)))]

    ck = torch.load(a.ckpt, map_location=DEVICE)
    # evaluate the EMA weights when present -- the original sampler was always
    # scored at EMA, and raw weights sample measurably rougher/noisier
    sd = ck.get("ema", ck["net"])
    if "ema" in ck:
        print("using EMA weights")
    net = DenoiseUNet(w=sd["inp.weight"].shape[0],
                      in_ch=sd["inp.weight"].shape[1],
                      scale_cond=any(k.startswith("smlp.") for k in sd)
                      ).to(DEVICE)
    net.load_state_dict(sd)
    net.eval()
    # the checkpoint's parameterisation, not a default: decoding a
    # v-trained net as eps produces garbage that LOOKS like a model failure
    par = ck0.get("param", "eps")
    diff = Diffusion(device=DEVICE, param=par)
    print(f"param={par}")
    print(f"\nloaded {a.ckpt}   (DDIM {a.steps} steps)\n")

    hdr = f"{'scale':>9} {'method':>9} |" + "".join(f"{k:>15}" for k in KEYS)
    print(hdr)
    print("-" * len(hdr))
    for path, mpp, km in specs:
        tag = f"{km:g}km"
        sel = va[src[va] == tag][:a.n]
        if len(sel) < 16:
            continue
        m_rows, h_rows = [], []
        for i in range(0, len(sel), a.batch):
            b = sel[i:i + a.batch]
            x0 = torch.tensor(Z[b])[:, None].to(DEVICE)
            sc = torch.tensor(S[b]).to(DEVICE)
            holes = (sample_holes_orig if a.holes == "orig"
                     else sample_holes)
            m = holes(len(b), Z.shape[-1], rng, DEVICE)
            if vnorm:
                v = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
                mu_ = (x0 * m).sum(dim=(1, 2, 3), keepdim=True) / v
                sd_ = ((((x0 - mu_) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                       / v).sqrt().clamp(min=0.5)
                x0 = ((x0 - mu_) / sd_).clamp(-8.0, 8.0)
            with torch.no_grad():
                harm = harmonic_torch(x0 * m, m)
                if a.residual:
                    # the net was trained on the RESIDUAL over harmonic, and
                    # was given harmonic as its conditioning channel -- so
                    # sampling must supply the same context and add the
                    # harmonic back
                    ctx = torch.cat([harm, m], dim=1)
                    # `known` is the ZERO residual, NOT x0*m. The net was
                    # trained on xt = noised(x0 - harmonic), and the residual
                    # is exactly 0 on valid pixels -- harmonic equals truth
                    # there. Forcing raw elevations into the residual state
                    # (the first version of this call) put the valid region
                    # metres off-distribution at every DDIM step; the
                    # signature was vario_ratio OVERSHOOTING to 2.6 with
                    # elev_rmse worse than harmonic. The original sample_k in
                    # test_diffusion_sampler.py states this in its comment.
                    zk = torch.zeros_like(x0)
                    draws = [harm + diff.ddim_cond(
                        net, zk, m, steps=a.steps, scale=sc,
                        ctx_override=ctx) for _ in range(a.best_of)]
                else:
                    draws = [diff.ddim_cond(net, x0 * m, m, steps=a.steps,
                                            scale=sc)
                             for _ in range(a.best_of)]
                if a.best_of > 1:
                    pred = torch.cat([rerank([d[j:j+1] for d in draws],
                                             x0[j:j+1], m[j:j+1])
                                      for j in range(len(b))], 0)
                else:
                    pred = draws[0]
            for j in range(len(b)):
                t_ = x0[j, 0].cpu().numpy()
                mk = m[j, 0].cpu().numpy()
                m_rows.append(score(t_, pred[j, 0].cpu().numpy(), mk))
                h_rows.append(score(t_, harm[j, 0].cpu().numpy(), mk))
        for name, rows in (("model", m_rows), ("harmonic", h_rows)):
            v = summarise(rows)
            print(f"{tag:>9} {name:>9} |" +
                  "".join(f"{v[k]:15.4f}" for k in KEYS))
    print("\n  psd_ratio and vario_ratio: CLOSER TO 1.0 IS BETTER.")
    print("  < 1 means too smooth -- the failure mode RMSE cannot see.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--holes", choices=["orig", "mixed"], default="orig",
                    help="hole distribution to SCORE on. The 0.810 anchor "
                         "was measured on the ORIGINAL distribution "
                         "(squares 1.5-11% + strokes); scoring on the "
                         "harder mixed one and comparing to 0.810 is a "
                         "cross-protocol comparison")
    ap.add_argument("--force-valid-norm", action="store_true",
                    help="for checkpoints from the ORIGINAL script, which "
                         "trained valid-normalised but predate the "
                         "valid_norm flag in the checkpoint dict")
    ap.add_argument("--residual", action="store_true",
                    help="checkpoint predicts the residual over harmonic")
    ap.add_argument("--best-of", type=int, default=1,
                    help="draw K samples and rerank by rim-matched slope")
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
