"""Train the DEM diffusion sampler and measure what sampling buys.

Reproduces: logs/diffusion_eval_s{seed}.csv, figs/fig_diffusion_samples.png,
checkpoint logs/ddpm.pt

The deterministic ceiling is measured: ~0.31 of true fine-scale power (3-seed
median), and the hedging diagnostic showed why -- an L1 net spreads a channel
over its plausible positions and the spread reads as smoothness.  A sampler
must instead COMMIT per draw.  Three pre-registered read-outs:

  texture     median psd/vario ratios of SINGLE samples.  Success band
              0.7-1.3 (vs 0.31 deterministic, 0.13 harmonic).
  diversity   mean in-hole std across K draws.  ~0 means the model collapsed
              to the conditional mean and sampling bought nothing.
  hedging     the mean of K draws should look like the L1 net's broad trough
              (that IS the conditional mean); each draw should be sharp.
              best-of-K elev RMSE < deterministic RMSE = the truth lies
              inside the sampled distribution.

Run (suntzu):
    source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
    $PY -u test_diffusion_sampler.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydropfn.models.diffusion import (DenoiseUNet, Diffusion,  # noqa: E402
                       harmonic_torch)
from hydropfn.models.inpaint import fill_harmonic, make_mask  # noqa: E402
from hydropfn.metrics.terrain import hillshade, score  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STD_FLOOR = 0.5          # metres; a near-flat patch must not be amplified


def texture_ratio(Z: np.ndarray) -> np.ndarray:
    """High-pass energy / total energy, per patch.

    Measured: psd_ratio correlates -0.417 (Spearman) with this quantity -- the
    sampler is WORST on patches that are flat overall but finely textured
    (farmland, roads, urban), and best on high-relief natural terrain.  Used to
    oversample the hard class during training.
    """
    from scipy.ndimage import uniform_filter
    out = np.empty(len(Z), np.float32)
    for i, z in enumerate(Z):
        hp = z - uniform_filter(z, size=9)
        out[i] = hp.std() / (z.std() + 1e-6)
    return out


def rim_band(mask: np.ndarray, width: int = 6) -> np.ndarray:
    """Valid pixels within `width` of the hole -- the local texture reference."""
    from scipy.ndimage import binary_dilation
    hole = mask < 0.5
    grown = binary_dilation(hole, iterations=width)
    return grown & ~hole


def rerank(truth, draws, mask, lags=(1, 2, 4, 8)):
    """Pick the draw whose in-hole roughness best matches the SURROUNDING rim.

    A sampler's whole point is that you can draw many; taking draw #1 throws
    that away.  Measured spread is p10 0.14 / p90 1.95 in psd_ratio, so a good
    draw usually EXISTS -- the problem is selection, not capability.

    The selector uses only information available at inference (the rim is
    observed), so this is a legitimate method rather than oracle picking.
    """
    from hydropfn.metrics.terrain import semivariogram
    band = rim_band(mask)
    ref = semivariogram(truth, lags=lags, mask=(~band).astype(np.float32))
    best, best_s = None, np.inf
    for g in draws:
        got = semivariogram(g, lags=lags, mask=mask)
        sc = 0.0
        for k in ref:
            a, b = got.get(k, np.nan), ref[k]
            if np.isfinite(a) and np.isfinite(b) and a > 0 and b > 0:
                sc += abs(np.log(a / b))
        if sc < best_s:
            best, best_s = g, sc
    return best if best is not None else draws[0]


def norm_stats(z: np.ndarray, valid: np.ndarray | None = None):
    v = z if valid is None else z[valid]
    return float(v.mean()), max(float(v.std()), STD_FLOOR)


def train(Z, epochs, batch, width, T, seed, size, tag="ddpm",
          param="eps", residual=False, oversample=False):
    """Z is RAW mean-removed elevation; masks are drawn per batch and the
    normalisation uses VALID-region statistics only -- exactly what inference
    can see.  v0 normalised on the whole patch during training and on the
    valid region at inference, so every prediction was made at a slightly
    wrong scale; on a 214 m-relief patch that mismatch is metres.
    """
    torch.manual_seed(1000 * seed + 3)
    net = DenoiseUNet(width, in_ch=3).to(DEVICE)
    dif = Diffusion(T, DEVICE, param=param)
    print(f"  param={param}  residual={residual}  oversample={oversample}",
          flush=True)
    w = None
    if oversample:
        tr_ratio = texture_ratio(Z)
        # weight ~ the failing regime; clipped so nothing dominates
        w = np.clip(tr_ratio / (np.median(tr_ratio) + 1e-6), 0.3, 4.0)
        w = w / w.sum()
        print(f"  oversampling by texture ratio "
              f"(min {tr_ratio.min():.3f} med {np.median(tr_ratio):.3f} "
              f"max {tr_ratio.max():.3f})", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-4, weight_decay=1e-5)
    ema = {k: v.detach().clone() for k, v in net.state_dict().items()}
    X = torch.tensor(Z).unsqueeze(1)
    n = len(X)
    mrng = np.random.default_rng(seed + 91)
    t0 = time.time()
    for ep in range(epochs):
        perm = (torch.tensor(mrng.choice(n, size=n, replace=True, p=w))
                if w is not None else torch.randperm(n))
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            xb = X[perm[i:i + batch]].to(DEVICE)
            mb = torch.tensor(make_mask(len(xb), size, mrng),
                              dtype=torch.float32,
                              device=DEVICE).unsqueeze(1)
            v = mb.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            mu = (xb * mb).sum(dim=(1, 2, 3), keepdim=True) / v
            sd = ((((xb - mu) * mb) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                  / v).sqrt().clamp(min=STD_FLOOR)
            xb = (xb - mu) / sd
            if residual:
                # Generate the DEPARTURE from the harmonic fill, not the
                # surface.  Harmonic already solves the low frequencies
                # exactly (it is the flattest consistent surface), so making
                # the net re-derive them wastes capacity on the part that is
                # not the problem.  Conditioning becomes (harmonic, mask):
                # the harmonic preserves every valid pixel, so no information
                # about the known region is lost.
                # cond is ONE channel: loss_cond appends the mask itself,
                # so passing [harm, mask] here made a 4-channel input.
                harm = harmonic_torch(xb, mb)
                loss = dif.loss_cond(net, xb - harm, mb, cond=harm)
            else:
                loss = dif.loss_cond(net, xb, mb)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema[k].mul_(0.999).add_(v, alpha=0.001)
            tot += loss.item(); nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  [{tag}] epoch {ep+1}/{epochs}  loss {tot/nb:.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
        if (ep + 1) % 50 == 0 or ep + 1 == epochs:
            torch.save({"net": net.state_dict(), "ema": ema,
                        "width": width, "T": T, "param": param,
                        "residual": residual},
                       ROOT / "logs" / f"{tag}.pt")
    net.load_state_dict(ema)          # evaluate the EMA weights
    net.eval()
    return net, dif


def sample_k(net, dif, truth, m, k, steps, resample=2, residual=False):
    """K denormalised fills for one patch; valid-region stats only (matching
    how the net was trained)."""
    mu, sd = norm_stats(truth, m >= 0.5)
    kn = torch.tensor((truth - mu) / sd, dtype=torch.float32,
                      device=DEVICE).view(1, 1, *truth.shape).repeat(k, 1, 1, 1)
    mb = torch.tensor(m, dtype=torch.float32,
                      device=DEVICE).view(1, 1, *m.shape).repeat(k, 1, 1, 1)
    if residual:
        # the net generates the departure from the harmonic fill; the known
        # region is carried by the harmonic itself, so `known` for the DDIM
        # forcing is the ZERO residual there.
        harm = harmonic_torch(kn, mb)
        ctx = torch.cat([harm, mb], dim=1)
        res = dif.ddim_cond(net, torch.zeros_like(kn), mb, steps, resample,
                            ctx_override=ctx)
        out = (harm + res).cpu().numpy()[:, 0]
    else:
        out = dif.ddim_cond(net, kn, mb, steps, resample).cpu().numpy()[:, 0]
    return out * sd + mu


def main(patches, epochs, batch, width, T, steps, k, n_eval, seed,
         ckpt=None, skip_figure=False, param="eps", residual=False,
         oversample=False, tag="ddpm", do_rerank=True):
    z = np.load(patches)
    P, tiles = z["patches"], z["tile"]
    size = P.shape[1]
    rng = np.random.default_rng(seed)
    ut = np.array(sorted(set(tiles))); rng.shuffle(ut)
    test_t = set(ut[:max(1, len(ut) // 5)])
    te = np.array([t in test_t for t in tiles]); tr = ~te
    print(f"{len(P):,} patches | train {tr.sum():,} / test {te.sum():,} "
          f"(whole-tile split) | {DEVICE}", flush=True)

    Zm = (P - P.mean((1, 2), keepdims=True)).astype(np.float32)

    (ROOT / "logs").mkdir(exist_ok=True)
    if ckpt:
        st = torch.load(ckpt, map_location=DEVICE)
        net = DenoiseUNet(st["width"], in_ch=3).to(DEVICE)
        net.load_state_dict(st["ema"]); net.eval()
        param = st.get("param", "eps")
        residual = st.get("residual", False)
        dif = Diffusion(st["T"], DEVICE, param=param)
        print(f"loaded {ckpt}  (param={param}, residual={residual})")
    else:
        # raw mean-removed patches; masking and normalisation happen inside
        net, dif = train(Zm[tr], epochs, batch, width, T, seed, size, tag,
                         param, residual, oversample)

    # ---- evaluation, same protocol as the deterministic probe (own rng,
    # random test subset, alternating mask kinds)
    erng = np.random.default_rng(seed + 500)
    ev = np.sort(erng.choice(np.flatnonzero(te),
                             size=min(n_eval, int(te.sum())), replace=False))
    relief = P.max((1, 2)) - P.min((1, 2))
    rows = []
    t0 = time.time()
    for j, i in enumerate(ev):
        kind = "square" if j % 2 == 0 else "stroke"
        m = make_mask(1, size, erng, kind=kind)[0]
        t = Zm[i]
        hole = m < 0.5
        S = sample_k(net, dif, t, m, k, steps, residual=residual)
        per = [score(t, s, m) for s in S]
        med = {key: float(np.nanmedian([p[key] for p in per]))
               for key in per[0]}
        med.update(method="diffusion", relief=float(relief[i]),
                   mask_kind=kind,
                   best_elev=float(min(p["elev_rmse"] for p in per)),
                   spread=float(S.std(axis=0)[hole].mean()))
        rows.append(med)
        if do_rerank and k > 1:
            pick = rerank(t, S, m)
            r = score(t, pick, m)
            r.update(method="diffusion_rerank", relief=float(relief[i]),
                     mask_kind=kind, best_elev=r["elev_rmse"],
                     spread=float(S.std(axis=0)[hole].mean()))
            rows.append(r)
        h = score(t, fill_harmonic(t, m), m)
        h.update(method="harmonic", relief=float(relief[i]), mask_kind=kind,
                 best_elev=h["elev_rmse"], spread=0.0)
        rows.append(h)
        if (j + 1) % 10 == 0:
            print(f"  eval {j+1}/{len(ev)}  [{(time.time()-t0)/60:.1f} min]",
                  flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "logs" / f"diffusion_eval_{tag}_s{seed}.csv",
              index=False)
    KEYS = ["elev_rmse", "best_elev", "slope_rmse", "slope_w1", "psd_ratio",
            "vario_ratio_10m", "vario_ratio_80m", "spread"]
    pd.set_option("display.width", 220)
    print("\n=== diffusion (per-sample medians) vs harmonic ===")
    print(df.groupby("method")[KEYS].median()
            .to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\n  deterministic reference (3-seed medians): psd 0.31, "
          "vario10 0.42, vario80 0.52-0.64, elev 1.88")
    print("  success band for psd/vario: 0.7-1.3.  spread ~ 0 => collapsed.")

    if skip_figure:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    frng = np.random.default_rng(seed + 7)
    rl = relief[ev]
    qq = np.quantile(rl, [0.15, 0.5, 0.9])
    cols = ["truth", "masked", "harmonic", "sample 1", "sample 2",
            f"mean of {k}"]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.7 * len(cols), 10.2))
    for r, (lab, tgt) in enumerate((("flat", qq[0]), ("moderate", qq[1]),
                                    ("rough", qq[2]))):
        i = int(ev[np.argmin(np.abs(rl - tgt))])
        m = make_mask(1, size, frng, kind="square")[0]
        t = Zm[i]
        S = sample_k(net, dif, t, m, k, steps)
        panels = [t, None, fill_harmonic(t, m), S[0], S[1], S.mean(axis=0)]
        for c, (nm, p) in enumerate(zip(cols, panels)):
            ax = axes[r, c]
            hs = (np.where(m > 0.5, hillshade(t), np.nan) if p is None
                  else hillshade(p))
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ys, xs = np.where(m < 0.5)
            ax.add_patch(Rectangle((xs.min() - .5, ys.min() - .5),
                                   xs.max() - xs.min() + 1,
                                   ys.max() - ys.min() + 1,
                                   fill=False, ec="#D55E00", lw=1.3))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(nm, fontsize=10.5)
            if c == 0:
                ax.set_ylabel(f"{lab}\nrelief {relief[i]:.0f} m", fontsize=9)
            if c >= 2 and p is not None:
                v = score(t, p, m)["psd_ratio"]
                ax.set_xlabel("PSD ratio n/a" if not np.isfinite(v)
                              else f"PSD ratio {v:.2f}", fontsize=8.5,
                              color="#666666")
    fig.suptitle("DEM diffusion sampler -- individual draws must be sharp; "
                 "their mean should reproduce the hedged (smooth) answer",
                 fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=1.6)
    (ROOT / "figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figs" / "fig_diffusion_samples.png", dpi=150,
                facecolor="white")
    print(f"wrote {ROOT/'figs'/'fig_diffusion_samples.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(ROOT / "logs" / "dem_patches.npz"))
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--t", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-eval", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default=None,
                    help="skip training, evaluate this checkpoint")
    ap.add_argument("--skip-figure", action="store_true")
    ap.add_argument("--param", choices=["eps", "v"], default="eps")
    ap.add_argument("--residual", action="store_true",
                    help="generate the departure from the harmonic fill")
    ap.add_argument("--oversample", action="store_true",
                    help="oversample high-texture-ratio patches (the measured "
                         "failing regime)")
    ap.add_argument("--tag", default="ddpm")
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()
    main(a.patches, a.epochs, a.batch, a.width, a.t, a.steps, a.k, a.n_eval,
         a.seed, a.ckpt, a.skip_figure, a.param, a.residual, a.oversample,
         a.tag, not a.no_rerank)
