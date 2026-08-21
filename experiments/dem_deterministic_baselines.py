"""Can a deterministic net produce terrain TEXTURE if the loss asks for it?

Reproduces: figs/fig_inpaint_texture.png, logs/inpaint_texture.csv

The first version of this probe trained on L1 and scored pointwise slope RMSE,
and concluded harmonic interpolation was best.  Both halves of that were biased
toward smoothness:

  * L1 yields the CONDITIONAL MEAN, which for terrain is smooth.  Structural,
    not a budget problem.
  * pointwise slope RMSE rewards flatness -- harmonic had the largest flattening
    bias (-0.0080) and still the best score, while IDW had nearly the right
    roughness (-0.0029) and the worst score, because its roughness was in the
    wrong places.

So this version changes both sides:

  LOSS     L1 + moment matching on |grad z| + a log-spectral term, so wrong
           roughness statistics are penalised rather than wrong pixels.
  METRICS  slope-distribution W1, short-wavelength PSD ratio, and semivariogram
           ratios -- all position-free, alongside the pointwise numbers so the
           two families can be compared directly.

Run (suntzu; gpuenv.sh handles the user-anaconda path, nvjitlink
LD_LIBRARY_PATH and CUDA_DEVICE_ORDER traps -- do not `conda activate`):

    source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
    $PY -u test_inpaint_probe.py --w-texture 1.0 --w-spec 0.3

Numbers are seed-dependent (two same-config runs differed by ~0.11 in
psd_ratio), so quote medians across >= 3 seeds, never a single run.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


from hydropfn.models.inpaint import (PConvUNet, fill_harmonic, fill_idw,  # noqa: E402
                     make_mask, texture_loss)
from hydropfn.metrics.terrain import hillshade, score  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def figure(examples, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cols = ["truth", "masked", "harmonic", "IDW", "U-Net L1", "U-Net +texture"]
    fig, axes = plt.subplots(len(examples), len(cols),
                             figsize=(2.7 * len(cols), 3.4 * len(examples)))
    axes = np.atleast_2d(axes)
    for r, ex in enumerate(examples):
        m = ex["mask"]
        panels = [ex["truth"], None, ex["harmonic"], ex["idw"],
                  ex["unet_l1"], ex["unet_tex"]]
        for c, (name, p) in enumerate(zip(cols, panels)):
            ax = axes[r, c]
            hs = (np.where(m > 0.5, hillshade(ex["truth"]), np.nan)
                  if p is None else hillshade(p))
            ax.imshow(hs, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ys, xs = np.where(m < 0.5)
            if len(ys):
                ax.add_patch(Rectangle((xs.min() - .5, ys.min() - .5),
                                       xs.max() - xs.min() + 1,
                                       ys.max() - ys.min() + 1,
                                       fill=False, ec="#D55E00", lw=1.3))
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(name, fontsize=10.5, color="#222222")
            if c == 0:
                ax.set_ylabel(f"{ex['terrain']}\nrelief {ex['relief']:.0f} m",
                              fontsize=9, color="#222222")
            if c >= 2 and name in ex["psd"]:
                v = ex["psd"][name]
                ax.set_xlabel("PSD ratio n/a" if not np.isfinite(v)
                              else f"PSD ratio {v:.2f}",
                              fontsize=8.5, color="#666666")
    fig.suptitle("Masked DEM inpainting on held-out 1° tiles — hillshade\n"
                 "PSD ratio = short-wavelength power vs truth; 1.0 = right "
                 "roughness, <<1 = too smooth",
                 fontsize=11.5, color="#222222")
    # h_pad: the per-panel PSD xlabel was being clipped by the row beneath it.
    fig.tight_layout(rect=[0, 0, 1, 0.93], h_pad=1.6)
    fig.savefig(out_png, dpi=150, facecolor="white")
    print(f"wrote {out_png}")


def train(Z, tr, size, epochs, batch, width, rng, w_tex, w_spec, tag, tseed):
    # B3: numpy was seeded but torch never was, so net init and batch order
    # differed every run -- the measured run-to-run spread (~0.11 psd_ratio)
    # exceeded the effect under study.  Each net gets its own derived seed so
    # the L1 and texture nets do not share an init.
    torch.manual_seed(tseed)
    net = PConvUNet(width).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    X = torch.tensor(Z[tr]).unsqueeze(1)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X))
        acc = np.zeros(3); nb = 0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            zb = X[idx].to(DEVICE)
            mb = torch.tensor(make_mask(len(idx), size, rng)).unsqueeze(1).to(DEVICE)
            pred = net(zb * mb, mb)
            l1 = F.l1_loss(pred * (1 - mb), zb * (1 - mb)) / (1 - mb).mean()
            loss = l1
            mom = spec = torch.tensor(0.0)
            if w_tex > 0 or w_spec > 0:
                mom, spec = texture_loss(pred, zb, mb)
                loss = l1 + w_tex * mom + w_spec * spec
            opt.zero_grad(); loss.backward(); opt.step()
            acc += [l1.item(), float(mom), float(spec)]; nb += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            a = acc / nb
            print(f"  [{tag}] epoch {ep+1}/{epochs}  L1 {a[0]:.3f}  "
                  f"moment {a[1]:.4f}  spec {a[2]:.3f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    # BatchNorm must leave training mode: predict() runs batches of 1, and in
    # train mode BN would normalise by that single sample's statistics, which
    # inflates error and manufactures noise (observed: elev RMSE 5.55 vs 1.11,
    # vario ratio 3.05 -- i.e. 3x too much roughness, from this alone).
    net.eval()
    return net


def predict(net, truth, m, size):
    with torch.no_grad():
        xb = torch.tensor(truth * m).view(1, 1, size, size).to(DEVICE)
        mb = torch.tensor(m).view(1, 1, size, size).to(DEVICE)
        p = net(xb, mb).cpu().numpy()[0, 0]
    return np.where(m < 0.5, p, truth)


def main(patches, epochs, batch, width, seed, n_eval, w_tex, w_spec,
         skip_figure=False):
    z = np.load(patches)
    P, tiles = z["patches"], z["tile"]
    n, size = P.shape[0], P.shape[1]
    rng = np.random.default_rng(seed)

    ut = np.array(sorted(set(tiles))); rng.shuffle(ut)
    test_t = set(ut[:max(1, len(ut) // 5)])
    te = np.array([t in test_t for t in tiles]); tr = ~te
    print(f"{n:,} patches {size}x{size} | train {tr.sum():,} / test {te.sum():,} "
          f"(whole-tile split) | {DEVICE}")

    Z = (P - P.mean((1, 2), keepdims=True)).astype(np.float32)
    relief = P.max((1, 2)) - P.min((1, 2))

    ck = ROOT / "logs"
    ck.mkdir(exist_ok=True)
    net_l1 = train(Z, tr, size, epochs, batch, width, rng, 0.0, 0.0, "L1",
                   1000 * seed + 1)
    torch.save(net_l1.state_dict(), ck / "unet_l1.pt")
    net_tx = train(Z, tr, size, epochs, batch, width, rng, w_tex, w_spec,
                   "texture", 1000 * seed + 2)
    torch.save(net_tx.state_dict(), ck / "unet_tex.pt")

    # Evaluation gets its OWN rng: (a) `[:n_eval]` of the test indices was a
    # geographically biased sample -- the patch array is ordered by tile name,
    # and 3DEP tile names sort by latitude, so "the first 250" meant "the
    # southernmost test tiles" (review B2); (b) drawing eval masks from the
    # training stream meant changing --epochs silently changed the eval masks.
    erng = np.random.default_rng(seed + 500)
    te_idx = np.flatnonzero(te)
    ev = np.sort(erng.choice(te_idx, size=min(n_eval, len(te_idx)),
                             replace=False))
    rows = []
    for j, i in enumerate(ev):
        # Alternate mask kinds EXPLICITLY and record which, so the psd_ratio
        # dilution on stroke-mask bounding boxes (review M1) is visible in the
        # output instead of silently averaged in.
        kind = "square" if j % 2 == 0 else "stroke"
        m = make_mask(1, size, erng, kind=kind)[0]
        t = Z[i]
        fills = {"harmonic": fill_harmonic(t, m), "idw": fill_idw(t, m),
                 "unet_l1": predict(net_l1, t, m, size),
                 "unet_tex": predict(net_tx, t, m, size)}
        for name, p in fills.items():
            s = score(t, p, m)
            s.update(method=name, relief=float(relief[i]), mask_kind=kind)
            rows.append(s)
    df = pd.DataFrame(rows)
    (ROOT / "logs").mkdir(exist_ok=True)
    df.to_csv(ROOT / "logs" / f"inpaint_texture_s{seed}.csv", index=False)

    q = df.relief.quantile([.33, .66]).to_list()
    df["terrain"] = np.where(df.relief < q[0], "flat",
                             np.where(df.relief < q[1], "moderate", "rough"))
    pd.set_option("display.width", 220)
    print("\n=== POINTWISE (structurally favours smooth fills) ===")
    print(df.groupby("method")[["elev_rmse", "slope_rmse", "slope_bias",
                                "rim_jump"]].median()
            .to_string(float_format=lambda v: f"{v:+.4f}"))
    print("\n=== DISTRIBUTIONAL (can reward real texture) ===")
    print(df.groupby("method")[["slope_w1", "psd_ratio", "vario_ratio_10m",
                                "vario_ratio_80m"]].median()
            .to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n  psd_ratio / vario_ratio: 1.0 = right roughness, <1 = too smooth")
    print("  slope_w1: lower = slope distribution closer to truth")

    # Stroke-mask crops below the 60% hole-fraction guard are NaN by design;
    # this shows how many survived and whether the two kinds agree.
    print("\n  psd_ratio by mask kind (dilution check, review M1):")
    print(df.pivot_table(index="method", columns="mask_kind",
                         values="psd_ratio", aggfunc=["median", "count"])
            .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\n=== by terrain class (median) ===")
    print(df.pivot_table(index="method", columns="terrain",
                         values=["psd_ratio", "slope_rmse"], aggfunc="median")
            .to_string(float_format=lambda v: f"{v:.3f}"))

    if skip_figure:
        return

    # worked examples for the figure
    exs, rl = [], relief[ev]
    qq = np.quantile(rl, [0.15, 0.5, 0.9])
    # One rng hoisted OUT of the loop: recreating default_rng(seed+7) per row
    # gave all three rows the identical mask (review M2).  Square masks only,
    # so the orange bbox drawn in the figure is exactly the hole and the PSD
    # caption is never a diluted stroke-bbox value.
    frng = np.random.default_rng(seed + 7)
    for lab, tgt in (("flat", qq[0]), ("moderate", qq[1]), ("rough", qq[2])):
        i = int(ev[np.argmin(np.abs(rl - tgt))])
        m = make_mask(1, size, frng, kind="square")[0]
        t = Z[i]
        f = {"harmonic": fill_harmonic(t, m), "idw": fill_idw(t, m),
             "unet_l1": predict(net_l1, t, m, size),
             "unet_tex": predict(net_tx, t, m, size)}
        # Annotate with the SAME hole-restricted metric the table reports.
        # Calling psd_ratio on the whole patch instead put ~0.8 under every
        # panel -- the identical surround swamps the hole and flatters every
        # method toward 1.0, which is exactly the dilution bug the scoring path
        # already fixes.  Two code paths, one metric.
        exs.append({"terrain": lab, "relief": float(relief[i]), "mask": m,
                    "truth": t, **f,
                    "psd": {"harmonic": score(t, f["harmonic"], m)["psd_ratio"],
                            "IDW": score(t, f["idw"], m)["psd_ratio"],
                            "U-Net L1": score(t, f["unet_l1"], m)["psd_ratio"],
                            "U-Net +texture": score(t, f["unet_tex"], m)["psd_ratio"]}})
    (ROOT / "figs").mkdir(exist_ok=True)
    figure(exs, ROOT / "figs" / "fig_inpaint_texture.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(ROOT / "logs" / "dem_patches.npz"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=250)
    ap.add_argument("--w-texture", type=float, default=1.0)
    ap.add_argument("--w-spec", type=float, default=0.3)
    ap.add_argument("--skip-figure", action="store_true",
                    help="metrics only; use for the seed sweep so only one "
                         "seed regenerates the figure")
    a = ap.parse_args()
    main(a.patches, a.epochs, a.batch, a.width, a.seed, a.n_eval,
         a.w_texture, a.w_spec, a.skip_figure)
