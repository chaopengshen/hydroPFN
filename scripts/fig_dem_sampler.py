"""Side-by-side: TRUE terrain vs SAMPLER-GENERATED terrain.

Reproduces: figs/fig_sampler_truth_vs_generated.png

The probe figure shows method columns; this one answers the simpler question
"does the generated map look like the real map" at a size where you can judge
it.  Four rows per patch column:

  1  true elevation            same colour scale within a column
  2  generated elevation       (hole filled by the conditional sampler)
  3  true hillshade, ZOOMED to the hole + margin
  4  generated hillshade, same zoom

Rows 3-4 are the ones that matter: at full-patch scale almost anything looks
right because 90% of the pixels are shared with truth by construction.

    source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
    $PY fig_sampler_truth_vs_generated.py --ckpt ../../logs/ddpm.pt
"""

from __future__ import annotations

import argparse
import sys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import torch

from hydropfn.models.diffusion import DenoiseUNet, Diffusion  # noqa: E402
from hydropfn.models.inpaint import fill_harmonic, make_mask  # noqa: E402
from hydropfn.metrics.terrain import hillshade, score  # noqa: E402
from hydropfn.train.train_dem_sampler import rerank, sample_k  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STD_FLOOR = 0.5


def main(patches, ckpt, n_col, steps, seed, out):
    z = np.load(patches)
    P, tiles = z["patches"], z["tile"]
    size = P.shape[1]
    rng = np.random.default_rng(seed)
    ut = np.array(sorted(set(tiles))); rng.shuffle(ut)
    test_t = set(ut[:max(1, len(ut) // 5)])
    te = np.flatnonzero(np.array([t in test_t for t in tiles]))

    st = torch.load(ckpt, map_location=DEVICE)
    net = DenoiseUNet(st["width"], in_ch=3).to(DEVICE)
    net.load_state_dict(st["ema"]); net.eval()
    param = st.get("param", "eps")
    residual = st.get("residual", False)
    dif = Diffusion(st["T"], DEVICE, param=param)
    print(f"loaded {ckpt}  (param={param}, residual={residual})")

    Zm = (P - P.mean((1, 2), keepdims=True)).astype(np.float32)
    relief = P.max((1, 2)) - P.min((1, 2))

    # Pick the MEDIAN patch of each relief tercile, not evenly-spaced
    # quantiles.  Quantiles 0.15..0.95 put 3 of 5 panels in the flat/moderate
    # regime, where the sampler is measurably weakest (psd 0.45 flat vs 0.84
    # rough) -- so the figure read as smoother than the 60-patch median and
    # misrepresented the model in BOTH directions.  One exemplar per terrain
    # class, each captioned with its class median, shows the real behaviour:
    # texture quality depends strongly on how much texture there is to find.
    rl = relief[te]
    edges = np.quantile(rl, [0, 1 / 3, 2 / 3, 1.0])
    picks, cls = [], []
    for k, name in enumerate(["flat", "moderate", "rough"]):
        sel = (rl >= edges[k]) & (rl <= edges[k + 1])
        sub = te[sel]
        picks.append(int(sub[np.argmin(np.abs(relief[sub] -
                                              np.median(relief[sub])))]))
        cls.append(name)
    # class medians measured over the full 60-patch evaluation
    CLASS_MED = {"flat": (0.452, 0.564), "moderate": (0.652, 0.778),
                 "rough": (0.841, 0.874)}

    frng = np.random.default_rng(seed + 3)
    n_draw = 3
    cols = []
    for ci, i in enumerate(picks):
        m = make_mask(1, size, frng, kind="square")[0]
        t = Zm[i]
        valid = m >= 0.5
        mu = float(t[valid].mean())
        sd = max(float(t[valid].std()), STD_FLOOR)
        g = sample_k(net, dif, t, m, n_draw, steps, residual=residual)
        pick = rerank(t, g, m)
        cols.append({"truth": t, "draws": g, "mask": m, "cls": cls[ci],
                     "pick": pick, "harm": fill_harmonic(t, m),
                     "relief": float(relief[i]),
                     "pick_score": score(t, pick, m),
                     "harm_score": score(t, fill_harmonic(t, m), m),
                     "scores": [score(t, gi, m) for gi in g]})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    nrow = 3 + n_draw
    fig, axes = plt.subplots(nrow, len(cols),
                             figsize=(3.4 * len(cols), 3.15 * nrow))
    axes = np.atleast_2d(axes)
    for c, ex in enumerate(cols):
        t, m = ex["truth"], ex["mask"]
        ys, xs = np.where(m < 0.5)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        pad = max(10, (y1 - y0) // 2)
        zy = slice(max(0, y0 - pad), min(size, y1 + pad + 1))
        zx = slice(max(0, x0 - pad), min(size, x1 + pad + 1))

        panels = ([(hillshade(t), "TRUTH", None),
                   (hillshade(ex["harm"]), "harmonic", ex["harm_score"])] +
                  [(hillshade(g), f"draw {k+1}", s)
                   for k, (g, s) in enumerate(zip(ex["draws"],
                                                  ex["scores"]))] +
                  [(hillshade(ex["pick"]), "RERANK pick", ex["pick_score"])])
        for r, (img, lab, s) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img[zy, zx], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
            ax.add_patch(Rectangle(
                (x0 - zx.start - .5, y0 - zy.start - .5),
                x1 - x0 + 1, y1 - y0 + 1, fill=False, ec="#D55E00", lw=1.6))
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(lab, fontsize=11, color="#222222")
            if s is not None:
                ax.set_xlabel(f"PSD {s['psd_ratio']:.2f}   "
                              f"RMSE {s['elev_rmse']:.2f} m",
                              fontsize=8.5, color="#666666")
        cm = CLASS_MED[ex["cls"]]
        axes[0, c].set_title(
            f"{ex['cls'].upper()} — relief {ex['relief']:.0f} m\n"
            f"class median (60 patches): PSD {cm[0]:.2f}, vario10 {cm[1]:.2f}",
            fontsize=10)

    fig.suptitle(
        "Best recipe: residual parameterisation + best-of-K reranking\n"
        "Draws vary; RERANK picks the one whose in-hole roughness matches "
        "the rim\n"
        "60 held-out patches: elev 0.891 m vs harmonic 2.028  ·  "
        "PSD 0.810  ·  variogram 0.842 / 0.980",
        fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90], h_pad=1.0)
    (ROOT / "figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figs" / out, dpi=150, facecolor="white")
    print(f"wrote {ROOT/'figs'/out}")
    for ex in cols:
        ps = [s["psd_ratio"] for s in ex["scores"]]
        rs = [s["elev_rmse"] for s in ex["scores"]]
        print(f"  {ex['cls']:8s} relief {ex['relief']:6.0f} m   "
              f"psd per draw {np.round(ps, 2)}   "
              f"RMSE per draw {np.round(rs, 2)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(ROOT / "logs" / "dem_patches.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "logs" / "residual.pt"))
    ap.add_argument("--n-col", type=int, default=5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig_sampler_best.png")
    a = ap.parse_args()
    main(a.patches, a.ckpt, a.n_col, a.steps, a.seed, a.out)
