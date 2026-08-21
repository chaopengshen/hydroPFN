"""Does the L1 net DELETE a linear feature, or merely HEDGE it?

Reproduces: figs/fig_hedging.png

Motivating observation: in `figs/fig_inpaint_examples.png`, the flat row has a
channel entering the mask from above.  A human continues it straight through.
No model does.  Three different reasons are suspected:

  harmonic  provably cannot -- it minimises integral|grad z|^2, and continuing an
            incision INCREASES that energy.  It is succeeding at the wrong goal.
  IDW       has no notion of direction: the nearest valid pixels lie all around
            the rim, so averaging them destroys orientation.
  U-Net     L1 gives the pixelwise conditional MEDIAN.  If the channel's lateral
            position is uncertain by more than its own width, it is present in
            under half the plausible continuations AT ANY GIVEN PIXEL, so the
            median says "no channel".  Hedging over position does not blur a
            narrow feature -- it deletes it.

That last one makes a falsifiable prediction: the net should leave a BROAD,
SHALLOW trough where the channel could be (hedged mass spread over plausible
positions), not a flat surface.  Hillshade cannot show this because it is
dominated by local gradient; an elevation transect can.

  wide shallow trough -> the model knows, and is hedging.  A generative
                         objective would sharpen it into a real channel.
  nothing at all      -> it never registered the feature; that is a capacity or
                         receptive-field problem needing a different fix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from hydropfn.models.inpaint import PConvUNet, fill_harmonic, fill_idw  # noqa: E402
from hydropfn.metrics.terrain import hillshade  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main(patches: str, ckpt: str, width: int, seed: int, out: str) -> None:
    if not Path(ckpt).exists():
        raise SystemExit(
            f"checkpoint not found: {ckpt}\n"
            "run test_inpaint_probe.py first (it writes logs/unet_l1.pt); do "
            "not point this at the pre-reorg channel_geometry checkpoint -- "
            "that is a differently-trained net (review B4).")
    z = np.load(patches)
    P, tiles = z["patches"], z["tile"]
    size = P.shape[1]
    Z = (P - P.mean((1, 2), keepdims=True)).astype(np.float32)
    relief = P.max((1, 2)) - P.min((1, 2))

    net = PConvUNet(width).to(DEVICE)
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()

    # A flat patch WITH a strong linear feature: low relief overall, but a
    # high-gradient tail.  That is the case the figure showed failing.
    rng = np.random.default_rng(seed)
    from hydropfn.metrics.terrain import slope_mag
    sl = np.array([np.percentile(slope_mag(Z[i]), 99) for i in range(len(Z))])
    cand = np.flatnonzero((relief < np.percentile(relief, 40)) &
                          (sl > np.percentile(sl, 80)))
    print(f"{len(cand)} low-relief patches with a strong linear feature")

    rows = []
    for i in cand[:6]:
        # Horizontal band mask, so the feature must be continued VERTICALLY.
        m = np.ones((size, size), np.float32)
        r0, h = size // 2 - 20, 40
        m[r0:r0 + h, :] = 0.0
        t = Z[i]
        with torch.no_grad():
            xb = torch.tensor(t * m).view(1, 1, size, size).to(DEVICE)
            mb = torch.tensor(m).view(1, 1, size, size).to(DEVICE)
            pu = net(xb, mb).cpu().numpy()[0, 0]
        pu = np.where(m < 0.5, pu, t)
        rows.append({"i": int(i), "mask": m, "truth": t, "unet": pu,
                     "harmonic": fill_harmonic(t, m), "idw": fill_idw(t, m),
                     "r0": r0, "h": h})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ex = rows[0]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7),
                             gridspec_kw={"height_ratios": [1.35, 1]})
    names = ["truth", "harmonic", "IDW", "U-Net (L1)"]
    fields = [ex["truth"], ex["harmonic"], ex["idw"], ex["unet"]]
    for ax, nm, f in zip(axes[0], names, fields):
        ax.imshow(hillshade(f), cmap="gray", vmin=0, vmax=1)
        ax.axhline(ex["r0"], color="#D55E00", lw=1.2)
        ax.axhline(ex["r0"] + ex["h"], color="#D55E00", lw=1.2)
        ax.set_title(nm, fontsize=11); ax.set_xticks([]); ax.set_yticks([])

    # Elevation transect ALONG the middle of the hole, across the feature.
    mid = ex["r0"] + ex["h"] // 2
    ax = axes[1, 0]
    for nm, f, col in (("truth", ex["truth"], "#222222"),
                       ("harmonic", ex["harmonic"], "#0072B2"),
                       ("IDW", ex["idw"], "#E69F00"),
                       ("U-Net (L1)", ex["unet"], "#D55E00")):
        ax.plot(np.arange(size) * 10, f[mid], lw=1.8 if nm == "truth" else 1.4,
                color=col, label=nm, ls="-" if nm == "truth" else "--")
    ax.set_xlabel("distance across patch (m)"); ax.set_ylabel("elevation (m)")
    ax.set_title(f"transect through the hole centre (row {mid})", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.25)

    # The diagnostic: DEPARTURE from a straight line across the hole.  A hedged
    # feature shows as a broad negative bump; a deleted one shows as ~zero.
    for k, (nm, f, col) in enumerate((("truth", ex["truth"], "#222222"),
                                      ("harmonic", ex["harmonic"], "#0072B2"),
                                      ("U-Net (L1)", ex["unet"], "#D55E00"))):
        ax = axes[1, k + 1]
        prof = f[mid]
        base = np.linspace(prof[0], prof[-1], size)
        ax.plot(np.arange(size) * 10, prof - base, color=col, lw=1.6)
        ax.axhline(0, color="#999999", lw=.8)
        ax.set_title(f"{nm}: departure from a straight fill", fontsize=9.5)
        ax.set_xlabel("distance (m)"); ax.grid(alpha=.25)
        if k == 0:
            ax.set_ylabel("elevation anomaly (m)")

    fig.suptitle("Does the L1 net hedge a linear feature, or delete it?\n"
                 "orange lines bound the masked band; the feature must be "
                 "continued across it", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    (ROOT / "figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figs" / out, dpi=150, facecolor="white")
    print(f"wrote {ROOT/'figs'/out}")

    # Quantify: amplitude of the anomaly inside the hole, truth vs fills.
    print("\namplitude of the in-hole elevation anomaly (m, p95-p05 of "
          "departure from a straight fill):")
    for nm in ("truth", "harmonic", "idw", "unet"):
        amps = []
        for r in rows:
            mid_r = r["r0"] + r["h"] // 2
            prof = r[nm][mid_r]
            base = np.linspace(prof[0], prof[-1], size)
            d = prof - base
            amps.append(np.percentile(d, 95) - np.percentile(d, 5))
        print(f"  {nm:9s} {np.median(amps):6.2f}")
    print("\n  truth >> unet ~ harmonic  -> the feature is deleted, not hedged")
    print("  unet between truth and harmonic -> hedging; a sampler would sharpen it")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(ROOT / "logs" / "dem_patches.npz"))
    ap.add_argument("--ckpt", default=str(ROOT / "logs" / "unet_l1.pt"))
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="fig_hedging.png")
    a = ap.parse_args()
    main(a.patches, a.ckpt, a.width, a.seed, a.out)
