"""Draw the StefaNP architecture: encoders / connector / decoders.

Reproduces: figs/fig_architecture.png

Drawn as code rather than exported from the deck so it stays in sync with the
design and can be regenerated after any change.

    python fig_architecture.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

INK, MUTE = "#1A1A1A", "#5A5F66"
ACC, GOOD, BAD, WARN = "#0B5F8A", "#1B7F4B", "#B03A2B", "#B57A0B"
BG = "#F4F6F8"


def box(ax, x, y, w, h, title, body, edge=ACC, fill="white", ts=10.5, bs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.4, edgecolor=edge,
                                facecolor=fill, zorder=2))
    ax.text(x + w / 2, y + h - 0.030, title, ha="center", va="top",
            fontsize=ts, color=edge, weight="bold", zorder=3)
    if body:
        ax.text(x + 0.012, y + h - 0.085, body, ha="left", va="top",
                fontsize=bs, color=INK, zorder=3, linespacing=1.5,
                family="DejaVu Sans Mono")


def arrow(ax, x1, y1, x2, y2, color=MUTE, lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw,
                                 color=color, zorder=1))


def main(out: str) -> None:
    fig, ax = plt.subplots(figsize=(15.5, 10.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.985, "StefaNP — a Transformer Neural Process for hydrology",
            ha="center", va="top", fontsize=19, weight="bold", color=INK)
    ax.text(0.5, 0.947,
            "d = 256 everywhere (StefaLand's width, so its weights load "
            "unchanged)   ·   B = tasks,  S = sites per task,  "
            "M = visits,  N = time patches",
            ha="center", va="top", fontsize=10.5, color=MUTE)

    # ---------------------------------------------------------- level 1
    ax.text(0.012, 0.912, "LEVEL 1 — per-site ENCODERS  (run on B·S flattened; "
            "no cross-site interaction here)", fontsize=11.5, color=ACC,
            weight="bold", va="top")
    units = [
        ("A · static", GOOD,
         "IN   (B,S,49)\nMLP  49→256→256\nOUT  (B,S,1,256)\n\n"
         "mask 15% of attrs,\nregress them back\n\n→ StefaLand weights"),
        ("B · terrain (DEM)", GOOD,
         "IN   (B,S,1,128,128)\n     10 m patches\nconv 128→16, pool\n"
         "OUT  (B,S,1..4,256)\n\nmasked recon\n(diffusion decoder)"),
        ("C · forcing-response", BAD,
         "IN   (B,S,T,V)\n     T≈3650 d, V≈10\n  + the A and B tokens\n"
         "     in the SAME seq\n16-d patches → N≈228\n     → pool 2..4\n"
         "OUT  (B,S,2..4,256)"),
        ("D · measurement", GOOD,
         "IN   (B,S,M,3)\n     var / value / cov\nsum of 3 embeddings\n"
         "OUT  (B,S,M,256)\n     M≈12\n\nmasked value"),
    ]
    x0, w, gap, y1, h1 = 0.012, 0.234, 0.016, 0.575, 0.292
    for i, (t, c, b) in enumerate(units):
        x = x0 + i * (w + gap)
        box(ax, x, y1, w, h1, t, b, edge=c)
        arrow(ax, x + w / 2, y1, x + w / 2, y1 - 0.032, c)

    ax.add_patch(FancyBboxPatch((0.012, 0.497), 0.976, 0.048,
                                boxstyle="round,pad=0.008", linewidth=1.3,
                                edgecolor=ACC, facecolor=BG, zorder=2))
    ax.text(0.5, 0.521,
            "per-site token bundle   (B, S, K, 256),   K = 4–10 tokens/site"
            "      ·      an absent modality is simply an ABSENT TOKEN "
            "(masked, never zero-filled)",
            ha="center", va="center", fontsize=10.5, color=ACC, weight="bold")
    arrow(ax, 0.5, 0.497, 0.5, 0.462, ACC, 2.0)

    # ---------------------------------------------------------- level 2
    ax.text(0.012, 0.452, "LEVEL 2 — CONNECTOR, cross-site transformer",
            fontsize=11.5, color=ACC, weight="bold", va="top")
    box(ax, 0.012, 0.286, 0.976, 0.128, "", "", edge=ACC)
    ax.text(0.5, 0.393,
            "8 layers · 8 heads · d=256 · FFN 1024",
            ha="center", fontsize=10, color=MUTE)
    ax.text(0.5, 0.363,
            "IN    (B, 1 + S·K, 256)   flatten sites, prepend [TASK], "
            "add geo-encoding, padding mask (B, 1+S·K)",
            ha="center", fontsize=10, color=INK, family="DejaVu Sans Mono")
    ax.text(0.5, 0.337,
            "OUT   (B, 1 + S·K, 256)   → summaries now carry CROSS-SITE "
            "information",
            ha="center", fontsize=10, color=INK, family="DejaVu Sans Mono")
    ax.text(0.5, 0.307,
            "permutation-invariant over sites (no positional index)   ·   "
            "query attends to context, context never attends to query",
            ha="center", fontsize=9.5, color=MUTE)
    arrow(ax, 0.5, 0.286, 0.5, 0.268, ACC, 2.0)

    # ---------------------------------------------------------- level 3
    ax.text(0.012, 0.260, "LEVEL 3 — per-site DECODERS  (loss in DATA space)",
            fontsize=11.5, color=GOOD, weight="bold", va="top")
    dec = [
        ("temporal decoder", "[ context-aware summary (K,256)\n"
         "  + unmasked patches ]\n→ (B, N_masked, 16) days"),
        ("scalar head", "[TASK] token → 64 bins\n→ p(y | query, context)\n"
         "bar distribution, per-variable borders"),
        ("DEM decoder", "summary + unmasked pixels\n→ (B,1,128,128)\n"
         "diffusion sampler"),
    ]
    wd = 0.316
    for i, (t, b) in enumerate(dec):
        x = 0.012 + i * (wd + 0.014)
        box(ax, x, 0.095, wd, 0.126, t, b, edge=GOOD, ts=10, bs=8.2)

    ax.text(0.5, 0.062,
            "Why three levels: a 40-year daily record is ~900 patch tokens — "
            "far too many for the connector, which must stay at 4–10 summary "
            "tokens per site.\nThe summary is the CONDITIONING VECTOR that "
            "carries cross-site information down to the per-site decoder; "
            "local detail never leaves the site.",
            ha="center", va="center", fontsize=9.5, color=MUTE,
            linespacing=1.6)

    ax.text(0.5, 0.036, "green = built and gated      red = not built",
            ha="center", va="center", fontsize=9, color=MUTE, style="italic")

    (ROOT / "figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figs" / out, dpi=150, facecolor="white",
                bbox_inches="tight")
    print(f"wrote {ROOT / 'figs' / out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="fig_architecture.png")
    main(ap.parse_args().out)
