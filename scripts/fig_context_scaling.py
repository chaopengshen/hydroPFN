"""Does accuracy actually RISE with more context? The canonical ICL curve.

Reproduces: figs/fig_context_scaling.png, logs/context_scaling.csv

The whole claim of an in-context model is that adding observations at inference
improves prediction WITHOUT retraining. A single before/after number cannot
show that; a monotone curve can, and a flat one would falsify it. Two sweeps
from one trained checkpoint, no training:

  own-site visits    0, 1, 2, 3, 5, 8, 12   (context sites held fixed)
                     -> at-a-station partial pooling
  context sites      0, 1, 2, 4, 8, 16      (own visits held at 0)
                     -> pure PUB: what do NEIGHBOURS alone buy?

Baselines on the same targets: the per-site power law (needs >= 3 own visits,
so it simply does not exist at the left of the curve -- which is the point) and
an attributes-only RF (flat by construction; it cannot use context at all).

Read it as:
  rising own-visit curve  -> the model uses a site's own history, as claimed
  flat own-visit curve    -> it memorised an attribute->y map; the ICL story is
                             dead regardless of what the headline gate said
  rising site curve       -> neighbours carry transferable signal
"""

from __future__ import annotations

import argparse
import sys
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import numpy as np
import pandas as pd
import torch

from hydropfn.models.measurement_pfn import VARS, HydroPFN, make_borders  # noqa: E402
from hydropfn.train.train_measurement_pfn import ATTRS, SiteStore, collate, r2  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(net, store, targets, tr_site, n_own, n_ctx, rng, batch=32):
    """Predict every target with exactly n_own own-site visits and n_ctx
    context sites."""
    preds, ys, qv = [], [], []
    for i in range(0, len(targets), batch):
        chunk = targets[i:i + batch]
        exs = []
        for si, tgt in chunk:
            vis = store.visits[si]
            own = vis[vis[:, 3] != tgt[3]]          # whole occasion excluded
            if n_own == 0:
                own = own[:0]
            elif len(own) > n_own:
                own = own[rng.choice(len(own), n_own, replace=False)]
            ctx = (rng.choice(tr_site, size=min(n_ctx, len(tr_site)),
                              replace=False) if n_ctx > 0
                   else np.array([], dtype=int))
            ex = store._pack(si, own, ctx, rng)
            ex["q_var"] = np.int64(tgt[0])
            ex["q_cov"] = np.float32(tgt[2])
            ex["y"] = np.float32(tgt[1])
            exs.append(ex)
        b = collate(exs, DEVICE)
        with torch.no_grad():
            preds.extend(net.bar.mean(net(b), b["q_var"]).cpu().numpy())
        ys.extend([float(t[1]) for _, t in chunk])
        qv.extend([int(t[0]) for _, t in chunk])
    return np.array(ys), np.array(preds), np.array(qv)


def main(table, ckpt, holdout, n_ctx_fixed, seed):
    rng = np.random.default_rng(seed + 900)
    df = pd.read_csv(table, low_memory=False)
    df = df[df.HUC2.notna() & df.site_no.notna()].reset_index(drop=True)
    df["HUC2"] = df.HUC2.apply(lambda h: f"{int(float(h)):02d}")

    st = torch.load(ckpt, map_location=DEVICE)
    d, depth = st.get("d", 128), st.get("depth", 6)
    # n_ctx must match the checkpoint's slot count; n_meas caps own visits
    store = SiteStore(df, n_ctx=max(n_ctx_fixed, 16), n_meas=12)
    borders = make_borders([df[v].to_numpy(float) for v in VARS]).to(DEVICE)
    net = HydroPFN(len(ATTRS), borders, d=d, depth=depth).to(DEVICE)
    net.load_state_dict(st["net"]); net.eval()

    te_site = np.flatnonzero(store.huc2.astype(str) == str(holdout))
    tr_site = np.flatnonzero(store.huc2.astype(str) != str(holdout))
    targets = []
    for si in te_site:
        vis = store.visits[si]
        if len(vis) < 6:
            continue
        for t in rng.choice(len(vis), size=min(3, len(vis)), replace=False):
            targets.append((int(si), vis[t]))
    print(f"holdout {holdout}: {len(targets):,} targets from "
          f"{len(set(t[0] for t in targets)):,} sites", flush=True)

    rows = []
    for n_own in [0, 1, 2, 3, 5, 8, 12]:
        y, p, qv = evaluate(net, store, targets, tr_site, n_own,
                            n_ctx_fixed, rng)
        for vi, v in enumerate(VARS):
            m = qv == vi
            if m.sum() >= 30:
                rows.append({"sweep": "own_visits", "n": n_own, "var": v,
                             "r2": r2(y[m], p[m]), "n_targets": int(m.sum())})
        print(f"  own={n_own:2d}  " + "  ".join(
            f"{v} {r2(y[qv==vi], p[qv==vi]):+.3f}"
            for vi, v in enumerate(VARS)), flush=True)

    for n_ctx in [0, 1, 2, 4, 8, 16]:
        y, p, qv = evaluate(net, store, targets, tr_site, 0, n_ctx, rng)
        for vi, v in enumerate(VARS):
            m = qv == vi
            if m.sum() >= 30:
                rows.append({"sweep": "context_sites", "n": n_ctx, "var": v,
                             "r2": r2(y[m], p[m]), "n_targets": int(m.sum())})
        print(f"  sites={n_ctx:2d}  " + "  ".join(
            f"{v} {r2(y[qv==vi], p[qv==vi]):+.3f}"
            for vi, v in enumerate(VARS)), flush=True)

    out = pd.DataFrame(rows)
    (ROOT / "logs").mkdir(exist_ok=True)
    out.to_csv(ROOT / "logs" / "context_scaling.csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"log_W": "#0072B2", "log_d": "#D55E00", "log_v": "#1B7F4B"}
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0))
    for ax, sweep, xl, ti in [
            (axes[0], "own_visits", "own-site visits in context",
             "At-a-station: the query site's OWN history"),
            (axes[1], "context_sites", "context sites (0 own visits)",
             "PUB: neighbouring sites only")]:
        sub = out[out.sweep == sweep]
        for v in VARS:
            s = sub[sub["var"] == v].sort_values("n")
            if len(s):
                ax.plot(s.n, s.r2, "o-", color=COL[v], label=v, lw=2, ms=6)
        ax.set_xlabel(xl); ax.set_ylabel("held-out R²")
        ax.set_title(ti, fontsize=11)
        ax.grid(alpha=.3); ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Does adding context at inference help?  (holdout HUC2 {holdout}, "
        "no retraining anywhere)\n"
        "A rising curve is the in-context claim; a flat one would falsify it",
        fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    (ROOT / "figs").mkdir(exist_ok=True)
    fig.savefig(ROOT / "figs" / "fig_context_scaling.png", dpi=150,
                facecolor="white")
    print(f"\nwrote {ROOT/'figs'/'fig_context_scaling.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--ckpt", default=str(ROOT / "logs" / "hydropfn_v1.pt"))
    ap.add_argument("--holdout", default="03")
    ap.add_argument("--n-ctx-fixed", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.table, a.ckpt, a.holdout, a.n_ctx_fixed, a.seed)
