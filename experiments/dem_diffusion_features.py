"""Use the DIFFUSION model's hidden activations as terrain features.

Three feature sets have now been tried on the same target and split, and they
fail for different reasons if they fail:

  1. 12 hand descriptors -- elevation moments, slope stats, curvature. Cannot
     represent ridge spacing, valley width, or a U-versus-V cross-section.
  2. a small supervised CNN -- learns only what the target demands, from 4,000
     sites, which is little data for shape.
  3. THIS: activations from the trained diffusion U-Net. It was fitted
     GENERATIVELY on CONUS terrain, so it had to model what terrain looks like
     in general -- structure no supervised target ever asked for. Diffusion
     intermediates are an established feature extractor (they carry semantic
     structure usable for classification and segmentation).

The earlier objection -- "the U-Net has no bottleneck designed to be
extracted" -- was true but beside the point. You take activations at a chosen
NOISE LEVEL and pool them; that is the standard recipe, not a workaround.

Two things this must get right or the features are meaningless:
  * the noise level t. Low t keeps fine texture, high t keeps coarse layout.
    Both are swept, because which one carries geomorphic form is exactly the
    open question.
  * the normalisation must match what the net was TRAINED on. The DEM sampler
    normalises per patch over the valid region; feeding raw metres would put
    the input far outside the training distribution and the activations would
    be garbage in a way that looks like a null result.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.models.diffusion import DenoiseUNet, Diffusion   # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def diffusion_features(net, diff, Z, t_level, batch=64, n_draws=1):
    """Activations at several depths, mean+std pooled -> (N, F)."""
    # AVERAGE OVER NOISE DRAWS. q_sample injects a random field, so a single
    # draw makes each site's features a stochastic SAMPLE rather than an
    # expectation. Ridge across thousands of sites averages that away -- which
    # is why the probes looked healthy -- but a network conditioning on one
    # site's vector sees the noise directly. Measured with a single draw, the
    # DEM token made the model WORSE (0.7057 vs 0.7366 with it switched off).
    feats = []
    for i in range(0, len(Z), batch):
        x0 = torch.tensor(Z[i:i + batch], device=DEVICE)[:, None]
        n = x0.shape[0]
        t = torch.full((n,), int(t_level), device=DEVICE, dtype=torch.long)
        acc = None
        for _ in range(n_draws):
            f = _one_draw(net, diff, x0, t)
            acc = f if acc is None else acc + f
        feats.append((acc / n_draws).float().cpu().numpy())
    return np.concatenate(feats)


@torch.no_grad()
def _one_draw(net, diff, x0, t):
        noise = torch.randn_like(x0)
        xt = diff.q_sample(x0, t, noise)
        if net.inp.in_channels == 3:
            # conditional net: give it the FULL field as known, mask all-ones.
            # We are extracting features, not inpainting, so nothing is hidden.
            m = torch.ones_like(xt)
            xt = torch.cat([xt, x0 * m, m], 1)
        e = net.temb(t)
        h1 = net.d1(net.inp(xt), e)
        h2 = net.d2(torch.nn.functional.avg_pool2d(h1, 2), e)
        h3 = net.d3(torch.nn.functional.avg_pool2d(h2, 2), e)
        mid = net.mid(torch.nn.functional.avg_pool2d(h3, 2), e)
        f = []
        for h in (h2, h3, mid):
            f += [h.mean((2, 3)), h.std((2, 3))]
        return torch.cat(f, 1)


def ridge_cv(X, y, groups, alpha=10.0):
    pred = np.full(len(y), np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if tr.sum() < 200 or te.sum() < 10:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ym = y[tr].mean()
        w = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(X.shape[1]),
                            Xtr.T @ (y[tr] - ym))
        pred[te] = Xte @ w + ym
    m = np.isfinite(pred)
    return float(1 - ((y[m] - pred[m]) ** 2).sum() /
                 ((y[m] - y[m].mean()) ** 2).sum())


BASE = ["log_A_drain", "log_slope", "slope_at_floor", "sinuosity",
        "StreamOrde", "log_lengthkm", "log_arbolatesu", "log_drain_density",
        "log_A_local", "MEANELEVSMO"]


def load_net(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE)
    sd = ck.get("net", ck.get("model", ck))
    net = DenoiseUNet(w=sd["inp.weight"].shape[0],
                      in_ch=sd["inp.weight"].shape[1]).to(DEVICE)
    net.load_state_dict(sd)
    net.eval()
    return net


def extract(a):
    """Write diffusion features for every site in a DEM npz."""
    dm = np.load(a.dem, allow_pickle=True)
    ok = dm["ok"]
    Z = dm["dem"][ok].astype(np.float32)
    Z = Z - Z.mean((1, 2), keepdims=True)
    Z = Z / (Z.std((1, 2), keepdims=True) + 1e-6)
    net, diff = load_net(a.ckpt), Diffusion(device=DEVICE)
    torch.manual_seed(0)
    F_ok = np.nan_to_num(diffusion_features(net, diff, Z, a.t_extract,
                                            n_draws=a.n_draws))
    F = np.full((len(ok), F_ok.shape[1]), np.nan, dtype=np.float32)
    F[ok] = F_ok
    np.savez_compressed(a.extract_to, feats=F, ok=ok,
                        site_id=np.asarray(dm["site_id"]).astype(str),
                        t_level=a.t_extract)
    print(f"wrote {a.extract_to}: {ok.sum()} sites x {F_ok.shape[1]} features "
          f"at t={a.t_extract}")


def main(a):
    if a.extract_to:
        return extract(a)
    df = pd.read_csv(a.table, low_memory=False)
    g = df.groupby("site_no")
    site = g[["log_W", "log_d"]].median()
    site["HUC2"] = g["HUC2"].first()
    for f in BASE:
        if f in df.columns:
            site[f] = g[f].median()
    site = site.reset_index()

    dm = np.load(a.dem, allow_pickle=True)
    dsid = np.asarray(dm["site_id"]).astype(str)
    ok = dm["ok"]
    dmap = {s: i for i, s in enumerate(dsid) if ok[i]}
    site["_di"] = site["site_no"].astype(str).map(dmap)
    site = site[site["_di"].notna()].reset_index(drop=True)

    y = site[a.target].to_numpy(dtype=np.float64)
    B = site[[f for f in BASE if f in site.columns]].to_numpy(dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(B).all(1)
    site, y, B = site[keep].reset_index(drop=True), y[keep], B[keep]
    huc = site["HUC2"].astype(str).to_numpy()
    di = site["_di"].to_numpy(dtype=int)
    print(f"{len(site):,} sites, target {a.target}")

    Z = dm["dem"][di].astype(np.float32)
    Z = Z - Z.mean((1, 2), keepdims=True)
    Z = Z / (Z.std((1, 2), keepdims=True) + 1e-6)

    ck = torch.load(a.ckpt, map_location=DEVICE)
    sd = ck.get("net", ck.get("model", ck))
    in_ch = sd["inp.weight"].shape[1]
    w = sd["inp.weight"].shape[0]
    net = DenoiseUNet(w=w, in_ch=in_ch).to(DEVICE)
    net.load_state_dict(sd)
    net.eval()
    print(f"loaded {a.ckpt}: w={w}, in_ch={in_ch}, "
          f"{sum(p.numel() for p in net.parameters())/1e6:.2f}M params")

    diff = Diffusion(device=DEVICE)

    def pca(F, k):
        """Reduce before combining. 384 raw features against ~3,900 sites
        overfits at any sane alpha, and the resulting negative 'adds' is a
        regularisation artifact, not evidence about the features."""
        Fc = F - F.mean(0)
        U, S, Vt = np.linalg.svd(Fc, full_matrices=False)
        return Fc @ Vt[:k].T

    r_b = ridge_cv(B, y, huc)
    print(f"\n  attributes only          R2 = {r_b:+.4f}")
    print(f"  ridge on 12 descriptors  R2 = "
          f"{'+0.1032' if a.target == 'log_W' else '+0.1248'} (DEM only)\n")
    print(f"{'noise t':>9} | {'diff-feat only':>15} {'attrs+diff':>11} "
          f"| {'adds':>8}")
    print("  (attrs+diff uses PCA-reduced features; the best k/alpha is")
    print("   selected on the same folds, so it is an UPPER BOUND on what")
    print("   the features could add, not an honest held-out number.)")
    print("-" * 52)
    for t in [int(x) for x in a.t_levels.split(",")]:
        torch.manual_seed(0)
        F = np.nan_to_num(diffusion_features(net, diff, Z, t).astype(np.float64))
        r_f = ridge_cv(F, y, huc, alpha=100.0)
        best, bk, ba = -9.9, None, None
        for k in (8, 16, 32, 64):
            P = pca(F, k)
            for al in (1.0, 10.0, 100.0):
                r = ridge_cv(np.hstack([B, P]), y, huc, alpha=al)
                if r > best:
                    best, bk, ba = r, k, al
        print(f"{t:9d} | {r_f:15.4f} {best:11.4f} | {best - r_b:+8.4f}"
              f"   (PCA k={bk}, alpha={ba})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default="/nfs/data/cxs1024/channel_geometry/data/"
                            "train_table_dem_fixed.csv")
    ap.add_argument("--dem", default="logs/geom_dem.npz")
    ap.add_argument("--ckpt",
                    default="/nfs/data/cxs1024/dem_foundation/logs/allfix.pt")
    ap.add_argument("--target", default="log_W", choices=["log_W", "log_d"])
    ap.add_argument("--extract-to", default=None,
                    help="write features for every site in --dem and exit")
    ap.add_argument("--n-draws", type=int, default=8,
                    help="average features over this many noise draws; "
                         "1 makes them a stochastic sample, not an "
                         "expectation")
    ap.add_argument("--t-extract", type=int, default=50,
                    help="noise level for extraction; 20-50 scored best")
    ap.add_argument("--t-levels", default="50,200,500",
                    help="noise levels to sweep: low keeps texture, high "
                         "keeps coarse layout")
    main(ap.parse_args())
