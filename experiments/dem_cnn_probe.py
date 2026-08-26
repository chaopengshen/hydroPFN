"""Do LEARNED terrain features beat 12 hand-built descriptors?

This separates two very different conclusions that the descriptor probes cannot
tell apart:

  * DEM carries no useful signal for this target, or
  * elevation moments discard the signal DEM carries.

Twelve moments cannot represent ridge spacing, valley width, or the U-versus-V
cross-section that marks a glaciated valley. A small CNN on the raw patch can.
So: same target, same split, same protocol -- only the feature extractor
changes. If the CNN cannot beat ridge-on-descriptors, the descriptors were not
the bottleneck and the DEM pathway is genuinely empty for this target.

Deliberately NOT reusing the diffusion U-Net. That network was built to
GENERATE terrain, and its only bottleneck is a 128x16x16 spatial feature map --
pooling it to a vector is exactly the operation that would discard the
cross-sectional structure this test is looking for. Describing and generating
want different bottlenecks.

Leave-HUC2-out, matching experiments/dem_geometry_probe.py.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TerrainNet(nn.Module):
    """128 -> 4 conv stages -> global pool -> scalar. ~0.2M params."""

    def __init__(self, w: int = 32, n_out: int = 1):
        super().__init__()
        def blk(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, 1, 1), nn.GroupNorm(8, co), nn.SiLU(),
                nn.Conv2d(co, co, 3, 1, 1), nn.GroupNorm(8, co), nn.SiLU(),
                nn.AvgPool2d(2))
        self.f = nn.Sequential(blk(1, w), blk(w, w * 2), blk(w * 2, w * 2),
                               blk(w * 2, w * 4))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(w * 4 * 2, 128),
                                  nn.SiLU(), nn.Linear(128, n_out))

    def forward(self, x):
        h = self.f(x)                                   # (B, 4w, 8, 8)
        # mean AND std pooling: std keeps texture information that mean alone
        # destroys, and texture is the whole point of using the raw patch
        h = torch.cat([h.mean((2, 3)), h.std((2, 3))], 1)
        return self.head(h)


def main(a):
    df = pd.read_csv(a.table, low_memory=False)
    g = df.groupby("site_no")
    site = g[["log_W", "log_d"]].median()
    site["HUC2"] = g["HUC2"].first()
    site = site.reset_index()

    dm = np.load(a.dem, allow_pickle=True)
    dsid = np.asarray(dm["site_id"]).astype(str)
    ok = dm["ok"]
    dmap = {s: i for i, s in enumerate(dsid) if ok[i]}
    site["_di"] = site["site_no"].astype(str).map(dmap)
    site = site[site["_di"].notna()].reset_index(drop=True)

    y_all = site[a.target].to_numpy(dtype=np.float32)
    keep = np.isfinite(y_all)
    site, y_all = site[keep].reset_index(drop=True), y_all[keep]
    di = site["_di"].to_numpy(dtype=int)
    huc = site["HUC2"].astype(str).to_numpy()
    print(f"{len(site):,} sites with DEM and {a.target}")

    Z = dm["dem"][di].astype(np.float32)
    # NORMALISE PER PATCH. Absolute elevation is already in the attribute
    # baseline (MEANELEVSMO) and would let the net rediscover it instead of
    # learning shape. Removing the mean forces it to use FORM.
    Z = Z - Z.mean((1, 2), keepdims=True)
    Z = Z / (Z.std((1, 2), keepdims=True) + 1e-6)

    preds = np.full(len(y_all), np.nan, dtype=np.float32)
    for h in np.unique(huc):
        tr, te = huc != h, huc == h
        if tr.sum() < 200 or te.sum() < 10:
            continue
        torch.manual_seed(0)
        net = TerrainNet(a.width).to(DEVICE)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
        Xtr = torch.tensor(Z[tr])[:, None]
        ytr = torch.tensor(y_all[tr])
        ym, ys = ytr.mean(), ytr.std() + 1e-6
        rng = np.random.default_rng(0)
        net.train()
        for step in range(a.steps):
            b = rng.choice(len(Xtr), size=min(a.batch, len(Xtr)),
                           replace=False)
            xb = Xtr[b].to(DEVICE)
            if a.augment:                       # terrain has no preferred
                k = int(rng.integers(4))        # orientation
                xb = torch.rot90(xb, k, (2, 3))
                if rng.random() < 0.5:
                    xb = torch.flip(xb, (3,))
            loss = ((net(xb).squeeze(-1) -
                     ((ytr[b].to(DEVICE) - ym) / ys)) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            p = []
            Xte = torch.tensor(Z[te])[:, None]
            for i in range(0, len(Xte), 256):
                p.append(net(Xte[i:i + 256].to(DEVICE)).squeeze(-1).cpu())
            preds[te] = (torch.cat(p) * ys + ym).numpy()
        print(f"  HUC2 {h}: {te.sum():4d} held out", flush=True)

    m = np.isfinite(preds)
    r2 = float(1 - ((y_all[m] - preds[m]) ** 2).sum() /
               ((y_all[m] - y_all[m].mean()) ** 2).sum())
    print(f"\n=== LEARNED terrain features, {a.target}, leave-HUC2-out ===")
    print(f"  CNN on raw patch      R2 = {r2:+.4f}  (n = {m.sum():,})")
    print(f"  ridge on 12 descriptors    = "
          f"{'+0.1032' if a.target == 'log_W' else '+0.1248'}")
    print("  If these are close, the descriptors were NOT the bottleneck and")
    print("  the DEM pathway is empty for this target.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table",
                    default="/nfs/data/cxs1024/channel_geometry/data/"
                            "train_table_dem_fixed.csv")
    ap.add_argument("--dem", default="logs/geom_dem.npz")
    ap.add_argument("--target", default="log_W", choices=["log_W", "log_d"])
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--augment", action="store_true", default=True)
    main(ap.parse_args())
