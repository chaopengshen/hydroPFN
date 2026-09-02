"""Paired test: do site attributes sharpen terrain generation?

One attr-conditioned checkpoint (trained with attr-dropout, so it has a
genuine unconditional mode) is sampled twice on IDENTICAL held-out tiles,
masks and noise seeds -- once with the gauge's true statics, once with the
null token. Any difference is attributable to the attributes alone. Tiles
are the HUC2-holdout gauges' patches, which the model never saw in training.

This is the existence proof for the DEM<->time-series connector in the
generative direction (the direction the future 2D-field arm uses: sparse
site data conditioning a continuous field), motivated by the reverse probe:
statics recover the leading terrain PCs at ~0.2 R^2, above location and
climate.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from hydropfn.metrics.terrain import score
from hydropfn.models.diffusion import DenoiseUNet, Diffusion, harmonic_torch
from hydropfn.train.train_dem_multiscale import sample_holes_orig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KEYS = ["psd_ratio", "vario_ratio_10m", "vario_ratio_80m", "slope_w1",
        "elev_rmse"]


def med(rows, k):
    v = np.array([r[k] for r in rows if np.isfinite(r.get(k, np.nan))])
    return float(np.median(v)) if len(v) else np.nan


def main(a):
    ck = torch.load(a.ckpt, map_location=DEVICE)
    sd = ck.get("ema", ck["net"])
    n_attr = sd["amlp.0.weight"].shape[1] if "amlp.0.weight" in sd else 0
    assert n_attr, "checkpoint has no attribute-conditioning pathway"
    net = DenoiseUNet(w=sd["inp.weight"].shape[0],
                      in_ch=sd["inp.weight"].shape[1],
                      scale_cond=any(k.startswith("smlp.") for k in sd),
                      attr_cond=n_attr).to(DEVICE)
    net.load_state_dict(sd)
    net.eval()
    diff = Diffusion(device=DEVICE, param=ck.get("param", "eps"))
    assert ck.get("residual") and ck.get("valid_norm"), \
        "this eval mirrors the parity recipe (residual + valid-norm) only"

    za = np.load(a.attr_npz, allow_pickle=True)
    zn = np.load(a.attr_norm, allow_pickle=True)
    amat = za["attrs"].astype(np.float64)
    amat = (np.where(np.isfinite(amat), amat, zn["mu"]) - zn["mu"]) / zn["sd"]
    table = {str(s): amat[i]
             for i, s in enumerate(np.asarray(za["site_id"]).astype(str))}

    pref = tuple(a.holdout.split(","))
    print(f"{'scale':>7} {'arm':>9} |"
          + "".join(f"{k:>15}" for k in KEYS))
    for path, mpp, km in [tuple(t.split(",")) for t in a.levels.split(";")]:
        mpp, km = float(mpp), float(km)
        z = np.load(path, allow_pickle=True)
        dem, ok = z["dem"], z["ok"].astype(bool)
        sid = np.asarray(z["site_id"]).astype(str)
        pick = ok & np.isfinite(dem).all((1, 2)) \
            & np.array([s.startswith(pref) for s in sid])
        dem, sid = dem[pick].astype(np.float32), sid[pick]
        print(f"\n{path}: {len(dem)} held-out gauge tiles")
        Z = dem - dem.mean((1, 2), keepdims=True)
        sc_row = np.array([np.log10(mpp), np.log10(km)], np.float32)

        rows = {"attrs": [], "null": [], "harmonic": []}
        rng = np.random.default_rng(a.seed)
        for i0 in range(0, min(len(Z), a.n), a.batch):
            b = slice(i0, min(i0 + a.batch, len(Z), a.n))
            x0 = torch.tensor(Z[b])[:, None].to(DEVICE)
            B = x0.shape[0]
            sc = torch.tensor(np.tile(sc_row, (B, 1))).to(DEVICE)
            at_true = torch.tensor(np.stack(
                [np.concatenate([table[s], [1.0]]) for s in sid[b]]
            ).astype(np.float32)).to(DEVICE)
            at_null = torch.zeros_like(at_true)
            m = sample_holes_orig(B, Z.shape[-1], rng, DEVICE)
            v = m.sum(dim=(1, 2, 3), keepdim=True).clamp(min=1.0)
            mu_ = (x0 * m).sum(dim=(1, 2, 3), keepdim=True) / v
            sd_ = ((((x0 - mu_) * m) ** 2).sum(dim=(1, 2, 3), keepdim=True)
                   / v).sqrt().clamp(min=0.5)
            x0 = ((x0 - mu_) / sd_).clamp(-8.0, 8.0)
            with torch.no_grad():
                harm = harmonic_torch(x0 * m, m)
                ctx = torch.cat([harm, m], dim=1)
                zk = torch.zeros_like(x0)
                preds = {}
                for name, at in (("attrs", at_true), ("null", at_null)):
                    # identical noise for both arms: the seed pins both the
                    # DDIM init and every RePaint re-noising draw
                    torch.manual_seed(a.seed * 100003 + i0)
                    preds[name] = harm + diff.ddim_cond(
                        net, zk, m, steps=a.steps, scale=sc,
                        ctx_override=ctx, attrs=at)
            for j in range(B):
                t_ = x0[j, 0].cpu().numpy()
                mk = m[j, 0].cpu().numpy()
                for name in ("attrs", "null"):
                    rows[name].append(score(t_, preds[name][j, 0]
                                            .cpu().numpy(), mk))
                rows["harmonic"].append(score(t_, harm[j, 0].cpu().numpy(),
                                              mk))
        for name in ("attrs", "null", "harmonic"):
            print(f"{km:>6g}k {name:>9} |"
                  + "".join(f"{med(rows[name], k):15.4f}" for k in KEYS))
        # paired per-tile deltas, attrs minus null (negative = attrs better
        # for slope_w1/elev_rmse; closer to 1 better for the ratios)
        for k in ("elev_rmse", "slope_w1"):
            d = np.array([ra[k] - rn[k] for ra, rn
                          in zip(rows["attrs"], rows["null"])
                          if np.isfinite(ra[k]) and np.isfinite(rn[k])])
            wins = int((d < 0).sum())
            print(f"        paired {k}: median delta {np.median(d):+.4f}  "
                  f"attrs better on {wins}/{len(d)} tiles")
        d = np.array([abs(ra["psd_ratio"] - 1) - abs(rn["psd_ratio"] - 1)
                      for ra, rn in zip(rows["attrs"], rows["null"])
                      if np.isfinite(ra["psd_ratio"])
                      and np.isfinite(rn["psd_ratio"])])
        print(f"        paired |psd-1|: median delta {np.median(d):+.4f}  "
              f"attrs better on {int((d < 0).sum())}/{len(d)} tiles")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--levels", required=True,
                    help="gauge-level npzs only: path,mpp,km;...")
    ap.add_argument("--attr-npz", default="logs/camels_attrs_by_site.npz")
    ap.add_argument("--attr-norm", required=True,
                    help="attr_norm_<tag>.npz written by the trainer")
    ap.add_argument("--holdout", default="01,11,17")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
