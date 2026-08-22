"""Train unit A (merged static + temporal encoder) on masked reconstruction.

Reproduces: logs/site_encoder_{tag}.csv, checkpoint logs/site_encoder_{tag}.pt

Objective: hide part of the series, reconstruct it. The four mask types are
sampled per example, so one training run produces gap filling, forecasting,
cross-variable inference and PUB from a single objective.

TWO PRE-REGISTERED CHECKS, because "the loss went down" proves nothing:

  A  attribute ablation.  Zero the static attributes at inference and measure
     the change in streamflow reconstruction. If it costs NOTHING, merging the
     static and temporal paths bought nothing and the simpler design wins.
     This is the whole justification for the merged unit, so it has to be
     tested rather than assumed.

  B  per-mask breakdown.  Report each mask type separately. A model can look
     fine on average while being useless at the one mode you care about
     (whole_site = PUB), and the average hides it.

Baselines on the same held-out sites and the same masks:
  climatology   per-variable training mean            (must beat)
  persistence   last visible value of that channel    (must beat for spans)

Splits are leave-region-out by SITE. With `--source synthetic` there are no
regions, so sites are split at random -- honest for a machinery test, NOT a
substitute for spatial validation on CAMELS.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from hydropfn.data.forcing import (CLIMATE_STATICS, MASK_KINDS,  # noqa: E402
                                   load_camels, patchify, sample_mask,
                                   synthetic)
from hydropfn.models.site_encoder import (SiteEncoder,  # noqa: E402
                                          load_stefaland_trunk,
                                          masked_mse)
from hydropfn.paths import LOGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def standardise(x, valid):
    """Per-variable z-scoring over the TRAINING sites only."""
    m = np.where(valid, x, np.nan)
    mu = np.nanmean(m, axis=(0, 1))
    sd = np.nanstd(m, axis=(0, 1)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def make_batch(Xp, A, valid_p, doy, idx, rng, win, kind=None, n_obs=1,
               fixed_start=None):
    """One batch: a window per site, plus a sampled mask.

    `fixed_start` pins the window so this evaluation is comparable to the PUB
    and LSTM runs, which both score at patch 200. Averaging over random windows
    instead answers a different question and the two numbers must not be
    compared.
    """
    N = Xp.shape[1]
    b_x, b_v, b_val, b_doy = [], [], [], []
    for i in idx:
        s = (fixed_start if fixed_start is not None
             else int(rng.integers(0, max(1, N - win))))
        sl = slice(s, s + win)
        b_x.append(Xp[i, sl])
        b_val.append(valid_p[i, sl])
        b_doy.append(doy[sl])
        b_v.append(sample_mask(win, Xp.shape[2], rng, kind, n_obs))
    t = lambda a, d=torch.float32: torch.tensor(np.stack(a), dtype=d,
                                                device=DEVICE)  # noqa: E731
    return {"attrs": t([A[i] for i in idx]), "series": t(b_x),
            "vis": t(b_v), "valid": t(b_val), "doy": t(b_doy)}


def evaluate(net, Xp, A, valid_p, doy, idx, rng, win, n_obs, obs_col, batch,
             clim_cols=None, phys_cols=None, fixed_start=None):
    """Per-mask-type reconstruction R2 on the observation channel, plus the
    attribute ablations.

    The ablation is SPLIT because "attributes help" is only interesting if the
    PHYSICAL half does the work. CAMELS statics include p_mean, aridity,
    frac_snow and the precipitation-frequency terms, all of which are
    aggregates of the forcing series the model already reads -- handing those
    over is not new information, it is a summary of its own input.
    """
    rows = []
    net.eval()
    for kind in MASK_KINDS:
        got = {"model": [], "no_attrs": [], "no_climate": [], "no_physical": [],
               "clim": [], "persist": []}
        truth = []
        erng = np.random.default_rng(0)
        for i in range(0, len(idx), batch):
            b = make_batch(Xp, A, valid_p, doy, idx[i:i + batch], erng, win,
                           kind, n_obs, fixed_start=fixed_start)
            with torch.no_grad():
                r = net(b)["recon"]
                b0 = dict(b); b0["attrs"] = torch.zeros_like(b["attrs"])
                r0 = net(b0)["recon"]
                if clim_cols is not None and len(clim_cols):
                    bc = dict(b); ac = b["attrs"].clone()
                    ac[:, clim_cols] = 0.0; bc["attrs"] = ac
                    rc = net(bc)["recon"]
                    bp = dict(b); ap_ = b["attrs"].clone()
                    ap_[:, phys_cols] = 0.0; bp["attrs"] = ap_
                    rp = net(bp)["recon"]
                else:
                    rc = rp = r
            w = ((1 - b["vis"]) * b["valid"])[..., obs_col].bool()
            if w.sum() == 0:
                continue
            y = b["series"][..., obs_col, :][w].cpu().numpy().ravel()
            truth.append(y)
            got["model"].append(r[..., obs_col, :][w].cpu().numpy().ravel())
            got["no_attrs"].append(r0[..., obs_col, :][w].cpu().numpy().ravel())
            got["no_climate"].append(rc[..., obs_col, :][w].cpu().numpy().ravel())
            got["no_physical"].append(rp[..., obs_col, :][w].cpu().numpy().ravel())
            got["clim"].append(np.zeros_like(y))       # standardised => mean 0
            # persistence: last visible patch of that channel, per sample
            vis = b["vis"][..., obs_col].cpu().numpy()
            ser = b["series"][..., obs_col, :].cpu().numpy()
            pv = np.zeros_like(ser)
            for s_i in range(ser.shape[0]):
                last = ser[s_i, 0]
                for n_i in range(ser.shape[1]):
                    if vis[s_i, n_i] > 0.5:
                        last = ser[s_i, n_i]
                    pv[s_i, n_i] = last
            got["persist"].append(pv[w.cpu().numpy()].ravel())
        if not truth:
            continue
        y = np.concatenate(truth)
        den = ((y - y.mean()) ** 2).sum()
        rec = {"mask": kind, "n": int(len(y))}
        for k, v in got.items():
            p = np.concatenate(v)
            rec[k] = float(1 - ((y - p) ** 2).sum() / den)
        rec["attr_gain"] = rec["model"] - rec["no_attrs"]
        rec["climate_gain"] = rec["model"] - rec["no_climate"]
        rec["physical_gain"] = rec["model"] - rec["no_physical"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    if a.source == "camels":
        d = load_camels(a.nc)
    else:
        d = synthetic(a.n_sites, a.n_days, a.seed)
    X, A_, valid = d["x"], d["attrs"], d["valid"]
    n_obs = sum(1 for v in d["series_vars"] if v == "QObs")
    print(f"{a.source}: {X.shape[0]} sites x {X.shape[1]} days x "
          f"{X.shape[2]} vars ({d['series_vars']}), "
          f"{A_.shape[1]} attributes | {DEVICE}", flush=True)

    S = X.shape[0]
    region = d["region"]
    if a.source == "camels":
        # LEAVE-REGION-OUT: hold out whole USGS drainage-basin groups. A random
        # basin split would leave neighbours of every test basin in training,
        # which is the easy question, not the one this model exists to answer.
        regs = sorted(set(region.tolist()))
        hold = a.holdout.split(",") if a.holdout else regs[:3]
        te = np.flatnonzero(np.isin(region, hold))
        tr = np.flatnonzero(~np.isin(region, hold))
        print(f"  holdout regions {hold}: {len(te)} basins held out, "
              f"{len(tr)} train", flush=True)
    else:
        perm = rng.permutation(S)
        n_te = max(20, S // 5)
        te, tr = perm[:n_te], perm[n_te:]
        print(f"  synthetic: random split, {len(te)} held out", flush=True)

    mu, sd = standardise(X[tr], valid[tr])
    Xs = (X - mu) / sd
    am, asd = A_[tr].mean(0), A_[tr].std(0) + 1e-6
    A_s = np.nan_to_num((A_ - am) / asd).astype(np.float32)

    Xp = patchify(np.nan_to_num(Xs), a.patch)
    valid_p = patchify(valid.astype(np.float32), a.patch).min(-1)
    doy = ((np.arange(Xp.shape[1]) * a.patch) % 365.25) / 365.25
    doy = doy.astype(np.float32)
    obs_col = Xp.shape[2] - 1

    net = SiteEncoder(A_s.shape[1], Xp.shape[2], a.patch, depth=a.depth,
                      d_ffd=a.d_ffd)
    if a.stefaland:
        # Only the trunk can transfer; see docs/stefaland_reuse.md. Whether it
        # HELPS is the open question this flag exists to answer -- their trunk
        # learned to mix per-timestep tokens from per-variable MLPs, ours mixes
        # patch projections plus variable-ID embeddings.
        load_stefaland_trunk(net, a.stefaland)
    net = net.to(DEVICE)
    print(f"  SiteEncoder {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"params, window {a.win} patches = {a.win*a.patch} days", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train()
        tot = 0.0
        for _ in range(a.steps):
            idx = rng.choice(tr, size=a.batch, replace=False)
            b = make_batch(Xp, A_s, valid_p, doy, idx, rng, a.win,
                           n_obs=n_obs)
            out = net(b)
            loss = masked_mse(out["recon"], b["series"], b["vis"], b["valid"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs}  masked MSE {tot/a.steps:.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    torch.save({"net": net.state_dict(), "n_attr": A_s.shape[1],
                "n_vars": Xp.shape[2], "patch": a.patch},
               LOGS / f"site_encoder_{a.tag}.pt")

    names = d["static_vars"]
    clim_cols = [i for i, n in enumerate(names) if n in CLIMATE_STATICS]
    phys_cols = [i for i, n in enumerate(names) if n not in CLIMATE_STATICS]
    df = evaluate(net, Xp, A_s, valid_p, doy, te, rng, a.win, n_obs, obs_col,
                  a.batch, clim_cols, phys_cols, fixed_start=a.eval_start)
    df.to_csv(LOGS / f"site_encoder_{a.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== streamflow reconstruction R2 on held-out sites, by mask ===")
    cols = ["mask", "n", "model", "no_attrs", "attr_gain"]
    if "climate_gain" in df.columns:
        cols += ["climate_gain", "physical_gain"]
    cols += ["clim", "persist"]
    print(df[cols].to_string(index=False,
                             float_format=lambda v: f"{v:+.4f}"))
    print("\n  model         : full input")
    print("  no_attrs      : ALL attributes zeroed at inference")
    print("  attr_gain     : model - no_attrs. If ~0, merging bought nothing.")
    print("  climate_gain  : cost of zeroing only the CLIMATE statics")
    print("                  (p_mean, aridity, frac_snow, prec freq/dur).")
    print("                  These duplicate the forcing series the model")
    print("                  already reads, so a large value here is NOT")
    print("                  evidence of catchment learning.")
    print("  physical_gain : cost of zeroing only the PHYSICAL statics")
    print("                  (soils, geology, slope, area). THIS is the")
    print("                  number that justifies the merged design.")
    print("  clim          : per-variable training mean (standardised => 0)")
    print("  persist       : last visible value of that channel")
    print(f"\nwrote {LOGS / f'site_encoder_{a.tag}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["synthetic", "camels"],
                    default="synthetic")
    ap.add_argument("--nc", default=None, help="CAMELS_Frederik.nc path")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--n-sites", type=int, default=400)
    ap.add_argument("--n-days", type=int, default=2048)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32, help="patches per window")
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="synth")
    ap.add_argument("--eval-start", type=int, default=None,
                    help="pin the evaluation window (patch index) so the "
                         "number is comparable to the PUB and LSTM runs, "
                         "which both score at 200")
    ap.add_argument("--d-ffd", type=int, default=512,
                    help="512 matches StefaLand's trunk exactly; 1024 (=4*d) "
                         "is the usual transformer default")
    ap.add_argument("--stefaland", default=None,
                    help="path to StefalandOriginalGlobal20.pt; initialises "
                         "the TRUNK only")
    ap.add_argument("--holdout", default=None,
                    help="comma-separated USGS region prefixes to hold out, "
                         "e.g. 01,11,17 (default: the first three)")
    main(ap.parse_args())
