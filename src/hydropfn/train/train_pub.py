"""THE test: do context basins help predict an UNGAUGED basin's streamflow?

Reproduces: logs/pub_{tag}.csv

Everything else in this repository is scaffolding around this claim, and it has
never been tested for time series. Context has been shown to help for POINT
measurements (unit D) — but there the context is the query site's OWN history,
which is a much easier setting. Here the query basin is genuinely ungauged:
every streamflow patch masked, in a region the model never trained on.

A task = 1 query basin + K context basins.
  query    streamflow FULLY masked (`whole_site`); forcings visible
  context  streamflow VISIBLE, drawn from TRAINING regions only

Context size is RANDOMISED during training. That is not a detail: in unit D a
fixed context size made the model calibrate off one neighbour and never
aggregate (−0.145 → 0.582 on the first neighbour, then flat to 16). Randomising
is the untested fix, and this is where it gets tested.

Baselines on identical query basins (docs/pub_test_plan.md):
  no_context   K = 0 through the same model      the internal bar
  nn_donor     copy the most attribute-similar context basin's flow.
               THE standard PUB method -- if we do not beat this we have
               reinvented it expensively.
  ctx_mean     mean of the context basins' flow. Catches "the model just
               learned regional climatology".
  climatology  training mean (0 after standardisation)

Retrieval ablation (`--retrieval`): `similar` | `random` | `geo`. If RANDOM
context works as well as SIMILAR, the model is not using site identity at all
and "retrieved context set" is a story we are telling ourselves.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from hydropfn.data.forcing import load_camels, patchify, sample_mask
from hydropfn.models.connector import PUBModel
from hydropfn.models.site_encoder import SiteEncoder
from hydropfn.paths import LOGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_task(Xp, A, valid_p, doy, q_idx, ctx_pool, K, rng, win, obs_col,
               retrieval="similar", nn_rank=None, fixed_start=None,
               geo_rank=None, start_lo=0, start_hi=None):
    """One task: query basin (streamflow hidden) + K context basins (visible).

    All sites share the SAME time window, so context and query are contemporary
    -- a context basin from a different decade would be a different question.
    """
    N, V = Xp.shape[1], Xp.shape[2]
    hi = (N - win) if start_hi is None else min(start_hi, N - win)
    s = fixed_start if fixed_start is not None else int(
        rng.integers(start_lo, max(start_lo + 1, hi)))
    sl = slice(s, s + win)

    if K == 0:
        ctx = np.array([], dtype=int)
    elif retrieval == "geo" and geo_rank is not None:
        # GEOGRAPHIC neighbours. This is the only retrieval that can supply
        # information the attributes cannot: a nearby gauged basin shares
        # STORMS with the query. Attribute-similar basins on the far side of
        # the continent see different weather on the same day, which is why
        # the first version of this test measured exactly zero.
        ctx = np.array([c for c in geo_rank[q_idx] if c != q_idx][:K])
    elif retrieval == "similar" and nn_rank is not None:
        ctx = nn_rank[q_idx][:K]
    else:
        ctx = rng.choice(ctx_pool, size=min(K, len(ctx_pool)), replace=False)

    sites = np.concatenate([[q_idx], ctx]).astype(int)
    S = len(sites)
    ser = Xp[sites][:, sl]                                   # (S, win, V, p)
    val = valid_p[sites][:, sl]
    vis = np.ones((S, win, V), np.float32)
    vis[0, :, obs_col] = 0.0                                 # query ungauged
    return {"attrs": A[sites], "series": ser, "vis": vis, "valid": val,
            "doy": np.tile(doy[sl], (S, 1)),
            "site_valid": np.ones(S, np.float32)}, sites, sl


def collate(tasks):
    out = {}
    for k in tasks[0]:
        out[k] = torch.tensor(np.stack([t[k] for t in tasks]),
                              dtype=torch.float32, device=DEVICE)
    return out


def r2(y, p):
    y, p = np.asarray(y).ravel(), np.asarray(p).ravel()
    m = np.isfinite(y) & np.isfinite(p)
    return float(1 - ((y[m] - p[m]) ** 2).sum() /
                 ((y[m] - y[m].mean()) ** 2).sum())


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_camels(a.nc)
    X, A_, valid, region = d["x"], d["attrs"], d["valid"], d["region"]

    hold = a.holdout.split(",")
    te = np.flatnonzero(np.isin(region, hold))
    tr = np.flatnonzero(~np.isin(region, hold))
    print(f"leave-region-out {hold}: {len(te)} query basins held out, "
          f"{len(tr)} available as context | {DEVICE}", flush=True)

    mu = np.nanmean(np.where(valid[tr], X[tr], np.nan), axis=(0, 1))
    sd = np.nanstd(np.where(valid[tr], X[tr], np.nan), axis=(0, 1)) + 1e-6
    Xs = np.nan_to_num((X - mu) / sd)
    am, asd = A_[tr].mean(0), A_[tr].std(0) + 1e-6
    A_s = np.nan_to_num((A_ - am) / asd).astype(np.float32)

    Xp = patchify(Xs, a.patch)
    valid_p = patchify(valid.astype(np.float32), a.patch).min(-1)
    doy = (((np.arange(Xp.shape[1]) * a.patch) % 365.25) / 365.25).astype(np.float32)
    obs_col = Xp.shape[2] - 1

    # attribute-similarity ranking: every basin's nearest TRAINING basins
    from scipy.spatial import cKDTree
    tree = cKDTree(A_s[tr])
    _, nn_idx = tree.query(A_s, k=min(64, len(tr)))
    nn_rank = tr[nn_idx]                                     # (S_all, 64)

    # Geographic ranking. `--context-pool all` lets a held-out query draw on
    # OTHER basins in its own region. That is not leakage: the model never
    # trained on them, and in the real PUB setting those gauges exist and
    # their records are available at inference. What must never be visible is
    # the QUERY's own streamflow, and it never is.
    ll = d["latlon"]
    pool_geo = np.arange(len(ll)) if a.context_pool == "all" else tr
    gtree = cKDTree(ll[pool_geo])
    _, g_idx = gtree.query(ll, k=min(64, len(pool_geo)))
    geo_rank = pool_geo[g_idx]
    dist_tr = cKDTree(ll[tr]).query(ll[te], k=1)[0]
    dist_all = gtree.query(ll[te], k=2)[0][:, 1]
    print(f"  held-out query -> nearest TRAINING basin: median "
          f"{np.median(dist_tr):.2f} deg; nearest ANY basin: median "
          f"{np.median(dist_all):.2f} deg", flush=True)

    # TEMPORAL SPLIT. Without --train-end, training windows are drawn from
    # the WHOLE record including the evaluation window, so the model has seen
    # that stretch of weather through the training-region basins. That leaves
    # period memorisation uncontrolled: the model could recall 1988-90 rather
    # than infer from context. Setting --train-end confines training to an
    # earlier period and --eval-start to a later one, which is the only way to
    # separate "conditions on neighbours" from "remembers this weather".
    train_end_patch = (a.train_end // a.patch) if a.train_end else None
    if train_end_patch:
        print(f"  TEMPORAL SPLIT: train windows start before patch "
              f"{train_end_patch} (day {a.train_end}); eval at patch "
              f"{a.eval_start} (day {a.eval_start * a.patch})", flush=True)
        if a.eval_start < train_end_patch:
            raise SystemExit("eval window is inside the training period")
    else:
        print("  WARNING: no temporal split -- training windows span the "
              "whole record INCLUDING the evaluation window. Period "
              "memorisation is uncontrolled. See Diagnosis.md.", flush=True)

    enc = SiteEncoder(A_s.shape[1], Xp.shape[2], a.patch, depth=a.depth,
                      d_ffd=a.d_ffd, k_summary=a.k_summary)
    net = PUBModel(enc, depth=a.conn_depth,
                   time_aligned=a.time_aligned).to(DEVICE)
    print(f"  PUBModel {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"params | context sizes sampled from {a.k_train}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
    k_train = [int(x) for x in a.k_train.split(",")]

    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for _ in range(a.steps):
            K = int(rng.choice(k_train))     # RANDOMISED context size
            tasks = []
            for _ in range(a.batch):
                q = int(rng.choice(tr))
                pool = tr[tr != q]
                t, _, _ = build_task(Xp, A_s, valid_p, doy, q, pool, K, rng,
                                     a.win, obs_col, a.retrieval, nn_rank,
                                     geo_rank=geo_rank,
                                     start_hi=train_end_patch)
                tasks.append(t)
            b = collate(tasks)
            rec = net(b)
            w = ((1 - b["vis"][:, 0]) * b["valid"][:, 0])[..., obs_col]
            tgt = b["series"][:, 0][..., obs_col, :]
            pred = rec[..., obs_col, :]
            loss = (((pred - tgt) ** 2) * w.unsqueeze(-1)).sum() / \
                w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs}  masked MSE {tot/a.steps:.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    torch.save({"net": net.state_dict()}, LOGS / f"pub_{a.tag}.pt")

    # ---------------- evaluation: the context-scaling curve + baselines
    net.eval()
    rows = []
    for K in [int(x) for x in a.k_eval.split(",")]:
        erng = np.random.default_rng(123)
        ys, ps, nn_ps, cm_ps = [], [], [], []
        for i0 in range(0, len(te), a.batch):
            chunk = te[i0:i0 + a.batch]
            tasks, metas = [], []
            for q in chunk:
                t, sites, sl = build_task(
                    Xp, A_s, valid_p, doy, int(q), tr, K, erng, a.win,
                    obs_col, a.retrieval, nn_rank, fixed_start=a.eval_start,
                    geo_rank=geo_rank)
                tasks.append(t); metas.append((sites, sl))
            b = collate(tasks)
            with torch.no_grad():
                rec = net(b)
            for j, (sites, sl) in enumerate(metas):
                y = Xp[sites[0], sl][..., obs_col, :].ravel()
                ys.append(y)
                ps.append(rec[j][..., obs_col, :].cpu().numpy().ravel())
                if K > 0:
                    donor = Xp[sites[1], sl][..., obs_col, :].ravel()
                    nn_ps.append(donor)
                    cm_ps.append(Xp[sites[1:], sl][..., obs_col, :]
                                 .mean(0).ravel())
        rec_row = {"K": K, "n": int(np.concatenate(ys).size),
                   "model": r2(np.concatenate(ys), np.concatenate(ps))}
        if K > 0:
            rec_row["nn_donor"] = r2(np.concatenate(ys), np.concatenate(nn_ps))
            rec_row["ctx_mean"] = r2(np.concatenate(ys), np.concatenate(cm_ps))
        rows.append(rec_row)
        print(f"  K={K:3d}  model {rec_row['model']:+.4f}"
              + (f"   nn_donor {rec_row['nn_donor']:+.4f}"
                 f"   ctx_mean {rec_row['ctx_mean']:+.4f}" if K > 0 else ""),
              flush=True)

    df = pd.DataFrame(rows)
    df["retrieval"] = a.retrieval
    df.to_csv(LOGS / f"pub_{a.tag}.csv", index=False)
    base = df[df.K == 0].model.iloc[0] if (df.K == 0).any() else np.nan
    best = df.model.max()
    print(f"\n  K=0 (no context): {base:+.4f}   best with context: {best:+.4f}"
          f"   gain {best-base:+.4f}")
    print("  RISING curve = the claim holds. STEP at K=1 then flat = 'one "
          "donor', not in-context learning.")
    print(f"\nwrote {LOGS / f'pub_{a.tag}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True)
    ap.add_argument("--holdout", default="01,11,17")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--conn-depth", type=int, default=4)
    ap.add_argument("--d-ffd", type=int, default=512)
    ap.add_argument("--k-summary", type=int, default=3)
    ap.add_argument("--k-train", default="0,1,2,4,8,16")
    ap.add_argument("--k-eval", default="0,1,2,4,8,16,32")
    ap.add_argument("--retrieval", choices=["similar", "random", "geo"],
                    default="similar")
    ap.add_argument("--time-aligned", action="store_true",
                    help="let each query patch attend to context patches at "
                         "the SAME time position, not just pooled summaries")
    ap.add_argument("--context-pool", choices=["train", "all"], default="train",
                    help="'all' lets a held-out query use OTHER basins in its "
                         "own region as context -- not leakage, since the "
                         "model never trained on them and those gauges exist")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--eval-start", type=int, default=200,
                    help="evaluation window start, in PATCHES")
    ap.add_argument("--train-end", type=int, default=None,
                    help="training windows must start before this DAY. "
                         "Without it there is no temporal split and period "
                         "memorisation is uncontrolled -- see Diagnosis.md")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pub")
    main(ap.parse_args())
