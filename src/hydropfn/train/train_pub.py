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
               geo_rank=None, start_lo=0, start_hi=None,
               ctx_start=None, latlon=None, self_da=0):
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

    # TWO CONTEXT MODES, and they answer different questions.
    #
    #   ctx_start None  -- context shares the query's window. The neighbours'
    #                      CURRENT discharge is visible. This is data
    #                      assimilation on an ungauged basin.
    #   ctx_start set   -- context comes from an earlier (training) period.
    #                      Only the neighbours' LONG-TERM behaviour is visible,
    #                      never today's. This is regionalisation, and the
    #                      time-aligned attention is meaningless here by
    #                      construction, since patch n of the query and patch n
    #                      of the context are different weeks.
    if ctx_start is None:
        ser = Xp[sites][:, sl]
        val = valid_p[sites][:, sl]
        dd = np.tile(doy[sl], (S, 1))
    else:
        csl = slice(ctx_start, ctx_start + win)
        ser = np.concatenate([Xp[sites[:1]][:, sl], Xp[sites[1:]][:, csl]], 0)
        val = np.concatenate([valid_p[sites[:1]][:, sl],
                              valid_p[sites[1:]][:, csl]], 0)
        dd = np.concatenate([np.tile(doy[sl], (1, 1)),
                             np.tile(doy[csl], (S - 1, 1))], 0)

    vis = np.ones((S, win, V), np.float32)
    if self_da:
        # SELF-ASSIMILATION MODE. The query keeps its OWN discharge history
        # and only the final `self_da` patches are hidden. This is the
        # information set of Jamaat et al. (2025) and Yang et al. (2026):
        # update using the target gauge's own recent record, then predict
        # forward. It is NOT the ungauged-basin question the rest of this
        # file asks -- it is a capability check on the same machinery.
        #
        # GRANULARITY CAVEAT: `vis` is per (patch, variable), so the finest
        # hideable unit is one 16-day patch. A true 1-day-lead comparison
        # would need sub-patch masking, which the token design cannot express
        # -- value_proj collapses all 16 days into a single token. So this is
        # a 16-day-ahead forecast given own history, not a 1-day-ahead one.
        vis[0, :, obs_col] = 1.0
        vis[0, -self_da:, obs_col] = 0.0
    else:
        vis[0, :, obs_col] = 0.0                             # query ungauged
    # NOTE the context window is `csl` above, which differs from the query's
    # `sl` whenever ctx_start is set. The caller reconstructs it, because the
    # BASELINES must read the same window the model's context read. Before
    # 2026-08-22 they always read `sl`, which in mode B handed
    # nn_donor/ctx_mean the neighbours' CONCURRENT discharge -- exactly the
    # information mode B withholds from the model. The tell: the baseline
    # columns were byte-identical between the mode A and mode B result files.
    task = {"attrs": A[sites], "series": ser, "vis": vis, "valid": val,
            "doy": dd, "site_valid": np.ones(S, np.float32)}
    # which query patches are actually being asked for -- eval scores ONLY
    # these, so self-DA is not credited for copying back its visible history
    task["_score"] = (vis[0, :, obs_col] == 0.0)
    if latlon is not None:
        # site 0 is the query; the connector encodes displacement FROM it
        task["latlon"] = latlon[sites].astype(np.float32)
    return task, sites, sl


def collate(tasks):
    out = {}
    for k in tasks[0]:
        if k.startswith("_"):
            continue
        out[k] = torch.tensor(np.stack([t[k] for t in tasks]),
                              dtype=torch.float32, device=DEVICE)
    return out


def nse_per_site(ys, ps):
    """Median per-basin NSE, and the fraction of basins above zero.

    NOT the same quantity as the pooled r2() below, and usually much lower.
    Pooling concatenates every basin before subtracting a SINGLE global mean,
    so between-basin variance in flow magnitude lands in the denominator and
    inflates the score -- a model that only knew each basin's mean flow would
    already post a respectable pooled R2. Per-basin NSE removes that free
    variance by using each basin's OWN mean.

    The hydrology literature reports median per-basin NSE (e.g. Kratzert 2019;
    Jamaat et al. 2025 report 0.75 -> 0.82 under variational DA on 531 CAMELS
    basins). Quoting our pooled number against theirs would be an apples-to-
    oranges comparison in our favour.
    """
    v = []
    for y, q in zip(ys, ps):
        y, q = np.asarray(y).ravel(), np.asarray(q).ravel()
        m = np.isfinite(y) & np.isfinite(q)
        den = ((y[m] - y[m].mean()) ** 2).sum()
        if den > 0:
            v.append(1 - ((y[m] - q[m]) ** 2).sum() / den)
    v = np.asarray(v)
    return float(np.median(v)), float((v > 0).mean()), int(v.size)


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
    # TWO geo rankings, and conflating them was a leak.
    #
    # TRAINING context must come from TRAINING basins only. Ranking over all
    # basins let a training query pull a HELD-OUT basin in as context, so the
    # model saw 70% of the eval basins' pre-cutoff streamflow (87 of 124, at
    # 1.5-4.1% of context slots). Their eval-period data was never exposed, so
    # the mode-A gain still comes from genuinely unseen information -- but any
    # claim about HISTORICAL context was contaminated, because the history had
    # already been absorbed into the weights.
    #
    # EVALUATION context may use whatever gauges exist, which is the
    # operational situation.
    gtrain = cKDTree(ll[tr])
    _, gi = gtrain.query(ll, k=min(64, len(tr)))
    geo_rank_train = tr[gi]

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
    # A window is `win` patches long, so a window STARTING at the cutoff would
    # extend past it. Subtract the window length so training genuinely ends by
    # --train-end. Without this our model trained to day 9472 while the LSTM
    # baseline stopped at 9000 -- 472 extra days, and the ones nearest the
    # evaluation period. An asymmetry in our own favour is the kind to be most
    # suspicious of.
    train_end_patch = ((a.train_end // a.patch) - a.win) if a.train_end else None
    if train_end_patch:
        print(f"  TEMPORAL SPLIT: train windows END by patch "
              f"{train_end_patch + a.win} (day {a.train_end}); eval at patch "
              f"{a.eval_start} (day {a.eval_start * a.patch}) -- gap "
              f"{(a.eval_start - train_end_patch - a.win) * a.patch} days",
              flush=True)
        if a.eval_start < train_end_patch:
            raise SystemExit("eval window is inside the training period")
    else:
        print("  WARNING: no temporal split -- training windows span the "
              "whole record INCLUDING the evaluation window. Period "
              "memorisation is uncontrolled. See Diagnosis.md.", flush=True)

    if a.eval_half:
        # Score EXACTLY the basins the neighbour-trained LSTM scores, using the
        # same rng and permutation. Comparing our 124-basin number against its
        # 62-basin number would confound the question with which basins landed
        # in the set -- and that subset is measurably harder (LSTM(b) gets
        # 0.7074 there vs 0.7553 on all 124).
        _rs = np.random.default_rng(0)
        te = _rs.permutation(te)[len(te) // 2:]
        print(f"  EVAL-HALF: scoring the same {len(te)} basins as the "
              f"neighbour-trained LSTM arm", flush=True)

    enc = SiteEncoder(A_s.shape[1], Xp.shape[2], a.patch, depth=a.depth,
                      d_ffd=a.d_ffd, k_summary=a.k_summary)
    net = PUBModel(enc, depth=a.conn_depth, time_aligned=a.time_aligned,
                   geo=a.geo, causal=a.causal,
                   no_pooled=a.no_pooled).to(DEVICE)
    print(f"  PUBModel {sum(p.numel() for p in net.parameters())/1e6:.1f}M "
          f"params | context sizes sampled from {a.k_train}", flush=True)
    if a.geo:
        print("  GEO: connector sees displacement from the query", flush=True)
    if a.no_pooled and not a.causal:
        print("  NO-POOLED: pooled summary path skipped, time NOT masked "
              "(control for --causal)", flush=True)
    if a.causal:
        print("  CAUSAL: site encoder time-masked; pooled summary path "
              "SKIPPED (summaries pool over the whole window). Only the "
              "time-aligned path survives.", flush=True)

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
                cs = (int(rng.integers(0, max(1, (train_end_patch or
                                                  Xp.shape[1] - a.win))))
                      if a.context_period == "train" else None)
                t, _, _ = build_task(Xp, A_s, valid_p, doy, q, pool, K, rng,
                                     a.win, obs_col, a.retrieval, nn_rank,
                                     geo_rank=geo_rank_train,
                                     start_hi=train_end_patch, ctx_start=cs,
                                     latlon=ll if a.geo else None,
                                     self_da=a.self_da)
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
                cs = (a.context_train_start if a.context_period == "train"
                      else None)
                t, sites, sl = build_task(
                    Xp, A_s, valid_p, doy, int(q), tr, K, erng, a.win,
                    obs_col, a.retrieval, nn_rank, fixed_start=a.eval_start,
                    geo_rank=geo_rank, ctx_start=cs,
                    latlon=ll if a.geo else None, self_da=a.self_da)
                # the window the CONTEXT was drawn from -- equals sl in mode A
                csl = sl if cs is None else slice(cs, cs + a.win)
                tasks.append(t); metas.append((sites, sl, csl, t["_score"]))
            b = collate(tasks)
            with torch.no_grad():
                rec = net(b)
            for j, (sites, sl, csl, sc) in enumerate(metas):
                # score ONLY the patches that were hidden from the query
                ys.append(Xp[sites[0], sl][sc][..., obs_col, :].ravel())
                ps.append(rec[j][sc][..., obs_col, :].cpu().numpy().ravel())
                if K > 0:
                    # csl, NOT sl. In mode A these are the same window and
                    # nothing changes; in mode B this is what stops the
                    # baselines from reading the future.
                    nn_ps.append(Xp[sites[1], csl][sc][..., obs_col, :].ravel())
                    cm_ps.append(Xp[sites[1:], csl][:, sc][..., obs_col, :]
                                 .mean(0).ravel())
        _y, _p = np.concatenate(ys), np.concatenate(ps)
        # DAILY is the honest headline; the 16-day mean is reported too because
        # aggregation raises R2 and any comparison must state which is which.
        n_sc = ys[0].size // a.patch          # scored patches per basin
        _ya = _y.reshape(-1, n_sc, a.patch).mean(-1)
        _pa = _p.reshape(-1, n_sc, a.patch).mean(-1)
        med, frac, nb = nse_per_site(ys, ps)
        rec_row = {"K": K, "n": int(_y.size), "model": r2(_y, _p),
                   "model_patch16": r2(_ya, _pa),
                   "nse_median": med, "nse_frac_pos": frac, "n_basins": nb}
        if K > 0:
            rec_row["nn_donor"] = r2(np.concatenate(ys), np.concatenate(nn_ps))
            rec_row["ctx_mean"] = r2(np.concatenate(ys), np.concatenate(cm_ps))
            rec_row["nn_donor_nse"] = nse_per_site(ys, nn_ps)[0]
            rec_row["ctx_mean_nse"] = nse_per_site(ys, cm_ps)[0]
        rows.append(rec_row)
        print(f"  K={K:3d}  pooled-daily {rec_row['model']:+.4f}"
              f"  16d {rec_row['model_patch16']:+.4f}"
              f"  | per-basin NSE med {med:+.4f} ({frac:.0%} of {nb} > 0)"
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
    ap.add_argument("--eval-half", action="store_true",
                    help="score the same half the neighbour-trained LSTM does")
    ap.add_argument("--context-period", choices=["current", "train"],
                    default="current",
                    help="'current': context shares the query window -- data "
                         "assimilation. 'train': context comes from the "
                         "training period -- long-term conditioning only")
    ap.add_argument("--context-train-start", type=int, default=100,
                    help="patch index for context when --context-period train")
    ap.add_argument("--train-end", type=int, default=None,
                    help="training windows must start before this DAY. "
                         "Without it there is no temporal split and period "
                         "memorisation is uncontrolled -- see Diagnosis.md")
    ap.add_argument("--self-da", type=int, default=0, metavar="P",
                    help="SELF-ASSIMILATION: query keeps its own discharge "
                         "except the last P patches, which are predicted and "
                         "scored. Comparable in information set to Jamaat "
                         "2025 / Yang 2026. 0 = ungauged (default).")
    ap.add_argument("--no-pooled", action="store_true",
                    help="skip the pooled-summary path WITHOUT time masking "
                         "-- the control that isolates --causal")
    ap.add_argument("--geo", action="store_true",
                    help="connector encodes displacement from the query")
    ap.add_argument("--causal", action="store_true",
                    help="forbid attending to the future -- makes the model a "
                         "FILTER rather than a smoother, comparable to the "
                         "causal LSTM baseline")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="pub")
    main(ap.parse_args())
