"""hydroPFN's PUBModel on the CAMELS-531 protocol.

The companion to `camels531_lstm.py`. Both read the same basins, the same
periods and the same warmup, and both are scored by the same vendored
`Metrics` class on raw mm/day -- which is the only way the two numbers may be
put in one table. Everything this file changes relative to `train_pub.py` is
protocol, not architecture: the model, the task construction and the context
retrieval are `hydropfn.models` and `hydropfn.train.train_pub.build_task`
unchanged.

What is different from every previous hydroPFN run:

  * **531 basins, PUB/PUR/temporal folds** instead of one hand-picked HUC2
    triple.
  * **The scored period is continuous** -- 1461 days (spatial) or 5479 days
    (temporal) per basin -- not 16-day tails whose NSE denominator barely
    varies. The eval span is tiled by non-overlapping windows so every day is
    predicted exactly once, and a warmup prefix is predicted and discarded.
  * **Raw mm/day.** Any target transform is inverted before scoring, so the
    metric no longer depends on it.

K=0 is the ungauged, no-neighbour arm and is the row directly comparable to
the LSTM baseline. K>0 additionally reads neighbouring gauges' concurrent
discharge at inference, which the LSTM structurally cannot do -- that is the
claim, and it is only meaningful stated next to the K=0 row and the donor
baselines printed beside it.

Note on `--context-pool all` under the two extents. In PUB the held-out basins
are scattered, so a query's geographic neighbours are mostly TRAINING basins:
K>0 is close to "an ungauged basin among gauged ones". In PUR the whole region
is held out, so the neighbours are mostly other held-out basins the model
never trained on. Same flag, materially different question; the per-fold
distance diagnostic below prints which one you are actually running.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P                        # noqa: E402
from hydropfn.data.forcing import load_camels                  # noqa: E402
from hydropfn.models.connector import PUBModel                 # noqa: E402
from hydropfn.models.site_encoder import SiteEncoder           # noqa: E402
from hydropfn.paths import LOGS                                # noqa: E402
from hydropfn.train.train_pub import build_task, collate       # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tile_starts(span_days, patch, win):
    """Patch indices that tile `span_days` with non-overlapping windows.

    The last window is pulled back so it ends on the final patch rather than
    running past it, so a few days near the end may be covered twice; the
    later prediction wins. Every day is covered at least once, which is what
    the continuous scored series requires.
    """
    n_patch = span_days // patch
    if n_patch < win:
        raise SystemExit(f"eval span {span_days} d is shorter than one "
                         f"{win * patch} d window")
    st = list(range(0, n_patch - win + 1, win))
    if st[-1] != n_patch - win:
        st.append(n_patch - win)
    return st


def predict_basins(net, Xp, A_s, valid_p, doy, te_idx, ctx_pool, K, a,
                   geo_rank, nn_rank, latlon, area, p0, starts, span_days,
                   obs_col, ctx_off=None, recent_obs=0):
    """Predicted QObs for `te_idx` over the eval span, indexed by DAY.

    Returns (n_basins, span_days) on the STANDARDISED scale, plus the donor
    baselines at K > 0.

    Predictions are written into a day-indexed buffer rather than
    concatenated. The final tile is pulled back to end on the last patch, so
    it OVERLAPS its predecessor; concatenating would silently shift every day
    after that point and the scored slice would read the wrong dates. Writing
    by day makes the overlap a harmless overwrite instead.
    """
    net.eval()
    erng = np.random.default_rng(123)
    n, wd = len(te_idx), a.win * a.patch
    if recent_obs:
        # RECENT-OBS MODE (suite iii/iv). The query basin is added as its own
        # context site with the last `recent_obs` patches hidden; each tile
        # contributes ONLY its final patch, so every scored day is predicted
        # from a tile where it lies BEYOND the observation cutoff. Tiles step
        # by one patch (final patches tile the span exactly) and may begin
        # before the span, drawing history from pre-eval observations -- the
        # legitimate operational setting.
        n_patch = span_days // a.patch
        starts = [st for st in range(-(a.win - 1), n_patch - a.win + 1)
                  if p0 + st >= 0]
    buf = {k: np.full((n, span_days), np.nan, np.float32)
           for k in ("p", "nn", "cm", "idw")}
    row = {int(q): i for i, q in enumerate(te_idx)}

    for st in starts:
        d0 = st * a.patch          # day offset from the start of the buffer
        for i0 in range(0, len(te_idx), a.batch):
            chunk = te_idx[i0:i0 + a.batch]
            tasks, metas = [], []
            for q in chunk:
                t, sites, sl = build_task(
                    Xp, A_s, valid_p, doy, int(q), ctx_pool, K, erng, a.win,
                    obs_col, a.retrieval, nn_rank, fixed_start=p0 + st,
                    geo_rank=geo_rank, latlon=latlon if a.geo else None,
                    area=area if a.area_scale else None,
                    ctx_start=(p0 + st - ctx_off) if ctx_off else None,
                    self_ctx=recent_obs)
                tasks.append(t)
                metas.append((int(q), sites, sl))
            b = collate(tasks)
            with torch.no_grad():
                rec = net(b)
            for j, (q, sites, sl) in enumerate(metas):
                i = row[q]
                if recent_obs:
                    lo = d0 + wd - recent_obs * a.patch
                    if lo < 0:
                        continue
                    buf["p"][i, lo:d0 + wd] = rec[j][
                        -recent_obs:, obs_col, :].cpu().numpy().ravel()
                else:
                    buf["p"][i, d0:d0 + wd] = rec[j][
                        :, obs_col, :].cpu().numpy().ravel()
                if K > 0:
                    # Baselines must read the SAME window the model's
                    # context read: for mode B that is the HISTORICAL
                    # window, not the eval slice -- else they are handed
                    # concurrent discharge the model was denied (the bug
                    # fixed in train_pub 2026-08-22, reintroduced by the
                    # port until this line).
                    csl = (slice(sl.start - ctx_off, sl.stop - ctx_off)
                           if ctx_off else sl)
                    b0 = 2 if recent_obs else 1
                    nb = Xp[sites[b0:], csl][..., obs_col, :]
                    if nb.shape[0] == 0:
                        continue
                    if recent_obs:
                        lo = d0 + wd - recent_obs * a.patch
                        if lo < 0:
                            continue
                        buf["nn"][i, lo:d0 + wd] = nb[0, -recent_obs:].ravel()
                        buf["cm"][i, lo:d0 + wd] = nb.mean(0)[
                            -recent_obs:].ravel()
                    else:
                        buf["nn"][i, d0:d0 + wd] = nb[0].ravel()
                        buf["cm"][i, d0:d0 + wd] = nb.mean(0).ravel()
                    rel = latlon[sites[b0:]] - latlon[sites[0]]
                    dist = np.sqrt(rel[:, 0] ** 2 + (rel[:, 1] * 0.766) ** 2)
                    w = 1.0 / (dist ** 2 + 1e-3)
                    w = (w / w.sum()).astype(np.float32)
                    if recent_obs:
                        buf["idw"][i, lo:d0 + wd] = (
                            nb * w[:, None, None]).sum(0)[
                            -recent_obs:].ravel()
                    else:
                        buf["idw"][i, d0:d0 + wd] = (
                            nb * w[:, None, None]).sum(0).ravel()

    return {k: v for k, v in buf.items()
            if k == "p" or (K > 0 and np.isfinite(v).any())}


def main(a):
    if a.smoke:
        # only fill values the user did NOT explicitly override --
        # the first version stomped an explicit --epochs 2, turning a
        # 2-epoch smoke check into a 40-epoch x 10-fold run
        import sys as _sys
        if "--epochs" not in _sys.argv:
            a.epochs = 40
        if "--steps" not in _sys.argv:
            a.steps = 150
        if "--tasks" not in _sys.argv:
            a.tasks = 4

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = load_camels(a.nc)
    sub, gage = P.load_531(d)
    win_p = P.windows(d["time"], a.protocol)
    print(P.describe(a.extent, a.protocol, gage, d["time"]), flush=True)
    print(f"  device {DEVICE} | seed {a.seed} | window "
          f"{a.win} x {a.patch} = {a.win * a.patch} d", flush=True)

    X, A_, valid = sub["x"], sub["attrs"], sub["valid"]
    ll, area_km = sub["latlon"], sub["area"]
    obs = X.shape[-1] - 1

    # Raw mm/day discharge, held aside as the scoring target. Everything the
    # model sees is standardised; everything reported is mapped back to this.
    q_raw = X[..., obs].astype(np.float32).copy()
    q_raw[valid[..., obs] == 0] = np.nan

    folds = P.folds(a.extent, gage)
    if a.max_folds:
        folds = folds[:a.max_folds]
        print(f"  PARTIAL: first {a.max_folds} fold(s) only -- NOT the "
              f"protocol result", flush=True)

    k_eval = [int(x) for x in a.k_eval.split(",")]
    acc = {K: {"p": [], "nn": [], "cm": [], "idw": []} for K in k_eval}
    targs, fold_of, gages = [], [], []

    for kf, te_idx in enumerate(folds):
        if a.extent == "temporal":
            tr_idx = np.arange(len(gage))
        else:
            tr_idx = np.setdiff1d(np.arange(len(gage)), te_idx)
        print(f"\n  fold {kf}: train {len(tr_idx)} basins, "
              f"score {len(te_idx)} basins", flush=True)

        tw = win_p["train"]
        mu = np.nanmean(np.where(valid[tr_idx][:, tw], X[tr_idx][:, tw],
                                 np.nan), axis=(0, 1))
        sd = np.nanstd(np.where(valid[tr_idx][:, tw], X[tr_idx][:, tw],
                                np.nan), axis=(0, 1)) + 1e-6
        Xs = np.nan_to_num((X - mu) / sd).astype(np.float32)
        am, asd = np.nanmean(A_[tr_idx], 0), np.nanstd(A_[tr_idx], 0) + 1e-6
        A_s = np.nan_to_num((A_ - am) / asd).astype(np.float32)

        Xp = np.ascontiguousarray(
            Xs[:, :(Xs.shape[1] // a.patch) * a.patch]
            .reshape(len(Xs), -1, a.patch, Xs.shape[-1]).transpose(0, 1, 3, 2))
        valid_p = (valid[:, :Xp.shape[1] * a.patch]
                   .reshape(len(valid), -1, a.patch, valid.shape[-1])
                   .transpose(0, 1, 3, 2).min(-1)).astype(np.float32)
        doy = (((np.arange(Xp.shape[1]) * a.patch) % 365.25)
               / 365.25).astype(np.float32)

        from scipy.spatial import cKDTree
        nn_rank = tr_idx[cKDTree(A_s[tr_idx]).query(
            A_s, k=min(64, len(tr_idx)))[1]]
        # Training context is drawn from TRAINING basins only -- ranking over
        # all basins let a training query pull a held-out basin in as context,
        # which is the leak recorded in Diagnosis.md.
        geo_train = tr_idx[cKDTree(ll[tr_idx]).query(
            ll, k=min(64, len(tr_idx)))[1]]
        pool = (np.arange(len(ll)) if a.context_pool == "all" else tr_idx)
        geo_eval = pool[cKDTree(ll[pool]).query(ll, k=min(64, len(pool)))[1]]

        d_tr = cKDTree(ll[tr_idx]).query(ll[te_idx], k=1)[0]
        d_any = cKDTree(ll[pool]).query(ll[te_idx], k=2)[0][:, 1]
        print(f"    held-out -> nearest TRAINING basin: median "
              f"{np.median(d_tr):.2f} deg; nearest ANY: median "
              f"{np.median(d_any):.2f} deg", flush=True)

        enc = SiteEncoder(A_s.shape[1], Xp.shape[2], a.patch, depth=a.depth,
                          d_ffd=a.d_ffd, k_summary=a.k_summary)
        net = PUBModel(enc, depth=a.conn_depth, time_aligned=a.time_aligned,
                       geo=a.geo, causal=a.causal).to(DEVICE)
        if kf == 0:
            print(f"    PUBModel "
                  f"{sum(t.numel() for t in net.parameters()) / 1e6:.1f}M "
                  f"params | k_train {a.k_train}", flush=True)

        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)
        k_train = [int(x) for x in a.k_train.split(",")]

        # Training windows must lie inside the training period. `build_task`
        # picks a START, so the bound is the last start whose window still
        # ends by the cutoff.
        lo_p = tw.start // a.patch
        hi_p = tw.stop // a.patch - a.win

        t0 = time.time()
        for ep in range(a.epochs):
            net.train()
            tot = 0.0
            for _ in range(a.steps):
                # per STEP, not per task: self-ctx adds a site, and a
                # per-task draw makes S ragged within the batch
                step_self = (int(rng.integers(1, a.win // 2))
                             if rng.random() < a.self_ctx_p else 0)
                K = int(rng.choice(k_train))
                tasks = []
                for _ in range(a.tasks):
                    q = int(rng.choice(tr_idx))
                    t, _, _ = build_task(
                        Xp, A_s, valid_p, doy, q, tr_idx[tr_idx != q], K, rng,
                        a.win, obs, a.retrieval, nn_rank,
                        geo_rank=geo_train, start_lo=lo_p, start_hi=hi_p,
                        latlon=ll if a.geo else None,
                        area=area_km if a.area_scale else None,
                        ctx_start=("align" if a.context_period
                                   == "train" else None),
                        self_ctx=step_self)
                    tasks.append(t)
                b = collate(tasks)
                rec = net(b)
                w = (1 - b["vis"][:, 0]) * b["valid"][:, 0]
                loss = (((rec - b["series"][:, 0]) ** 2)
                        * w.unsqueeze(-1)).sum() / w.sum().clamp(min=1.0)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            if (ep + 1) % 10 == 0 or ep == 0:
                print(f"      epoch {ep + 1}/{a.epochs}  masked MSE "
                      f"{tot / a.steps:.4f}  [{(time.time() - t0) / 60:.1f} "
                      f"min]", flush=True)

        # ---- evaluation: tile the eval span, keep only the scored days
        ev = win_p["eval_in"]
        p0 = ev.start // a.patch
        day0 = p0 * a.patch                 # buffer day 0, in record days
        # Round the span UP to a whole patch: rounding down truncates the last
        # days of the scored period, which must be covered in full.
        span = -(-(ev.stop - day0) // a.patch) * a.patch
        if day0 + span > X.shape[1]:
            raise SystemExit("record ends before the padded eval span")
        starts = tile_starts(span, a.patch, a.win)
        keep = slice(win_p["score"].start - day0, win_p["score"].stop - day0)
        if keep.stop > span:
            raise SystemExit("tiling does not cover the scored period")
        if kf == 0:
            print(f"    tiling: {len(starts)} windows covering "
                  f"{starts[-1] * a.patch + a.win * a.patch} d from "
                  f"{d['time'][day0]}; scoring days "
                  f"{keep.start}..{keep.stop}", flush=True)

        targs.append(q_raw[te_idx][:, win_p["score"]].astype(np.float32))
        fold_of.append(np.full(len(te_idx), kf))
        gages.append(gage["gage"].to_numpy()[te_idx])

        for K in k_eval:
            got = predict_basins(net, Xp, A_s, valid_p, doy, te_idx, tr_idx,
                                 K, a, geo_eval, nn_rank, ll, area_km,
                                 p0, starts, span, obs,
                ctx_off=(137 if a.context_period == "train" else None),
                recent_obs=a.recent_obs)
            for name, arr in got.items():
                # standardised -> mm/day, the scale everything is scored on
                acc[K][name].append(
                    (arr[:, keep] * sd[obs] + mu[obs]).astype(np.float32))
            m = P.nse_table(acc[K]["p"][-1], targs[-1])
            print(f"    fold {kf} K={K:2d}  median NSE "
                  f"{np.nanmedian(m.nse):+.4f}", flush=True)

    targ = np.concatenate(targs, 0)
    tag = a.tag or f"pub_{a.extent}_{a.protocol}_s{a.seed}"
    outdir = LOGS / "camels531" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "targ.npy", targ)
    np.save(outdir / "gage.npy", np.concatenate(gages))
    np.save(outdir / "fold.npy", np.concatenate(fold_of))

    summary = {}
    print("\n=== PUBModel, CAMELS-531 ===", flush=True)
    for K in k_eval:
        row = {}
        for name in ("p", "nn", "cm", "idw"):
            if not acc[K][name]:
                continue
            arr = np.concatenate(acc[K][name], 0)
            row[name] = float(np.nanmedian(P.nse_table(arr, targ).nse))
            if name == "p":
                np.save(outdir / f"pred_K{K}.npy", arr)
                (outdir / f"K{K}").mkdir(exist_ok=True)
                P.nse_table(arr, targ).dump_metrics(str(outdir / f"K{K}"))
        summary[f"K={K}"] = row
        extra = "".join(f"  {n} {row[n]:+.4f}"
                        for n in ("nn", "cm", "idw") if n in row)
        print(f"  K={K:2d}  median NSE {row['p']:+.4f}{extra}", flush=True)

    with open(outdir / "run.json", "w") as f:
        json.dump({"model": "PUBModel", "extent": a.extent,
                   "protocol": a.protocol, "seed": a.seed,
                   "n_basins": int(targ.shape[0]),
                   "n_days": int(targ.shape[1]),
                   "epochs": a.epochs, "steps": a.steps, "tasks": a.tasks,
                   "k_train": a.k_train, "context_pool": a.context_pool,
                   "time_aligned": a.time_aligned, "geo": a.geo,
                   "causal": a.causal, "median_nse": summary}, f, indent=2)
    print(f"\nwrote {outdir}", flush=True)
    print("  nn = nearest donor, cm = context mean, idw = inverse-distance "
          "weighting -- the baselines K>0 must beat.", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=P.CAMELS_NC)
    ap.add_argument("--extent", choices=["PUB", "PUR", "temporal"],
                    default="PUB")
    ap.add_argument("--protocol", choices=["spatial", "temporal"], default=None)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--conn-depth", type=int, default=4)
    ap.add_argument("--d-ffd", type=int, default=512)
    ap.add_argument("--k-summary", type=int, default=3)
    ap.add_argument("--time-aligned", action="store_true", default=True)
    ap.add_argument("--no-time-aligned", dest="time_aligned",
                    action="store_false")
    ap.add_argument("--geo", action="store_true", default=True)
    ap.add_argument("--no-geo", dest="geo", action="store_false")
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--area-scale", action="store_true")
    ap.add_argument("--retrieval", choices=["geo", "similar", "random"],
                    default="geo")
    ap.add_argument("--context-pool", choices=["all", "train"], default="all")
    # NOTE: the fixed 137-patch (6-year) eval offset keeps historical
    # context inside the training period for the SPATIAL protocols (eval
    # 1995-99 -> context 1989-93). On the TEMPORAL extent late tiles would
    # reach into the eval span; mode B there needs a per-tile multiple of
    # 137 large enough to clear train_end. Not yet implemented -- do not
    # run --context-period train with --extent temporal for real results.
    ap.add_argument("--self-ctx-p", type=float, default=0.0,
                    help="TRAIN: probability a step adds the query as its "
                         "own context site with a random hidden tail")
    ap.add_argument("--recent-obs", type=int, default=0, metavar="P",
                    help="EVAL: own gauge visible to P patches before each "
                         "scored day; stride-1 final-patch scoring")
    ap.add_argument("--context-period", choices=["current", "train"],
                    default="current",
                    help="train = MODE B: context windows are HISTORICAL, "
                         "same-DOY whole-year offsets (align draw in "
                         "training, fixed 137-patch offset at eval)")
    ap.add_argument("--k-train", default="0,0,0,1,2,4,8,16")
    ap.add_argument("--k-eval", default="0,4")
    ap.add_argument("--lr", type=float, default=3e-4)
    # DEFAULTS ARE THE CONVERGED BUDGET. The old 40/150/4 (24k task-views)
    # were smoke-test defaults from train_pub.py, sized for 5-minute
    # architecture debugging; the converged budget only ever lived in
    # command-line overrides, so every run at defaults reproduced the
    # undertrained regime (K=0 0.265 at e40 vs 0.63+ at e800, fold 0).
    ap.add_argument("--smoke", action="store_true",
                    help="tiny debugging budget (the old defaults)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--tasks", type=int, default=8, help="tasks per step")
    ap.add_argument("--batch", type=int, default=8, help="basins per eval batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.protocol is None:
        args.protocol = "temporal" if args.extent == "temporal" else "spatial"
    main(args)
