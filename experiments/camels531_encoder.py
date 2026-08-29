"""Unit A standalone — the SiteEncoder with no context machinery at all.

The third CAMELS arm, and the one the first protocol rerun omitted. It exists
because **PUBModel's K=0 is not an honest no-context number.** Diagnosis.md
records the finding directly: capacity spent on the context pathway costs the
no-context pathway, the same encoder reaches 0.788 trained standalone against
0.526 as PUBModel's K=0 arm, and the recorded deployment conclusion is "do not
ship one model for both regimes."

So the comparison this file completes is:

    regional LSTM        forcings + statics, causal          the field's bar
    unit A standalone    forcings + statics, bidirectional   THIS FILE
    PUBModel K=0         the same encoder, but trained
                         alongside a context pathway
    PUBModel K>0         + neighbouring gauges at inference

`unit A standalone` vs `PUBModel K=0` is the arm that separates *the connector
damages the per-site path* from *the run was undertrained*. Both are trained
here at the same budget on the same folds, so the difference is attributable.

The three arms differ ONLY in architecture and context. All are trained on the
same objective — hide the query's streamflow for the whole window, reconstruct
it — over the same folds, the same periods, the same 365-day warmup, and are
scored by the same vendored `Metrics` on raw mm/day. `--mask-mix` switches to
the four-conditional mixture `train_site_encoder.py` uses; it is OFF by
default so this arm trains on exactly the task it is scored on, as the LSTM
does.
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
from hydropfn.data.forcing import MASK_KINDS, load_camels, sample_mask  # noqa: E402
from hydropfn.models.site_encoder import SiteEncoder, masked_mse  # noqa: E402
from hydropfn.paths import LOGS                                # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def tile_starts(span_days, patch, win):
    """Patch offsets tiling `span_days` with non-overlapping windows.

    The last window is pulled back to end on the final patch, so it may
    overlap its predecessor; predictions are written by day, never
    concatenated, so the overlap is a harmless overwrite.
    """
    n_patch = span_days // patch
    if n_patch < win:
        raise SystemExit(f"eval span {span_days} d shorter than one window")
    st = list(range(0, n_patch - win + 1, win))
    if st[-1] != n_patch - win:
        st.append(n_patch - win)
    return st


def make_batch(Xp, A_s, valid_p, doy, idx, starts, win, obs_col, rng,
               mask_mix=False):
    """One batch of single-site tasks with the query's streamflow hidden."""
    ser = np.stack([Xp[b, s:s + win] for b, s in zip(idx, starts)])
    val = np.stack([valid_p[b, s:s + win] for b, s in zip(idx, starts)])
    dd = np.stack([doy[s:s + win] for s in starts])
    V = ser.shape[2]
    vis = np.ones((len(idx), win, V), np.float32)
    if mask_mix:
        for j in range(len(idx)):
            vis[j] = sample_mask(win, V, rng, kind=str(rng.choice(MASK_KINDS)),
                                 n_obs=1)
    else:
        vis[:, :, obs_col] = 0.0                      # ungauged: hide it all
    t = lambda x: torch.tensor(x, dtype=torch.float32, device=DEVICE)  # noqa: E731
    return {"attrs": t(A_s[idx]), "series": t(ser), "vis": t(vis),
            "valid": t(val), "doy": t(dd)}


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = load_camels(a.nc)
    sub, gage = P.load_531(d)
    win_p = P.windows(d["time"], a.protocol)
    print(P.describe(a.extent, a.protocol, gage, d["time"]), flush=True)
    print(f"  device {DEVICE} | seed {a.seed} | window "
          f"{a.win} x {a.patch} = {a.win * a.patch} d"
          + ("  | MASK MIXTURE" if a.mask_mix else ""), flush=True)

    X, A_, valid = sub["x"], sub["attrs"], sub["valid"]
    obs = X.shape[-1] - 1
    q_raw = X[..., obs].astype(np.float32).copy()
    q_raw[valid[..., obs] == 0] = np.nan

    folds = P.folds(a.extent, gage)
    if a.max_folds:
        folds = folds[:a.max_folds]
        print(f"  PARTIAL: first {a.max_folds} fold(s) -- NOT the protocol "
              f"result", flush=True)

    preds, targs, fold_of, gages = [], [], [], []

    for kf, te_idx in enumerate(folds):
        tr_idx = (np.arange(len(gage)) if a.extent == "temporal"
                  else np.setdiff1d(np.arange(len(gage)), te_idx))
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

        net = SiteEncoder(A_s.shape[1], Xp.shape[2], a.patch, depth=a.depth,
                          d_ffd=a.d_ffd, k_summary=a.k_summary).to(DEVICE)
        if kf == 0:
            print(f"    SiteEncoder "
                  f"{sum(t.numel() for t in net.parameters()) / 1e6:.1f}M "
                  f"params (no connector, no context)", flush=True)

        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)

        lo_p, hi_p = tw.start // a.patch, tw.stop // a.patch - a.win
        t0 = time.time()
        for ep in range(a.epochs):
            net.train()
            tot = 0.0
            for _ in range(a.steps):
                b_idx = rng.choice(tr_idx, size=min(a.batch, len(tr_idx)),
                                   replace=False)
                st = rng.integers(lo_p, hi_p, size=len(b_idx))
                b = make_batch(Xp, A_s, valid_p, doy, b_idx, st, a.win, obs,
                               rng, a.mask_mix)
                out = net(b)
                loss = masked_mse(out["recon"], b["series"], b["vis"],
                                  b["valid"])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += loss.item()
            if (ep + 1) % 20 == 0 or ep == 0:
                print(f"      epoch {ep + 1}/{a.epochs}  masked MSE "
                      f"{tot / a.steps:.4f}  [{(time.time() - t0) / 60:.1f} "
                      f"min]", flush=True)

        # ---- evaluation, identical tiling and warmup handling to the others
        ev = win_p["eval_in"]
        p0 = ev.start // a.patch
        day0 = p0 * a.patch
        span = -(-(ev.stop - day0) // a.patch) * a.patch
        if day0 + span > X.shape[1]:
            raise SystemExit("record ends before the padded eval span")
        starts = tile_starts(span, a.patch, a.win)
        keep = slice(win_p["score"].start - day0, win_p["score"].stop - day0)
        if keep.stop > span:
            raise SystemExit("tiling does not cover the scored period")

        net.eval()
        buf = np.full((len(te_idx), span), np.nan, np.float32)
        with torch.no_grad():
            for st in starts:
                for i0 in range(0, len(te_idx), a.batch):
                    chunk = te_idx[i0:i0 + a.batch]
                    b = make_batch(Xp, A_s, valid_p, doy, chunk,
                                   np.full(len(chunk), p0 + st), a.win, obs,
                                   rng, mask_mix=False)
                    rec = net(b)["recon"][:, :, obs, :].cpu().numpy()
                    buf[i0:i0 + len(chunk),
                        st * a.patch:st * a.patch + a.win * a.patch] = \
                        rec.reshape(len(chunk), -1)

        p = (buf[:, keep] * sd[obs] + mu[obs]).astype(np.float32)
        preds.append(p)
        targs.append(q_raw[te_idx][:, win_p["score"]].astype(np.float32))
        fold_of.append(np.full(len(te_idx), kf))
        gages.append(gage["gage"].to_numpy()[te_idx])
        m = P.nse_table(p, targs[-1])
        print(f"    fold {kf} median NSE {np.nanmedian(m.nse):+.4f}",
              flush=True)

    pred = np.concatenate(preds, 0)
    targ = np.concatenate(targs, 0)
    met = P.nse_table(pred, targ)
    met.print_summary()

    tag = a.tag or f"unitA_{a.extent}_{a.protocol}_s{a.seed}"
    outdir = LOGS / "camels531" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "pred.npy", pred)
    np.save(outdir / "targ.npy", targ)
    np.save(outdir / "gage.npy", np.concatenate(gages))
    np.save(outdir / "fold.npy", np.concatenate(fold_of))
    met.dump_metrics(str(outdir))
    with open(outdir / "run.json", "w") as f:
        json.dump({"model": "SiteEncoder (unit A standalone)",
                   "extent": a.extent, "protocol": a.protocol, "seed": a.seed,
                   "median_nse": float(np.nanmedian(met.nse)),
                   "n_basins": int(pred.shape[0]),
                   "n_days": int(pred.shape[1]),
                   "epochs": a.epochs, "steps": a.steps,
                   "mask_mix": a.mask_mix}, f, indent=2)
    print(f"\nwrote {outdir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=P.CAMELS_NC)
    ap.add_argument("--extent", choices=["PUB", "PUR", "temporal"],
                    default="PUB")
    ap.add_argument("--protocol", choices=["spatial", "temporal"], default=None)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--d-ffd", type=int, default=512)
    ap.add_argument("--k-summary", type=int, default=3)
    ap.add_argument("--mask-mix", action="store_true",
                    help="train on the four-conditional mixture instead of "
                         "the whole_site task the arm is scored on")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.protocol is None:
        args.protocol = "temporal" if args.extent == "temporal" else "spatial"
    main(args)
