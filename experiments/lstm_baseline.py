"""Regional LSTM baseline — the U3 gate from docs/architecture.md.

Reproduces: logs/lstm_baseline_{tag}.csv

The PUB model beats `ctx_mean` and `nn_donor`, which are the right PUB
baselines. But `docs/architecture.md` specifies the gate as "must beat a
regional LSTM on the same basins under leave-region-out", and an LSTM is what
the hydrology literature will ask about. Without it, our numbers float free of
the field's reference point.

This is the standard CAMELS setup (Kratzert-style): one LSTM trained across all
training basins, forcings plus static attributes at every timestep, predicting
streamflow. It has NO access to context basins — that is precisely the point.
The comparison it licenses:

    LSTM            vs   our K=0     does the per-site pathway hold its own
                                     against the field's workhorse?
    LSTM            vs   our K>0     does conditioning on nearby gauges beat
                                     the workhorse -- the claim that matters

Evaluation is deliberately made COMPARABLE to the PUB runs rather than
maximally flattering to the LSTM: the same held-out basins, the same window,
the same log1p target. BOTH resolutions are reported -- daily and 16-day
means -- because train_pub.py scores DAILY values, and an earlier version of
this file scored only 16-day means while claiming to match it. Aggregation
generally raises R², so the resolution must be stated with every number.

Two asymmetries remain, both FAVOURING our model, and both must be disclosed
alongside any head-to-head:
  * this LSTM is CAUSAL; our model is a bidirectional smoother that sees the
    whole 512-day window, including days after the one it predicts.
  * our model reads context basins' CONCURRENT discharge; this one cannot.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data.forcing import load_camels  # noqa: E402
from hydropfn.paths import LOGS  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RegionalLSTM(nn.Module):
    """Forcings + broadcast statics -> streamflow, one model for all basins."""

    def __init__(self, n_forcing: int, n_attr: int, hidden: int = 256,
                 layers: int = 1, dropout: float = 0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_forcing + n_attr, hidden, layers,
                            batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, a):
        a = a[:, None, :].expand(-1, x.shape[1], -1)
        h, _ = self.lstm(torch.cat([x, a], dim=-1))
        return self.head(self.drop(h)).squeeze(-1)


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = load_camels(a.nc)
    X, A_, region = d["x"], d["attrs"], d["region"]

    hold = a.holdout.split(",")
    te = np.flatnonzero(np.isin(region, hold))
    tr = np.flatnonzero(~np.isin(region, hold))
    if a.eval_half:
        # Evaluate on EXACTLY the basins --train-on-neighbours evaluates on,
        # without training on the other half. Same rng, same permutation, so
        # the two arms differ ONLY in whether the neighbours were trained on.
        # Comparing 62-basin and 124-basin scores would confound the question
        # with which basins happened to be in the set.
        rs = np.random.default_rng(0)
        te = rs.permutation(te)[len(te) // 2:]
        print(f"  EVAL-HALF: scoring the same {len(te)} basins as the "
              f"neighbour-trained arm, but WITHOUT training on the others",
              flush=True)
    if a.train_on_neighbours:
        # THE ARM THAT SEPARATES THE TWO ROUTES TO NEIGHBOUR INFORMATION.
        # Split the held-out region in half: TRAIN on the training basins plus
        # half of it, EVALUATE on the other half. Evaluation basins are still
        # never trained on, but their geographic neighbours now ARE.
        #
        # This isolates what our in-context model does. Training on a
        # neighbour gives you its climatology and response function, baked
        # into weights. READING it at inference gives you its actual discharge
        # this week. Weights cannot know it rained last Tuesday; context can.
        # If this arm lands near LSTM(b), the advantage is real-time STATE.
        # If it lands near our model, the advantage was just more training
        # data and the in-context machinery is unnecessary.
        rs = np.random.default_rng(0)
        perm = rs.permutation(te)
        half = len(te) // 2
        nb_train, te = perm[:half], perm[half:]
        tr = np.concatenate([tr, nb_train])
        print(f"  NEIGHBOUR-TRAINED ARM: {len(nb_train)} held-out basins moved "
              f"INTO training; evaluating on the other {len(te)}", flush=True)
    elif a.include_test_gauges:
        # THE GAUGED CEILING. This LSTM trains on the test basins too, so it
        # answers "how well can this gauge be predicted if you HAVE its
        # record". Our PUB model must be read against it: beating it is not
        # automatically wrong (real-time neighbours are a different
        # information set from historical calibration at the gauge) but it
        # demands an explanation.
        tr = np.arange(len(region))
        print("  CEILING ARM: test gauges INCLUDED in training", flush=True)
    print(f"leave-region-out {hold}: {len(te)} held out / {len(tr)} train "
          f"| {DEVICE}", flush=True)

    mu, sd = X[tr].mean((0, 1)), X[tr].std((0, 1)) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)
    am, asd = A_[tr].mean(0), A_[tr].std(0) + 1e-6
    As = np.nan_to_num((A_ - am) / asd).astype(np.float32)
    obs = Xs.shape[-1] - 1
    forc = np.delete(Xs, obs, axis=-1)
    q = Xs[..., obs]

    T = X.shape[1]
    seq = a.patch * a.win                      # same span as the PUB window
    if a.train_end:
        print(f"  TEMPORAL SPLIT: training windows end by day {a.train_end}; "
              f"eval at day {a.eval_start * a.patch}", flush=True)
        if a.eval_start * a.patch < a.train_end:
            raise SystemExit("eval window is inside the training period")
    net = RegionalLSTM(forc.shape[-1], As.shape[1], a.hidden,
                       a.layers).to(DEVICE)
    print(f"  RegionalLSTM {sum(p.numel() for p in net.parameters())/1e6:.2f}M "
          f"params, seq {seq} d, warmup {a.warmup} d", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * a.steps, pct_start=0.1)

    t0 = time.time()
    for ep in range(a.epochs):
        net.train(); tot = 0.0
        for _ in range(a.steps):
            b = rng.choice(tr, size=a.batch, replace=False)
            hi = (a.train_end - seq) if a.train_end else (T - seq)
            s = rng.integers(0, max(1, hi))
            x = torch.tensor(forc[b, s:s + seq], device=DEVICE)
            y = torch.tensor(q[b, s:s + seq], device=DEVICE)
            aa = torch.tensor(As[b], device=DEVICE)
            p = net(x, aa)
            # warmup steps are excluded: the hidden state has to fill first,
            # and scoring them would flatter the baseline's competitors
            loss = ((p[:, a.warmup:] - y[:, a.warmup:]) ** 2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{a.epochs}  MSE {tot/a.steps:.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    # ---- evaluate on the SAME window the PUB runs use, aggregated to patches
    net.eval()
    s0 = a.eval_start * a.patch
    ys, ps = [], []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            b = te[i:i + 64]
            x = torch.tensor(forc[b, s0:s0 + seq], device=DEVICE)
            y = q[b, s0:s0 + seq]
            aa = torch.tensor(As[b], device=DEVICE)
            p = net(x, aa).cpu().numpy()
            # Keep DAILY values and aggregate at scoring time, so BOTH
            # resolutions get reported. The previous version scored only
            # 16-day means while train_pub.py scored daily values, under a
            # comment claiming the two matched. They did not, and aggregation
            # generally raises R2, so those numbers were never comparable.
            ys.append(y)
            ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)

    def _r2(u, v):
        u, v = u.ravel(), v.ravel()
        return float(1 - ((u - v) ** 2).sum() / ((u - u.mean()) ** 2).sum())

    # Median per-basin NSE -- the metric the hydrology literature reports,
    # and NOT the same quantity as the pooled R2 below. See nse_per_site in
    # train_pub.py for why pooling inflates the score.
    den = ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
    nse = 1 - ((y - p) ** 2).sum(1) / np.where(den > 0, den, np.nan)
    nse = nse[np.isfinite(nse)]
    nse_med, nse_pos = float(np.median(nse)), float((nse > 0).mean())

    n = seq // a.patch
    r2_daily = _r2(y, p)
    r2_patch = _r2(y.reshape(-1, n, a.patch).mean(-1),
                   p.reshape(-1, n, a.patch).mean(-1))
    r2 = r2_daily
    print(f"\n=== regional LSTM, held-out basins ===")
    print(f"  R2 DAILY (pooled) = {r2_daily:+.4f}   (n = {y.size:,})")
    print(f"  R2 16-d means     = {r2_patch:+.4f}")
    print(f"  per-basin NSE med = {nse_med:+.4f}  "
          f"({nse_pos:.0%} of {nse.size} basins > 0)")
    print("  train_pub.py scores DAILY -- compare against that one.")
    print("\n  compare to the PUB runs on the same holdout:")
    print("    our K=0 (no context)   ~0.46      per-site pathway only")
    print("    our K=4 (with context) ~0.85      the claim")
    pd.DataFrame([{"holdout": a.holdout, "seed": a.seed,
                   "r2_daily": r2_daily, "r2_patch16": r2_patch,
                   "nse_median": nse_med, "nse_frac_pos": nse_pos,
                   "n": int(y.size)}]).to_csv(
        LOGS / f"lstm_baseline_{a.tag}.csv", index=False)
    print(f"\nwrote {LOGS / f'lstm_baseline_{a.tag}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True)
    ap.add_argument("--holdout", default="01,11,17")
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=120)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--eval-start", type=int, default=200,
                    help="evaluation window start, in PATCHES")
    ap.add_argument("--train-end", type=int, default=None,
                    help="training windows must end by this DAY")
    ap.add_argument("--eval-half", action="store_true",
                    help="score the same half the neighbour-trained arm "
                         "scores, without training on the other half")
    ap.add_argument("--train-on-neighbours", action="store_true",
                    help="train on half the held-out region, evaluate on the "
                         "other half -- neighbours trained-on, query not")
    ap.add_argument("--include-test-gauges", action="store_true",
                    help="train on the held-out basins too -- the GAUGED "
                         "CEILING arm")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="lstm")
    main(ap.parse_args())
