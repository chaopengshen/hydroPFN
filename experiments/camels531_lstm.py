"""Regional LSTM baseline on the CAMELS-531 protocol.

Replaces `experiments/lstm_baseline.py`, which is kept only for provenance.
Every difference is a correction to something that made the old number
incomparable to published CAMELS results:

  1. **Scores raw mm/day, not log1p.** The old file trained and scored on
     `log1p(QObs)` z-scored, then compared the result against Kratzert's and
     Jamaat's NSE, which are on raw discharge. Different metric, different
     number, neither bounding the other.
  2. **Warmup is excluded from scoring.** The old file dropped 120 days from
     the LOSS and scored them anyway, with a cold hidden state, under a
     docstring saying warmup was "excluded from both loss and scoring". Here
     the model is fed `warmup + scored` days and only the scored days count.
  3. **531 basins**, the Newman/Addor quality-filtered subset, not 671.
  4. **The published periods** -- train 1980-10-01..1999-09-30, score
     1995-10-01..1999-09-30 -- not a single 512-day window at day 9600.
  5. **PUB (10 groups) and PUR (7 regions)** leave-one-out, not one hand-picked
     HUC2 triple, and the full continuous test series per basin rather than
     16-day tails whose NSE denominator barely varies.
  6. **Metrics come from dmg_dev's own `Metrics` class**, vendored verbatim.

Hyperparameters follow dmg_dev's `conf/lstm.yaml`: hidden 256, dropout 0.5,
Adadelta at lr 1.0, batch 128, rho 365, warmup 365.

Two asymmetries against hydroPFN's own model remain, and both must be stated
wherever the two are compared:
  * this LSTM is CAUSAL; PUBModel is a bidirectional smoother unless run with
    --causal.
  * this LSTM cannot read another basin's concurrent discharge; PUBModel can.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P                       # noqa: E402
from hydropfn.data.forcing import load_camels                 # noqa: E402
from hydropfn.paths import LOGS                               # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class RegionalLSTM(nn.Module):
    """Forcings + broadcast statics -> streamflow, one model for all basins."""

    def __init__(self, n_forcing, n_attr, hidden=256, layers=1, dropout=0.5):
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


def fit_fold(forc, att, q, tr_idx, win, a, rng):
    """Train one LSTM on `tr_idx` basins over the training window.

    Sequences are `warmup + rho` = 730 days long and the loss covers only the
    last `rho` = 365, matching dmg_dev's HydroSampler
    (`select_subset(..., warmup)` prepends spin-up; the target is rho only).
    """
    net = RegionalLSTM(forc.shape[-1], att.shape[1], a.hidden, a.layers,
                       a.dropout).to(DEVICE)
    opt = torch.optim.Adadelta(net.parameters(), lr=a.lr)
    seq = a.warmup_train + P.RHO
    t0, lo, hi = time.time(), win["train"].start, win["train"].stop - seq
    if hi <= lo:
        raise SystemExit(f"training window shorter than one {seq}-day sequence")

    # Steps per epoch follow dmg_dev's rule unless overridden, so "50 epochs"
    # means the same data exposure as the reference runs.
    steps = a.steps or P.iters_per_epoch(
        len(tr_idx), win["train"].stop - win["train"].start, a.batch)
    print(f"    {steps} iter/ep x {a.epochs} ep = {steps * a.epochs} steps"
          f" (dmg_dev rule)" if not a.steps else
          f"    {steps} iter/ep x {a.epochs} ep (overridden)", flush=True)

    for ep in range(a.epochs):
        net.train()
        tot = 0.0
        for _ in range(steps):
            b = rng.choice(tr_idx, size=min(a.batch, len(tr_idx)),
                           replace=False)
            s = int(rng.integers(lo, hi))
            x = torch.tensor(forc[b, s:s + seq], device=DEVICE)
            y = torch.tensor(q[b, s:s + seq], device=DEVICE)
            aa = torch.tensor(att[b], device=DEVICE)
            p = net(x, aa)
            # Spin-up carries no gradient, exactly as it carries no score.
            yv, pv = y[:, a.warmup_train:], p[:, a.warmup_train:]
            m = torch.isfinite(yv).float()
            loss = torch.sqrt(
                (((pv - torch.nan_to_num(yv)) ** 2) * m).sum()
                / m.sum().clamp(min=1.0))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"    epoch {ep + 1}/{a.epochs}  RMSE {tot / steps:.4f}"
                  f"  [{(time.time() - t0) / 60:.1f} min]", flush=True)
    return net


def predict_fold(net, forc, att, te_idx, win, chunk=32):
    """Predictions for held-out basins over the SCORED days only.

    The model is fed WARMUP days before the scored period so its hidden state
    is spun up; those days are then dropped. This is the fix for the old
    file's cold-start scoring.
    """
    net.eval()
    out = []
    sl = win["eval_in"]
    with torch.no_grad():
        for i in range(0, len(te_idx), chunk):
            b = te_idx[i:i + chunk]
            x = torch.tensor(forc[b, sl], device=DEVICE)
            aa = torch.tensor(att[b], device=DEVICE)
            out.append(net(x, aa)[:, P.WARMUP:].cpu().numpy())
    return np.concatenate(out, 0)


def main(a):
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    d = load_camels(a.nc)                       # raw mm/day, no log transform
    n_bas = P.BENCHMARKS[a.protocol]["basins"] \
        if a.protocol in P.BENCHMARKS else 531
    sub, gage = P.load_subset(d, n_bas)
    win = P.windows(d["time"], a.protocol)
    print(P.describe(a.extent, a.protocol, gage, d["time"]), flush=True)
    print(f"  device {DEVICE} | seed {a.seed}", flush=True)

    X, A_ = sub["x"], sub["attrs"]
    obs = X.shape[-1] - 1
    forc_raw = np.delete(X, obs, axis=-1)
    q_raw = X[..., obs].astype(np.float32)          # mm/day, the scored scale
    q_raw[sub["valid"][..., obs] == 0] = np.nan

    folds = P.folds(a.extent, gage)
    if a.max_folds:
        # Smoke-test escape hatch. The aggregate is then over a SUBSET of the
        # 531 and is not the protocol number, so it is labelled as such.
        folds = folds[:a.max_folds]
        print(f"  PARTIAL: first {a.max_folds} fold(s) only -- NOT the "
              f"protocol result", flush=True)
    preds, targs, fold_of = [], [], []

    for k, te_idx in enumerate(folds):
        if a.extent == "temporal":
            tr_idx = np.arange(len(gage))       # same basins, earlier period
        else:
            tr_idx = np.setdiff1d(np.arange(len(gage)), te_idx)
        print(f"\n  fold {k}: train {len(tr_idx)} basins, "
              f"score {len(te_idx)} basins", flush=True)

        # Normalisation statistics come from the TRAINING basins over the
        # TRAINING window only. Computing them over the whole record would
        # leak the test period's climate into the held-out basins' inputs.
        tw = win["train"]
        mu = np.nanmean(forc_raw[tr_idx][:, tw], (0, 1))
        sd = np.nanstd(forc_raw[tr_idx][:, tw], (0, 1)) + 1e-6
        forc = np.nan_to_num((forc_raw - mu) / sd).astype(np.float32)

        am = np.nanmean(A_[tr_idx], 0)
        asd = np.nanstd(A_[tr_idx], 0) + 1e-6
        att = np.nan_to_num((A_ - am) / asd).astype(np.float32)

        # The TARGET is standardised for the loss and mapped straight back
        # afterwards, so training is well-conditioned while every reported
        # number lives on the mm/day scale.
        qmu = float(np.nanmean(q_raw[tr_idx][:, tw]))
        qsd = float(np.nanstd(q_raw[tr_idx][:, tw]) + 1e-6)
        qn = ((q_raw - qmu) / qsd).astype(np.float32)

        net = fit_fold(forc, att, qn, tr_idx, win, a, rng)
        n_par = sum(t.numel() for t in net.parameters())
        p = predict_fold(net, forc, att, te_idx, win) * qsd + qmu

        preds.append(p.astype(np.float32))
        targs.append(q_raw[te_idx][:, win["score"]].astype(np.float32))
        fold_of.append(np.full(len(te_idx), k))

        m = P.nse_table(p, targs[-1])
        print(f"    fold {k} median NSE {np.nanmedian(m.nse):+.4f}", flush=True)

    pred = np.concatenate(preds, 0)
    targ = np.concatenate(targs, 0)
    met = P.nse_table(pred, targ)
    met.print_summary()

    tag = a.tag or f"lstm_{a.extent}_{a.protocol}_s{a.seed}"
    outdir = LOGS / "camels531" / tag
    outdir.mkdir(parents=True, exist_ok=True)
    np.save(outdir / "pred.npy", pred)
    np.save(outdir / "targ.npy", targ)
    # Basin identity travels WITH the arrays. dmg_dev's aggregated .npy files
    # are holdout blocks concatenated, so column j is not basin j of the
    # subset file -- an assumption that already mislabelled every basin in two
    # case-study figures. Writing the order down costs nothing.
    np.save(outdir / "gage.npy",
            np.concatenate([gage["gage"].to_numpy()[f] for f in folds]))
    np.save(outdir / "fold.npy", np.concatenate(fold_of))
    met.dump_metrics(str(outdir))
    with open(outdir / "run.json", "w") as f:
        json.dump({"model": "RegionalLSTM", "extent": a.extent,
                   "protocol": a.protocol, "seed": a.seed,
                   "median_nse": float(np.nanmedian(met.nse)),
                   "n_basins": int(pred.shape[0]),
                   "n_days": int(pred.shape[1]),
                   "epochs": a.epochs, "steps": a.steps,
                   "hidden": a.hidden, "dropout": a.dropout,
                   "params": int(n_par)}, f, indent=2)
    print(f"\nwrote {outdir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", default=P.CAMELS_NC)
    ap.add_argument("--extent", choices=["PUB", "PUR", "temporal"],
                    default="PUB")
    ap.add_argument("--protocol", default=None,
                    choices=["spatial", "temporal"] + list(P.BENCHMARKS),
                    help="named benchmark (see protocol.BENCHMARKS) "
                         "or legacy spatial/temporal")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=1.0)      # Adadelta, dmg_dev
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--steps", type=int, default=0,
                    help="iterations per epoch; 0 = dmg_dev rule "
                         "ceil(log .01 / log(1 - batch*rho/N/(T-warmup)))")
    ap.add_argument("--warmup-train", type=int, default=P.WARMUP,
                    help="days of each training sequence excluded from loss")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-folds", type=int, default=0,
                    help="smoke test: run only the first N folds")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    if args.protocol is None:
        args.protocol = "temporal" if args.extent == "temporal" else "spatial"
    if args.protocol in P.BENCHMARKS:
        ok = P.BENCHMARKS[args.protocol]["extents"]
        if args.extent not in ok:
            raise SystemExit(
                f"benchmark {args.protocol} defines extents {ok}, "
                f"not {args.extent!r}")
    main(args)
