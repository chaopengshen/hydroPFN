"""Matched daily DI baseline and frozen-weight adaptation on unseen basins.

Use --extent PUB --protocol temporal to hold out BOTH basins and the future
period. Campaign observations end before scoring; daily observations arrive
strictly before each prediction. This is a recurrent baseline for hydroPFN's
observation interface, not a claim that the transformer has been repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels
from hydropfn.data.observations import training_di_views, evaluation_di_features
from hydropfn.paths import LOGS
from hydropfn.train.objectives import basin_normalized_mse, training_variances
from camels531_lstm import RegionalLSTM

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fit(forc, attrs, q, tr, windows, variances, a, rng, availability_rng, training_log=None):
    net = RegionalLSTM(forc.shape[-1] + 4, attrs.shape[-1], a.hidden,
                       dropout=a.dropout).to(DEVICE)
    opt = torch.optim.Adadelta(net.parameters(), lr=a.lr)
    seq = P.WARMUP + P.RHO
    lo, hi = windows["train"].start, windows["train"].stop - seq
    steps = a.steps or P.iters_per_epoch(len(tr), windows["train"].stop - lo, a.batch)
    views_per_step = 2 if a.mode == "paired" else 1
    print(f"{steps} updates/epoch; {views_per_step} observation views/update; "
          "one averaged loss and one optimizer update per sampled batch", flush=True)
    start = time.time()
    for epoch in range(a.epochs):
        net.train()
        total = 0.
        view_totals = np.zeros(views_per_step)
        for _ in range(steps):
            b = rng.choice(tr, size=min(a.batch, len(tr)), replace=False)
            s = int(rng.integers(lo, hi))
            qs = q[b, s:s + seq]
            at = torch.tensor(attrs[b], device=DEVICE)
            target = torch.tensor(qs[:, P.WARMUP:], device=DEVICE)
            valid = torch.isfinite(target)
            views = training_di_views(qs, a.mode, availability_rng, a.paired_second)
            opt.zero_grad()
            for vi, (label, di) in enumerate(views):
                x = torch.tensor(np.concatenate([forc[b, s:s + seq], di], -1), device=DEVICE)
                pred = net(x, at)[:, P.WARMUP:]
                if a.loss == "nse":
                    loss = basin_normalized_mse(pred, target, valid,
                                                torch.tensor(variances[b], device=DEVICE))
                else:
                    err = torch.where(valid, pred - torch.nan_to_num(target), 0.)
                    loss = (err.square().sum() / valid.sum().clamp(min=1)).sqrt()
                # Sequential backward frees each graph while accumulating the
                # exact mean-loss gradient. Clip and step ONCE after both views.
                (loss / len(views)).backward()
                view_totals[vi] += loss.item()
                total += loss.item() / len(views)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
        row = {"epoch": epoch + 1, "mean_loss": total / steps,
               "view_losses": {f"view{i}_{label}": float(view_totals[i] / steps)
                               for i, (label, _) in enumerate(views)},
               "optimizer_updates": (epoch + 1) * steps,
               "sampled_window_views": (epoch + 1) * steps * min(a.batch, len(tr)) * views_per_step,
               "elapsed_seconds": time.time() - start}
        if training_log is not None:
            with Path(training_log).open("a") as log:
                log.write(json.dumps(row) + "\n")
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f"epoch {epoch + 1}/{a.epochs} {a.loss}={total / steps:.5f} "
                  f"views={row['view_losses']} "
                  f"elapsed={(time.time() - start) / 60:.1f} min", flush=True)
    net.training_budget = {"optimizer_updates": a.epochs * steps,
                           "views_per_update": views_per_step,
                           "sampled_basin_windows": a.epochs * steps * min(a.batch, len(tr)),
                           "sampled_window_views": row["sampled_window_views"]}
    return net


@torch.no_grad()
def predict(net, forc, attrs, q, indices, windows, scenario, chunk=32):
    net.eval()
    output = []
    sl = windows["eval_in"]
    for offset in range(0, len(indices), chunk):
        b = indices[offset:offset + chunk]
        di = evaluation_di_features(q[b, sl], scenario)
        x = np.concatenate([forc[b, sl], di], -1)
        pred = net(torch.tensor(x, device=DEVICE), torch.tensor(attrs[b], device=DEVICE))
        output.append(pred[:, P.WARMUP:].cpu().numpy())
    return np.concatenate(output)


def main(a):
    sub, gage = P.load_531(load_camels(a.nc))
    windows = P.windows(sub["time"], a.protocol)
    print(P.describe(a.extent, a.protocol, gage, sub["time"]), flush=True)
    print(f"daily DI: device={DEVICE}, mode={a.mode}, loss={a.loss}", flush=True)
    X, raw_attrs = sub["x"], sub["attrs"]
    q_raw = X[..., -1].astype(np.float32).copy()
    q_raw[sub["valid"][..., -1] == 0] = np.nan
    folds = P.folds(a.extent, gage)
    if a.max_folds:
        folds = folds[:a.max_folds]
        print("PARTIAL fold pilot; not the full 531-basin result", flush=True)
    scenarios = a.scenarios.split(",")
    predictions = {s: [] for s in scenarios}
    targets, ids, fold_ids = [], [], []
    tag = a.tag or f"daily_di_{a.mode}_{a.extent}_{a.protocol}_{a.loss}_s{a.seed}"
    out = LOGS / "camels531" / tag
    if out.exists() and not a.resume:
        raise ValueError(f"refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=a.resume)
    config = {k: v for k, v in vars(a).items() if k != "resume"}
    config["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_path = out / "config.json"
    if a.resume:
        if json.loads(config_path.read_text()) != config:
            raise ValueError("resume configuration or driver source differs")
        if (out / "run.json").exists():
            raise ValueError("this experiment is already complete")
    else:
        config_path.write_text(json.dumps(config, indent=2))
    for k, te in enumerate(folds):
        # Independent fold seeds make interrupted cross-validation resumable
        # while keeping the forward/daily/mixed draws identical within a fold.
        torch.manual_seed(a.seed + 1009 * k)
        rng = np.random.default_rng(a.seed + 1009 * k)
        availability_rng = np.random.default_rng(a.seed + 1009 * k + 10000)
        target = q_raw[te, windows["score"]]
        fold_record = out / f"fold{k}_complete.json"
        if a.resume and fold_record.exists():
            record = json.loads(fold_record.read_text())
            if record["gages"] != gage["gage"].to_numpy()[te].astype(str).tolist():
                raise ValueError("resume gauge IDs differ")
            np.testing.assert_allclose(np.load(out / f"fold{k}_target.npy"), target,
                                       rtol=0, atol=0, equal_nan=True)
            for scenario in scenarios:
                predictions[scenario].append(np.load(out / f"fold{k}_pred_{scenario}.npy"))
            targets.append(target)
            ids.append(gage["gage"].to_numpy()[te])
            fold_ids.append(np.full(len(te), k))
            print(f"fold {k}: reused completed artifacts", flush=True)
            continue
        tr = (np.arange(len(gage)) if a.extent == "temporal"
              else np.setdiff1d(np.arange(len(gage)), te))
        tw = windows["train"]
        raw_forc = X[..., :-1]
        mean, std = np.nanmean(raw_forc[tr, tw], (0, 1)), np.nanstd(raw_forc[tr, tw], (0, 1)) + 1e-6
        forc = np.nan_to_num((raw_forc - mean) / std).astype(np.float32)
        am, ast = np.nanmean(raw_attrs[tr], 0), np.nanstd(raw_attrs[tr], 0) + 1e-6
        attrs = np.nan_to_num((raw_attrs - am) / ast).astype(np.float32)
        qm, qs = float(np.nanmean(q_raw[tr, tw])), float(np.nanstd(q_raw[tr, tw]) + 1e-6)
        q = ((q_raw - qm) / qs).astype(np.float32)
        variance = training_variances(q_raw, tr, tw, qs, a.nse_min_std)
        print(f"fold {k}: {len(tr)} train, {len(te)} test basins", flush=True)
        net = fit(forc, attrs, q, tr, windows, variance, a, rng, availability_rng,
                  training_log=out / f"fold{k}_training.jsonl")
        torch.save(net.state_dict(), out / f"fold{k}.pt")
        np.savez(out / f"fold{k}_normalization.npz", mean=mean, std=std,
                 attr_mean=am, attr_std=ast, q_mean=qm, q_std=qs,
                 train_gage=gage["gage"].to_numpy()[tr].astype(str))
        fold_summary = {}
        for scenario in scenarios:
            pred = predict(net, forc, attrs, q, te, windows, scenario) * qs + qm
            predictions[scenario].append(pred.astype(np.float32))
            np.save(out / f"fold{k}_pred_{scenario}.npy", pred.astype(np.float32))
            fold_summary[scenario] = float(np.nanmedian(P.nse_table(pred, target).nse))
            print(f"fold {k} {scenario}: median NSE="
                  f"{fold_summary[scenario]:.4f}", flush=True)
        np.save(out / f"fold{k}_target.npy", target)
        fold_record.write_text(json.dumps({"gages": gage["gage"].to_numpy()[te].astype(str).tolist(),
                                           "median_nse": fold_summary,
                                           "training_budget": net.training_budget}, indent=2))
        targets.append(target)
        ids.append(gage["gage"].to_numpy()[te])
        fold_ids.append(np.full(len(te), k))
    target = np.concatenate(targets)
    np.save(out / "targ.npy", target)
    np.save(out / "gage.npy", np.concatenate(ids))
    np.save(out / "fold.npy", np.concatenate(fold_ids))
    summary = {}
    for scenario, values in predictions.items():
        pred = np.concatenate(values)
        np.save(out / f"pred_{scenario}.npy", pred)
        metric = P.nse_table(pred, target)
        summary[scenario] = {"median_nse": float(np.nanmedian(metric.nse))}
        # Fixed early-year adaptation readout, separate from the full protocol.
        if target.shape[1] >= 365:
            summary[scenario]["first_year_median_nse"] = float(np.nanmedian(
                P.nse_table(pred[:, :365], target[:, :365]).nse))
    metadata = {"model": "DailyDILSTM", "extent": a.extent, "protocol": a.protocol,
                "n_basins": len(target), "n_days": target.shape[1], "seed": a.seed,
                "information": "Q observations strictly before prediction day; fixed weights across scenarios",
                "partial_folds": bool(a.max_folds), "median_nse": summary, "args": vars(a)}
    (out / "run.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=P.CAMELS_NC)
    p.add_argument("--extent", choices=["temporal", "PUB", "PUR"], default="temporal")
    p.add_argument("--protocol", choices=["temporal", "spatial"], default="temporal")
    p.add_argument("--mode", choices=["forward", "daily", "mixed", "paired"], default="daily")
    p.add_argument("--paired-second", choices=["mixed", "daily"], default="mixed",
                   help="paired mode always includes daily; daily as second view is the compute control")
    p.add_argument("--loss", choices=["rmse", "nse"], default="rmse")
    p.add_argument("--nse-min-std", type=float, default=0.1)
    p.add_argument("--scenarios", default="none,daily,lag7,lag16,weekly,campaign30,campaign90,campaign365")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=1.)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-folds", type=int, default=0)
    p.add_argument("--tag")
    p.add_argument("--resume", action="store_true", help="reuse completed folds with identical configuration")
    main(p.parse_args())
