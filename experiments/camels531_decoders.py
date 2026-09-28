"""Frozen-PFN readout and raw-input LSTM/TCN comparison.

Preparation caches evaluation and fixed-control 512-day PFN windows. Daily
sampling recomputes frozen features for fresh crops; training scores their final
256 days. Every evaluated day also has at least 256 real preceding days. This
is a controlled temporal K=0 hindcast pilot, not the original 730-day LSTM
benchmark or a causal-DI experiment. PFN features retain noncausal forcing
attention; all query discharge values are removed before encoding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels
from hydropfn.data.decoder_windows import training_starts, evaluation_windows
from hydropfn.models.site_encoder import SiteEncoder
from hydropfn.models.connector import PUBModel
from hydropfn.models.temporal_decoders import (DailySequenceDecoder,
                                             PatchReconstructionDecoder, lift_patch_features)
from hydropfn.train.objectives import basin_normalized_mse, training_variances
from hydropfn.paths import LOGS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def encoder_batch(forcing, attrs, start, length, patch, device, time_offset=0):
    """Construct K=0 inputs from forcing ONLY; target values cannot enter."""
    B, _, F = forcing.shape
    if length % patch:
        raise ValueError("window must contain whole patches")
    x = forcing[:, start:start + length]
    if x.shape[1] != length:
        raise ValueError("window exceeds forcing record")
    valid = np.isfinite(x)
    x = np.concatenate([np.nan_to_num(x), np.zeros((B, length, 1), np.float32)], -1)
    valid = np.concatenate([valid, np.ones((B, length, 1), bool)], -1)
    N, V = length // patch, F + 1
    series = x.reshape(B, N, patch, V).transpose(0, 1, 3, 2)
    real = valid.reshape(B, N, patch, V).all(2)
    vis = np.ones((B, N, V), np.float32)
    vis[:, :, -1] = 0
    doy = ((start + time_offset + np.arange(N) * patch) % 365.25) / 365.25
    values = dict(series=series[:, None], attrs=attrs[:, None],
                  valid=real[:, None].astype(np.float32), vis=vis[:, None],
                  doy=np.broadcast_to(doy, (B, 1, N)).copy(),
                  site_valid=np.ones((B, 1), np.float32),
                  # K=0 relative geometry is identically zero, but the learned
                  # Fourier projection still has a nonzero constant response.
                  latlon=np.zeros((B, 1, 2), np.float32))
    return {k: torch.tensor(v, device=device, dtype=torch.float32) for k, v in values.items()}


def prepare(a):
    out = Path(a.cache)
    if out.exists():
        raise ValueError(f"refusing to overwrite cache {out}")
    source = Path(a.source)
    meta = json.loads((source / "run.json").read_text())
    if meta["extent"] != "temporal" or meta["protocol"] != "temporal":
        raise ValueError("this pilot requires the temporal PFN checkpoint")
    cfg = meta["args"]
    if cfg.get("dem_npz") or cfg.get("area_scale"):
        raise ValueError("DEM/area checkpoints are not supported by this K=0 probe")
    sub, gage = P.load_531(load_camels(a.nc))
    ids = gage["gage"].to_numpy().astype(str)
    w = P.windows(sub["time"], "temporal")
    norm = np.load(source / "fold0_normalization.npz")
    np.testing.assert_array_equal(norm["train_gage"].astype(str), ids)
    patch, length = cfg["patch"], cfg["patch"] * cfg["win"]
    if not 0 < a.warmup < length:
        raise ValueError("warmup must be positive and shorter than input window")
    tr_starts = training_starts(w["train"].start, w["train"].stop, length, a.train_stride)
    ev = evaluation_windows(w["score"].start, w["score"].stop, length, a.warmup)
    raw = np.where(sub["valid"], sub["x"], np.nan).astype(np.float32)
    q_raw = raw[..., -1].copy()
    forcing = ((raw[..., :-1] - norm["mean"][:-1]) / norm["std"][:-1]).astype(np.float32)
    attrs = np.nan_to_num((sub["attrs"] - norm["attr_mean"]) / norm["attr_std"]).astype(np.float32)
    enc = SiteEncoder(attrs.shape[-1], raw.shape[-1], patch=patch, depth=cfg["depth"],
                      k_summary=cfg["k_summary"], d_ffd=cfg["d_ffd"])
    net = PUBModel(enc, depth=cfg["conn_depth"], time_aligned=cfg["time_aligned"],
                   geo=cfg["geo"], causal=cfg["causal"], no_pooled=cfg.get("no_pooled", False))
    net.load_state_dict(torch.load(source / "fold0.pt", map_location="cpu", weights_only=True))
    net.to(DEVICE).eval().requires_grad_(False)
    out.mkdir(parents=True)
    begin = time.time()
    old_pred = np.full((len(ids), w["score"].stop - w["score"].start), np.nan, np.float32)
    counts = np.zeros(old_pred.shape[1], int)
    for split, starts in (("train", tr_starts), ("eval", [v[0] for v in ev])):
        features = np.lib.format.open_memmap(out / f"{split}_features.npy", mode="w+",
                    dtype=np.float32, shape=(len(starts), len(ids), length // patch, enc.d))
        for wi, start in enumerate(starts):
            for lo in range(0, len(ids), a.chunk):
                ix = slice(lo, min(lo + a.chunk, len(ids)))
                batch = encoder_batch(forcing[ix], attrs[ix], start, length, patch, DEVICE)
                with torch.no_grad():
                    latent = net(batch, return_hidden=True)[:, :, -1]
                    features[wi, ix] = latent.cpu().numpy()
                    if split == "eval":
                        pred = net.head(latent).flatten(1).cpu().numpy()
                        _, score_lo, score_hi = ev[wi]
                        take = slice(score_lo - start, score_hi - start)
                        dest = slice(score_lo - w["score"].start, score_hi - w["score"].start)
                        old_pred[ix, dest] = pred[:, take] * norm["std"][-1] + norm["mean"][-1]
            if split == "eval":
                counts[dest] += 1
            if wi == 0 or (wi + 1) % 10 == 0 or wi + 1 == len(starts):
                print(f"cache {split} {wi+1}/{len(starts)} elapsed={time.time()-begin:.1f}s", flush=True)
        features.flush()
        del features
    if not np.all(counts == 1):
        raise ValueError("scoring dates must be covered exactly once")
    target = q_raw[:, w["score"]]
    variance = training_variances(q_raw, np.arange(len(ids)), w["train"],
                                  float(norm["std"][-1]), 0.1)
    np.savez(out / "data.npz", forcing=forcing, attrs=attrs, q_raw=q_raw,
             q_mean=norm["mean"][-1], q_std=norm["std"][-1], variance=variance,
             gage=ids, train_starts=np.array(tr_starts), eval_windows=np.array(ev))
    np.save(out / "original_head_pred.npy", old_pred)
    np.save(out / "targ.npy", target)
    record = dict(source=str(source), source_weights_sha256=sha(source / "fold0.pt"),
                  source_normalization_sha256=sha(source / "fold0_normalization.npz"),
                  data_file=str(a.nc), data_sha256=sha(a.nc), n_basins=len(ids),
                  length=length, patch=patch, warmup=a.warmup, score_days=target.shape[1],
                  train_start=w["train"].start, train_stop=w["train"].stop,
                  score_start=w["score"].start, score_stop=w["score"].stop,
                  train_stride=a.train_stride, n_train_windows=len(tr_starts),
                  n_eval_windows=len(ev), latent_dim=enc.d,
                  original_head_median_nse=float(np.nanmedian(P.nse_table(old_pred, target).nse)),
                  information="K=0, no Q inputs; frozen noncausal forcing encoder; hindcast only",
                  normalization="unchanged training-only checkpoint normalization",
                  query_valid="Q token always valid, Q value always zero and masked",
                  original_head_note="Same inputs and score windows as new decoders; not exact historical tiling",
                  prepare_seconds=time.time()-begin,
                  files_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    (out / "manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


def make_inputs(data, features, wi, basins, length, patch, device, context=True):
    start = int(data["train_starts"][wi])
    forcing = np.nan_to_num(data["forcing"][basins, start:start+length])
    x = torch.tensor(forcing, device=device)
    attrs = torch.tensor(data["attrs"][basins], device=device)
    z = (lift_patch_features(torch.tensor(np.array(features[wi, basins]), device=device), patch)
         if context else None)
    return x, attrs, z, start


def sample_start(rng, starts, manifest, sampling):
    """One uniform draw in either arm preserves the paired basin RNG stream."""
    u = float(rng.random())
    if sampling == "fixed":
        wi = min(int(u * len(starts)), len(starts)-1)
        return int(starts[wi]), wi, u
    if sampling != "daily":
        raise ValueError("sampling must be fixed or daily")
    count = manifest["train_stop"] - manifest["length"] - manifest["train_start"] + 1
    if count < 1:
        raise ValueError("training interval is shorter than one window")
    return manifest["train_start"] + min(int(u * count), count-1), None, u


def load_frozen_encoder(source, manifest, data, device):
    """Recreate precisely the encoder that produced the evaluation cache."""
    source = Path(source)
    for filename, key in (("fold0.pt", "source_weights_sha256"),
                          ("fold0_normalization.npz", "source_normalization_sha256")):
        if sha(source / filename) != manifest[key]:
            raise ValueError(f"source does not match cache: {filename}")
    meta = json.loads((source / "run.json").read_text())
    if meta["extent"] != "temporal" or meta["protocol"] != "temporal":
        raise ValueError("temporal source checkpoint required")
    cfg = meta["args"]
    if cfg.get("dem_npz") or cfg.get("area_scale"):
        raise ValueError("DEM/area source not supported")
    if cfg["patch"] != manifest["patch"] or cfg["patch"]*cfg["win"] != manifest["length"]:
        raise ValueError("source window geometry differs from cache")
    with np.load(source / "fold0_normalization.npz") as norm:
        np.testing.assert_array_equal(norm["train_gage"].astype(str), data["gage"].astype(str))
    enc = SiteEncoder(data["attrs"].shape[-1], data["forcing"].shape[-1]+1,
                      patch=cfg["patch"], depth=cfg["depth"],
                      k_summary=cfg["k_summary"], d_ffd=cfg["d_ffd"])
    net = PUBModel(enc, depth=cfg["conn_depth"], time_aligned=cfg["time_aligned"],
                   geo=cfg["geo"], causal=cfg["causal"], no_pooled=cfg.get("no_pooled", False))
    net.load_state_dict(torch.load(source / "fold0.pt", map_location="cpu", weights_only=True))
    return net.to(device).eval().requires_grad_(False)


@torch.no_grad()
def online_inputs(encoder, data, basins, start, manifest, chunk, device):
    """Encode the actual crop; never slice contextual features from another crop."""
    if chunk < 1 or start < manifest["train_start"] or start+manifest["length"] > manifest["train_stop"]:
        raise ValueError("invalid chunk or training crop outside training interval")
    length, patch = manifest["length"], manifest["patch"]
    raw = data["forcing"][basins, start:start+length]
    attrs = data["attrs"][basins]
    x = torch.tensor(np.nan_to_num(raw), device=device)
    a = torch.tensor(attrs, device=device)
    z = None
    if encoder is not None:
        if encoder.training or any(p.requires_grad for p in encoder.parameters()):
            raise ValueError("online feature extraction requires a frozen eval-mode encoder")
        parts = []
        for lo in range(0, len(basins), chunk):
            batch = encoder_batch(raw[lo:lo+chunk], attrs[lo:lo+chunk], 0,
                                  length, patch, device, time_offset=start)
            parts.append(encoder(batch, return_hidden=True)[:, :, -1])
        z = lift_patch_features(torch.cat(parts), patch)
    return x, a, z, start


def weights_sha256(net):
    h = hashlib.sha256()
    for name, value in net.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def evaluate(net, data, features, manifest, chunk, device, ablate=False):
    net.eval()
    length, patch = manifest["length"], manifest["patch"]
    pred = np.full((manifest["n_basins"], manifest["score_days"]), np.nan, np.float32)
    for wi, (start, lo, hi) in enumerate(data["eval_windows"]):
        for b0 in range(0, len(pred), chunk):
            b = slice(b0, b0+chunk)
            x = torch.tensor(np.nan_to_num(data["forcing"][b, start:start+length]), device=device)
            attrs = torch.tensor(data["attrs"][b], device=device)
            z = None
            if net.latent_dim:
                z = lift_patch_features(torch.tensor(np.array(features[wi, b]), device=device), patch)
                if ablate:
                    z.zero_()
            p = net(x, attrs, z).cpu().numpy()
            pred[b, lo-manifest["score_start"]:hi-manifest["score_start"]] = p[:, lo-start:hi-start]
    return pred * float(data["q_std"]) + float(data["q_mean"])


def train(a):
    sampling = getattr(a, "sampling", "fixed")
    feature_mode = getattr(a, "feature_mode", "auto")
    if feature_mode == "auto":
        feature_mode = "online" if sampling == "daily" else "cached"
    if sampling == "daily" and feature_mode == "cached":
        raise ValueError("daily starts require online features; fixed cached crops cannot be sliced")
    checkpoint_epochs = sorted(set(int(v) for v in
                              getattr(a, "checkpoint_epochs", "50").split(",") if v))
    if any(v < 1 for v in checkpoint_epochs):
        raise ValueError("checkpoint epochs must be positive")
    checkpoint_epochs = [v for v in checkpoint_epochs if v <= a.epochs]
    if getattr(a, "evaluate_checkpoints", False) and not checkpoint_epochs:
        raise ValueError("checkpoint evaluation requires a checkpoint within the run")
    if a.decoder == "patch" and a.inputs != "latent":
        raise ValueError("original patch head requires --inputs latent")
    cache = Path(a.cache)
    m = json.loads((cache / "manifest.json").read_text())
    # Verify cache provenance once for each process before training.
    for name, expected in m["files_sha256"].items():
        if sha(cache / name) != expected:
            raise ValueError(f"cache changed: {name}")
    data = dict(np.load(cache / "data.npz"))
    features = np.load(cache / "train_features.npy", mmap_mode="r")
    eval_features = np.load(cache / "eval_features.npy", mmap_mode="r")
    out = LOGS / "camels531" / a.tag
    if out.exists():
        raise ValueError(f"refusing to overwrite {out}")
    encoder = None
    encoder_chunk = getattr(a, "encoder_chunk", 32)
    if feature_mode == "online" and a.inputs != "raw":
        source = getattr(a, "source", None) or m.get("source")
        if not source:
            raise ValueError("daily PFN sampling needs the original encoder checkpoint")
        encoder = load_frozen_encoder(source, m, data, DEVICE)
        # The load/initialization must not change decoder initialization or
        # subsequent dropout draws relative to the fixed-window control.
    encoder_digest = weights_sha256(encoder) if encoder is not None else None
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    net = (PatchReconstructionDecoder(m["latent_dim"], m["patch"]) if a.decoder == "patch"
           else DailySequenceDecoder(data["forcing"].shape[-1], data["attrs"].shape[-1],
            kind=a.decoder, latent_dim=m["latent_dim"] if a.inputs != "raw" else 0,
            use_raw=a.inputs != "latent", hidden=a.hidden, tcn_width=a.tcn_width,
            tcn_blocks=a.tcn_blocks, dropout=a.dropout, patch=m["patch"])).to(DEVICE)
    if net.receptive_field is not None and net.receptive_field-1 > m["warmup"]:
        raise ValueError("TCN history exceeds real warmup; increase common history")
    out.mkdir(parents=True)
    config = dict(vars(a), cache_manifest_sha256=sha(cache / "manifest.json"),
                  n_parameters=sum(p.numel() for p in net.parameters()),
                  receptive_field=net.receptive_field, device=DEVICE,
                  gpu_name=torch.cuda.get_device_name() if DEVICE == "cuda" else None,
                  runtime=dict(python=platform.python_version(), torch=str(torch.__version__),
                               numpy=np.__version__, cuda=torch.version.cuda,
                               cudnn=torch.backends.cudnn.version(), host=platform.node(),
                               slurm_job_id=os.environ.get("SLURM_JOB_ID")),
                  checkpoint_epochs=checkpoint_epochs)
    config.update(sampling=sampling, feature_mode=feature_mode, initial_weights_sha256=weights_sha256(net),
                  sampler="paired_uniform_v1", encoder_frozen=True,
                  online_encoder_initial_sha256=encoder_digest,
                  n_available_train_starts=(len(data["train_starts"]) if sampling == "fixed"
                    else m["train_stop"]-m["length"]-m["train_start"]+1))
    (out / "config.json").write_text(json.dumps(config, indent=2))
    opt = torch.optim.Adadelta(net.parameters(), lr=a.lr)
    rho = m["length"] - m["warmup"]
    steps = a.steps or P.iters_per_epoch(m["n_basins"], m["train_stop"]-m["train_start"],
                                       a.batch, rho=rho, warmup=m["warmup"])
    start_time = time.time()
    pair_stream = hashlib.sha256()
    start_counts = np.zeros(m["train_stop"]-m["length"]-m["train_start"]+1, dtype=np.int64)
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(a.epochs):
        net.train()
        total = 0.
        for _ in range(steps):
            b = rng.choice(m["n_basins"], size=min(a.batch, m["n_basins"]), replace=False)
            start, wi, u = sample_start(rng, data["train_starts"], m, sampling)
            pair_stream.update(np.asarray(b, dtype="<i8").tobytes())
            pair_stream.update(np.asarray([u], dtype="<f8").tobytes())
            start_counts[start-m["train_start"]] += 1
            if feature_mode == "cached":
                x, attrs, z, start = make_inputs(data, features, wi, b, m["length"], m["patch"],
                                               DEVICE, context=bool(net.latent_dim))
            else:
                x, attrs, z, start = online_inputs(encoder, data, b, start, m, encoder_chunk, DEVICE)
            target = (data["q_raw"][b, start+m["warmup"]:start+m["length"]]
                      - data["q_mean"]) / data["q_std"]
            y = torch.tensor(target, device=DEVICE)
            pred = net(x, attrs, z)[:, m["warmup"]:]
            loss = basin_normalized_mse(pred, y, torch.isfinite(y),
                                        torch.tensor(data["variance"][b], device=DEVICE))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite loss")
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            total += loss.item()
        row = dict(epoch=epoch+1, loss=total/steps, updates=(epoch+1)*steps,
                   elapsed_seconds=time.time()-start_time,
                   unique_train_starts=int(np.count_nonzero(start_counts)))
        with (out / "training.jsonl").open("a") as f:
            f.write(json.dumps(row)+"\n")
        if epoch == 0 or (epoch+1) % 10 == 0:
            print(json.dumps(row), flush=True)
        if epoch+1 in checkpoint_epochs:
            torch.save(net.state_dict(), out / f"epoch{epoch+1}.pt")
    training_seconds = time.time()-start_time
    np.savez(out / "sampling_audit.npz", starts=np.arange(len(start_counts))+m["train_start"],
             counts=start_counts)
    torch.save(net.state_dict(), out / "fold0.pt")
    begin = time.time()
    pred = evaluate(net, data, eval_features, m, a.eval_chunk, DEVICE)
    seconds = time.time()-begin
    target = np.load(cache / "targ.npy")
    result = dict(model=f"DailyDecoder_{a.decoder}", inputs=a.inputs, seed=a.seed,
                  median_nse=float(np.nanmedian(P.nse_table(pred, target).nse)),
                  n_basins=len(pred), n_days=pred.shape[1], training_seconds=training_seconds,
                  inference_seconds=seconds, optimizer_updates=steps*a.epochs,
                  sampled_basin_windows=steps*a.epochs*min(a.batch,m["n_basins"]),
                  peak_memory_bytes=torch.cuda.max_memory_allocated() if DEVICE == "cuda" else None,
                  information=m["information"] if net.latent_dim else "K=0, causal raw forcing and attributes",
                  history=m["warmup"], config=config)
    result.update(paired_sample_stream_sha256=pair_stream.hexdigest(),
                  unique_train_starts=int(np.count_nonzero(start_counts)),
                  encoder_unchanged=(weights_sha256(encoder) == encoder_digest
                                     if encoder is not None else None))
    if encoder is not None and not result["encoder_unchanged"]:
        raise RuntimeError("frozen encoder weights changed during training")
    if net.latent_dim:
        # A distribution-shift diagnostic, not a trained no-context fallback.
        ablated = evaluate(net, data, eval_features, m, a.eval_chunk, DEVICE, ablate=True)
        result["zero_latent_median_nse"] = float(np.nanmedian(P.nse_table(ablated, target).nse))
        np.save(out / "pred_zero_latent.npy", ablated)
    np.save(out / "pred.npy", pred)
    np.save(out / "targ.npy", target)
    np.save(out / "gage.npy", data["gage"])
    (out / "run.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    if getattr(a, "evaluate_checkpoints", False):
        # Evaluate only after training ends: these diagnostics cannot change
        # the optimizer, dropout RNG, sample stream, or stopping epoch.
        scores, rows = [], []
        for epoch in checkpoint_epochs:
            if epoch == a.epochs:
                checkpoint_pred = pred
            else:
                net.load_state_dict(torch.load(out / f"epoch{epoch}.pt",
                                               map_location=DEVICE, weights_only=True))
                checkpoint_pred = evaluate(net, data, eval_features, m, a.eval_chunk, DEVICE)
                np.save(out / f"pred_epoch{epoch}.npy", checkpoint_pred)
            nse = np.asarray(P.nse_table(checkpoint_pred, target).nse)
            scores.append(nse)
            row = dict(epoch=epoch, optimizer_updates=steps*epoch,
                       median_nse=float(np.nanmedian(nse)))
            if len(scores) > 1:
                good = np.isfinite(nse) & np.isfinite(scores[0])
                baseline = scores[0][good]
                current = nse[good]
                row.update(reference_epoch=checkpoint_epochs[0],
                           n_paired_basins=int(good.sum()),
                           median_difference=float(np.median(current)-np.median(baseline)),
                           median_paired_delta=float(np.median(current-baseline)),
                           fraction_better=float(np.mean(current > baseline)))
            rows.append(row)
        np.savez(out / "checkpoint_basin_nse.npz", gage=data["gage"],
                 epochs=np.array(checkpoint_epochs), nse=np.array(scores))
        (out / "checkpoint_scores.json").write_text(json.dumps(dict(scores=rows,
            note="Prespecified test diagnostics; no checkpoint selection or early stopping."), indent=2))
        print(json.dumps(dict(checkpoint_scores=rows), indent=2), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    subp = p.add_subparsers(dest="command", required=True)
    prep = subp.add_parser("prepare")
    prep.add_argument("--cache", required=True)
    prep.add_argument("--source", required=True)
    prep.add_argument("--nc", default=P.CAMELS_NC)
    prep.add_argument("--warmup", type=int, default=256)
    prep.add_argument("--train-stride", type=int, default=64)
    prep.add_argument("--chunk", type=int, default=32)
    fit = subp.add_parser("train")
    fit.add_argument("--cache", required=True)
    fit.add_argument("--tag", required=True)
    fit.add_argument("--decoder", choices=["lstm", "tcn", "pointwise", "patch"], required=True)
    fit.add_argument("--inputs", choices=["raw", "hybrid", "latent"], default="hybrid")
    fit.add_argument("--sampling", choices=["daily", "fixed"], default="daily",
                     help="Random daily starts by default; fixed reproduces the cache-grid ablation")
    fit.add_argument("--source", help="Original frozen PFN directory; defaults to cache provenance path")
    fit.add_argument("--encoder-chunk", type=int, default=32)
    fit.add_argument("--feature-mode", choices=["auto", "online", "cached"], default="auto",
                     help="auto uses online encoding for daily starts; use online in both paired arms")
    fit.add_argument("--seed", type=int, default=0)
    fit.add_argument("--epochs", type=int, default=100)
    fit.add_argument("--checkpoint-epochs", default="50", help="Comma-separated fixed epochs to save")
    fit.add_argument("--evaluate-checkpoints", action="store_true",
                     help="Score all requested checkpoints after training; never select on test")
    fit.add_argument("--steps", type=int, default=0)
    fit.add_argument("--batch", type=int, default=128)
    fit.add_argument("--eval-chunk", type=int, default=32)
    fit.add_argument("--hidden", type=int, default=256)
    fit.add_argument("--tcn-width", type=int, default=64)
    fit.add_argument("--tcn-blocks", type=int, default=6)
    fit.add_argument("--dropout", type=float, default=0.1)
    fit.add_argument("--lr", type=float, default=1.)
    args = p.parse_args()
    prepare(args) if args.command == "prepare" else train(args)
