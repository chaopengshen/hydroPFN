"""Finite stage-2 experiment batch, released only after the stage-1 pilot.

The NSE objective is prespecified, not selected on future-period performance.
All three temporal arms have identical recurrent capacity. New-basin controls
and mixed-availability models use all ten PUB folds and future-year scoring.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_recovery_stage1 import stamp, verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "recovery_stage2"


def queues():
    common = ["--protocol", "temporal", "--loss", "nse", "--epochs", "100", "--seed", "0"]
    def job(extent, mode, scenarios=None):
        tag = f"stage2_{extent}_{mode}_nse_s0"
        args = common + ["--extent", extent, "--mode", mode, "--tag", tag]
        if scenarios:
            args += ["--scenarios", scenarios]
        return tag, args
    return [
        [job("temporal", "forward", "none"), job("temporal", "mixed"),
         job("PUB", "mixed")],
        [job("temporal", "daily", "daily"), job("PUB", "forward", "none")],
    ]


def write_status(tag, data):
    tmp = OUT / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(OUT / f"{tag}.json")


def free_gpu(gpu):
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                     "--format=csv,noheader,nounits"], text=True)
    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.splitlines()}
    return used[gpu] <= 100


def worker(gpu, jobs):
    for tag, args in jobs:
        deadline = time.monotonic() + 86400
        while not free_gpu(gpu):
            write_status(tag, {"state": "waiting_for_free_gpu", "gpu": gpu, "updated": stamp()})
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU {gpu} remained occupied for 24 h")
            time.sleep(30)
        source_hash = verify_sources()
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HYDROPFN_CAMELS_ROOT="/nfs/data/cxs1024/hydroPFN/data")
        command = [sys.executable, "-u", str(ROOT / "experiments/camels531_daily_di.py"), *args]
        state = {"tag": tag, "state": "running", "gpu": gpu, "started": stamp(),
                 "source_manifest_sha256": source_hash, "command": command}
        with (OUT / f"{tag}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["pid"] = proc.pid
            write_status(tag, state)
            print(f"{stamp()} started {tag} GPU={gpu} PID={proc.pid}", flush=True)
            code = proc.wait()
        state.update(state="complete" if code == 0 else "failed", exit_code=code, finished=stamp())
        write_status(tag, state)
        if code:
            raise RuntimeError(f"{tag} failed; inspect its log and completed-fold caches")
        print(f"{stamp()} completed {tag}", flush=True)


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    if a.detach:
        with (OUT / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__, "--after", a.after,
                                     "--gpus", a.gpus], cwd=ROOT, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (OUT / "supervisor_pid.txt").write_text(str(proc.pid))
        print(json.dumps({"pid": proc.pid, "state": "waiting_for_stage1", "after": a.after}))
        return
    gpu_ids = [int(v) for v in a.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("two distinct GPUs required")
    dependency = Path(a.after)
    if not dependency.is_dir() or not (dependency / "supervisor_pid.txt").exists():
        raise ValueError("stage-1 dependency must be an existing supervisor directory")
    for gpu, jobs in zip(gpu_ids, queues()):
        for tag, args in jobs:
            write_status(tag, {"tag": tag, "gpu": gpu, "state": "waiting_for_stage1",
                               "args": args, "created": stamp()})
    print(f"Waiting for successful stage 1: {dependency}", flush=True)
    deadline = time.monotonic() + 86400
    while not (dependency / "complete.json").exists():
        for path in dependency.glob("stage1_*.json"):
            status = json.loads(path.read_text())
            if status.get("state") == "failed":
                raise RuntimeError(f"stage 1 failed: {path}; dependent batch not released")
        if time.monotonic() > deadline:
            raise RuntimeError("stage 1 did not finish within 24 h")
        time.sleep(30)
    # Preserve the paired stage-1 report before any dependent model starts.
    stage1_runs = dependency.parent / "camels531"
    command = [sys.executable, str(ROOT / "experiments/temporal_diagnostics.py"),
               "--reference", "lstm_rmse", "--out", str(OUT / "stage1_diagnostics")]
    for label, filename in [("lstm_rmse", "pred.npy"), ("lstm_nse", "pred.npy"),
                            ("pfn_mse", "pred_K0.npy"), ("pfn_nse", "pred_K0.npy"),
                            ("pfn_nopooled", "pred_K0.npy")]:
        command += ["--run", f"{label}={stage1_runs / ('stage1_' + label + '_s0')}={filename}"]
    subprocess.run(command, cwd=ROOT, check=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, g, q) for g, q in zip(gpu_ids, queues())]
        for future in futures:
            future.result()
    command = [sys.executable, str(ROOT / "experiments/daily_di_diagnostics.py"),
               "--runs", str(ROOT / "logs/camels531"), "--out", str(OUT / "diagnostics")]
    subprocess.run(command, cwd=ROOT, check=True)
    (OUT / "complete.json").write_text(json.dumps({"completed": stamp()}))
    print("Stage 2 seed-0 pilot complete; replication and PFN integration remain decisions.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--after", required=True)
    p.add_argument("--gpus", default="2,4")
    p.add_argument("--detach", action="store_true")
    args = p.parse_args()
    try:
        main(args)
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failed.json").write_text(json.dumps({"failed": stamp(), "error": str(exc)}, indent=2))
        raise
