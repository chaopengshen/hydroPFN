"""Run the authorized seed-0 objective/architecture pilot on suntzu.

Each GPU runs one job at a time. The source manifest is checked before every
job; logs and per-job status persist after the SSH connection closes.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "recovery_stage1"


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def verify_sources():
    manifest = json.loads((ROOT / "source_sha256.json").read_text())
    for relative, expected in manifest.items():
        path = ROOT / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"source changed after snapshot: {relative}")
    return hashlib.sha256((ROOT / "source_sha256.json").read_bytes()).hexdigest()


def save_status(tag, payload):
    tmp = OUT / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT / f"{tag}.json")


def jobs():
    pfn = ["--extent", "temporal", "--epochs", "800", "--steps", "150",
           "--tasks", "8", "--seed", "0", "--save-ckpt", "--k-eval", "0,4"]
    lstm = ["--extent", "temporal", "--epochs", "100", "--seed", "0", "--save-ckpt"]
    return [
        [("stage1_pfn_mse_s0", "camels531_pub.py", pfn + ["--loss", "mse"]),
         ("stage1_pfn_nopooled_s0", "camels531_pub.py", pfn + ["--loss", "mse", "--no-pooled"])],
        [("stage1_lstm_rmse_s0", "camels531_lstm.py", lstm + ["--loss", "rmse"]),
         ("stage1_lstm_nse_s0", "camels531_lstm.py", lstm + ["--loss", "nse"]),
         ("stage1_pfn_nse_s0", "camels531_pub.py", pfn + ["--loss", "nse"])],
    ]


def worker(gpu, queue):
    for tag, script, args in queue:
        source_hash = verify_sources()
        result_dir = ROOT / "logs" / "camels531" / tag
        if result_dir.exists():
            raise RuntimeError(f"refusing to overwrite existing experiment {result_dir}")
        command = [sys.executable, "-u", str(ROOT / "experiments" / script),
                   *args, "--tag", tag]
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HYDROPFN_CAMELS_ROOT="/nfs/data/cxs1024/hydroPFN/data")
        status = {"tag": tag, "gpu": gpu, "command": command, "state": "running",
                  "started": stamp(), "source_manifest_sha256": source_hash}
        with (OUT / f"{tag}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
            status["pid"] = proc.pid
            save_status(tag, status)
            print(f"{stamp()} started {tag} GPU={gpu} PID={proc.pid}", flush=True)
            code = proc.wait()
        status.update(exit_code=code, finished=stamp(),
                      state="complete" if code == 0 else "failed")
        save_status(tag, status)
        print(f"{stamp()} {status['state']} {tag} exit={code}", flush=True)
        if code != 0:
            raise RuntimeError(f"{tag} failed; see {OUT / (tag + '.log')}")
    return gpu


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    if a.detach:
        if (OUT / "supervisor.log").exists():
            raise RuntimeError("this snapshot was already launched; inspect its status")
        with (OUT / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__, "--gpus", a.gpus],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (OUT / "supervisor_pid.txt").write_text(str(proc.pid))
        print(json.dumps({"supervisor_pid": proc.pid, "logs": str(OUT)}))
        return
    gpu_ids = [int(x) for x in a.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("supply two distinct GPUs")
    usage = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True)
    used = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.splitlines()}
    if any(used[g] > 100 for g in gpu_ids):
        raise RuntimeError(f"selected GPU is occupied; refusing to compete: {used}")
    queues = jobs()
    for gpu, queue in zip(gpu_ids, queues):
        for tag, script, args in queue:
            save_status(tag, {"tag": tag, "gpu": gpu, "state": "queued",
                              "script": script, "args": args, "created": stamp()})
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(worker, gpu, queue) for gpu, queue in zip(gpu_ids, queues)]
        for future in futures:
            future.result()
    (OUT / "complete.json").write_text(json.dumps({"completed": stamp()}))
    print("Stage 1 pilot complete. Inspect paired results before seed replication.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="2,4")
    parser.add_argument("--detach", action="store_true")
    main(parser.parse_args())
