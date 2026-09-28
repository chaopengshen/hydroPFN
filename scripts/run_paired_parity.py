"""Finite paired-training pilot. No architecture or downstream-task expansion.

Compare equal daily exposure, equal observation-view budget, and a duplicated
daily specialist with identical passes/updates. Seed-0 diagnostic only.
"""
from concurrent.futures import ThreadPoolExecutor
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_recovery_stage1 import stamp, verify_sources
from run_recovery_stage2 import free_gpu

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs/paired_parity"


def jobs():
    common = ["--extent", "temporal", "--protocol", "temporal", "--loss", "nse",
              "--mode", "paired", "--seed", "0"]
    return [
        [("paired_daily_mixed_e100_s0", common + ["--epochs", "100", "--paired-second", "mixed"]),
         ("paired_daily_daily_e100_s0", common + ["--epochs", "100", "--paired-second", "daily",
                                                  "--scenarios", "daily"])],
        [("paired_daily_mixed_e50_s0", common + ["--epochs", "50", "--paired-second", "mixed"])],
    ]


def status(tag, payload):
    tmp = OUT / f"{tag}.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(OUT / f"{tag}.json")


def worker(gpu, queue):
    for tag, args in queue:
        deadline = time.monotonic() + 86400
        while not free_gpu(gpu):
            status(tag, {"state": "waiting_for_gpu", "gpu": gpu, "updated": stamp()})
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU {gpu} remained busy for 24 h")
            time.sleep(30)
        manifest = verify_sources()
        command = [sys.executable, "-u", str(ROOT / "experiments/camels531_daily_di.py"),
                   *args, "--tag", tag]
        env = os.environ.copy()
        env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
                   PYTHONPATH=str(ROOT / "src"), OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
                   HYDROPFN_CAMELS_ROOT="/nfs/data/cxs1024/hydroPFN/data")
        state = {"state": "running", "gpu": gpu, "tag": tag, "command": command,
                 "started": stamp(), "source_manifest_sha256": manifest}
        with (OUT / f"{tag}.log").open("x") as log:
            proc = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            state["pid"] = proc.pid
            status(tag, state)
            print(f"{stamp()} started {tag} GPU={gpu} PID={proc.pid}", flush=True)
            code = proc.wait()
        state.update(state="complete" if code == 0 else "failed", finished=stamp(), exit_code=code)
        status(tag, state)
        if code:
            raise RuntimeError(f"{tag} failed")
        print(f"{stamp()} complete {tag}", flush=True)


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    if a.detach:
        with (OUT / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__, "--gpus", a.gpus,
                                     "--reference-root", a.reference_root], cwd=ROOT,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                    start_new_session=True)
        (OUT / "supervisor_pid.txt").write_text(str(proc.pid))
        print(json.dumps({"pid": proc.pid, "logs": str(OUT)}))
        return
    gpu_ids = [int(x) for x in a.gpus.split(",")]
    if len(gpu_ids) != 2 or len(set(gpu_ids)) != 2:
        raise ValueError("two distinct GPUs required")
    for gpu, queue in zip(gpu_ids, jobs()):
        for tag, args in queue:
            status(tag, {"state": "queued", "gpu": gpu, "args": args})
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, gpu, queue) for gpu, queue in zip(gpu_ids, jobs())]
        for future in futures:
            future.result()
    command = [sys.executable, str(ROOT / "experiments/paired_parity_diagnostics.py"),
               "--runs", str(ROOT / "logs/camels531"), "--references", a.reference_root,
               "--out", str(OUT / "diagnostics")]
    subprocess.run(command, cwd=ROOT, check=True)
    (OUT / "complete.json").write_text(json.dumps({"completed": stamp()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpus", default="2,4")
    p.add_argument("--reference-root", required=True)
    p.add_argument("--detach", action="store_true")
    try:
        main(p.parse_args())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "failed.json").write_text(json.dumps({"failed": stamp(), "error": str(exc)}))
        raise
