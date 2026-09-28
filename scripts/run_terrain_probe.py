"""Run one finite CPU terrain probe after the verified 531-basin extraction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_recovery_stage1 import stamp, verify_sources

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "logs" / "recovery_terrain"


def main(a):
    OUT.mkdir(parents=True, exist_ok=True)
    verify_sources()
    if a.detach:
        with (OUT / "supervisor.log").open("x") as log:
            proc = subprocess.Popen([sys.executable, "-u", __file__, "--terrain", a.terrain],
                                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        (OUT / "supervisor_pid.txt").write_text(str(proc.pid))
        print(json.dumps({"pid": proc.pid, "state": "waiting_for_terrain"}))
        return
    terrain = Path(a.terrain)
    launch = json.loads((terrain / "launch.json").read_text())
    script = Path(launch["command"][2])
    if hashlib.sha256(script.read_bytes()).hexdigest() != launch["script_sha256"]:
        raise ValueError("terrain source changed during extraction")
    deadline = time.monotonic() + 86400
    (OUT / "status.json").write_text(json.dumps({"state": "waiting_for_terrain", "started": stamp()}))
    while True:
        log = (terrain / "extraction.log").read_text()
        if log.rstrip().endswith("descriptors retain physical units and verified coverage."):
            break
        if "Traceback (most recent call last)" in log or time.monotonic() > deadline:
            raise RuntimeError("terrain extraction failed or timed out; inspect extraction.log")
        time.sleep(30)
    # Revalidate the immutable source and the finished source-side extraction.
    verify_sources()
    if len(list((terrain / "basins").glob("*.json"))) != 531:
        raise ValueError("full polygon extraction did not finish")
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONPATH=str(ROOT / "src"),
               HYDROPFN_CAMELS_ROOT="/nfs/data/cxs1024/hydroPFN/data",
               OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    command = [sys.executable, "-u", str(ROOT / "experiments/basin_terrain_probe.py"),
               "--basin", str(terrain / "basin_terrain.npz"),
               "--outlet", str(terrain / "outlet_terrain.npz"),
               "--out", str(OUT / "probe")]
    (OUT / "status.json").write_text(json.dumps({"state": "running", "started": stamp(), "command": command}))
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    (OUT / "status.json").write_text(json.dumps({"state": "complete", "finished": stamp()}))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--terrain", required=True)
    p.add_argument("--detach", action="store_true")
    try:
        main(p.parse_args())
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "status.json").write_text(json.dumps({"state": "failed", "error": str(exc), "failed": stamp()}))
        raise
