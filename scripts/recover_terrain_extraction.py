"""Resume cached terrain extraction with a hard timeout per isolated basin.

Uses the original, unchanged extractor and descriptor settings. A hung GDAL
request can no longer prevent all later basins from being attempted.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


def main(a):
    out = Path(a.out)
    subsets, logs = out / "isolated_subsets", out / "isolated_logs"
    subsets.mkdir(exist_ok=True)
    logs.mkdir(exist_ok=True)
    ids = sorted(str(int(x)).zfill(8) for x in json.loads(Path(a.subset).read_text()))
    env = os.environ.copy()
    env.update(GDAL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    for i, gage in enumerate(ids, 1):
        record_path = out / "basins" / f"{gage}.json"
        if record_path.exists():
            record = json.loads(record_path.read_text())
            if "basin" in record and "outlet" in record:
                continue
        subset = subsets / f"{gage}.json"
        subset.write_text(json.dumps([int(gage)]))
        command = [sys.executable, "-u", a.extractor, "--subset", str(subset), "--out", str(out)]
        error = None
        with (logs / f"{gage}.log").open("a") as log:
            try:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                               timeout=a.timeout, check=True)
            except subprocess.TimeoutExpired:
                error = f"isolated extraction timed out after {a.timeout}s"
            except subprocess.CalledProcessError as exc:
                error = f"isolated extraction exited {exc.returncode}"
        if error and not record_path.exists():
            record_path.write_text(json.dumps({"gage": gage, "basin_error": error,
                "outlet_error": error, "config": {"resolution_m": 30., "max_pixels": 5000000,
                "min_coverage": .98, "outlet_width_km": 12.8}}, indent=2))
        record = json.loads(record_path.read_text())
        print(f"recovery {i}/{len(ids)} {gage}: basin={'basin' in record} "
              f"outlet={'outlet' in record} error={error}", flush=True)
    # Child arrays contain a single basin. Rebuild the final aligned inventory
    # only after all child processes have exited, before releasing the probe.
    spec = importlib.util.spec_from_file_location("original_terrain", a.extractor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = [json.loads((out / "basins" / f"{g}.json").read_text()) for g in ids]
    module.write_arrays(records, out)
    print(f"Wrote {out}; descriptors retain physical units and verified coverage.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--extractor", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--subset", default="/nfs/data/cxs1024/hydroPFN/data/1-camels/531sub_id.txt")
    p.add_argument("--timeout", type=int, default=180)
    main(p.parse_args())
