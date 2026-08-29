"""Write every finished CAMELS-531 run to one long-format CSV.

Long format on purpose: one row per (run, K, series), so the file can be
pivoted or filtered without reshaping. `series` distinguishes the model's own
prediction from the three donor baselines computed on the same neighbours.

    python scripts/results_csv.py [-o results/camels531_results.csv]

Columns
    model        RegionalLSTM | SiteEncoder (unit A standalone) | PUBModel
    extent       PUB | PUR | temporal
    protocol     spatial | temporal
    epochs       training epochs per fold
    seed
    K            neighbouring gauges read at inference; blank for non-PUB models
    series       model | idw | ctx_mean | nn_donor
    nse          median per-basin NSE on raw mm/day  <- the headline
    kge, corr, bias_rel, rmse, fhv, flv               dmg_dev's other metrics
    n_basins, n_days
    tag          the run directory under logs/camels531/

Reference rows from dmg_dev (3-seed means, identical protocol) are appended
with model="[reference] ..." so the file is self-contained.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.paths import LOGS, ROOT                          # noqa: E402

FIELDS = ["model", "extent", "protocol", "epochs", "seed", "K", "series",
          "nse", "kge", "corr", "bias_rel", "rmse", "fhv", "flv",
          "n_basins", "n_days", "tag"]

# dmg_dev, scratch/Extended1980_1999/extended_1980_1999_seed_averaged.csv
REFERENCE = [
    ("LSTM+HBV", "PUB", 0.699950), ("LSTM+HBV", "PUR", 0.609453),
    ("Embedding adapter (Stefaland, Annually)", "PUB", 0.705698),
    ("Embedding adapter (Stefaland, Annually)", "PUR", 0.626596),
    ("Condensed embedding (Stefaland, daily)", "PUB", 0.721042),
    ("Condensed embedding (Stefaland, daily)", "PUR", 0.637768),
]


def agg(path):
    """median of each metric from a metrics_agg.json, or {} if absent."""
    f = Path(path) / "metrics_agg.json"
    if not f.exists():
        return {}
    m = json.load(open(f))
    return {k: round(v["median"], 6) for k, v in m.items()
            if isinstance(v, dict) and "median" in v}


def main(a):
    rows = []
    for run in sorted(Path(a.logs).glob("*/run.json")):
        d = run.parent
        r = json.load(open(run))
        base = {"model": r["model"], "extent": r["extent"],
                "protocol": r["protocol"], "epochs": r["epochs"],
                "seed": r["seed"], "n_basins": r["n_basins"],
                "n_days": r["n_days"], "tag": d.name}
        med = r["median_nse"]

        if not isinstance(med, dict):                    # LSTM / unit A
            m = agg(d)
            rows.append({**base, "K": "", "series": "model",
                         "nse": round(med, 6), "kge": m.get("kge"),
                         "corr": m.get("corr"), "bias_rel": m.get("bias_rel"),
                         "rmse": m.get("rmse"), "fhv": m.get("fhv"),
                         "flv": m.get("flv")})
            continue

        for kk, v in med.items():                        # PUBModel, per K
            K = int(kk.split("=")[1])
            m = agg(d / f"K{K}")
            rows.append({**base, "K": K, "series": "model",
                         "nse": round(v["p"], 6), "kge": m.get("kge"),
                         "corr": m.get("corr"), "bias_rel": m.get("bias_rel"),
                         "rmse": m.get("rmse"), "fhv": m.get("fhv"),
                         "flv": m.get("flv")})
            for src, name in (("idw", "idw"), ("cm", "ctx_mean"),
                              ("nn", "nn_donor")):
                if src in v:
                    rows.append({**base, "K": K, "series": name,
                                 "nse": round(v[src], 6)})

    for name, ext, nse in REFERENCE:
        rows.append({"model": f"[reference] {name}", "extent": ext,
                     "protocol": "spatial", "epochs": 50, "seed": "3-seed mean",
                     "K": "", "series": "model", "nse": nse,
                     "n_basins": 531, "n_days": 1461, "tag": "dmg_dev"})

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows)")

    # A compact pivot to stdout, because the point of the file is to be read.
    print(f"\n{'tag':<30}{'ep':>5}{'K':>4}  {'model':>8}{'idw':>9}")
    print("-" * 58)
    for r in rows:
        if r["series"] != "model" or r["tag"] == "dmg_dev":
            continue
        idw = next((x["nse"] for x in rows
                    if x["tag"] == r["tag"] and x["K"] == r["K"]
                    and x["series"] == "idw"), None)
        print(f"{r['tag']:<30}{r['epochs']:>5}{str(r['K']):>4}"
              f"  {r['nse']:>8.4f}{(f'{idw:>9.4f}' if idw is not None else ''):>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(LOGS / "camels531"))
    ap.add_argument("-o", "--out",
                    default=str(ROOT / "results" / "camels531_results.csv"))
    main(ap.parse_args())
