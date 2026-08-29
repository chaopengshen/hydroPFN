"""Collect finished CAMELS-531 runs into one table, with the protocol shown.

Reads `logs/camels531/*/run.json` and prints median NSE per run alongside the
dmg_dev reference values, which are on the identical protocol and are
therefore the only numbers on this page that may be compared to ours.

Anything from before 2026-08-28 is on a different protocol (671 basins, a
hand-picked HUC2 holdout, log1p discharge, a 512-day window) and must not be
placed in this table. That is what docs/all_results.md is for.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.paths import LOGS                                # noqa: E402

# dmg_dev, seed-averaged over 3 seeds, same 531 basins / periods / metric.
# Source: scratch/Extended1980_1999/extended_1980_1999_seed_averaged.csv
REFERENCE = {
    ("LSTM+HBV", "PUB"): 0.700,
    ("LSTM+HBV", "PUR"): 0.609,
    ("Embedding adapter (Stefaland, Annually)", "PUB"): 0.706,
    ("Embedding adapter (Stefaland, Annually)", "PUR"): 0.627,
    ("Condensed embedding (Stefaland, daily)", "PUB"): 0.721,
    ("Condensed embedding (Stefaland, daily)", "PUR"): 0.638,
}


def main(a):
    root = Path(a.logs)
    rows = []
    for f in sorted(root.glob("*/run.json")):
        try:
            r = json.load(open(f))
        except json.JSONDecodeError:
            print(f"  (skipping unreadable {f})")
            continue
        r["tag"] = f.parent.name
        rows.append(r)

    if not rows:
        print(f"no finished runs under {root}")
        return

    print("=== hydroPFN, CAMELS-531 protocol "
          "(531 basins, raw mm/day NSE, warmup 365 excluded) ===\n")
    print(f"{'run':<34}{'extent':<10}{'protocol':<10}"
          f"{'seed':>5}{'basins':>8}{'days':>7}  median NSE")
    print("-" * 96)
    for r in rows:
        med = r["median_nse"]
        if isinstance(med, dict):                       # PUBModel: one per K
            head = f"{r['tag']:<34}{r['extent']:<10}{r['protocol']:<10}" \
                   f"{r['seed']:>5}{r['n_basins']:>8}{r['n_days']:>7}"
            print(head)
            for k, v in med.items():
                base = "".join(
                    f"  {n} {v[n]:+.4f}" for n in ("nn", "cm", "idw")
                    if n in v)
                print(f"{'':<34}{k:<10}{'':<10}{'':>5}{'':>8}{'':>7}"
                      f"  {v['p']:+.4f}{base}")
        else:
            print(f"{r['tag']:<34}{r['extent']:<10}{r['protocol']:<10}"
                  f"{r['seed']:>5}{r['n_basins']:>8}{r['n_days']:>7}"
                  f"  {med:+.4f}")

    print("\n=== dmg_dev reference, identical protocol, 3-seed mean ===")
    for (name, ext), v in REFERENCE.items():
        print(f"  {name:<44}{ext:<6}{v:+.4f}")
    print("\nComparable to our spatial rows. The temporal rows are a "
          "different protocol\n(train 1980-1995, score 1995-2010) and have no "
          "reference value here.")
    print("For PUBModel, K=0 is the row comparable to the LSTM; K>0 "
          "additionally reads\nneighbouring gauges at inference and must be "
          "read against nn/cm/idw beside it.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(LOGS / "camels531"))
    main(ap.parse_args())
