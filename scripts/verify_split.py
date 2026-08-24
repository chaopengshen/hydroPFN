"""Verify the train/test split is what the papers claim it is.

Run this before trusting any number in docs/benchmarks.md. It checks the four
things that have actually gone wrong in this project at least once each:

  1. the split is by REGION (PUR), not random basins
  2. train and eval sites do not overlap -- by site_no, never by COMID/index
  3. the temporal split leaves a real gap, and eval is AFTER training
  4. how far context sites actually are, since "ungauged region" and "no gauge
     nearby" are different claims and only the first is true here

Usage:
    python scripts/verify_split.py --nc data/CAMELS_Frederik.nc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data.forcing import load_camels  # noqa: E402


def main(a):
    d = load_camels(a.nc)
    region, sid, ll = d["region"], d["site_id"], d["latlon"]
    hold = a.holdout.split(",")
    te = np.flatnonzero(np.isin(region, hold))
    tr = np.flatnonzero(~np.isin(region, hold))

    print("\n1. SPLIT TYPE")
    print(f"   held-out HUC2 regions : {hold}")
    print(f"   eval basins           : {len(te)}")
    print(f"   train basins          : {len(tr)}")
    held_regions = sorted(set(region[te]))
    train_regions = sorted(set(region[tr]))
    overlap = set(held_regions) & set(train_regions)
    print(f"   regions in BOTH arms  : {sorted(overlap) or 'NONE -> this is PUR'}")

    print("\n2. SITE OVERLAP")
    inter = set(sid[tr]) & set(sid[te])
    print(f"   site_ids in both arms : {len(inter)}  "
          f"{'OK' if not inter else 'LEAK'}")
    print(f"   unique ids total      : {len(set(sid))} of {len(sid)} rows")

    print("\n3. TEMPORAL SPLIT")
    tr_end, ev_start, win = a.train_end, a.eval_start * a.patch, a.win * a.patch
    print(f"   train windows END by  : day {tr_end}")
    print(f"   eval window           : day {ev_start} .. {ev_start + win}")
    print(f"   gap                   : {ev_start - tr_end} days  "
          f"{'OK' if ev_start >= tr_end else 'EVAL INSIDE TRAINING'}")
    print(f"   record length         : {d['x'].shape[1]} days")

    print("\n4. HOW FAR IS CONTEXT, REALLY")
    kt = cKDTree(ll[tr])
    dtr, _ = kt.query(ll[te], k=1)
    ka = cKDTree(np.delete(ll, te[:0], axis=0))
    dall, _ = ka.query(ll[te], k=2)
    print(f"   eval -> nearest TRAINING basin : median {np.median(dtr):.2f} deg")
    print(f"   eval -> nearest ANY basin      : median "
          f"{np.median(dall[:, 1]):.2f} deg")
    print("   A random k-fold holdout would put a TRAINING basin ~0.3 deg away.")
    print("   The large first number is what makes this PUR. The small second")
    print("   number is why --context-pool all is 'ungauged basin in an")
    print("   ungauged region WITH nearby untrained gauges', not 'no gauges'.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", required=True)
    ap.add_argument("--holdout", default="01,11,17")
    ap.add_argument("--train-end", type=int, default=9000)
    ap.add_argument("--eval-start", type=int, default=600)
    ap.add_argument("--patch", type=int, default=16)
    ap.add_argument("--win", type=int, default=32)
    main(ap.parse_args())
