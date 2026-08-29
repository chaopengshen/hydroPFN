"""Prove the CAMELS-531 protocol matches dmg_dev, against dmg_dev's own output.

Run this before trusting any number from `camels531_lstm.py` or
`camels531_pub.py`. It does not reason about the protocol -- it compares the
observation series this repository would score against the observation series
dmg_dev actually scored, read from a completed run's
`spatial_aggregated_*/aggregated_targets.npy`.

That file is the ground truth for four things at once: which basins, in which
order, over which days, on which scale. Every CAMELS mistake in this project's
history would have been caught by this check:

  * scoring log1p while the comparison scored raw         -> scale mismatch
  * scoring 16-day means while the comparison scored daily -> length mismatch
  * scoring a 512-day window at day 9600                  -> length mismatch
  * assuming column j is basin j of `531sub_id.txt`       -> permutation found

Basins are matched by correlating every column of dmg_dev's array against
every basin's record. For this dataset that is a clean bijection at r = 1.0 to
~1e-14, so it is asserted rather than trusted. The two arrays are known to
differ by an exact per-basin constant (the runs read different source files);
NSE is invariant to it, so the check reports the offset rather than failing on
it.

Usage:
    python scripts/verify_camels531.py \
        --dmg-run /storage/home/nrk5343/scratch/Hbv_1_1p_Triton/\
Extended1980_1999_Camels531/Camels_531_condensed_embeddings_daily
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P                        # noqa: E402
from hydropfn.data.forcing import load_camels                  # noqa: E402

DEFAULT_RUN = ("/storage/home/nrk5343/scratch/Hbv_1_1p_Triton/"
               "Extended1980_1999_Camels531/"
               "Camels_531_condensed_embeddings_daily")


def match_columns(ref, mine):
    """For each column of `ref`, the row of `mine` it corresponds to.

    Correlation, not equality: the arrays differ by a per-basin constant.
    """
    r = ref - ref.mean(0, keepdims=True)
    m = mine - mine.mean(1, keepdims=True)
    r /= np.linalg.norm(r, axis=0) + 1e-12
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    corr = m @ r                                    # (n_mine, n_ref)
    best = corr.argmax(0)
    return best, corr[best, np.arange(corr.shape[1])]


def main(a):
    ok = True
    d = load_camels(P.CAMELS_NC)
    sub, gage = P.load_531(d)
    win = P.windows(d["time"], "spatial")
    obs = sub["x"].shape[-1] - 1
    q = sub["x"][..., obs].astype(np.float64)
    q[sub["valid"][..., obs] == 0] = np.nan

    print(f"our scored period : {d['time'][win['score'].start]} .. "
          f"{d['time'][win['score'].stop - 1]}  "
          f"({win['score'].stop - win['score'].start} d)")

    for ext in ("PUB", "PUR"):
        f = Path(a.dmg_run) / ext / f"spatial_aggregated_{ext}" / \
            "aggregated_targets.npy"
        if not f.exists():
            print(f"\n{ext}: no dmg_dev run at {f} -- skipped")
            continue
        ref = np.load(f).squeeze().astype(np.float64)          # (days, basins)
        print(f"\n{ext}: dmg_dev targets {ref.shape}")

        n_days = win["score"].stop - win["score"].start
        if ref.shape[0] != n_days:
            print(f"  FAIL length: dmg_dev scores {ref.shape[0]} d, "
                  f"we score {n_days} d")
            ok = False
            continue
        print(f"  OK length   : both score {n_days} d")

        mine = q[:, win["score"]]                              # (basins, days)
        row, r = match_columns(ref, mine)

        if len(np.unique(row)) != ref.shape[1]:
            print(f"  FAIL matching: {ref.shape[1] - len(np.unique(row))} "
                  f"columns collide -- not a bijection")
            ok = False
        elif r.min() < 0.999999:
            print(f"  FAIL matching: worst correlation {r.min():.6f}")
            ok = False
        else:
            print(f"  OK basins   : bijection, min correlation "
                  f"{r.min():.10f}")

        # The fold structure dmg_dev concatenated, recovered from the mapping.
        folds = P.folds(ext, gage)
        expect = np.concatenate(folds)
        if np.array_equal(row, expect):
            print(f"  OK order    : column j is fold-block basin j "
                  f"(widths {[len(x) for x in folds]})")
        else:
            same = int((row == expect).sum())
            print(f"  NOTE order  : our fold concatenation matches "
                  f"{same}/{len(row)} columns; dmg_dev's block order differs, "
                  f"so join on gage id, never on position")

        off = np.nanmean(ref.T - mine[row], axis=1)
        spread = np.nanmax(np.nanstd(ref.T - mine[row], axis=1))
        print(f"  scale       : per-basin offset {np.nanmin(off):+.3f} .. "
              f"{np.nanmax(off):+.3f}, max within-basin spread {spread:.2e}")
        if spread > 1e-3:
            print("  FAIL scale  : offset is not constant within a basin -- "
                  "these are different quantities, not a unit shift")
            ok = False
        else:
            print("  OK scale    : differs by an exact per-basin constant "
                  "only; NSE is invariant to it")

    print("\n" + ("PROTOCOL VERIFIED" if ok else "PROTOCOL MISMATCH -- "
                  "do not quote numbers until this passes"))
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dmg-run", default=DEFAULT_RUN,
                    help="a completed dmg_dev experiment directory holding "
                         "PUB/ and PUR/ subdirectories")
    sys.exit(main(ap.parse_args()))
