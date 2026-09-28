"""Paired daily DI and unseen-basin adaptation reports for the stage-2 pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels
from temporal_diagnostics import load_run, gauge_ids


def verify_unseen_folds(path, gage):
    ids = gauge_ids(np.load(path / "gage.npy", allow_pickle=True))
    assignments = np.load(path / "fold.npy")
    expected = P.folds("PUB", gage)
    for k, te in enumerate(expected):
        norm = np.load(path / f"fold{k}_normalization.npz")
        train = set(gauge_ids(norm["train_gage"]))
        test = set(gauge_ids(gage["gage"].to_numpy()[te]))
        if set(ids[assignments == k]) != test or train & test:
            raise ValueError(f"unseen-basin separation failed: {path}, fold {k}")
        if train | test != set(gauge_ids(gage["gage"])):
            raise ValueError("normalization inventory is incomplete")


def main(a):
    sub, gage = P.load_531(load_camels(a.nc))
    window = P.windows(sub["time"], "temporal")
    target = sub["x"][:, window["score"], -1].copy()
    target[sub["valid"][:, window["score"], -1] == 0] = np.nan
    ids = gauge_ids(gage["gage"])
    runs = Path(a.runs)
    rows, curves = [], []
    basins = pd.DataFrame({"gage": ids})
    blocks = [("days_1_30", 0, 30), ("days_31_90", 30, 90),
              ("days_91_365", 90, 365), ("year_2", 365, 730),
              ("years_3_15", 730, target.shape[1])]
    for extent in ("temporal", "PUB"):
        baseline_path = runs / f"stage2_{extent}_forward_nse_s0"
        forward, metadata = load_run(baseline_path, "pred_none.npy", ids, target)
        if metadata["extent"] != extent or metadata.get("partial_folds"):
            raise ValueError("full matched evaluation required")
        ref_nse = P.nse_table(forward, target).nse
        arms = [("forward", "none")]
        if extent == "temporal":
            arms.append(("daily", "daily"))
        arms += [("mixed", s) for s in
                 ("none", "daily", "lag7", "lag16", "weekly", "campaign30", "campaign90", "campaign365")]
        if extent == "PUB":
            verify_unseen_folds(baseline_path, gage)
            verify_unseen_folds(runs / "stage2_PUB_mixed_nse_s0", gage)
        for mode, scenario in arms:
            path = runs / f"stage2_{extent}_{mode}_nse_s0"
            pred, meta = load_run(path, f"pred_{scenario}.npy", ids, target)
            if meta["extent"] != extent or meta.get("partial_folds"):
                raise ValueError(f"mismatched extent or incomplete folds: {path}")
            metrics = P.nse_table(pred, target)
            delta = metrics.nse - ref_nse
            key = f"{extent}_{mode}_{scenario}"
            basins[f"{key}_nse"] = metrics.nse
            basins[f"{key}_delta"] = delta
            rows.append({"extent": extent, "mode": mode, "scenario": scenario,
                         "n": len(ids), "median_nse": float(np.nanmedian(metrics.nse)),
                         "median_paired_delta": float(np.nanmedian(delta)),
                         "fraction_better": float(np.mean(delta > 0)),
                         "median_rmse_mm_day": float(np.nanmedian(metrics.rmse))})
            # The full-record denominator stays fixed across lead blocks;
            # short-block NSE would change the metric along with the lead.
            variance = np.maximum(np.nanvar(target, axis=1), .01)
            error = (pred - target) ** 2 / variance[:, None]
            ref_error = (forward - target) ** 2 / variance[:, None]
            for label, start, stop in blocks:
                err = np.nanmean(error[:, start:stop], axis=1)
                ref = np.nanmean(ref_error[:, start:stop], axis=1)
                curves.append({"extent": extent, "mode": mode, "scenario": scenario,
                               "block": label, "days": stop - start,
                               "median_normalized_mse": float(np.nanmedian(err)),
                               "median_paired_error_reduction": float(np.nanmedian(ref - err))})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "scores.csv", index=False)
    pd.DataFrame(curves).to_csv(out / "adaptation_persistence.csv", index=False)
    basins.to_csv(out / "paired_basins.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "gauge_and_raw_target_alignment": "verified", "PUB_training_disjoint_from_test_basins": "verified",
        "seed": 0, "pilot_only": True,
        "interpretation": "Recurrent observation-interface baseline; does not establish PFN integration or multi-seed robustness."
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nc", default=P.CAMELS_NC)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
