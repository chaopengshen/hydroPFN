"""Report specialist parity per information regime, not an averaged score."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from hydropfn.data import protocol as P
from hydropfn.data.forcing import load_camels
from temporal_diagnostics import gauge_ids, load_run


def paired_interval(candidate, reference, rng):
    indices = rng.integers(0, len(candidate), size=(2000, len(candidate)))
    draws = np.nanmedian(candidate[indices], 1) - np.nanmedian(reference[indices], 1)
    return np.percentile(draws, [2.5, 97.5]).tolist()


def main(a):
    sub, gage = P.load_531(load_camels(P.CAMELS_NC))
    window = P.windows(sub["time"], "temporal")["score"]
    target = sub["x"][:, window, -1].copy()
    target[sub["valid"][:, window, -1] == 0] = np.nan
    ids = gauge_ids(gage["gage"])
    runs, refs = Path(a.runs), Path(a.references)
    def score(path, scenario):
        pred, metadata = load_run(path, f"pred_{scenario}.npy", ids, target)
        if metadata["extent"] != "temporal" or metadata.get("partial_folds"):
            raise ValueError("full same-basin temporal predictions required")
        return P.nse_table(pred, target).nse
    specialist = score(refs / "stage2_temporal_daily_nse_s0", "daily")
    double = score(runs / "paired_daily_daily_e100_s0", "daily")
    table = pd.DataFrame({"gage": ids, "daily_specialist": specialist, "daily_double_control": double})
    scenarios = ["none", "daily", "lag7", "lag16", "weekly", "campaign30", "campaign90", "campaign365"]
    rows = []
    rng = np.random.default_rng(119)
    for tag in ["paired_daily_mixed_e100_s0", "paired_daily_mixed_e50_s0"]:
        for scenario in scenarios:
            values = score(runs / tag, scenario)
            reference = specialist if scenario == "daily" else score(refs / "stage2_temporal_mixed_nse_s0", scenario)
            interval = paired_interval(values, reference, rng)
            row = {"model": tag, "scenario": scenario, "median_nse": float(np.nanmedian(values)),
                   "reference": "daily_specialist" if scenario == "daily" else "original_mixed_same_scenario",
                   "reference_nse": float(np.nanmedian(reference)),
                   "median_difference": float(np.nanmedian(values) - np.nanmedian(reference)),
                   "median_paired_delta": float(np.nanmedian(values - reference)),
                   "fraction_better": float(np.mean(values > reference)),
                   "basin_bootstrap_lower": interval[0], "basin_bootstrap_upper": interval[1]}
            if scenario == "daily":
                row["same_two_pass_control_nse"] = float(np.nanmedian(double))
                row["difference_vs_two_pass_control"] = float(np.nanmedian(values) - np.nanmedian(double))
            rows.append(row)
            table[f"{tag}_{scenario}"] = values
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(out / "regime_scores.csv", index=False)
    table.to_csv(out / "paired_basins.csv", index=False)
    (out / "audit.json").write_text(json.dumps({
        "target_and_gauge_alignment": "verified", "seed": 0, "replication_required": True,
        "parity_certified": False,
        "interval_limitation": "Resamples basins, not seeds; basin spatial dependence is not modeled.",
        "budget_control": "e100 paired and two-pass daily share GPU, updates, capacity, and view count; e50 paired matches original e100 view count with half as many updates.",
        "interpretation": "Evaluate each information regime separately. This tests the recurrent observation formulation, not original PFN forward parity."
    }, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", required=True)
    p.add_argument("--references", required=True)
    p.add_argument("--out", required=True)
    main(p.parse_args())
