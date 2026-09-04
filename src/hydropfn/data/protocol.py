"""The CAMELS-531 evaluation protocol, matched to dmg_dev.

Every number this project has quoted for CAMELS used a protocol of its own
invention -- a 671-basin set, a HUC2 holdout picked by hand, `log1p` discharge,
a 512-day evaluation window, and NSE computed on the transformed scale. None of
that is comparable to the published values it was being compared against.

This module replaces it with the protocol the comparison studies actually use,
read off dmg_dev's configs rather than reconstructed from a paper:

    conf/Extended1980_1999_531/LSTMPUB.yaml   (and LSTMPUR.yaml)
    conf/lstm.yaml                            (the temporal split)

    basins     the 531-basin Newman/Addor subset, `531sub_id.txt`
    PUB        10 random groups, leave-one-group-out   (`PUB_ID` 1..10)
    PUR        7 contiguous regions, leave-one-region-out
    spatial    train 1980-10-01..1999-09-30, score 1995-10-01..1999-09-30
    temporal   train 1980-10-01..1995-09-30, score 1995-10-01..2010-09-30
    warmup     365 days of spin-up before the scored period, never scored
    metric     NSE on RAW mm/day, per basin, median over basins

The scored period overlaps the training period in the spatial protocols. That
is not a leak and it is not an oversight: the held-out basins are disjoint from
the training basins, which is the whole point of PUB/PUR, and it is what Feng
et al. (2021, 2023) and Li et al. (2025) do. The temporal protocol is the one
that separates periods.

**Metrics are not reimplemented here.** `hydropfn.metrics.dmg_metrics` is a
verbatim copy of `dmg/core/calc/metrics.py` (one import line rewritten), so NSE
is computed by the same code that produced the numbers we compare against.
Reimplementing it is exactly how this project ended up scoring 16-day means
against daily values under a comment claiming they matched.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

# dmg_dev's `observations: camels_531` and the shared StefaLand data root.
# Overridable by environment variable so the module is not machine-bound.
DATA_ROOT = os.environ.get(
    "HYDROPFN_CAMELS_ROOT",
    "/storage/group/cxs1024/default/nrk5343/StefaLandData")
SUBSET_531 = os.path.join(DATA_ROOT, "1-camels", "531sub_id.txt")
GAGE_SPLIT = os.path.join(DATA_ROOT, "gages_list_with_pub.csv")
CAMELS_NC = os.path.join(DATA_ROOT, "CAMELS_Frederik.nc")

# test.huc_regions, verbatim from the dmg_dev configs. NOTE: these NAME the
# holdout blocks but do not describe them -- the `huc` column of the gage CSV
# disagrees with the USGS part number of the station beside it for 467 of the
# 531 basins. The blocks are still disjoint, still cover all 531, and each
# still occupies a coherent longitude band, so this is a valid regional
# holdout; only the labels are wrong. Region 5 is Colorado/Great Basin/Oregon
# and region 6 is Pacific NW + California, NOT what the numbers imply.
# Do not "fix" the CSV -- the trained splits depend on it.
HUC_REGIONS = [[1, 2], [3, 6], [4, 5, 7], [9, 10],
               [8, 11, 12, 13], [14, 15, 16, 18], [17]]
PUB_IDS = list(range(1, 11))

WARMUP = 365          # model.warmup
RHO = 365             # model.rho -- training sequence length

# ---------------------------------------------------------------- benchmarks
#
# Each entry is a NAMED published protocol. The name is carried into every
# run.json and every CSV row, because the single most expensive mistake in
# this project's history was putting two protocols in one table -- so a bare
# NSE with no protocol beside it is not a result here, it is a liability.
#
# `basins`  531 = the Newman/Addor quality-filtered subset; 671 = all of CAMELS
# `extents` which holdouts the benchmark defines
BENCHMARKS = {
    "peijun_li": {
        "label": "Peijun Li Ensemble",
        "basins": 531, "extents": ["PUB", "PUR"],
        "train": ("1980-10-01", "1999-09-30"),
        "test": ("1995-10-01", "1999-09-30"),
        "note": "spatial generalization; train and test periods overlap, "
                "which is not a leak because the basins are disjoint",
    },
    "feng2023": {
        "label": "Dapeng Feng 2023",
        "basins": 531, "extents": ["PUB", "PUR"],
        "train": ("1989-10-01", "1999-09-30"),
        "test": ("1989-10-01", "1999-09-30"),
        "note": "trained and tested in the SAME period, different basins",
    },
    "leo10_531": {
        "label": "10 years 531 Leo",
        "basins": 531, "extents": ["temporal"],
        "train": ("1999-10-01", "2008-09-30"),
        "test": ("1989-10-01", "1999-09-30"),
        "note": "the test period PRECEDES the training period -- backwards "
                "in time, and deliberately so",
    },
    "leo15_671": {
        "label": "15 years 671 Leo",
        "basins": 671, "extents": ["temporal"],
        "train": ("1980-10-01", "1995-09-30"),
        "test": ("1995-10-01", "2010-09-30"),
        "note": "all 671 CAMELS basins, not the 531 subset",
    },
    # The 531-basin twin of leo15_671, already run before the registry
    # existed. Kept so those results stay addressable by name.
    "leo15_531": {
        "label": "15 years 531 (legacy temporal)",
        "basins": 531, "extents": ["temporal"],
        "train": ("1980-10-01", "1995-09-30"),
        "test": ("1995-10-01", "2010-09-30"),
        "note": "same periods as leo15_671 on the 531 subset",
    },
}

# Legacy names, kept so already-written run.json files stay readable.
PERIODS = {
    "spatial": (*BENCHMARKS["peijun_li"]["train"],
                *BENCHMARKS["peijun_li"]["test"]),
    "temporal": (*BENCHMARKS["leo15_531"]["train"],
                 *BENCHMARKS["leo15_531"]["test"]),
}
for _k, _b in BENCHMARKS.items():
    PERIODS[_k] = (*_b["train"], *_b["test"])


def load_subset(d, n_basins=531):
    """Restrict a `load_camels` dict to 531 or 671 basins, in file order.

    Returns (subset_dict, gage_table). The gage table carries `PUB_ID` and
    `huc` aligned to the subset rows, so the split functions never have to
    re-join anything.

    For 671 the order is the netCDF's own station order; `gages_list_with_pub.csv`
    covers all 671, so both PUB and PUR folds are defined there too:
        671 PUB [59, 68, 55, 61, 79, 75, 61, 74, 73, 66]
        671 PUR [102, 109, 109, 79, 87, 94, 91]
    """
    if n_basins == 671:
        want = [int(s) for s in d["site_id"]]
    elif n_basins == 531:
        with open(SUBSET_531) as f:
            want = [int(x) for x in json.load(f)]
    else:
        raise ValueError(f"n_basins must be 531 or 671, got {n_basins}")

    sid_int = np.array([int(s) for s in d["site_id"]])
    pos = {v: i for i, v in enumerate(sid_int)}
    missing = [g for g in want if g not in pos]
    if missing:
        raise ValueError(f"{len(missing)} of the 531 basins are absent from "
                         f"the netCDF, e.g. {missing[:5]}")
    idx = np.array([pos[g] for g in want])

    out = {}
    for k, v in d.items():
        if k in ("time", "series_vars", "static_vars"):
            out[k] = v
        else:
            out[k] = np.asarray(v)[idx]
    out["subset_index"] = idx

    g = pd.read_csv(GAGE_SPLIT, dtype={"huc": int, "gage": str, "PUB_ID": int})
    g["gage_int"] = g["gage"].astype(int)
    g = g.set_index("gage_int").reindex(want)
    if g["PUB_ID"].isna().any():
        raise ValueError(f"gages_list_with_pub.csv does not cover all "
                         f"{n_basins} basins")
    return out, g.reset_index()


def load_531(d):
    """Backward-compatible alias for `load_subset(d, 531)`."""
    return load_subset(d, 531)


def folds(extent, gage_table):
    """Held-out basin indices per fold, as positions into the 531 subset.

    `extent` is 'PUB' (10 folds), 'PUR' (7 folds) or 'temporal' (1 fold that
    holds out nothing -- every basin is both trained on and scored, and the
    periods are what separate them).
    """
    if extent == "temporal":
        return [np.arange(len(gage_table))]
    if extent == "PUB":
        return [np.flatnonzero((gage_table["PUB_ID"] == p).to_numpy())
                for p in PUB_IDS]
    if extent == "PUR":
        return [np.flatnonzero(gage_table["huc"].isin(r).to_numpy())
                for r in HUC_REGIONS]
    raise ValueError(f"extent must be PUB, PUR or temporal; got {extent!r}")


def date_index(time, date):
    """Position of `date` in the record, raising rather than clipping."""
    t = np.asarray(time).astype("datetime64[D]")
    d = np.datetime64(str(date)[:10], "D")
    j = int(np.searchsorted(t, d))
    if j >= len(t) or t[j] != d:
        raise ValueError(f"{date} is not in the record "
                         f"({t[0]} .. {t[-1]})")
    return j


def windows(time, protocol):
    """Index windows for one protocol.

    Returns a dict with
        train      slice over which training windows may be drawn
        score      slice that is SCORED -- exactly the config's test period
        eval_in    slice fed to the model: `score` with WARMUP days prepended

    The distinction between `eval_in` and `score` is the fix for the bug where
    the old baseline excluded its 120-day warmup from the loss but scored it
    anyway, with a cold hidden state, under a docstring claiming otherwise.
    """
    if protocol in BENCHMARKS:
        b = BENCHMARKS[protocol]
        tr0, tr1, te0, te1 = (*b["train"], *b["test"])
    else:
        tr0, tr1, te0, te1 = PERIODS[protocol]
    i_tr0, i_tr1 = date_index(time, tr0), date_index(time, tr1)
    i_te0, i_te1 = date_index(time, te0), date_index(time, te1)
    if i_te0 - WARMUP < 0:
        raise ValueError("not enough record before the test period for warmup")
    return {"train": slice(i_tr0, i_tr1 + 1),
            "score": slice(i_te0, i_te1 + 1),
            "eval_in": slice(i_te0 - WARMUP, i_te1 + 1),
            "labels": {"train": (tr0, tr1), "test": (te0, te1)}}


def iters_per_epoch(n_basins, n_days, batch, rho=RHO, warmup=WARMUP):
    """dmg_dev's iterations-per-epoch rule, copied from `create_training_grid`.

        n_iter = ceil( log(0.01) / log(1 - batch*rho / n_basins / (n_t - warmup)) )

    It is the number of minibatches needed to touch 99% of the training
    samples once, so "50 epochs" means the same amount of data exposure here
    as it does in the reference runs. Hardcoding a fixed step count instead
    silently trains a different model: at 100 steps/epoch the LSTM baseline
    received 5,000 gradient steps per fold against dmg_dev's 15,350 -- a 3.1x
    shortfall, which understates the baseline every other arm is measured
    against.
    """
    frac = batch * rho / n_basins / (n_days - warmup)
    if not 0 < frac < 1:
        raise ValueError(f"degenerate sampling fraction {frac}")
    return int(np.ceil(np.log(0.01) / np.log(1 - frac)))


def nse_table(pred, obs):
    """Per-basin metrics via dmg_dev's own Metrics class.

    `pred` and `obs` are (n_basins, n_days) on the RAW mm/day scale. Returns
    the Metrics object; `.nse` is the per-basin array and
    `np.nanmedian(m.nse)` is the headline.

    dmg_dev calls this with `np.swapaxes(arr.squeeze(), 0, 1)` on
    (time, basin, 1) arrays, i.e. basins-first -- the same orientation used
    here.
    """
    from hydropfn.metrics.dmg_metrics import Metrics

    pred = np.asarray(pred, dtype=np.float32)
    obs = np.asarray(obs, dtype=np.float32)
    if pred.shape != obs.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {obs.shape}")
    # Metrics refuses NaN predictions outright ("check your gradient chain").
    # NaN in the TARGET is normal (gauge outages) and is handled inside.
    if not np.isfinite(pred).all():
        raise ValueError("non-finite predictions; refusing to score")
    return Metrics(pred, obs)


def describe(extent, protocol, gage_table, time):
    """One-line-per-fold description, printed at the top of every run.

    A run that silently scores the wrong basins over the wrong days looks
    exactly like one that does not, which is the failure mode this whole
    module exists to close.
    """
    w = windows(time, protocol)
    fs = folds(extent, gage_table)
    n_score = w["score"].stop - w["score"].start
    b = BENCHMARKS.get(protocol)
    lines = [
        f"benchmark {protocol}"
        + (f"  [{b['label']}, {b['basins']} basins]" if b else "")
        + f" / extent {extent}",
    ] + ([f"  note         : {b['note']}"] if b else []) + [
        f"  train window : {w['labels']['train'][0]} .. "
        f"{w['labels']['train'][1]}  ({w['train'].stop - w['train'].start} d)",
        f"  scored       : {w['labels']['test'][0]} .. "
        f"{w['labels']['test'][1]}  ({n_score} d, warmup {WARMUP} d prepended "
        f"and NOT scored)",
        f"  folds        : {len(fs)}  sizes {[len(f) for f in fs]}"
        f"  total {sum(len(f) for f in fs)}",
    ]
    if extent != "temporal":
        seen = np.concatenate(fs)
        if len(np.unique(seen)) != len(seen):
            lines.append("  ERROR: folds overlap")
        if len(seen) != len(gage_table):
            lines.append(f"  ERROR: folds cover {len(seen)} of "
                         f"{len(gage_table)} basins")
    return "\n".join(lines)
