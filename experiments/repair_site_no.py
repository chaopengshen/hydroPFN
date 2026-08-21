"""Recover the missing `site_no` labels in train_table_dem.csv.

Reproduces: data/train_table_dem_fixed.csv

The defect: `site_no` is NaN on 39,141 of 64,797 rows (60%). It is inherited
all the way from `train_table.csv`, and propagates by row order into
train_table_{net,basin,dem}.csv -- all four share the identical NaN mask. The
pattern is NOT positional (35-43% missing at every within-COMID rank), so it is
a genuine merge miss rather than a "first row only" artefact. 3,165 COMIDs have
rows both with and without, i.e. the loss is row-level.

Cost: Phase-1 StefaNP trains on 25,656 of 64,797 visits, and every site-grouped
analysis silently drops the unattributed rows.

The repair: `train_table_v3.csv` carries `site_no` complete (65,341 rows,
5,023 sites). Join it on a MEASUREMENT key -- COMID plus the observed
quantities -- and copy the label across.

**Only the label moves.** CLAUDE.md forbids mixing feature encodings across a
train/test or base/augment boundary (v3's inherited attributes vs v4's
recomputed ones made "more data" look like it degraded the model by 0.07
depth). That rule is about FEATURES. `site_no` is an identifier: it cannot
leak information into a model that never sees it as an input, and it is used
only to group rows for splitting. No v3 feature column is read here.

Verification built in: rows that ALREADY have site_no are re-derived from v3
and compared. If the key were ambiguous or the tables misaligned, that
agreement would not be ~1.0, and the repair is refused.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

KEY_NUM = ["Q_cms", "d_mean_m", "v_ms"]


def build_key(df: pd.DataFrame) -> pd.Series:
    """COMID + the measured triple, rounded so float round-trips through CSV
    cannot break equality."""
    parts = [df.COMID.astype("Int64").astype(str)]
    for c in KEY_NUM:
        parts.append(df[c].astype(float).round(6).astype(str))
    return pd.Series(["|".join(t) for t in zip(*parts)], index=df.index)


def main(table: str, source: str, out: str) -> None:
    dem = pd.read_csv(table, low_memory=False)
    v3 = pd.read_csv(source, low_memory=False)
    missing_before = int(dem.site_no.isna().sum())
    print(f"target {table}\n  {len(dem):,} rows, site_no NaN {missing_before:,}"
          f" ({missing_before/len(dem):.1%})")
    print(f"source {source}\n  {len(v3):,} rows, site_no NaN "
          f"{int(v3.site_no.isna().sum()):,}, {v3.site_no.nunique():,} sites")

    for c in KEY_NUM:
        if c not in dem.columns or c not in v3.columns:
            raise SystemExit(f"key column {c} missing from one of the tables")

    v3k = v3.assign(_k=build_key(v3)).dropna(subset=["site_no"])
    # a key that maps to more than one site_no is unusable
    amb = v3k.groupby("_k").site_no.nunique()
    bad = set(amb[amb > 1].index)
    if bad:
        print(f"  dropping {len(bad):,} ambiguous keys (multiple site_no)")
    lut = (v3k[~v3k._k.isin(bad)]
           .drop_duplicates("_k").set_index("_k").site_no)

    dem["_k"] = build_key(dem)
    recovered = dem._k.map(lut)

    # --- verification on rows that already have a label
    have = dem.site_no.notna() & recovered.notna()
    if have.sum() < 200:
        raise SystemExit(f"only {have.sum()} rows to verify against; refusing")
    agree = float((dem.loc[have, "site_no"].astype(str) ==
                   recovered[have].astype(str)).mean())
    print(f"\n  verification: {have.sum():,} rows had BOTH a label and a "
          f"recovered value")
    print(f"  agreement = {agree:.4f}")
    if agree < 0.99:
        raise SystemExit("agreement below 0.99 -- the key is not reliable, "
                         "refusing to write. Do not force this.")

    fill = dem.site_no.isna() & recovered.notna()
    dem["site_no"] = dem.site_no.where(dem.site_no.notna(), recovered)
    n_exact = int(fill.sum())

    # --- fallback: COMID -> site_no, but ONLY where v3 maps that COMID to
    # exactly one site.  The exact key recovers little because v3 and this
    # table are different QC passes, so the measured values differ in the last
    # decimals -- yet 4,016 of the 4,103 unrecovered COMIDs ARE in v3.  A
    # one-to-one COMID is unambiguous evidence of which gage the row belongs
    # to; a COMID carrying several gages is skipped rather than guessed.
    per = v3.dropna(subset=["site_no"]).groupby("COMID").site_no.nunique()
    uniq = per[per == 1].index
    c2s = (v3[v3.COMID.isin(uniq)].drop_duplicates("COMID")
           .set_index("COMID").site_no)
    cand = dem.COMID.map(c2s)
    # verify the fallback the same way, on rows that already have a label
    hv = dem.site_no.notna() & cand.notna()
    agree2 = float((dem.loc[hv, "site_no"].astype(str) ==
                    cand[hv].astype(str)).mean()) if hv.sum() > 200 else np.nan
    print(f"\n  fallback check: {int(hv.sum()):,} labelled rows have a "
          f"1:1-COMID candidate; agreement = {agree2:.4f}")
    if np.isfinite(agree2) and agree2 >= 0.99:
        fill2 = dem.site_no.isna() & cand.notna()
        dem["site_no"] = dem.site_no.where(dem.site_no.notna(), cand)
        print(f"  fallback filled {int(fill2.sum()):,} rows")
    else:
        print("  fallback REJECTED (agreement < 0.99) -- exact key only")
    print(f"  exact key filled {n_exact:,} rows")

    dem = dem.drop(columns=["_k"])
    after = int(dem.site_no.isna().sum())
    print(f"  site_no NaN: {missing_before:,} -> {after:,} "
          f"({after/len(dem):.1%})")
    print(f"  usable rows: {len(dem)-missing_before:,} -> {len(dem)-after:,} "
          f"(+{(missing_before-after)/max(len(dem)-missing_before,1):.0%})")
    print(f"  sites: {dem.site_no.nunique():,}")

    # HUC2 sanity: a recovered site should not straddle regions
    if "HUC2" in dem.columns:
        span = dem.dropna(subset=["site_no"]).groupby("site_no").HUC2.nunique()
        print(f"  sites spanning >1 HUC2: {int((span > 1).sum()):,} "
              f"(should be ~0)")

    dem.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.table, a.source, a.out)
