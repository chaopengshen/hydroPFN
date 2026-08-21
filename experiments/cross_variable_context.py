"""T2: does knowing a site's WIDTH help predict its DEPTH at other flows?

Reproduces: results/cross_variable_context.csv

This is the cheap feasibility test for the hydrology-PFN proposal's use case 3
(docs/PROPOSAL_SEED_hydrology_PFN.md in dem_foundation): context rows carrying
one variable (width) informing a query for another (depth) at the same site.
HYDRoSWOT's real structure is exactly this -- width 89% populated, mean depth
26% -- so if the benefit exists it is harvestable at scale.

It is also the constructive counterpart of the residual-learnability gate.
That gate showed site-level geometry residuals are NOT predictable from any
terrain or tabular feature (all cells negative, two fold structures).  Here we
ask whether they are predictable FROM EACH OTHER: a site anomalously wide for
its attributes should be anomalously shallow and/or slow (W*d*v = Q), and how
that trade-off splits between d and v is precisely the site-level shape
information a cross-variable model could transfer.  Residual-from-terrain
failing does not preclude residual-from-residual succeeding -- the latter uses
the query site's own measurements, which no x can substitute for.

Design (everything leave-HUC2-out, splits by site via COMID grouping upstream):

  stage W   tabular attributes -> log_W on rows with measured width; OOF
            residuals give each width row an anomaly.
  site anomaly
            per depth row: median W-residual over the SAME SITE's width rows,
            EXCLUDING the current row.  Same-row exclusion is not hygiene
            theatre: measured W passed a +-5% continuity gate against Q/(d*v),
            so a same-visit W residual determines the d+v residual almost
            exactly, and including it would smuggle the identity in as skill.
            Cross-visit anomalies share no such identity.
  stage D   predict log_d from tabular alone (A0) vs tabular + w_anom (A1),
            OOF under the same folds.  Fair comparison on identical row sets.

Read-out:
  corr(w_anom, d-residual)      the raw coupling, one interpretable number
  A1 - A0 on rows with anomaly  the harvestable gain
  same split for log_v          where does the trade-off land?
  subset without same-row W     the true HYDRoSWOT use case (W measured on
                                other visits only)
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from hydropfn.data.folds import assign_region_folds, r2
from experiments.residual_learnability import TABULAR, _oof


def leave_one_out_site_median(site: np.ndarray, resid: np.ndarray,
                              has: np.ndarray) -> np.ndarray:
    """For every row, the median of `resid` over its site's rows where `has`,
    excluding the row itself.  NaN where nothing remains."""
    out = np.full(len(site), np.nan)
    df = pd.DataFrame({"site": site, "resid": np.where(has, resid, np.nan)})
    for s, g in df.groupby("site", sort=False):
        vals = g.resid.to_numpy()
        ok = np.isfinite(vals)
        if not ok.any():
            continue
        idx = g.index.to_numpy()
        for k, i in enumerate(idx):
            m = ok.copy()
            m[k] = False
            if m.any():
                out[i] = np.median(vals[m])
    return out


def main(table: str, out_csv: str, seed: int) -> None:
    df = pd.read_csv(table, low_memory=False)
    fold, regions = assign_region_folds(df)
    keep = fold >= 0
    df, fold = df[keep].reset_index(drop=True), fold[keep]

    Xt = np.nan_to_num(df[TABULAR].to_numpy(float), nan=0.0)
    site = df.site_no.to_numpy()
    yw = df.log_W.to_numpy(float)
    has_w = np.isfinite(yw)
    n_sites_w = df.site_no[has_w].nunique()
    print(f"{len(df):,} rows / {df.site_no.nunique():,} sites; measured width "
          f"on {has_w.sum():,} rows / {n_sites_w:,} sites")

    # ---- stage W: OOF width model on measured-width rows
    pw = np.full(len(df), np.nan)
    pw[has_w] = _oof(Xt[has_w], yw[has_w], fold[has_w], seed)
    rw = yw - pw                                    # width residual (OOF)
    print(f"  stage-W  log_W R2 = {r2(yw[has_w], pw[has_w]):.3f}")

    # ---- site-level width anomaly, leave-current-row-out
    w_anom = leave_one_out_site_median(site, rw, has_w & np.isfinite(rw))
    has_a = np.isfinite(w_anom)
    print(f"  rows with a cross-visit width anomaly: {has_a.sum():,} "
          f"({df.site_no[has_a].nunique():,} sites)")

    rows = []
    for tgt in ("log_d", "log_v"):
        y = df[tgt].to_numpy(float)

        # A0: tabular only.  A1: tabular + anomaly.  Same rows (has_a) so the
        # comparison is never confounded by an easier population.
        p0 = _oof(Xt[has_a], y[has_a], fold[has_a], seed)
        X1 = np.hstack([Xt, w_anom[:, None]])
        X1 = np.nan_to_num(X1, nan=0.0)
        p1 = _oof(X1[has_a], y[has_a], fold[has_a], seed)
        r0, r1 = r2(y[has_a], p0), r2(y[has_a], p1)

        # raw coupling: corr of the anomaly with A0's residual
        res0 = y[has_a] - p0
        m = np.isfinite(res0)
        cc = float(np.corrcoef(w_anom[has_a][m], res0[m])[0, 1])

        # the true HYDRoSWOT pattern: depth rows WITHOUT same-row width.
        # p0/p1 live on the has_a-compressed axis, so build the sub-mask there.
        idx_a = np.flatnonzero(has_a)
        sub_c = ~has_w[idx_a]                       # compressed-axis mask
        n_sub = int(sub_c.sum())
        s0 = r2(y[idx_a][sub_c], p0[sub_c]) if n_sub > 500 else np.nan
        s1 = r2(y[idx_a][sub_c], p1[sub_c]) if n_sub > 500 else np.nan

        print(f"\n=== {tgt} ===  (n={has_a.sum():,})")
        print(f"  corr(w_anom, residual)      {cc:+.3f}")
        print(f"  A0 tabular                  R2 = {r0:.4f}")
        print(f"  A1 tabular + width anomaly  R2 = {r1:.4f}   gain {r1-r0:+.4f}")
        if np.isfinite(s0):
            print(f"  no-same-row-W subset (n={n_sub:,}): "
                  f"A0 {s0:.4f} -> A1 {s1:.4f}   gain {s1-s0:+.4f}")
        rows.append({"target": tgt, "n": int(has_a.sum()), "corr": cc,
                     "r2_A0": r0, "r2_A1": r1, "gain": r1 - r0,
                     "n_nosame": n_sub, "r2_A0_nosame": s0,
                     "r2_A1_nosame": s1})

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n  gain ~ 0 on log_d AND log_v -> use case 3 is dead; the PFN "
          "proposal\n  leans on cases 1, 2, 4, 5.  A real gain here is the "
          "first evidence that\n  cross-variable context transfers site-level "
          "channel-shape information.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="cross_variable_context.csv")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    main(a.table, a.out, a.seed)
