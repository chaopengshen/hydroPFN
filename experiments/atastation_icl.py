"""T1: is at-a-station prediction better as IN-CONTEXT LEARNING?

Reproduces: results/t1_atastation_icl.csv

The hydrology-PFN proposal's use case 1 (docs/PROPOSAL_SEED_hydrology_PFN.md
in dem_foundation): a site's own few measurements, used as context, should act
as amortized partial pooling -- no functional form imposed, shrinkage toward
the prior for data-poor sites.  The claims to beat are measured: a per-site
power law scores 0.121 held-out per-site R^2 (width), and forcing the
exponential form on a free model costs ~0.06 R^2.

Within each evaluation site, measurements are split 70/30.  Contenders see
IDENTICAL information, so differences are model class, not data access:

  powerlaw     OLS log y ~ a + b log Q on the site's own 70%.  The classical
               at-a-station fit.
  rf_pooled    one RF per target on all non-eval-site rows PLUS the eval
               sites' 70% rows; features = attributes + log_Q.  The strong
               train-once baseline (attributes let it recognise the site).
  tabpfn_site  TabPFN, context = the site's own 70% only, X = log_Q.  Pure
               single-site ICL -- the direct replacement for the power law.
  tabpfn_nbr   TabPFN, context = 30 attribute-nearest sites' rows only (no
               own rows), X = attributes + log_Q.  Pure PUB: what context
               from NEIGHBOURS alone is worth.
  tabpfn_ctx   TabPFN, context = own 70% + the 30 neighbours, X = attributes
               + log_Q.  The proposal's actual mode.

Pre-registered read-outs: pooled R^2 over held-out rows and median per-site
R^2.  tabpfn_site > powerlaw = the free functional form + prior earns its
keep at a single site.  tabpfn_ctx > rf_pooled = in-context conditioning
beats train-once given identical information -- the T1 gate for the proposal.

Runs on a 2080 Ti (11 GB); context capped well under TabPFN limits.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd

from hydropfn.data.folds import r2
from experiments.residual_learnability import TABULAR


def tp_fit_predict(reg, Xc, yc, Xq, fails: dict, tag: str) -> np.ndarray:
    """TabPFN fit+predict with two guards.

    A site whose training rows share one discharge value gives an all-constant
    feature matrix, which TabPFN rejects (observed: TabPFNValidationError
    killed the first full run at site ~?/400).  Constant context -> the honest
    prediction is the context mean.  Any other per-site failure records NaN
    and is COUNTED rather than silently swallowed, so a systematic problem
    still surfaces in the summary."""
    if np.allclose(Xc.std(axis=0), 0.0):
        return np.full(len(Xq), float(np.mean(yc)))
    try:
        reg.fit(Xc, yc)
        return np.asarray(reg.predict(Xq), dtype=float)
    except Exception as e:                                  # noqa: BLE001
        fails[tag] = fails.get(tag, 0) + 1
        fails.setdefault("_last", f"{tag}: {type(e).__name__}: {e}")
        return np.full(len(Xq), np.nan)


def main(table: str, out_csv: str, n_sites: int, min_rows: int, n_nbr: int,
         seed: int, device: str) -> None:
    from sklearn.ensemble import RandomForestRegressor
    from tabpfn import TabPFNRegressor

    rng = np.random.default_rng(seed)
    df = pd.read_csv(table, low_memory=False)
    Xt = np.nan_to_num(df[TABULAR].to_numpy(float), nan=0.0)
    lq = df.log_Q.to_numpy(float)

    counts = df.groupby("site_no").size()
    rich = counts[counts >= min_rows].index.to_numpy()
    eval_sites = rng.choice(rich, size=min(n_sites, len(rich)), replace=False)
    eval_set = set(eval_sites)
    print(f"{len(df):,} rows; {len(rich):,} sites with >= {min_rows} rows; "
          f"evaluating {len(eval_sites)}")

    # per-row 70/30 split within eval sites (seeded, site-local)
    is_test = np.zeros(len(df), bool)
    site_rows = {s: np.flatnonzero(df.site_no.to_numpy() == s)
                 for s in eval_sites}
    for s, idx in site_rows.items():
        srng = np.random.default_rng(seed + int(str(abs(hash(s)))[-8:]))
        pick = srng.permutation(idx)
        is_test[pick[:max(2, int(0.3 * len(idx)))]] = True

    # neighbour lookup on standardised site-median attributes
    med = df.groupby("site_no")[TABULAR].median()
    M = np.nan_to_num(med.to_numpy(float), nan=0.0)
    M = (M - M.mean(0)) / (M.std(0) + 1e-9)
    site_ids = med.index.to_numpy()
    pos = {s: i for i, s in enumerate(site_ids)}
    from scipy.spatial import cKDTree
    tree = cKDTree(M)

    in_eval = df.site_no.isin(eval_set).to_numpy()
    train_mask_global = ~is_test          # everything except eval 30%

    results = {}
    for tgt in ("log_d", "log_v"):
        y = df[tgt].to_numpy(float)
        preds = {m: np.full(len(df), np.nan) for m in
                 ("powerlaw", "rf_pooled", "tabpfn_site", "tabpfn_nbr",
                  "tabpfn_ctx")}

        # --- rf_pooled: one fit per target
        Xg = np.column_stack([Xt, lq])
        ok = train_mask_global & np.isfinite(y)
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed)
        rf.fit(Xg[ok], y[ok])
        preds["rf_pooled"][is_test] = rf.predict(Xg[is_test])

        reg = TabPFNRegressor(device=device)
        fails: dict = {}
        t0 = time.time()
        for si, s in enumerate(eval_sites):
            idx = site_rows[s]
            tr = idx[~is_test[idx]]
            te = idx[is_test[idx]]
            if len(tr) < 3 or len(te) < 1:
                continue

            # power law.  lstsq on a constant log_Q column returns the
            # least-norm solution (slope ~ 0, intercept ~ mean), which is the
            # same degenerate-context answer tp_fit_predict gives -- so the
            # comparison stays fair on those sites rather than crashing one
            # contender and not the other.
            A = np.column_stack([np.ones(len(tr)), lq[tr]])
            coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
            preds["powerlaw"][te] = coef[0] + coef[1] * lq[te]

            # tabpfn_site: own rows, Q only
            preds["tabpfn_site"][te] = tp_fit_predict(
                reg, lq[tr, None], y[tr], lq[te, None], fails, "site")

            # neighbours: nearest non-eval sites by attributes
            _, nb = tree.query(M[pos[s]], k=n_nbr * 4)
            nb_sites = [site_ids[j] for j in np.atleast_1d(nb)
                        if site_ids[j] != s and site_ids[j] not in eval_set]
            nb_sites = nb_sites[:n_nbr]
            nb_rows = np.flatnonzero(df.site_no.isin(nb_sites).to_numpy())
            nb_rows = nb_rows[np.isfinite(y[nb_rows])]
            if len(nb_rows) > 1500:                     # context cap
                nb_rows = rng.choice(nb_rows, 1500, replace=False)

            if len(nb_rows) >= 50:
                preds["tabpfn_nbr"][te] = tp_fit_predict(
                    reg, Xg[nb_rows], y[nb_rows], Xg[te], fails, "nbr")
                ctx = np.concatenate([nb_rows, tr])
                preds["tabpfn_ctx"][te] = tp_fit_predict(
                    reg, Xg[ctx], y[ctx], Xg[te], fails, "ctx")

            if (si + 1) % 50 == 0:
                print(f"  {tgt}: site {si+1}/{len(eval_sites)} "
                      f"[{(time.time()-t0)/60:.1f} min]", flush=True)

        # --- scoring
        if fails:
            print(f"  {tgt} per-site TabPFN failures: "
                  f"{ {k: v for k, v in fails.items() if k != '_last'} }  "
                  f"last: {fails.get('_last')}", flush=True)
        rows = []
        # COMMON row set: every method scored on identical rows.  A method
        # that NaN'd on some sites would otherwise be scored on an easier
        # subset -- the same population confound that made v4 look +0.090
        # better than v3 on depth when the true gain was +0.024.
        finite_all = np.ones(len(df), bool)
        for p in preds.values():
            finite_all &= np.isfinite(p)
        te_all = np.flatnonzero(is_test & np.isfinite(y) & finite_all)
        print(f"  {tgt} scoring on {len(te_all):,} held-out rows common to "
              f"all methods", flush=True)
        for m, p in preds.items():
            pooled = r2(y[te_all], p[te_all])
            per = []
            for s in eval_sites:
                te = site_rows[s][is_test[site_rows[s]]]
                te = te[np.isfinite(y[te]) & finite_all[te]]
                if len(te) >= 3 and np.nanstd(y[te]) > 0:
                    per.append(r2(y[te], p[te]))
            rows.append({"target": tgt, "method": m, "pooled_r2": pooled,
                         "median_site_r2": float(np.nanmedian(per)),
                         "n_sites": len(per)})
            print(f"  {tgt} {m:12s} pooled R2 {pooled:+.4f}   "
                  f"median site R2 {np.nanmedian(per):+.4f} "
                  f"(n={len(per)})", flush=True)
        results[tgt] = rows

    out = pd.DataFrame([r for v in results.values() for r in v])
    out.to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}")
    print("\n  tabpfn_site > powerlaw   -> free-form + prior beats the "
          "exponential fit at a single site")
    print("  tabpfn_ctx  > rf_pooled  -> in-context conditioning beats "
          "train-once on identical information (the T1 gate)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", default="t1_atastation_icl.csv")
    ap.add_argument("--n-sites", type=int, default=400)
    ap.add_argument("--min-rows", type=int, default=8)
    ap.add_argument("--n-nbr", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    main(a.table, a.out, a.n_sites, a.min_rows, a.n_nbr, a.seed, a.device)
