"""Train mini-HydroPFN v1 and run its three pre-registered acceptance tests.

Reproduces: logs/hydropfn_v1_s{seed}.csv, checkpoint logs/hydropfn_v1.pt

Phase 1 of docs/BUILD_PLAN_hydropfn.md.  The gates were fixed BEFORE any run:

  A1 cross-variable (reproduce T2).  Predict log_v at a query site whose
     context holds ONLY that site's log_W visits.  Must beat the same model
     given no own-site context by >= +0.03 R2.  The hand-built cross-visit
     width-anomaly feature got +0.041 with a plain RF; the point of a PFN is
     to find that coupling itself, so anything near zero means the
     architecture is not earning its complexity.
  A2 at-a-station (T1).  Context = the query site's own visits of the SAME
     variable.  Must beat a per-site power-law fit on median per-site R2.
  A3 graceful degradation.  With no context tokens at all, must not be worse
     than an attributes-only RF -- this is the "only x, no neighbours" mode
     that has to work wherever nothing else is measured.

Splits are leave-HUC2-out by SITE.  Evaluation sites are never in training,
and a query's target visit is never in its own context.

Run (suntzu):
    source /nfs/data/cxs1024/dem_foundation/gpuenv.sh
    $PY -u test_hydropfn_v1.py --table <train_table_dem.csv>
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from hydropfn.models.measurement_pfn import VARS, HydroPFN, make_borders  # noqa: E402

from hydropfn.paths import ROOT  # noqa: E402
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ATTRS = ["log_A_drain", "log_slope", "log_q_mm_day", "sinuosity",
         "log_lengthkm", "log_arbolatesu", "log_drain_density", "log_A_local",
         "NDVI", "AI", "Forest", "Agriculture", "Developed",
         "b0_sand", "b0_clay", "b10_clay", "b10_sand", "Silt_101"]


# ------------------------------------------------------------------ data

class SiteStore:
    """Per-site attributes and visit lists, plus the samplers that build
    training examples and the three evaluation regimes."""

    def __init__(self, df: pd.DataFrame, n_ctx: int, n_meas: int):
        self.n_ctx, self.n_meas = n_ctx, n_meas
        self.sites = df.site_no.unique()
        self.idx = {s: i for i, s in enumerate(self.sites)}

        A = df.groupby("site_no")[ATTRS].median().reindex(self.sites)
        self.mu = np.nanmean(A.to_numpy(float), 0)
        self.sd = np.nanstd(A.to_numpy(float), 0) + 1e-6
        self.attrs = np.nan_to_num((A.to_numpy(float) - self.mu) / self.sd)

        self.huc2 = df.groupby("site_no").HUC2.first().reindex(self.sites) \
                      .to_numpy()
        # visits: (var, value, log_Q) per site, only finite values
        self.visits: list[np.ndarray] = []
        # groupby DROPS NaN keys while .unique() keeps them, so build the site
        # list from the groups themselves rather than trusting the two agree.
        g = dict(tuple(df.groupby("site_no")))
        for s in self.sites:
            sub = g[s]
            rows = []
            # Column 3 is the OCCASION id (the source row).  One field visit
            # yields up to three variables sharing one discharge, and
            # W*d*v = Q is an identity in these data -- so leaving a target's
            # siblings in context lets the model do arithmetic and call it
            # cross-variable inference.  The first run of this test did
            # exactly that: `crossvar` beat `own` on all three variables,
            # which is impossible for honest inference because own's context
            # is a superset.  Occasions are excluded wholesale below.
            occ = np.arange(len(sub))
            for vi, v in enumerate(VARS):
                y = sub[v].to_numpy(float)
                q = sub.log_Q.to_numpy(float)
                ok = np.isfinite(y) & np.isfinite(q)
                if ok.any():
                    rows.append(np.column_stack(
                        [np.full(ok.sum(), vi), y[ok], q[ok], occ[ok]]))
            self.visits.append(np.concatenate(rows) if rows
                               else np.zeros((0, 4)))

    # -- assembling one example -------------------------------------------

    def _pack(self, site_i, own_visits, ctx_sites, rng):
        """own_visits: (k,3) array already filtered of the target row."""
        S = 1 + self.n_ctx
        M = self.n_meas
        attrs = np.zeros((S, len(ATTRS)), np.float32)
        a_valid = np.zeros(S, np.float32)
        m_var = np.zeros((S, M), np.int64)
        m_val = np.zeros((S, M), np.float32)
        m_cov = np.zeros((S, M), np.float32)
        m_valid = np.zeros((S, M), np.float32)

        def put(slot, si, vis):
            attrs[slot] = self.attrs[si]
            a_valid[slot] = 1.0
            if len(vis):
                if len(vis) > M:
                    vis = vis[rng.choice(len(vis), M, replace=False)]
                k = len(vis)
                m_var[slot, :k] = vis[:, 0].astype(np.int64)
                m_val[slot, :k] = vis[:, 1]
                m_cov[slot, :k] = vis[:, 2]
                m_valid[slot, :k] = 1.0

        put(0, site_i, own_visits)
        for j, cs in enumerate(ctx_sites[:self.n_ctx]):
            put(1 + j, cs, self.visits[cs])
        return dict(attrs=attrs, a_valid=a_valid, m_var=m_var, m_val=m_val,
                    m_cov=m_cov, m_valid=m_valid)

    def train_example(self, site_i, rng, p_drop_own=0.25, pool=None):
        vis = self.visits[site_i]
        if len(vis) < 1:
            return None
        t = rng.integers(len(vis))
        target = vis[t]
        # drop the WHOLE occasion, not just the target visit (see __init__)
        own = vis[vis[:, 3] != target[3]]
        # Curriculum over the three regimes we will be graded on: sometimes
        # hide all own-site visits (A3), sometimes hide same-variable ones so
        # only the OTHER variables remain (A1), otherwise keep everything (A2).
        r = rng.random()
        if r < p_drop_own:
            own = own[:0]
        elif r < p_drop_own + 0.35 and len(own):
            own = own[own[:, 0] != target[0]]
        ctx = rng.choice(pool, size=min(self.n_ctx, len(pool)), replace=False)
        ex = self._pack(site_i, own, ctx, rng)
        ex["q_var"] = np.int64(target[0])
        ex["q_cov"] = np.float32(target[2])
        ex["y"] = np.float32(target[1])
        return ex

    def eval_example(self, site_i, target, mode, rng, pool):
        """mode: 'own' (same var), 'crossvar' (other vars only), 'none'."""
        vis = self.visits[site_i]
        own = vis[vis[:, 3] != target[3]]        # whole occasion excluded
        if mode == "none":
            own = own[:0]
        elif mode == "crossvar":
            own = own[own[:, 0] != target[0]]
        ctx = rng.choice(pool, size=min(self.n_ctx, len(pool)), replace=False)
        ex = self._pack(site_i, own, ctx, rng)
        ex["q_var"] = np.int64(target[0])
        ex["q_cov"] = np.float32(target[2])
        ex["y"] = np.float32(target[1])
        return ex


def collate(exs, device):
    out = {}
    for k in exs[0]:
        arr = np.stack([e[k] for e in exs])
        dt = torch.long if arr.dtype == np.int64 else torch.float32
        out[k] = torch.tensor(arr, dtype=dt, device=device)
    return out


def r2(o, p):
    o, p = np.asarray(o), np.asarray(p)
    m = np.isfinite(o) & np.isfinite(p)
    return float(1 - ((o[m] - p[m]) ** 2).sum() /
                 ((o[m] - o[m].mean()) ** 2).sum())


# ------------------------------------------------------------------ main

def main(table, holdout, epochs, steps, batch, n_ctx, n_meas, d, depth,
         seed, lr):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    df = pd.read_csv(table, low_memory=False)
    n0 = len(df)
    df = df[df.HUC2.notna() & df.site_no.notna()].reset_index(drop=True)
    if len(df) < n0:
        print(f"dropped {n0-len(df):,} rows with no HUC2 or no site_no")
    df["HUC2"] = df.HUC2.apply(
        lambda h: f"{int(float(h)):02d}" if pd.notna(h) else h)
    store = SiteStore(df, n_ctx, n_meas)

    te_site = np.flatnonzero(store.huc2.astype(str) == str(holdout))
    tr_site = np.flatnonzero(store.huc2.astype(str) != str(holdout))
    print(f"{len(df):,} rows | {len(store.sites):,} sites | "
          f"train {len(tr_site):,} / holdout HUC2 {holdout} {len(te_site):,} "
          f"| {DEVICE}", flush=True)
    if len(te_site) < 30:
        raise SystemExit(f"holdout HUC2 {holdout} has too few sites")

    borders = make_borders(
        [df[v].to_numpy(float) for v in VARS]).to(DEVICE)
    net = HydroPFN(len(ATTRS), borders, d=d, depth=depth).to(DEVICE)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"  HydroPFN {n_par/1e6:.1f}M params, {n_ctx} context sites x "
          f"{n_meas} visits", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=lr, total_steps=epochs * steps, pct_start=0.1)

    t0 = time.time()
    for ep in range(epochs):
        net.train()
        tot, nb = 0.0, 0
        for _ in range(steps):
            exs = []
            while len(exs) < batch:
                si = int(rng.choice(tr_site))
                e = store.train_example(si, rng, pool=tr_site)
                if e is not None:
                    exs.append(e)
            b = collate(exs, DEVICE)
            logits = net(b)
            loss = net.bar.loss(logits, b["y"], b["q_var"])
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  epoch {ep+1}/{epochs}  CE {tot/nb:.4f}  "
                  f"[{(time.time()-t0)/60:.1f} min]", flush=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    torch.save({"net": net.state_dict(), "d": d, "depth": depth},
               ROOT / "logs" / "hydropfn_v1.pt")

    # ---------------------------------------------------------- evaluation
    net.eval()
    erng = np.random.default_rng(seed + 500)
    targets = []                       # (site_i, visit row)
    for si in te_site:
        vis = store.visits[si]
        if len(vis) < 4:
            continue
        for t in erng.choice(len(vis), size=min(3, len(vis)), replace=False):
            targets.append((int(si), vis[t]))
    print(f"\n  {len(targets):,} held-out query visits from "
          f"{len(set(t[0] for t in targets)):,} sites", flush=True)

    preds = {m: [] for m in ("own", "crossvar", "none")}
    ys, qvars, qsites = [], [], []
    with torch.no_grad():
        for i in range(0, len(targets), batch):
            chunk = targets[i:i + batch]
            for mode in preds:
                exs = [store.eval_example(si, tgt, mode, erng, tr_site)
                       for si, tgt in chunk]
                b = collate(exs, DEVICE)
                mu = net.bar.mean(net(b), b["q_var"]).cpu().numpy()
                preds[mode].extend(mu.tolist())
            ys.extend([float(t[1]) for _, t in chunk])
            qvars.extend([int(t[0]) for _, t in chunk])
            qsites.extend([si for si, _ in chunk])
    ys = np.array(ys); qvars = np.array(qvars); qsites = np.array(qsites)

    # baselines on the same targets
    from sklearn.ensemble import RandomForestRegressor
    rf_pred = np.full(len(ys), np.nan)
    pl_pred = np.full(len(ys), np.nan)
    tr_rows = df[~df.site_no.isin(store.sites[te_site])]
    for vi, v in enumerate(VARS):
        sel = qvars == vi
        if not sel.any():
            continue
        sub = tr_rows[np.isfinite(tr_rows[v].to_numpy(float))]
        Xtr = np.nan_to_num(
            np.column_stack([(sub[ATTRS].to_numpy(float) - store.mu) / store.sd,
                             sub.log_Q.to_numpy(float)]))
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed)
        rf.fit(Xtr, sub[v].to_numpy(float))
        qi = np.flatnonzero(sel)
        Xq = np.nan_to_num(np.column_stack([
            store.attrs[qsites[qi]],
            np.array([targets[j][1][2] for j in qi])]))
        rf_pred[qi] = rf.predict(Xq)
        # per-site power law on the site's own same-variable visits
        for j in qi:
            si, tgt = targets[j]
            vis = store.visits[si]
            # same occasion exclusion as the model gets, so the baseline is
            # handicapped identically
            keep = (vis[:, 0] == vi) & (vis[:, 3] != tgt[3])
            o = vis[keep]
            if len(o) >= 3 and np.std(o[:, 2]) > 1e-6:
                A = np.column_stack([np.ones(len(o)), o[:, 2]])
                c, *_ = np.linalg.lstsq(A, o[:, 1], rcond=None)
                pl_pred[j] = c[0] + c[1] * tgt[2]
            elif len(o):
                pl_pred[j] = o[:, 1].mean()

    common = np.isfinite(rf_pred) & np.isfinite(pl_pred)
    for m in preds:
        preds[m] = np.array(preds[m])
        common &= np.isfinite(preds[m])
    print(f"  scoring on {common.sum():,} targets common to all methods")

    rows = []
    print("\n=== pooled R2 by query variable (holdout HUC2 "
          f"{holdout}) ===")
    for vi, v in enumerate(VARS):
        sel = common & (qvars == vi)
        if sel.sum() < 30:
            continue
        rec = {"var": v, "n": int(sel.sum()),
               "pfn_own": r2(ys[sel], preds["own"][sel]),
               "pfn_crossvar": r2(ys[sel], preds["crossvar"][sel]),
               "pfn_none": r2(ys[sel], preds["none"][sel]),
               "rf_attrs": r2(ys[sel], rf_pred[sel]),
               "powerlaw": r2(ys[sel], pl_pred[sel])}
        rows.append(rec)
        print(f"  {v:7s} n={rec['n']:5d}  own {rec['pfn_own']:+.4f}  "
              f"crossvar {rec['pfn_crossvar']:+.4f}  none "
              f"{rec['pfn_none']:+.4f}  | RF {rec['rf_attrs']:+.4f}  "
              f"powerlaw {rec['powerlaw']:+.4f}")

    out = pd.DataFrame(rows)
    out["seed"] = seed
    out["holdout"] = str(holdout)
    tag = f"s{seed}_h{holdout}"
    out.to_csv(ROOT / "logs" / f"hydropfn_v1_{tag}.csv", index=False)

    print("\n=== ACCEPTANCE (pre-registered) ===")
    vrow = next((r for r in rows if r["var"] == "log_v"), None)
    if vrow:
        gain = vrow["pfn_crossvar"] - vrow["pfn_none"]
        print(f"  A1 cross-variable (log_v): crossvar - none = {gain:+.4f}"
              f"   [gate >= +0.030; hand-built RF feature got +0.041]  "
              f"{'PASS' if gain >= 0.03 else 'FAIL'}")
    for r in rows:
        d1 = r["pfn_own"] - r["powerlaw"]
        d2 = r["pfn_none"] - r["rf_attrs"]
        print(f"  A2 at-a-station ({r['var']}): own - powerlaw = {d1:+.4f}  "
              f"{'PASS' if d1 > 0 else 'FAIL'}")
        print(f"  A3 graceful ({r['var']}): none - RF = {d2:+.4f}  "
              f"{'PASS' if d2 > -0.02 else 'FAIL'}")
    print(f"\nwrote {ROOT/'logs'/f'hydropfn_v1_{tag}.csv'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--holdout", default="03")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n-ctx", type=int, default=16)
    ap.add_argument("--n-meas", type=int, default=12)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    main(a.table, a.holdout, a.epochs, a.steps, a.batch, a.n_ctx, a.n_meas,
         a.d, a.depth, a.seed, a.lr)
