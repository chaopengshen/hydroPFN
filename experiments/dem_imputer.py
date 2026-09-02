"""Stage 1 of the DEM-as-attribute-imputer test: can each input tier
reconstruct the curated statics on held-out PUB groups?

Tiers (cumulative ladder, plus DEM alone for reference):
  CLIM       climate indices computed from the basin's own forcings --
             free anywhere on earth, the null competitor
  KRIG       statics IDW-interpolated from TRAINING basins by lat/lon --
             the location fingerprint made into an honest baseline
  CLIM+KRIG  what a global operator has without any terrain model
  +DEM       add the 768-dim multi-scale diffusion features (parity ckpt,
             12.8 km footprint) -- THE CLAIM: its increment over CLIM+KRIG

Out-of-fold ridge (alpha by inner 3-fold CV), leave-one-PUB-group-out over
all 10 groups, per-attribute R^2 pooled over held-out basins. Reported per
attribute block and stratified by distance to the nearest training basin.
DEM's value = the (+DEM) - (CLIM+KRIG) column, nothing else.
"""
import os

os.environ["HYDROPFN_CAMELS_ROOT"] = "/nfs/data/cxs1024/hydroPFN/data"

import sys                                                   # noqa: E402

sys.path.insert(0, "src")
import numpy as np                                           # noqa: E402
import xarray as xr                                          # noqa: E402
from scipy.spatial import cKDTree                            # noqa: E402
from sklearn.linear_model import Ridge                       # noqa: E402
from sklearn.model_selection import KFold                    # noqa: E402

from hydropfn.data import protocol as P                      # noqa: E402
from hydropfn.data.forcing import load_camels                # noqa: E402

d = load_camels(P.CAMELS_NC)
sub, gage = P.load_531(d)
A = np.asarray(sub["attrs"], np.float64)          # (531, n_attr)
X = np.asarray(sub["x"])                          # (531, T, V) forcings+Q
ll = np.asarray(sub["latlon"], np.float64)
sid = np.asarray([int(s) for s in sub["site_id"]])
pub = np.asarray(gage["PUB_ID"])

from hydropfn.data.forcing import DEFAULT_STATICS  # noqa: E402
names = list(DEFAULT_STATICS)
assert len(names) == A.shape[1]
print(f"{A.shape[0]} basins, {A.shape[1]} attrs: {names}")

BLOCKS = {
    "climate": ("p_mean", "pet", "p_season", "frac_snow", "arid",
                "high_prec", "low_prec"),
    "topo": ("elev", "slope", "area"),
    "veg": ("forest", "lai", "gvf", "land_cover", "root"),
    "soil": ("soil", "sand", "silt", "clay", "water_content"),
    "geol": ("geol", "glim", "carbonate"),
}


def block_of(n):
    for b, keys in BLOCKS.items():
        if any(k in n.lower() for k in keys):
            return b
    return "other"


blocks = np.array([block_of(n) for n in names])

# ---- CLIM: indices from the basin's own forcings, train period only
tw = P.windows(sub["time"], "spatial")
tr_sl = slice(0, tw["score"].start)               # everything before scoring
F = X[:, tr_sl, :5]                               # prcp srad tmax tmin vp
prcp, srad, tmax, tmin, vp = (F[..., i] for i in range(5))
tmean = 0.5 * (tmax + tmin)
doy = np.arange(F.shape[1]) % 365
month = (doy // 30.44).astype(int) % 12
pm = np.stack([np.array([p[month == m].mean() for m in range(12)])
               for p in prcp])
clim = np.stack([
    prcp.mean(1), prcp.std(1),
    (prcp < 1.0).mean(1),                              # low-prec freq
    (prcp > 5 * prcp.mean(1, keepdims=True)).mean(1),  # high-prec freq
    pm.std(1) / np.clip(pm.mean(1), 1e-6, None),       # seasonality
    np.where(tmean < 0, prcp, 0).sum(1)
    / np.clip(prcp.sum(1), 1e-6, None),                # frac snow
    tmean.mean(1), tmax.mean(1) - tmin.mean(1),
    srad.mean(1), vp.mean(1),
    srad.mean(1) / np.clip(prcp.mean(1), 1e-6, None),  # aridity proxy
], -1)
print(f"CLIM {clim.shape[1]} indices")

# ---- DEM features (parity multi-scale ckpt, 12.8 km, 8-draw averaged)
z = np.load("logs/camels_demfeat13_parity.npz", allow_pickle=True)
fmap = {int(s): i for i, s in enumerate(np.asarray(z["site_id"]))}
ok_all = np.asarray(z["ok"]).astype(bool)
dem = np.zeros((len(sid), z["feats"].shape[1]), np.float64)
dem_ok = np.zeros(len(sid), bool)
for j, s in enumerate(sid):
    i = fmap.get(int(s))
    if i is not None and ok_all[i]:
        dem[j] = z["feats"][i]
        dem_ok[j] = True
print(f"DEM features present for {dem_ok.sum()}/{len(sid)} basins")

pts = np.stack([ll[:, 0], ll[:, 1] * 0.766], -1)


def krig_attrs(tr_mask, K=8):
    """IDW of training basins' attrs, and distance to nearest train basin."""
    tree = cKDTree(pts[tr_mask])
    dist, idx = tree.query(pts, k=K)
    tr_idx = np.where(tr_mask)[0]
    w = 1.0 / (dist ** 2 + 1e-4)
    kg = (A[tr_idx[idx]] * w[..., None]).sum(1) / w.sum(1)[:, None]
    return kg, dist[:, 0]


def fit_tier(Xf, tr, te):
    mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-9
    Xs = (Xf - mu) / sd
    best, bscore = None, -1e18
    for al in (1.0, 10.0, 100.0, 1000.0):
        sc = 0.0
        for itr, iva in KFold(3, shuffle=True, random_state=0).split(tr):
            r = Ridge(alpha=al).fit(Xs[tr[itr]], Ya[tr[itr]])
            sc += r.score(Xs[tr[iva]], Ya[tr[iva]])
        if sc > bscore:
            bscore, best = sc, al
    r = Ridge(alpha=best).fit(Xs[tr], Ya[tr])
    return r.predict(Xs[te])


Amu, Asd = A.mean(0), A.std(0) + 1e-9
Ya = (A - Amu) / Asd

tiers = ["CLIM", "KRIG", "CLIM+KRIG", "CLIM+KRIG+DEM", "DEM"]
oof = {t: np.full_like(Ya, np.nan) for t in tiers}
near = np.full(len(sid), np.nan)

for g in np.unique(pub):
    te = np.where(pub == g)[0]
    tr = np.where(pub != g)[0]
    kg, dist0 = krig_attrs(pub != g)
    near[te] = dist0[te]
    kgz = (kg - Amu) / Asd
    # DEM rows missing features: fill with train mean (fair: an operator
    # without DEM for a basin falls back to the prior)
    dmu = dem[tr[dem_ok[tr]]].mean(0)
    demf = np.where(dem_ok[:, None], dem, dmu)
    feats = {"CLIM": clim, "KRIG": kgz, "CLIM+KRIG": np.hstack([clim, kgz]),
             "CLIM+KRIG+DEM": np.hstack([clim, kgz, demf]), "DEM": demf}
    for t in tiers:
        oof[t][te] = fit_tier(feats[t], tr, te)

def r2_cols(pred, rows):
    out = []
    for j in range(Ya.shape[1]):
        y, p = Ya[rows, j], pred[rows, j]
        m = np.isfinite(y) & np.isfinite(p)
        den = ((y[m] - y[m].mean()) ** 2).sum()
        out.append(1 - ((y[m] - p[m]) ** 2).sum() / den if den > 0 else np.nan)
    return np.array(out)


allr = np.arange(len(sid))
print(f"\n=== out-of-fold attribute R^2 (median over attrs) ===")
print(f"{'tier':>15} {'all':>7}", *(f"{b:>8}" for b in BLOCKS), sep="")
for t in tiers:
    r2 = r2_cols(oof[t], allr)
    row = [f"{t:>15} {np.nanmedian(r2):>7.3f}"]
    for b in BLOCKS:
        row.append(f"{np.nanmedian(r2[blocks == b]):>8.3f}")
    print("".join(row))

print("\n=== DEM increment over CLIM+KRIG, by distance to nearest train basin ===")
med = np.nanmedian(near)
for label, rows in (("near half", allr[near <= med]),
                    ("far half", allr[near > med])):
    base = r2_cols(oof["CLIM+KRIG"], rows)
    plus = r2_cols(oof["CLIM+KRIG+DEM"], rows)
    srt = np.argsort(-(plus - base))
    for j in srt[:5]:
        print(f"{'':>13}top: {names[j]:<22} {base[j]:+.3f} -> {plus[j]:+.3f}")
    print(f"{label:>10} (n={len(rows)}): CLIM+KRIG {np.nanmedian(base):.3f}"
          f" -> +DEM {np.nanmedian(plus):.3f}"
          f"   delta {np.nanmedian(plus - base):+.3f}")
    for b in BLOCKS:
        db = np.nanmedian((plus - base)[blocks == b])
        print(f"{'':>14}{b}: {db:+.3f}")


# ---- Stage 1b: donor-exclusion radius curve -----------------------------
# Manufactures the global-sparse regime on CONUS: kriging donors must lie
# farther than R km from each target. PCA-32 on the DEM block (fit per train
# fold) keeps the joint ridge alpha from having to shrink 768 raw dims;
# without it the +DEM tier is crippled by regularisation and reads falsely
# negative (measured: -0.05 at every radius with raw dims, ~0 with PCA).
from sklearn.decomposition import PCA

DEG = 111.0
print(f"\n{'excl km':>8} {'CLIM+KRIG':>10} {'+DEM':>7} {'delta':>7} "
      f"{'soil d':>7} {'geol d':>7} {'clim d':>7}")
sg = np.isin(blocks, ["soil", "geol"])
for R in (0, 50, 100, 200, 400):
    oofb, oofd = np.full_like(Ya, np.nan), np.full_like(Ya, np.nan)
    for g in np.unique(pub):
        te = np.where(pub == g)[0]
        tr = np.where(pub != g)[0]
        tree = cKDTree(pts[tr])
        dist, idx = tree.query(pts, k=len(tr))
        kg = np.zeros_like(A)
        trmean = A[tr].mean(0)
        for i in range(len(pts)):
            m = dist[i] * DEG > R
            if m.sum() == 0:
                kg[i] = trmean
                continue
            dd, ii = dist[i][m][:8], idx[i][m][:8]
            w = 1.0 / (dd ** 2 + 1e-4)
            kg[i] = (A[tr[ii]] * w[:, None]).sum(0) / w.sum()
        kgz = (kg - Amu) / Asd
        dmu = dem[tr[dem_ok[tr]]].mean(0)
        demf = np.where(dem_ok[:, None], dem, dmu)
        demp = PCA(n_components=32, random_state=0).fit(
            demf[tr]).transform(demf)
        oofb[te] = fit_tier(np.hstack([clim, kgz]), tr, te)
        oofd[te] = fit_tier(np.hstack([clim, kgz, demp]), tr, te)
    b, p_ = r2_cols(oofb, allr), r2_cols(oofd, allr)
    print(f"{R:>8} {np.nanmedian(b):>10.3f} {np.nanmedian(p_):>7.3f} "
          f"{np.nanmedian(p_ - b):>+7.3f} "
          f"{np.nanmedian((p_ - b)[sg]):>+7.3f} "
          f"{np.nanmedian((p_ - b)[blocks == 'geol']):>+7.3f} "
          f"{np.nanmedian((p_ - b)[blocks == 'climate']):>+7.3f}")
