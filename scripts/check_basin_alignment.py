"""Does hydroPFN's CAMELS data line up, basin for basin?

A port of adw/tools/check_basin_alignment.py to this project's layout. The
logic is theirs and the verdict is comparative, not a threshold: a permutation
shows up as some OTHER ordering tracking the attribute better than the stored
one. Their tool reads attributes.csv + forcing_*.npy; we hold everything in
one netCDF, so the same test is applied at the three places this pipeline
could drift:

  1. the raw netCDF          attributes vs forcings, same basin dimension
  2. after load_531()        the 531-subset reindex
  3. the gage side-file      gages_list_with_pub.csv -> PUB_ID / huc, which is
                             the one file assembled from a basin list rather
                             than from the arrays -- "where to look", per ADW

Signals: mean precipitation must track p_mean (primary), and the fraction of
precipitation falling on sub-zero days must track frac_snow (secondary; we
carry no PET forcing, so their PET signal is replaced rather than skipped).
"""
import os
import sys

os.environ.setdefault("HYDROPFN_CAMELS_ROOT", "/nfs/data/cxs1024/hydroPFN/data")
sys.path.insert(0, "src")

import numpy as np                                            # noqa: E402
import pandas as pd                                           # noqa: E402

from hydropfn.data import protocol as P                       # noqa: E402
from hydropfn.data.forcing import DEFAULT_STATICS, load_camels  # noqa: E402

MARGIN = 0.05
STRONG = 0.9
FAIL = []


def verdict(label, means, values, ids, primary=True):
    """means/values are per basin in STORED order; ids are the stored ids."""
    order = np.argsort(np.asarray(ids).astype(np.int64))
    as_written = np.corrcoef(means, values)[0, 1]
    as_sorted = np.corrcoef(means, values[order])[0, 1]
    print(f"  {label:<34} stored r={as_written:+.3f} | sorted-ID r={as_sorted:+.3f}")
    if as_sorted > as_written + MARGIN:
        FAIL.append(f"{label}: tracks better sorted ({as_sorted:.3f}) than stored "
                    f"({as_written:.3f}) -- rows permuted relative to arrays")
    elif primary and as_written < STRONG:
        FAIL.append(f"{label}: weak as stored (r={as_written:.3f}) and no simple "
                    f"reordering fixes it -- check units and basin set by hand")


def signals(x, attrs, series_names):
    """(per-basin forcing statistic, matching attribute value, label, primary)."""
    prcp = x[..., series_names.index("prcp_daymet")]
    tmax = x[..., series_names.index("tmax_daymet")]
    tmin = x[..., series_names.index("tmin_daymet")]
    tmean = 0.5 * (tmax + tmin)
    out = [("mean prcp vs p_mean", prcp.mean(1),
            attrs[:, DEFAULT_STATICS.index("p_mean")], True)]
    snow = np.where(tmean < 0, prcp, 0).sum(1) / np.clip(prcp.sum(1), 1e-9, None)
    out.append(("sub-zero prcp frac vs frac_snow", snow,
                attrs[:, DEFAULT_STATICS.index("frac_snow")], False))
    return out


print("=" * 68)
print("1. RAW netCDF:", P.CAMELS_NC)
d = load_camels(P.CAMELS_NC)
sid = np.asarray(d["site_id"]).astype(str)
names = ["prcp_daymet", "srad_daymet", "tmax_daymet", "tmin_daymet",
         "vp_daymet", "QObs"]
print(f"  {len(sid)} basins, arrays {d['x'].shape}, attrs {d['attrs'].shape}")
mis = [i for i in range(1, len(sid)) if int(sid[i]) < int(sid[i - 1])]
print(f"  station_ids are {'sorted' if not mis else f'NOT sorted ({len(mis)} out of place)'}")
for label, m, v, primary in signals(d["x"], d["attrs"], names):
    verdict(label, m, v, sid, primary)

print("\n2. AFTER load_531() (the 531-basin reindex)")
sub, gage = P.load_531(d)
ssid = np.asarray(sub["site_id"]).astype(str)
print(f"  {len(ssid)} basins, arrays {sub['x'].shape}")
mis = [i for i in range(1, len(ssid)) if int(ssid[i]) < int(ssid[i - 1])]
print(f"  subset ids are {'sorted' if not mis else f'NOT sorted ({len(mis)} out of place)'}"
      f"  <- file order, not sortedness, is what must match")
for label, m, v, primary in signals(sub["x"], sub["attrs"], names):
    verdict(label, m, v, ssid, primary)

print("\n3. GAGE SIDE-FILE: gages_list_with_pub.csv")
g = pd.read_csv(P.GAGE_SPLIT, dtype={"huc": int, "gage": str, "PUB_ID": int})
print(f"  columns: {list(g.columns)}")
gs = np.asarray(gage["gage"]).astype(str)
same = (gs.astype(np.int64) == ssid.astype(np.int64))
print(f"  gage-table row i == subset row i (by id): {same.sum()}/{len(same)}")
if not same.all():
    FAIL.append("gage table rows do not correspond to subset rows")
# does any OTHER pairing of PUB_ID to basins explain geography better? folds
# should be spatially coherent; a permuted join would scatter them
lat, lon = sub["latlon"][:, 0], sub["latlon"][:, 1]
pid = np.asarray(gage["PUB_ID"])
within = np.mean([lon[pid == k].std() for k in np.unique(pid)])
rng = np.random.default_rng(0)
shuf = np.mean([np.mean([lon[rng.permutation(pid) == k].std()
                         for k in np.unique(pid)]) for _ in range(20)])
print(f"  PUB fold longitude spread: stored {within:.2f} deg vs shuffled {shuf:.2f} deg"
      f"  ({'PUB folds are random by design' if within > 0.8 * shuf else 'spatially blocked'})")
pur = P.folds("PUR", gage)
lab = np.full(len(ssid), -1)
for k, te in enumerate(pur):
    lab[te] = k
purw = np.mean([lon[lab == k].std() for k in np.unique(lab)])
print(f"  PUR fold longitude spread: stored {purw:.2f} deg vs shuffled {shuf:.2f} deg")
if purw > 0.8 * shuf:
    FAIL.append("PUR folds are not spatially coherent -- the huc join may be permuted")
sizes = [len(t) for t in pur]
print(f"  PUR fold sizes {sizes}, total {sum(sizes)}, disjoint="
      f"{len(set(np.concatenate(pur))) == sum(sizes)}")

print("\n" + "=" * 68)
if FAIL:
    print("MISALIGNED:")
    for f in FAIL:
        print("  FAIL", f)
else:
    print("OK  attributes, forcings, subset and gage table all line up")
raise SystemExit(1 if FAIL else 0)
