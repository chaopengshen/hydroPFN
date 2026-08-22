# Compute environments

## ICDS (`ssh icds` -> submit.hpc.psu.edu)

Needs an **interactive MFA login first**; the cached credential lasts ~8 h.
Never pass `-o BatchMode=yes` — it suppresses the cached-credential fallback.
Do NOT add `ControlMaster`/`ControlPath` for this host: Windows OpenSSH cannot
multiplex and it breaks login outright (confirmed twice).

**The login node's default Python is 3.6 with no scientific stack, and `pip`
cannot build `netCDF4` there.** The fix is `uv`, not the system Python
(Nicholas Kraabel, 2026-08-21) — a user-local install, no admin needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/uvi.sh && sh /tmp/uvi.sh
export PATH="$HOME/.local/bin:$PATH"          # add to ~/.bashrc
uv venv --python 3.11 ~/work/ncenv
uv pip install --python ~/work/ncenv/bin/python xarray netCDF4 numpy pandas
```

Note: `curl -LsSf ... | sh` piped directly failed here; downloading to a file
first and running it worked. Verified: `~/work/ncenv/bin/python` opens
CAMELS_Frederik.nc.

Key paths:

```
StefaLand code     /storage/group/cxs1024/default/shared_model_code/MFFormer
StefaLand weights  .../shared_model_code/FoundationModels/StefalandOriginalGlobal20.pt
StefaLand embeds   .../shared_model_code/EmbeddingDatasets
CAMELS             /storage/group/cxs1024/default/nrk5343/StefaLandData/CAMELS_Frederik.nc
```

Why this matters beyond data loading: verifying our StefaLand checkpoint load
means running **their** `eval_probe.py` against `probe_configs/camels_spatial.yaml`
and matching their numbers — and that has to happen where the checkpoint lives.

## suntzu (`ssh suntzu` -> suntzu.cee.psu.edu)

Where the GPU work runs. Two Pythons, and picking the wrong one wastes hours:

| use | interpreter |
|---|---|
| torch / xarray / netCDF4 / sklearn | `/data/cxs1024/tools/anaconda3/envs/pytorch_gpu/bin/python` |
| rasterio / pyogrio / geopandas (DEM) | `/nfs/data/cxs1024/envs/demenv/bin/python` (a venv; invisible to `conda env list`; no torch, no matplotlib) |

`source gpuenv.sh` handles the torch env and its two traps:

1. `pytorch_gpu` lives under a **user** anaconda, not the system conda base, so
   `conda activate pytorch_gpu` fails with `EnvironmentNameNotFound` — and
   inside a `bash -lc` chain it silently leaves you on base CPU torch 1.12.
   The only symptom is the word `cpu` in the run banner and ~16x slower epochs.
   Call the interpreter by absolute path.
2. `libcusparse` needs the pip `nvjitlink` shim first on `LD_LIBRARY_PATH`, and
   the path must be **literal** — an unexpanded `python3.*` glob fails the same
   way. Without it, `import torch` dies on `__nvJitLinkAddData_12_1`.
3. `CUDA_DEVICE_ORDER=PCI_BUS_ID` is required, else `CUDA_VISIBLE_DEVICES=2`
   lands on an 11 GB 2080 Ti instead of the 24 GB 3090 Ti.

GPUs: 0,1,3,4 are RTX 2080 Ti (11 GB); **2 is an RTX 3090 Ti (24 GB)**.

Local data:

```
/nfs/data/cxs1024/hydroPFN/data/CAMELS_Frederik.nc        copied 2026-08-21
/nfs/data/cxs1024/dem_foundation/logs/dem_patches.npz     6,000 gage-centred
/nfs/data/cxs1024/dem_foundation/logs/dem_patches_uniform.npz  8,000 / 677 tiles
```

Moving a large file ICDS -> suntzu without landing it locally:

```bash
ssh icds 'cat /path/on/icds' | ssh suntzu 'cat > /path/on/suntzu'
```
