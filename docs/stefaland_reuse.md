# Reusing StefaLand weights in unit A

Short answer: **yes, substantially — the trunk and both input paths transfer;
only the observation channels and the summary queries are new.**

## What StefaLand is, as inspected

Code and weights on ICDS (Nicholas Kraabel):

```
code     /storage/group/cxs1024/default/shared_model_code/MFFormer
weights  /storage/group/cxs1024/default/shared_model_code/FoundationModels/
           StefalandOriginalGlobal20.pt   (+ .yaml, 57 MB, self-contained)
data     /storage/group/cxs1024/default/nrk5343/StefaLandData/CAMELS_Frederik.nc
```

Config (`StefalandOriginalGlobal20.yaml`):

```
hidden_size 256 · num_heads 4 · num_enc_layers 4 · d_ffd 512 · dropout 0.1
time_series_variables : P, RelHum, SWd, Tmax, Tmin        (5 inputs)
target_variables      : QObs                              (streamflow, OUTPUT)
static_variables      : ~49  (GMTED terrain, HWSD + SoilGrid2 soils, MSWX
                              climate, land cover, catchsize, porosity, ...)
```

Tensors: `batch_x [B, T, F]` dynamic, `batch_c [B, C]` static — where **its `B`
indexes sites**. It has no notion of a second site; that is exactly what the
connector adds.

## The one thing it cannot do

`time_series_variables` (read) and `target_variables` (predicted) are
**separate lists**. Observations are what StefaLand *outputs*, never what it
*reads*. So it can never use "this basin's observed streamflow" as context for
inferring anything else at that basin — which is the entire premise of an
in-context model.

That is the extension, and it is the only structural change: **move the
observation channels into the input stack and make them maskable.**

## What transfers, tensor by tensor

Our `SiteEncoder` (`models/site_encoder.py`) against StefaLand's encoder:

| our tensor | shape | transfers? | note |
|---|---|---|---|
| `static_mlp` | 49 → 256 → 256 | **yes** | same 49 attributes, same width |
| `value_proj` | patch → 256 | **partly** | same width; re-init if their patching differs |
| `var_emb` rows for `P, RelHum, SWd, Tmax, Tmin` | 5 × 256 | **yes** | identical variables |
| `var_emb` row for `QObs` | 1 × 256 | **no** | new — it was never an input |
| `trunk` (4 layers, 4 heads, d 256, FFN 512) | — | **yes** | shapes chosen to match exactly |
| `pos_emb`, `doy_proj` | — | **partly** | transfer if their windowing matches |
| `summary_q`, `mask_tok`, `head` | — | **no** | new; StefaLand has no summary tokens |

So the great majority of parameters — the trunk plus both input paths — load
unchanged. Fresh parameters are one embedding row, the summary queries, the
mask token, and the reconstruction head.

**This is why `d = 256`.** The width was not chosen for capacity; it was chosen
so this table has "yes" in it.

## How to load

Nicholas (2026-08-20): the yaml's `pretrained_model:` path is a **stale naming
artefact — ignore it**; the checkpoint is self-contained and loads directly via
the resume-checkpoint path. `eval_probe.py`'s `run_dir` layout is not required.
Run dirs go to scratch and may not persist; indexing files are not saved but
regenerate from the repo script.

Practical recipe:

1. `torch.load` the checkpoint, inspect `state_dict` keys.
2. Map by shape and name into `SiteEncoder`, `strict=False`.
3. **Print what did and did not load.** A silent `strict=False` that transfers
   nothing looks identical to one that transfers everything.
4. Verify before trusting: reproduce their own `probe_configs/camels_spatial.yaml`
   numbers with `eval_probe.py`. If our load cannot match their published
   probe, it is wrong and everything downstream is noise.

## Measured caveats — do not reuse blind

- Its embeddings reconstruct absolute elevation at R² 0.936. That is **not**
  learned memorisation: `GMTED_elevation` is one of its static *inputs*. So
  de-locating is a one-line ablation, not a research problem — but it does mean
  the representation carries a location fingerprint, which is a hazard for
  spatial transfer. Evaluate leave-region-out.
- Its embeddings added **−0.004** to channel geometry on top of 20 tabular
  attributes. Expected — that task has a tabular shortcut (drainage area,
  slope, discharge are the theoretically correct predictors and we measure all
  three). It says nothing about label-sparse tasks, which is where a
  representation should pay.
- It is **attribute-based, not image-based**, so it cannot work where no
  attribute table exists — i.e. most of the globe. That is what unit B (DEM
  pixels) is for. The two are complements, not alternatives.

## Also already in the MFFormer repo — reuse, do not rebuild

- `eval_probe.py` + `probe_configs/{camels,gmd}_{spatial,temporal}.yaml` — a
  RankMe + linear-probe harness that is **leave-region-out by construction**
  (80 train / 20 held-out basins) and already probes `geol_1st_class` (GLiM
  geology) and `dom_land_cover`. This *is* the U3 gate, prebuilt.
- An **image pathway**: `models/StefaLand_withImage_PatchTokens.py`,
  `models/image_encoders/`, `data_provider/zarr_image_loader.py`,
  `Image_guide.txt`, `StefaLandImagesAppend{128,256}.sh`. DEM patches should
  plug in there rather than through a parallel path.
