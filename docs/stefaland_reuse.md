# Reusing StefaLand weights in unit A

Short answer: **the TRUNK transfers — 2.108M of 4.771M parameters (~44%). The
input and output embeddings do not**, because StefaLand uses one MLP per named
variable while we use a shared projection plus a variable-ID embedding, and its
variable names are the global dataset's rather than CAMELS's.

An earlier version of this page claimed both input paths transferred. That was
written from the config files; inspecting the actual tensors showed it was
wrong. The corrected table is below.

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

**CORRECTED 2026-08-21 by inspecting the actual checkpoint.** An earlier
version of this table was written from the config files and was wrong in two
important ways. Read the tensors, not the yaml.

The checkpoint is `{"epoch", "model_state_dict", "optim_state_dict", ...}`,
**485 tensors, 4.771M parameters**:

| module | tensors | params | what it actually is |
|---|---|---|---|
| `encoder.transformer_encoder` | 48 | 2.108M | **4 standard `nn.TransformerEncoderLayer`s**, d=256, FFN **512** |
| `static_embedding` | 193 | 0.853M | **one MLP PER ATTRIBUTE** (1→64→256), 48 named attributes |
| `static_projection` | 192 | 0.793M | per-attribute reconstruction heads |
| `decoder` | 4 | 0.526M | **an LSTM** (`weight_ih_l0` (1024,256)), not an MLP |
| `positional_encoding` | 1 | 0.256M | learned `position_embedding` (1000, 256) |
| `time_series_embedding` | 21 | 0.085M | **one MLP PER VARIABLE** (1→64→256), 5 variables |
| `time_series_projection` | 20 | 0.083M | per-variable reconstruction heads |
| `enc_2_dec_embedding` | 2 | 0.066M | 256→256 linear |
| `encoder_norm`, `decoder_norm` | 4 | 0.002M | LayerNorms |

### The two things the earlier table got wrong

**1. There is no shared static MLP and no variable-ID embedding table.**
StefaLand embeds **each named variable with its own 2-layer MLP**:

```
time_series_embedding.embeddings1.P.weight        (64, 1)
time_series_embedding.embeddings2.P.weight        (256, 64)
static_embedding.numerical_embeddings1.MSWEP_P.weight  (64, 1)
```

So "the static path is 49→256→256" was wrong — it is 48 separate 1→64→256
MLPs keyed by attribute NAME. And "`var_emb` rows for P/RelHum/SWd/Tmax/Tmin
transfer" was wrong — there is no embedding table to take rows from.

The names are also the GLOBAL dataset's (`MSWEP_P`, `GMTED_elevation`,
`catchsize`, …), **not CAMELS's** (`prcp_daymet`, `elev_mean`, …). Even where
concepts overlap, the tensors are keyed by strings that do not match.

**2. It tokenises per TIMESTEP, not per patch.** `embeddings1.P.weight` is
(64, 1) — it consumes a scalar. With `seq_len` 365 that is 365 tokens per
variable, and the learned `position_embedding` is (1000, 256) accordingly. Our
encoder uses 16-day patches, so even the positional semantics differ.

### What this leaves

| our tensor | transfers? | condition |
|---|---|---|
| **trunk** (4 layers, d 256) | **YES — 2.108M** | must set FFN to **512**, not 4·d=1024, and match `norm_first` |
| `encoder_norm` | **yes** | — |
| `positional_encoding` | **only if** we switch to per-timestep tokens | we use patches, so no |
| `static_mlp` (shared 26→256→256) | **no** | they have 48 per-attribute MLPs, different names |
| `value_proj` (shared patch→256) | **no** | they have 5 per-variable MLPs on scalars |
| `var_emb` (variable-ID table) | **no** | no such table exists in their design |
| `summary_q`, `mask_tok`, `head` | **no** | new |

**Transferable: the trunk, ~2.11M of 4.77M ≈ 44% of parameters** (50% if the
positional encoding could be used). That is the part encoding *how to mix
tokens* — arguably the genuinely reusable knowledge — while the embeddings are
dataset-specific plumbing we would have to relearn for CAMELS regardless.

### The design trade-off this exposes

Their per-variable MLPs and our variable-ID embeddings are not just different
implementations of the same idea:

| | StefaLand: MLP per variable | ours: shared projection + variable-ID |
|---|---|---|
| new variable | a new MLP, retrain | one embedding row |
| any subset / any order | fixed name set | native |
| **weight transfer** | **their 5 forcing MLPs are reusable** | **not reusable** |

So adopting their parameterisation would buy the forcing embeddings but cost
the flexibility that motivates the whole design (a variable is an ID, not a
slot). **Recommendation: transfer the trunk, keep our embeddings fresh.**
That is why `d_ffd` is now 512 in `SiteEncoder` rather than 4·d.

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
