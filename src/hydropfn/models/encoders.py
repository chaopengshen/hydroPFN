"""TEMPLATE — per-site encoders (level 1), sharing one trunk.

PARTIALLY BUILT:

    B  terrain      `models/diffusion.py`         masked DEM reconstruction
    D  measurement  `models/measurement_pfn.py`   built, acceptance gates pass
    A  static       template here (initialise from StefaLand)
    C  forcing      template here

DESIGN DECISION, recorded because it is not obvious. Unit A (static attributes)
and unit C (time series) are **neither separate modules nor one fused module**.
They share a TRUNK, with per-modality input embedders and per-role output
tokens:

    attrs   (B,49)         --MLP------.
    DEM     (B,1,128,128)  --conv-----+--> [attr_tok, dem_tok, ts_1..ts_N]
    series  (B,T,V)        --patchify-'                 |
                                              SHARED TRUNK transformer
                                                        |
                              pool by role -->  t_attr, t_dem, t_ts

*Why not separate*: the rainfall-runoff response depends on catchment
properties at EVERY timescale -- soil depth sets recession, slope sets
flashiness, snow fraction sets seasonality. If A enters only at C's last layer,
every temporal representation below it was computed catchment-blind.

*Why not fully fused*: a site with attributes but no series must still emit
`t_attr` (the 5,057 channel-geometry gages are exactly that case); U3 gating
needs separable outputs to tell whether attributes contributed at all; and the
two objectives need explicit loss weighting.

TRAP -- we have already been bitten by its twin. Many statics are deterministic
aggregates of the forcings: `meanP`, `meanTa`, `seasonality_P`, `MSWX_*`.
Masking `meanP` and "predicting" it from the precipitation series is
ARITHMETIC, not learning. This is the same failure as the `W*d*v = Q` identity,
which inflated the measurement unit's cross-variable gain from +0.108 to +0.141
until whole measurement occasions were excluded from context. So the
masked-attribute loss must partition the statics:

    derivable-from-forcings   excluded, or reported separately and NEVER as
                              evidence of learning
    genuinely independent     soils, geology, permeability, land cover,
                              catchsize, soil_depth -- the real objective
"""

from __future__ import annotations

import torch.nn as nn

D_MODEL = 256      # StefaLand's hidden_size, so its weights load unchanged


class StaticEmbedder(nn.Module):
    """(B, n_attr) -> (B, 1, D_MODEL). Initialise from the StefaLand checkpoint."""

    def __init__(self, n_attr: int, d: int = D_MODEL):
        super().__init__()
        raise NotImplementedError


class TemporalEmbedder(nn.Module):
    """(B, T, V) -> (B, N, D_MODEL).

    Channel-independent patch tokens plus a variable-ID embedding per channel
    and day-of-year features. Variable identity is a LEARNED VECTOR the token
    carries, never a slot index -- that is what lets any subset of variables,
    in any order, with any missing, work without imputation.
    """

    def __init__(self, n_vars: int, patch: int = 16, d: int = D_MODEL):
        super().__init__()
        raise NotImplementedError


class SiteTrunk(nn.Module):
    """Shared transformer over the concatenated modality tokens.

    Load StefaLand here. Its inputs are `batch_x [B,T,F]` and `batch_c [B,C]`
    where ITS `B` indexes sites -- so our `(B, S, ...)` flattens to
    `(B*S, ...)` before this call and reshapes after. StefaLand has no `S`
    dimension because it never looks at another site; that is precisely what
    the connector adds.
    """

    def __init__(self, d: int = D_MODEL, depth: int = 4, heads: int = 4):
        super().__init__()
        raise NotImplementedError
