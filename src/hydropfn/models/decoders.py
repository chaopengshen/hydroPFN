"""TEMPLATE — level 3: per-modality decoders. Loss is computed in DATA space.

NOT IMPLEMENTED except the two that already exist:

    scalar head   `models/measurement_pfn.py`   bar distribution over a value
    DEM decoder   `models/diffusion.py`         conditional diffusion sampler

Every unit needs an encoder AND a decoder for masked reconstruction to work.
The connector only ever sees 4-10 summary tokens per site, so a full series
cannot be decoded from it. The decoder takes the CONTEXT-AWARE SUMMARY plus the
site's own unmasked data:

    temporal   [ summary (K,256) + unmasked patches ] -> (B, N_masked, 16) days
    scalar     [TASK] token -> 64 bins -> p(y | query, context)      [BUILT]
    DEM        summary + unmasked pixels -> (B, 1, 128, 128)         [BUILT]

Loss in DATA space, not token space: token-space losses are easy to game by
collapsing the representation.
"""

from __future__ import annotations

import torch.nn as nn


class TemporalDecoder(nn.Module):
    """Reconstruct masked time-series patches from summary + visible patches."""

    def __init__(self, d: int = 256, patch: int = 16):
        super().__init__()
        raise NotImplementedError
