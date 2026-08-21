"""TEMPLATE — the assembled model: encoders -> connector -> decoders.

NOT IMPLEMENTED. This is the integration point. Every component it needs is
either built (`models/diffusion.py`, `models/measurement_pfn.py`) or templated
(`encoders.py`, `connector.py`, `decoders.py`).

ONE OBJECTIVE drives everything: **masked reconstruction across modalities.**
Mask any subset of static attributes, DEM patches, observation series, forcing
series, or point measurements; reconstruct from whatever remains — including
from other SITES.

THE MASK IS THE QUERY. There is no mode switch at inference; you choose which
positions to hide, and that choice *is* the question:

    your data situation             what you mask            = training mask
    ---------------------------------------------------------------------
    forcings only, no observations  all observation slots     whole-site
    some observations exist         only what you want        whole-site
                                    predicted; the rest        (partial)
                                    stay VISIBLE
    record has gaps                 the gap positions         random span
    want the future                 everything after t        causal tail
    one variable never measured     that whole series         whole-variable

The key consequence: **being UNMASKED *is* being context.** There is no
separate context slot. A site with three years of streamflow and a gap in year
two has three years unmasked (context) and the gap masked (query), in the same
sequence, in one forward pass. The four training masks exist so that every
visible/hidden combination the real world produces has been seen in training.

Zero context is a valid input, not a failure case: with no context tokens the
model falls back to attributes + terrain, which is the "only x" mode that
global and ungauged deployment actually runs in.
"""

from __future__ import annotations

import torch.nn as nn


class StefaNP(nn.Module):
    """encoders (level 1) -> connector (level 2) -> decoders (level 3)."""

    def __init__(self, cfg):
        super().__init__()
        raise NotImplementedError("see docs/architecture.md")

    def forward(self, batch):
        raise NotImplementedError
