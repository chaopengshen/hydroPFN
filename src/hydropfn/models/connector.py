"""TEMPLATE — level 2: the cross-site transformer.

NOT IMPLEMENTED as a general connector. A working single-modality version
exists in `models/measurement_pfn.py` (`HydroPFN`) — that is what produced the
measured cross-variable and at-a-station results. This is its generalisation to
the full multimodal token bundle.

    IN    (B, 1 + S*K, 256)   sites flattened, [TASK] prepended, geo-encoding
                              added, padding mask (B, 1 + S*K)
    OUT   (B, 1 + S*K, 256)   read position 0 for a scalar query; read the
                              per-site positions to condition level-3 decoders

`B` indexes TASKS, `S` indexes sites within a task. **They cannot collapse**:
attention must stay inside a task, or a site from task 1 attends to a site from
task 2. The per-site encoders have no cross-site interaction, so they run on
the flattened `(B*S)`.

Permutation-invariant over sites — no positional index. Query attends to
context; context never attends to query.

MEASURED LIMITATION to fix here. With a FIXED context size during training, the
model calibrates off one neighbour and never learns to aggregate. The
context-scaling curve (`figs/fig_context_scaling.png`) shows own-site visits
producing a graded rise (velocity R² 0.575 -> 0.906 over 0 -> 8 visits) while
neighbour sites produce a STEP (-0.145 -> 0.582 on the first neighbour, then
flat out to 16). **Randomise the context size during training.**

`K` is the per-site information bottleneck: `K * 256` numbers is everything a
site can say to the connector — for a site with 40 years of daily data across
10 variables (~146,000 numbers), `K = 4` is ~140x compression. That is
survivable only because the summary need not carry detail: the level-3 decoder
has the site's own unmasked data locally, so the summary carries what OTHER
sites contribute, which is genuinely low-dimensional (basin similarity,
regional wetness, calibration offset). Pick `K` empirically — sweep it and find
where downstream performance saturates.
"""

from __future__ import annotations

import torch.nn as nn


class CrossSiteConnector(nn.Module):
    def __init__(self, d: int = 256, depth: int = 8, heads: int = 8):
        super().__init__()
        raise NotImplementedError("see module docstring and docs/architecture.md")
