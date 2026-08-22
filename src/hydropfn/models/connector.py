"""Level 2 — the cross-site connector, and the PUB model built on it.

This is the load-bearing claim of the whole design: **given K gauged context
basins, can we predict streamflow at an UNGAUGED basin better than without
them?** See `docs/pub_test_plan.md` for the ladder and the pre-registered
failure modes.

Note what is NOT yet shown when this file is written. Context has been shown to
help for POINT MEASUREMENTS (unit D, +0.127 R² cross-variable, 9/9 across
regions and seeds) — but that context is the query site's OWN history, a much
easier setting. For time series, unit A has no cross-site attention at all, and
its R² 0.769 comes from forcings and attributes at a single site.

`B` indexes TASKS and `S` indexes sites within a task. They cannot collapse:
attention must stay inside a task, or a site from task 1 attends to a site from
task 2. The per-site encoder has no cross-site interaction, so it runs on the
flattened `(B*S)`; only this module needs the `(B, S)` structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossSiteConnector(nn.Module):
    """Attention over site-SUMMARY tokens.

    IN   (B, S, K, d)   site 0 is the query; sites 1.. are context
    OUT  (B, S, K, d)   summaries now carry cross-site information

    Permutation-invariant over sites: there is no positional index across the
    site axis, only a role embedding marking query vs context. Two sites with
    the same summary are interchangeable, which is what makes a *retrieved set*
    the right abstraction rather than an ordered list.
    """

    def __init__(self, d: int = 256, depth: int = 4, heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.role = nn.Embedding(2, d)          # 0 = context, 1 = query
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok: torch.Tensor, site_valid: torch.Tensor):
        """tok (B,S,K,d); site_valid (B,S) 1.0 where the site slot is real."""
        B, S, K, d = tok.shape
        role = torch.zeros(B, S, dtype=torch.long, device=tok.device)
        role[:, 0] = 1
        x = tok + self.role(role)[:, :, None, :]
        x = x.reshape(B, S * K, d)
        pad = site_valid[:, :, None].expand(B, S, K).reshape(B, S * K).bool()
        h = self.enc(self.norm(x), src_key_padding_mask=~pad)
        return h.reshape(B, S, K, d)


class PUBModel(nn.Module):
    """SiteEncoder (level 1) -> connector (level 2) -> query decoder (level 3).

    The query basin has EVERY streamflow patch masked. Context basins have
    theirs visible. The query's own token stream then cross-attends to the
    context-aware summaries and reconstructs its masked streamflow.

    This is the minimal honest instantiation of the three-level design: the
    connector never sees raw timesteps (only K summary tokens per site), and
    the query's local detail never leaves the query site — the summary is a
    CONDITIONING VECTOR, not a reconstruction.
    """

    def __init__(self, encoder, d: int = 256, depth: int = 4, heads: int = 8):
        super().__init__()
        self.encoder = encoder
        self.connector = CrossSiteConnector(d, depth, heads)
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_c = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, encoder.patch))

    def forward(self, batch: dict) -> torch.Tensor:
        """batch fields are (B, S, ...) with site 0 the query.

        Returns the query's reconstruction (B, N, V, patch).
        """
        B, S = batch["vis"].shape[:2]
        flat = {k: v.reshape(B * S, *v.shape[2:]) for k, v in batch.items()
                if k in ("attrs", "series", "vis", "valid", "doy")}
        out = self.encoder(flat, return_hidden=True)

        K = 1 + self.encoder.k
        summ = torch.cat([out["t_static"], out["t_series"]], dim=1)
        summ = summ.reshape(B, S, K, -1)
        summ = self.connector(summ, batch["site_valid"])

        # query stream attends to ALL context-aware summaries
        d = summ.shape[-1]
        hq = out["h_series"].reshape(B, S, -1, d)[:, 0]        # (B, N*V, d)
        ctx = summ.reshape(B, S * K, d)
        a, _ = self.cross(self.norm_q(hq), self.norm_c(ctx), self.norm_c(ctx),
                          need_weights=False)
        z = hq + a
        N, V = batch["series"].shape[2], batch["series"].shape[3]
        return self.head(z).reshape(B, N, V, self.encoder.patch)
