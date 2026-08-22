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


class GeoEncoding(nn.Module):
    """Fourier encoding of each site's DISPLACEMENT from the query site.

    Factored out because it must be attachable to whichever path actually
    mixes context. It was originally wired only into CrossSiteConnector, and
    measured nothing (+0.003) -- because the connector turns out to carry
    almost no signal: dropping it entirely costs 0.001 median NSE at K=4. The
    time-aligned path does the work, and it had no geometry. Testing geo on
    the dead path was a mis-placed experiment, not a refuted hypothesis.

    Longitude is scaled by a FIXED reference latitude, never the query's own:
    cos(query_lat) is the better metric but makes the encoding a function of
    ABSOLUTE latitude, a region-identifying channel that would undermine
    leave-region-out. Pinned by test_connector_geo_translation_invariant.
    """

    LON_SCALE = 0.766                    # cos(40 deg), CONUS mid-latitude

    def __init__(self, d: int, n_freq: int = 8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float() * 0.5)
        self.proj = nn.Linear(4 * n_freq + 1, d)

    def forward(self, latlon: torch.Tensor) -> torch.Tensor:
        """latlon (B,S,2) degrees, site 0 = query -> (B,S,d)."""
        rel = latlon - latlon[:, :1]
        rel = torch.stack([rel[..., 0], rel[..., 1] * self.LON_SCALE], dim=-1)
        ang = rel[..., None] * self.freqs
        feat = torch.cat([ang.sin().flatten(-2), ang.cos().flatten(-2),
                          rel.norm(dim=-1, keepdim=True)], dim=-1)
        return self.proj(feat)


class CrossSiteConnector(nn.Module):
    """Attention over site-SUMMARY tokens.

    IN   (B, S, K, d)   site 0 is the query; sites 1.. are context
    OUT  (B, S, K, d)   summaries now carry cross-site information

    Permutation-invariant over sites: there is no positional index across the
    site axis, only a role embedding marking query vs context. Two sites with
    the same summary are interchangeable, which is what makes a *retrieved set*
    the right abstraction rather than an ordered list.

    `geo` adds a RELATIVE-POSITION encoding: each context site gets a Fourier
    embedding of its displacement from the QUERY. Until 2026-08-21 this module
    had no geometry at all -- context was chosen by distance and then treated
    as exchangeable, so the model could not tell a gauge 20 km away from one
    300 km away. That is the obvious suspect for the decay at large K, since
    without distance the only way to discount a far gauge is to discount all
    of them. Displacement is relative, not absolute lat/lon, so the encoding
    stays translation-invariant and cannot be used to memorise regions --
    which matters under leave-region-out.
    """

    def __init__(self, d: int = 256, depth: int = 4, heads: int = 8,
                 dropout: float = 0.1, geo: bool = False, n_freq: int = 8):
        super().__init__()
        self.role = nn.Embedding(2, d)          # 0 = context, 1 = query
        self.geo = GeoEncoding(d, n_freq) if geo else None
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)

    def forward(self, tok: torch.Tensor, site_valid: torch.Tensor,
                latlon: torch.Tensor | None = None):
        """tok (B,S,K,d); site_valid (B,S); latlon (B,S,2) degrees, optional."""
        B, S, K, d = tok.shape
        role = torch.zeros(B, S, dtype=torch.long, device=tok.device)
        role[:, 0] = 1
        x = tok + self.role(role)[:, :, None, :]
        if self.geo is not None and latlon is not None:
            x = x + self.geo(latlon)[:, :, None, :]
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

    def __init__(self, encoder, d: int = 256, depth: int = 4, heads: int = 8,
                 time_aligned: bool = False, geo: bool = False,
                 causal: bool = False, no_pooled: bool = False):
        """`time_aligned` adds a second cross-attention in which each query
        PATCH attends to the context patches at the SAME time position.

        Why it exists. The connector sees only time-POOLED summaries -- the
        scaling guardrail. That is right for transferring basin character and
        wrong for transferring today's weather: a neighbour's flow at patch n
        predicts the query's flow at patch n, and no summary pooled over the
        whole window can carry that. Measured: with nearby context, averaging
        the neighbours' hydrographs scores 0.829 while the summary-only model
        scores 0.556. The information is present and the architecture cannot
        reach it.
        """
        super().__init__()
        self.encoder = encoder
        self.time_aligned = time_aligned
        # In causal mode the connector and the pooled cross-attention are
        # SKIPPED, not masked. Site summaries pool over the whole window by
        # construction, so any path that consumes them carries the future.
        # What survives is the time-aligned path: query patch n attends to
        # context patch n, built from data <= n by the causal site encoder.
        self.causal = causal
        # `no_pooled` skips the same path WITHOUT masking time. It exists so
        # the causal ablation is not confounded: causal mode changes TWO
        # things at once (time masking AND dropping the pooled path), and the
        # pooled path is already known to cost ~0.18 at K=0. Comparing causal
        # against a both-paths control would credit causality with that.
        self.no_pooled = no_pooled or causal
        self.connector = CrossSiteConnector(d, depth, heads, geo=geo)
        # geo on the path that actually mixes context. Context tokens are
        # tagged with their displacement from the query BEFORE tcross, so the
        # attention can weight a neighbour by how far away it is.
        self.tgeo = GeoEncoding(d) if geo else None
        self.cross = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(d)
        self.norm_c = nn.LayerNorm(d)
        if time_aligned:
            self.tcross = nn.MultiheadAttention(d, heads, batch_first=True)
            self.norm_tq = nn.LayerNorm(d)
            self.norm_tc = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, encoder.patch))

    def forward(self, batch: dict) -> torch.Tensor:
        """batch fields are (B, S, ...) with site 0 the query.

        Returns the query's reconstruction (B, N, V, patch).
        """
        B, S = batch["vis"].shape[:2]
        flat = {k: v.reshape(B * S, *v.shape[2:]) for k, v in batch.items()
                if k in ("attrs", "series", "vis", "valid", "doy")}
        out = self.encoder(flat, return_hidden=True, causal=self.causal)

        K = 1 + self.encoder.k
        d = out["h_series"].shape[-1]
        hq = out["h_series"].reshape(B, S, -1, d)[:, 0]        # (B, N*V, d)
        if self.no_pooled:
            z = hq
        else:
            summ = torch.cat([out["t_static"], out["t_series"]], dim=1)
            summ = summ.reshape(B, S, K, -1)
            summ = self.connector(summ, batch["site_valid"],
                                  batch.get("latlon"))
            # query stream attends to ALL context-aware summaries
            ctx = summ.reshape(B, S * K, d)
            a, _ = self.cross(self.norm_q(hq), self.norm_c(ctx),
                              self.norm_c(ctx), need_weights=False)
            z = hq + a
        N, V = batch["series"].shape[2], batch["series"].shape[3]

        if self.time_aligned and S > 1:
            # Each query patch attends to the CONTEXT patches at the same time
            # position -- (B*N) independent attentions with (S-1)*V keys each.
            h = out["h_series"].reshape(B, S, N, V, d)
            qs = h[:, 0].reshape(B * N, V, d)
            hc = h[:, 1:]
            if self.tgeo is not None and batch.get("latlon") is not None:
                hc = hc + self.tgeo(batch["latlon"])[:, 1:, None, None, :]
            ks = hc.permute(0, 2, 1, 3, 4).reshape(B * N, (S - 1) * V, d)
            ta, _ = self.tcross(self.norm_tq(qs), self.norm_tc(ks),
                                self.norm_tc(ks), need_weights=False)
            z = z + ta.reshape(B, N, V, d).reshape(B, N * V, d)

        return self.head(z).reshape(B, N, V, self.encoder.patch)
