"""Unit A — the merged static + temporal site encoder.

This REPLACES the earlier plan of separate units A (static) and C (temporal).
The reasoning, because the reversal matters:

**StefaLand already merges them.** Its inputs are `batch_x [B,T,F]` (dynamic)
and `batch_c [B,C]` (static), fused in one masked autoencoder. My earlier
argument for keeping them separable was that 5,057 gages have attributes but no
series — but StefaLand is a *masked* autoencoder, so feeding it attributes with
every series channel masked is exactly what it was trained to handle. That
argument was wrong, and with it gone there is no reason to split.

**But StefaLand cannot do the thing we need.** Its config keeps
`time_series_variables` (inputs) and `target_variables` (e.g. `['QObs']`,
streamflow) as SEPARATE lists — observations are what it predicts, never what
it reads. So it can never use "this basin's observed streamflow" as context for
predicting something else at that basin. That is the whole point of an
in-context model.

So: **merge, but extend.** One unit, following StefaLand's fused design, with

  1. observation channels as MASKABLE INPUTS alongside the forcings — the mask
     sampler does not distinguish them, so the model learns both directions
     (forcing->obs, and obs->forcing, which is precipitation correction);
  2. a variable-ID embedding on every channel, so any subset of variables in
     any order works, a missing channel is an absent token rather than a zero,
     and a NEW variable costs one embedding row instead of a retrain;
  3. initialisation from StefaLand wherever shapes match (see
     `docs/stefaland_reuse.md` for exactly which tensors transfer).

Token layout, all d=256:

    [ STATIC ]  [ (patch 0, var 0) ... (patch N-1, var V-1) ]
      1 token             N * V tokens
                          each = value_emb + var_emb[v] + pos_emb[n] + doy_emb

Masked tokens keep their var/pos/doy and have the VALUE replaced by a learned
[MASK] vector, so the model always knows what it is being asked to infer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

D_MODEL = 256          # StefaLand's hidden_size, so its weights load unchanged


class SiteEncoder(nn.Module):
    """Static attributes + multivariate series -> tokens, with a patch decoder.

    forward(batch) expects
        attrs   (B, n_attr)        float, already standardised; NaN -> 0
        series  (B, N, V, patch)   float, patchified and standardised
        vis     (B, N, V)          1.0 = visible to the model, 0.0 = hidden
        valid   (B, N, V)          1.0 = real data (vs padding / outage)
        doy     (B, N)             day-of-year of each patch start, in [0,1)

    Returns dict with
        recon    (B, N, V, patch)  reconstruction for every patch
        t_static (B, 1, d)         the static summary token
        t_series (B, K, d)         pooled temporal summaries
    """

    def __init__(self, n_attr: int, n_vars: int, patch: int = 16,
                 d: int = D_MODEL, depth: int = 4, heads: int = 4,
                 k_summary: int = 3, dropout: float = 0.1,
                 d_ffd: int = 512):
        """`d_ffd` defaults to 512, not the usual 4*d = 1024, so the trunk has
        the SAME shape as StefaLand's `encoder.transformer_encoder` and its
        2.108M parameters load directly. Verified against the checkpoint --
        see docs/stefaland_reuse.md. The input embeddings do NOT transfer
        (StefaLand uses one MLP per named variable; we use a shared projection
        plus a variable-ID embedding), so the trunk is the reusable half.
        """
        super().__init__()
        self.d, self.patch, self.n_vars, self.k = d, patch, n_vars, k_summary

        self.static_mlp = nn.Sequential(
            nn.Linear(n_attr, d), nn.GELU(), nn.Linear(d, d))
        self.value_proj = nn.Linear(patch, d)
        self.var_emb = nn.Embedding(n_vars, d)
        self.mask_tok = nn.Parameter(torch.zeros(d))
        self.static_role = nn.Parameter(torch.zeros(d))
        self.doy_proj = nn.Linear(2, d)          # sin/cos of day-of-year
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, d))
        nn.init.normal_(self.pos_emb, std=0.02)

        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=d_ffd, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.trunk = nn.TransformerEncoder(layer, depth)

        # learned query tokens that pool the temporal stream into k summaries
        self.summary_q = nn.Parameter(torch.zeros(1, k_summary, d))
        nn.init.normal_(self.summary_q, std=0.02)

        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, patch))

    def forward(self, batch: dict) -> dict:
        a, x = batch["attrs"], batch["series"]
        vis, valid, doy = batch["vis"], batch["valid"], batch["doy"]
        B, N, V, _ = x.shape
        dev = x.device

        # ---- static token
        st = self.static_mlp(a) + self.static_role                # (B, d)

        # ---- series tokens: value (or [MASK]) + variable id + position + doy
        val = self.value_proj(x)                                  # (B,N,V,d)
        show = (vis * valid).unsqueeze(-1)
        val = show * val + (1.0 - show) * self.mask_tok

        vid = torch.arange(V, device=dev)
        tok = val + self.var_emb(vid)[None, None, :, :]
        tok = tok + self.pos_emb[:, :N][:, :, None, :]
        ang = 2 * math.pi * doy
        tok = tok + self.doy_proj(
            torch.stack([ang.sin(), ang.cos()], -1))[:, :, None, :]
        tok = tok.reshape(B, N * V, self.d)

        q = self.summary_q.expand(B, -1, -1)
        seq = torch.cat([st[:, None, :], q, tok], dim=1)

        # padding: never let the trunk attend to slots that are not real data.
        # NOTE this is the "absent" that is never scored; a MASKED-but-valid
        # token stays in the sequence, because the model must infer it.
        pad = torch.cat([
            torch.ones(B, 1 + self.k, device=dev, dtype=torch.bool),
            valid.reshape(B, N * V).bool()], dim=1)

        h = self.trunk(self.norm_in(seq), src_key_padding_mask=~pad)
        t_static = h[:, :1]
        t_series = h[:, 1:1 + self.k]
        recon = self.head(h[:, 1 + self.k:]).reshape(B, N, V, self.patch)
        return {"recon": recon, "t_static": t_static, "t_series": t_series}


def masked_mse(recon, target, vis, valid):
    """Score ONLY tokens we deliberately hid and that hold real data.

    Three kinds of absent, never conflated: padding and genuine gaps have
    valid=0 and are never scored; deliberately masked tokens have vis=0,
    valid=1 and are the training signal; visible tokens are context.
    """
    w = ((1.0 - vis) * valid).unsqueeze(-1)
    return ((recon - target) ** 2 * w).sum() / w.sum().clamp(min=1.0)
