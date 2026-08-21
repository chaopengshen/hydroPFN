"""Mini-HydroPFN: measurement unit + cross-site core.

Phase 1 of docs/BUILD_PLAN_hydropfn.md -- the smallest thing that is genuinely
a prior-fitted network for hydrology, built so its acceptance tests can fail.

Shape of the idea: a CONTEXT POINT IS A SITE, not a row.  Each site becomes a
small bundle of tokens (one attribute token + one token per measurement visit);
a transformer then attends across sites.  A [TASK] token carries "which
variable, at which discharge" with its value masked, and the head emits a full
distribution over that value.

One objective -- masked-measurement modelling -- yields every mode we want:

  own-site visits visible          -> at-a-station rating (T1)
  only OTHER variables visible     -> cross-variable transfer (T2, measured at
                                      +0.041 R2 on velocity with a hand-built
                                      feature; the model must find it itself)
  no context tokens at all         -> attributes-only prediction (the "just x"
                                      mode, which must not be worse than an RF)

Deliberately absent in v1: DEM tokens, time series, network topology.  Each
arrives through the U0-U4 template rather than by growing this file.

Distributional head is a bar (discretised) distribution, as in TabPFN: bin
edges from the training marginal per variable, cross-entropy on the bin, and
the point prediction is the bin-probability-weighted mean.  Regression by
classification gives calibrated multi-modality for free -- and at-a-station
uncertainty is a deliverable, not a nicety.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VARS = ("log_W", "log_d", "log_v")
N_VARS = len(VARS)


# --------------------------------------------------------------- bar head

class BarDistribution(nn.Module):
    """Discretised predictive distribution with fixed per-variable borders."""

    def __init__(self, borders: torch.Tensor):
        # borders: (N_VARS, B+1)
        super().__init__()
        self.register_buffer("borders", borders)
        self.n_bins = borders.shape[1] - 1
        self.register_buffer("centres", (borders[:, :-1] + borders[:, 1:]) / 2)

    def bin_of(self, y: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
        b = self.borders[var]                              # (N, B+1)
        idx = (y[:, None] > b[:, 1:-1]).sum(dim=1)
        return idx.clamp(0, self.n_bins - 1)

    def loss(self, logits, y, var):
        return F.cross_entropy(logits, self.bin_of(y, var))

    def mean(self, logits, var):
        p = logits.softmax(-1)
        return (p * self.centres[var]).sum(-1)


def make_borders(y_by_var: list[np.ndarray], n_bins: int = 64) -> torch.Tensor:
    """Quantile borders per variable -- equal-mass bins, so resolution follows
    where the data actually are instead of being wasted on empty tails."""
    out = []
    for y in y_by_var:
        y = y[np.isfinite(y)]
        q = np.quantile(y, np.linspace(0, 1, n_bins + 1))
        q[0] -= 1e-3
        q[-1] += 1e-3
        q = np.maximum.accumulate(q)                       # strictly sorted
        eps = 1e-6 * np.arange(len(q))
        out.append(q + eps)
    return torch.tensor(np.stack(out), dtype=torch.float32)


# ------------------------------------------------------------ the network

class HydroPFN(nn.Module):
    """Two-level attention: tokens within a site, then attention across sites.

    v1 flattens both levels into one sequence and lets a shared transformer do
    the work -- with only ~300 tokens per example the separation buys nothing
    but code, and a single stack lets a query visit attend directly to a
    context site's visit (which is exactly the T2 pathway).
    """

    def __init__(self, n_attr: int, borders: torch.Tensor, d: int = 128,
                 depth: int = 6, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d = d
        self.bar = BarDistribution(borders)
        self.attr = nn.Sequential(nn.Linear(n_attr, d), nn.GELU(),
                                  nn.Linear(d, d))
        self.var_emb = nn.Embedding(N_VARS, d)
        self.val = nn.Linear(1, d)
        self.cov = nn.Linear(1, d)                         # log_Q
        # role: 0 = attribute token, 1 = observed visit, 2 = the query
        self.role = nn.Embedding(3, d)
        # is this token from the query site or a context site?
        self.own = nn.Embedding(2, d)
        self.mask_val = nn.Parameter(torch.zeros(d))
        self.norm_in = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d, heads, dim_feedforward=4 * d, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                  nn.Linear(d, self.bar.n_bins))

    def forward(self, batch: dict) -> torch.Tensor:
        """batch tensors, all (B, ...):
             attrs   (B, S, A)   per-site attributes, S = 1 + n_context
             a_valid (B, S)
             m_var   (B, S, M)   visit variable ids
             m_val   (B, S, M)
             m_cov   (B, S, M)   log_Q
             m_valid (B, S, M)
             q_var   (B,)  q_cov (B,)      the [TASK] token
        """
        B, S, M = batch["m_var"].shape
        dev = batch["m_var"].device

        own_site = torch.zeros(B, S, dtype=torch.long, device=dev)
        own_site[:, 0] = 1                                 # site 0 = query

        a = self.attr(batch["attrs"]) + self.role(
            torch.zeros(B, S, dtype=torch.long, device=dev)) \
            + self.own(own_site)                           # (B, S, d)

        m = (self.var_emb(batch["m_var"])
             + self.val(batch["m_val"].unsqueeze(-1))
             + self.cov(batch["m_cov"].unsqueeze(-1))
             + self.role(torch.ones(B, S, M, dtype=torch.long, device=dev))
             + self.own(own_site)[:, :, None, :])          # (B, S, M, d)

        q = (self.var_emb(batch["q_var"])
             + self.cov(batch["q_cov"].unsqueeze(-1))
             + self.mask_val                               # value withheld
             + self.role(torch.full((B,), 2, dtype=torch.long, device=dev))
             + self.own(torch.ones(B, dtype=torch.long, device=dev)))

        tok = torch.cat([q[:, None, :], a, m.reshape(B, S * M, self.d)], 1)
        pad = torch.cat([torch.ones(B, 1, device=dev, dtype=torch.bool),
                         batch["a_valid"].bool(),
                         batch["m_valid"].reshape(B, S * M).bool()], 1)
        h = self.enc(self.norm_in(tok), src_key_padding_mask=~pad)
        return self.head(h[:, 0])                          # logits at [TASK]
