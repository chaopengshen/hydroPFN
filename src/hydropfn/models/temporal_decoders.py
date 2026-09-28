"""Daily readouts for raw inputs and contextual PFN patch features.

LSTM, pointwise and gated residual TCN share one feature interface. Causality
is with respect to supplied daily inputs: a noncausal encoder remains
noncausal even when followed by these decoders.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    """Left padding only; no temporal normalization or future pooling."""

    def __init__(self, n_in, n_out, kernel, dilation=1):
        if kernel < 1 or dilation < 1:
            raise ValueError("positive kernel and dilation required")
        super().__init__(n_in, n_out, kernel, dilation=dilation, padding=0)
        self.history = (kernel - 1) * dilation

    def forward(self, x):
        return super().forward(F.pad(x, (self.history, 0)))


class GatedTemporalBlock(nn.Module):
    """Two causal GLU convolutions, with static FiLM and a residual path."""

    def __init__(self, width, n_attr, kernel, dilation, dropout):
        super().__init__()
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.convs = nn.ModuleList([
            CausalConv1d(width, 2 * width, kernel, dilation) for _ in range(2)])
        self.drop = nn.Dropout(dropout)
        self.film = nn.Linear(n_attr, 2 * width) if n_attr else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, x, attrs):
        h = x
        for norm, conv in zip(self.norms, self.convs):
            # LayerNorm acts over channels at EACH time, never over time.
            h = norm(h.transpose(1, 2)).transpose(1, 2)
            h = self.drop(F.glu(conv(h), dim=1))
        if self.film is not None:
            scale, shift = self.film(attrs).chunk(2, dim=-1)
            h = h * (1 + scale[..., None]) + shift[..., None]
        return x + h


class DailySequenceDecoder(nn.Module):
    """(B,T,F), static attributes, optional (B,T,Z) -> (B,T) discharge."""

    def __init__(self, n_forcing, n_attr, kind="lstm", latent_dim=0,
                 latent_width=64, hidden=256, tcn_width=64, tcn_blocks=6,
                 kernel=3, dropout=0.1, use_raw=True, phase_dim=8, patch=16):
        super().__init__()
        if kind not in ("lstm", "tcn", "pointwise"):
            raise ValueError(f"unknown decoder {kind}")
        if not use_raw and not latent_dim:
            raise ValueError("decoder needs raw or latent inputs")
        if tcn_blocks < 1 or patch < 1:
            raise ValueError("positive block count and patch size required")
        self.kind, self.use_raw, self.latent_dim = kind, use_raw, latent_dim
        self.patch, self.n_attr = patch, n_attr
        self.latent_proj = (nn.Sequential(nn.LayerNorm(latent_dim),
                            nn.Linear(latent_dim, latent_width), nn.GELU())
                            if latent_dim else None)
        self.phase = nn.Embedding(patch, phase_dim) if latent_dim else None
        n_input = (n_forcing if use_raw else 0) + n_attr
        n_input += latent_width + phase_dim if latent_dim else 0
        if kind == "lstm":
            self.core = nn.LSTM(n_input, hidden, batch_first=True)
            self.readout = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
            self.receptive_field = None
        elif kind == "tcn":
            self.input_proj = nn.Conv1d(n_input, tcn_width, 1)
            self.blocks = nn.ModuleList([
                GatedTemporalBlock(tcn_width, n_attr, kernel, 2**j, dropout)
                for j in range(tcn_blocks)])
            self.readout = nn.Sequential(nn.LayerNorm(tcn_width), nn.Linear(tcn_width, 1))
            self.receptive_field = 1 + 2 * (kernel - 1) * (2**tcn_blocks - 1)
        else:
            self.core = nn.Sequential(nn.Linear(n_input, hidden), nn.GELU(),
                                     nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU())
            self.readout = nn.Linear(hidden, 1)
            self.receptive_field = 1

    def forward(self, forcing, attrs, latent=None, phase=None):
        B, T, _ = forcing.shape
        if attrs.shape != (B, self.n_attr):
            raise ValueError("attribute shape mismatch")
        parts = [attrs[:, None].expand(-1, T, -1)]
        if self.use_raw:
            parts.insert(0, forcing)
        if self.latent_dim:
            if latent is None or latent.shape != (B, T, self.latent_dim):
                raise ValueError("daily latent shape mismatch")
            if phase is None:
                phase = (torch.arange(T, device=forcing.device) % self.patch)[None].expand(B, -1)
            if phase.shape != (B, T):
                raise ValueError("phase shape mismatch")
            parts.extend([self.latent_proj(latent), self.phase(phase)])
        elif latent is not None:
            raise ValueError("latent supplied to a raw-only decoder")
        x = torch.cat(parts, dim=-1)
        if self.kind == "lstm":
            h, _ = self.core(x)
        elif self.kind == "tcn":
            h = self.input_proj(x.transpose(1, 2))
            for block in self.blocks:
                h = block(h, attrs)
            h = h.transpose(1, 2)
        else:
            h = self.core(x)
        return self.readout(h).squeeze(-1)


class PatchReconstructionDecoder(nn.Module):
    """Original PUBModel patch head, refitted on its frozen query features.

    Raw forcing/attributes are already represented by the frozen encoder;
    this head has no additional daily raw-input branch or temporal decoder.
    The shared driver supplies features repeated at daily resolution.
    """

    def __init__(self, latent_dim, patch=16):
        super().__init__()
        if latent_dim < 1 or patch < 1:
            raise ValueError("positive latent dimension and patch required")
        self.latent_dim, self.patch = latent_dim, patch
        self.receptive_field = None
        self.head = nn.Sequential(nn.LayerNorm(latent_dim),
                                  nn.Linear(latent_dim, latent_dim), nn.GELU(),
                                  nn.Linear(latent_dim, patch))

    def forward(self, forcing, attrs, latent=None):
        B, T, _ = forcing.shape
        if T % self.patch or latent is None or latent.shape != (B, T, self.latent_dim):
            raise ValueError("whole patches of daily lifted latent features required")
        return self.head(latent[:, ::self.patch]).flatten(1)


def lift_patch_features(features, patch):
    """Repeat (B,N,D) at daily resolution; phase embedding distinguishes days."""
    if features.ndim != 3 or patch < 1:
        raise ValueError("expected (batch, patch_time, features) and positive patch")
    return features.repeat_interleave(patch, dim=1)
