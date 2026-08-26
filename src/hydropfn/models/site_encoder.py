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

D_MODEL = 256          # StefaLand's hidden_size; with d_ffd=512 the TRUNK
                       # loads directly (2.108M params). The embeddings do
                       # not -- see docs/stefaland_reuse.md.


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
                 n_dem: int = 0,
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
        self.n_dem = n_dem

        # 2*n_attr in: the (masked) values AND a visibility indicator per
        # attribute. Zeroing a masked attribute without the indicator is the
        # classic absent/zero conflation -- a genuine 0.0 after
        # standardisation means "at the mean", not "missing".
        self.static_mlp = nn.Sequential(
            nn.Linear(2 * n_attr, d), nn.GELU(), nn.Linear(d, d))
        # CROSS-MODULE RECONSTRUCTION: statics predicted back from the static
        # token, which has attended over the whole time series. Until this
        # existed, attributes only ever flowed INTO the model -- no loss term
        # asked it to represent geology beyond what moves the hydrograph, so
        # the static embedding was a runoff-relevant projection of geology
        # rather than geology. This is the head that makes probing them
        # meaningful, and the first reconstruction that crosses modules
        # (time series -> statics) rather than staying within one.
        self.attr_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_attr))
        # DEM token. Terrain enters as ITS OWN TOKEN rather than being
        # concatenated into the static vector, so that "no DEM here" is an
        # ABSENT TOKEN -- the same object as an absent series variable --
        # instead of a zero that means "average terrain". 3DEP is CONUS-only
        # and the global tier is 30-90 m, so missing/degraded DEM is the
        # normal case, not an edge case.
        self.dem_proj = nn.Linear(n_dem, d) if n_dem else None
        self.dem_absent = nn.Parameter(torch.zeros(d))
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

    def _causal_mask(self, N: int, V: int, dev) -> torch.Tensor:
        """(L,L) bool attention mask, True = FORBIDDEN. Time flows one way.

        Sequence layout is [static, k summaries, N*V series], series index
        n*V+v, so patch(idx) = idx // V.

        Three rules, and each one is load-bearing:
          * series token at patch p attends to series tokens at patch <= p.
          * the STATIC token attends only to itself. If it could attend
            everywhere it would become a global pool of the whole window, and
            every series token attending to it would inherit the future
            through the back door.
          * NOTHING may attend to the summary tokens, for the same reason:
            they pool over all time by construction. They are still computed
            (they read everything) but in causal mode they are dead ends, and
            PUBModel skips the pooled path that consumes them.
        """
        n_head = 1 + self.k + (1 if self.dem_proj is not None else 0)
        L = n_head + N * V
        m = torch.ones(L, L, dtype=torch.bool, device=dev)      # all forbidden
        s0 = n_head
        idx = torch.arange(N * V, device=dev)
        patch = idx // V                                        # (N*V,)
        m[s0:, s0:] = patch[None, :] > patch[:, None]           # future = True
        m[s0:, 0] = False                                       # static is a key
        m[0, 0] = False                                         # static: self only
        m[1:s0, :] = False                                      # summaries read all
        m[:, 1:s0] = True                                       # ...but are never read
        m[1:s0, 1:s0] = False                                   # (keep rows legal)
        if self.dem_proj is not None:
            # The DEM token is STATIC in time, so it is a legal KEY for every
            # series token -- terrain does not change between patch 3 and 30.
            #
            # But it must READ nothing except itself. The summary rule above
            # (`m[1:s0, :] = False`) would otherwise let it read the whole
            # series, and combining that with being a universal key opens
            # future -> DEM -> past. Measured when this was wrong: the leak
            # test moved early patches by 1.4e-03 instead of exactly 0.0.
            # Same reasoning as the static token, which reads only itself.
            d0 = s0 - 1
            m[d0, :] = True
            m[d0, d0] = False
            m[s0:, d0] = False
        return m

    def forward(self, batch: dict, return_hidden: bool = False,
                causal: bool = False) -> dict:
        a, x = batch["attrs"], batch["series"]
        vis, valid, doy = batch["vis"], batch["valid"], batch["doy"]
        B, N, V, _ = x.shape
        dev = x.device

        # ---- static token
        av = batch.get("attr_vis")
        if av is None:
            av = torch.ones_like(a)
        st = self.static_mlp(torch.cat([a * av, av], dim=-1)) + self.static_role

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
        parts = [st[:, None, :], q]
        if self.dem_proj is not None:
            dv = batch.get("dem")
            dm = batch.get("dem_vis")          # 1 = present, 0 = absent
            if dv is None:
                dt = self.dem_absent.expand(B, 1, -1)
            else:
                dt = self.dem_proj(torch.nan_to_num(dv))[:, None, :]
                if dm is not None:
                    k = dm.view(B, 1, 1)
                    dt = k * dt + (1 - k) * self.dem_absent
            parts.append(dt)
        parts.append(tok)
        seq = torch.cat(parts, dim=1)

        # padding: never let the trunk attend to slots that are not real data.
        # NOTE this is the "absent" that is never scored; a MASKED-but-valid
        # token stays in the sequence, because the model must infer it.
        n_head = 1 + self.k + (1 if self.dem_proj is not None else 0)
        pad = torch.cat([
            torch.ones(B, n_head, device=dev, dtype=torch.bool),
            valid.reshape(B, N * V).bool()], dim=1)

        attn_mask = self._causal_mask(N, V, dev) if causal else None
        h = self.trunk(self.norm_in(seq), mask=attn_mask,
                       src_key_padding_mask=~pad)
        t_static = h[:, :1]
        t_series = h[:, 1:1 + self.k]
        h_series = h[:, n_head:]
        recon = self.head(h_series).reshape(B, N, V, self.patch)
        out = {"recon": recon, "t_static": t_static, "t_series": t_series,
               "attr_recon": self.attr_head(t_static.squeeze(1))}
        if return_hidden:
            # the connector's level-3 decoder needs the per-token stream, not
            # just the summaries -- local detail stays at the site
            out["h_series"] = h_series
        return out


def masked_mse(recon, target, vis, valid):
    """Score ONLY tokens we deliberately hid and that hold real data.

    Three kinds of absent, never conflated: padding and genuine gaps have
    valid=0 and are never scored; deliberately masked tokens have vis=0,
    valid=1 and are the training signal; visible tokens are context.
    """
    w = ((1.0 - vis) * valid).unsqueeze(-1)
    return ((recon - target) ** 2 * w).sum() / w.sum().clamp(min=1.0)


def load_stefaland_trunk(net: "SiteEncoder", ckpt_path: str,
                         verbose: bool = True) -> dict:
    """Initialise this encoder's trunk from a StefaLand checkpoint.

    ONLY the trunk transfers -- 2.108M of StefaLand's 4.771M parameters. Its
    input embeddings are one MLP per NAMED variable (keyed to a global dataset:
    `P`, `MSWEP_P`, `GMTED_elevation`), whereas ours are a shared projection
    plus a variable-ID table; there is nothing to map between them. See
    docs/stefaland_reuse.md for the tensor-by-tensor account.

    Requires `d_ffd=512` (StefaLand's, not the usual 4*d).

    OPEN QUESTION this function exists to settle: their trunk learned to mix
    tokens built from per-TIMESTEP scalars by per-variable MLPs. Ours mixes
    16-day PATCH projections plus variable-ID embeddings. The shapes match; the
    representation space does not. So transfer may be neutral or harmful, and
    the honest test is two runs identical but for the init.

    Returns a report; ALWAYS printed, because a `strict=False` load that
    transfers nothing looks exactly like one that transfers everything.
    """
    import torch as _t
    ck = _t.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)

    src = {k[len("encoder.transformer_encoder."):]: v
           for k, v in sd.items()
           if k.startswith("encoder.transformer_encoder.")}
    tgt = net.trunk.state_dict()
    take, skip = {}, []
    for k, v in tgt.items():
        s = src.get(k)
        if s is not None and tuple(s.shape) == tuple(v.shape):
            take[k] = s
        else:
            skip.append((k, tuple(v.shape),
                         tuple(s.shape) if s is not None else None))
    net.trunk.load_state_dict({**tgt, **take})

    n_take = sum(v.numel() for v in take.values())
    n_tot = sum(v.numel() for v in tgt.values())
    rep = {"loaded_tensors": len(take), "trunk_tensors": len(tgt),
           "loaded_params": n_take, "trunk_params": n_tot,
           "skipped": skip[:8]}
    if verbose:
        print(f"  StefaLand trunk init: {len(take)}/{len(tgt)} tensors, "
              f"{n_take/1e6:.3f}M/{n_tot/1e6:.3f}M params "
              f"({n_take/max(n_tot,1):.0%})", flush=True)
        if skip:
            print(f"    NOT loaded ({len(skip)}), first few: {skip[:3]}",
                  flush=True)
        if n_take == 0:
            print("    WARNING: nothing transferred -- check d_ffd and depth",
                  flush=True)
    # encoder_norm -> our norm_in, if shapes agree
    if "encoder_norm.weight" in sd and \
            sd["encoder_norm.weight"].shape == net.norm_in.weight.shape:
        net.norm_in.load_state_dict({"weight": sd["encoder_norm.weight"],
                                     "bias": sd["encoder_norm.bias"]})
        if verbose:
            print("    encoder_norm -> norm_in: loaded", flush=True)
    return rep
