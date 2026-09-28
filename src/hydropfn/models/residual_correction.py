"""An additive correction that starts exactly at its frozen baseline."""
import torch
from torch import nn


class ResidualCorrection(nn.Module):
    def __init__(self, n_forcing, n_attr, latent_dim=0, width=32, dropout=0.5):
        super().__init__()
        self.latent_dim = latent_dim
        self.norm = nn.LayerNorm(latent_dim) if latent_dim else None
        self.body = nn.Sequential(nn.Linear(n_forcing+n_attr+1+latent_dim, width),
                                  nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(width, 1)
        # Only the final layer is zero: gradients can learn a correction.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, baseline, forcing, attrs, latent=None):
        parts = [baseline.detach().unsqueeze(-1), forcing,
                 attrs[:, None].expand(-1, forcing.shape[1], -1)]
        if self.latent_dim:
            if latent is None or latent.shape[:2] != forcing.shape[:2]:
                raise ValueError('aligned PFN features required')
            parts.append(self.norm(latent.detach()))
        elif latent is not None:
            raise ValueError('raw correction must not receive PFN features')
        return baseline.detach() + self.output(self.body(torch.cat(parts, -1))).squeeze(-1)
