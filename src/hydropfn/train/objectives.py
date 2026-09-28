"""Discharge objectives with training-only, per-basin normalization.

The NSE surrogate averages daily squared errors within each sampled basin,
then averages basins. It targets mean normalized error, not median NSE.
"""

from __future__ import annotations

import numpy as np
import torch


def training_variances(q_raw, train_indices, train_window, global_std,
                       min_std_mm_day=0.1):
    """Return variances in standardized-Q units; unseen basins remain NaN.

The variance is fixed over the full TRAINING period, never the minibatch or
test period. A physical-unit standard-deviation floor bounds dry-basin weights.
"""
    if global_std <= 0 or min_std_mm_day <= 0:
        raise ValueError("global_std and min_std_mm_day must be positive")
    q = np.asarray(q_raw[train_indices, train_window], dtype=np.float64)
    if np.any(np.isfinite(q).sum(axis=1) < 2):
        raise ValueError("each training basin needs at least two valid targets")
    variance = np.nanvar(q, axis=1)
    result = np.full(len(q_raw), np.nan, dtype=np.float32)
    result[train_indices] = (np.maximum(variance, min_std_mm_day ** 2)
                             / global_std ** 2).astype(np.float32)
    return result


def basin_normalized_mse(prediction, target, valid, variance):
    """Equal weight per nonempty sampled basin, regardless of missing days."""
    if prediction.shape != target.shape or valid.shape != target.shape:
        raise ValueError("prediction, target and valid must have the same shape")
    if variance.shape != (prediction.shape[0],):
        raise ValueError("one training variance is required per sampled basin")
    if not torch.isfinite(variance).all() or (variance <= 0).any():
        raise ValueError("invalid or held-out basin variance in training loss")
    mask = valid.bool() & torch.isfinite(target)
    error = torch.where(mask, prediction - torch.nan_to_num(target), 0.0)
    count = mask.flatten(1).sum(1)
    per_basin = error.square().flatten(1).sum(1) / count.clamp(min=1)
    keep = count > 0
    if not keep.any():
        raise ValueError("batch contains no valid supervised discharge")
    return (per_basin[keep] / variance[keep]).mean()
