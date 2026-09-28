"""Causal daily observation features and reproducible availability scenarios."""
from __future__ import annotations

import numpy as np


def lagged_observations(q, observed=None, delay=1, max_age=365):
    """At day t, use only observations dated <= t-delay.

Channels: most recent standardized Q, indicator for an observation at t-1,
log-scaled age, and indicator that usable history exists. Absence never means
zero discharge. Values older than max_age are marked absent, not imputed.
"""
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or delay < 1 or max_age < 1:
        raise ValueError("q must be (basin,time), delay and max_age >= 1")
    observed = np.isfinite(q) if observed is None else np.asarray(observed, bool) & np.isfinite(q)
    if observed.shape != q.shape:
        raise ValueError("observation mask shape differs from Q")
    n, days = q.shape
    dates = np.broadcast_to(np.arange(days), q.shape)
    latest = np.maximum.accumulate(np.where(observed, dates, -1), axis=1)
    source = np.full((n, days), -1, dtype=int)
    if delay < days:
        source[:, delay:] = latest[:, :-delay]
    age = dates - source
    has = (source >= 0) & (age <= max_age)
    values = np.take_along_axis(np.nan_to_num(q), np.maximum(source, 0), axis=1)
    values = np.where(has, values, 0.)
    age_feature = np.log1p(np.where(has, age, max_age)) / np.log1p(max_age)
    return np.stack([values, has & (age == 1), age_feature, has], axis=-1).astype(np.float32)


def training_di_features(q, mode, rng, warmup=365, max_age=365):
    """Mixture over daily gauges, missing gauges, sparse visits and campaigns.

The availability RNG must be separate from the basin/window sampler so the
matched forward, daily and mixed experiments see identical target windows.
"""
    if mode == "forward":
        return lagged_observations(q, np.zeros_like(q, bool), max_age=max_age)
    if mode == "daily":
        return lagged_observations(q, max_age=max_age)
    if mode != "mixed":
        raise ValueError(mode)
    observed = np.zeros_like(q, bool)
    # Equal-sized architectural inputs in all arms. One quarter of the
    # mixture is an explicit whole-record absence.
    choices = rng.choice(4, size=len(q), p=[0.25, 0.35, 0.20, 0.20])
    for b, kind in enumerate(choices):
        if kind == 1:
            observed[b] = True
            # Include genuine long outages, independent of target values.
            if rng.random() < 0.5:
                start = int(rng.integers(0, max(1, q.shape[1] - 90)))
                observed[b, start:start + int(rng.integers(7, 91))] = False
        elif kind == 2:
            stride = int(rng.choice([2, 7, 30]))
            phase = int(rng.integers(stride))
            observed[b, phase::stride] = True
        elif kind == 3:
            length = int(rng.choice([30, 90, 365]))
            observed[b, max(0, warmup - length):warmup] = True
    features = lagged_observations(q, observed, max_age=max_age)
    # Reporting delay is trained separately from missing measurements.
    delayed = np.flatnonzero((choices == 1) & (rng.random(len(q)) < 0.25))
    for b in delayed:
        delay = int(rng.choice([2, 7, 16]))
        features[b:b + 1] = lagged_observations(
            q[b:b + 1], observed[b:b + 1], delay=delay, max_age=max_age)
    return features


def evaluation_di_features(q, scenario, warmup=365, max_age=365):
    observed = np.isfinite(q)
    delay = 1
    if scenario == "none":
        observed[:] = False
    elif scenario == "daily":
        pass
    elif scenario.startswith("lag"):
        delay = int(scenario[3:])
    elif scenario == "weekly":
        observed &= (np.arange(q.shape[1]) % 7 == 0)[None]
    elif scenario.startswith("campaign"):
        length = int(scenario[8:])
        observed[:] = False
        observed[:, max(0, warmup - length):warmup] = True
    else:
        raise ValueError(f"unknown observation scenario {scenario}")
    return lagged_observations(q, observed, delay=delay, max_age=max_age)


def training_di_views(q, mode, rng, paired_second="mixed", warmup=365):
    """Views of the SAME basin/window; paired mode never replaces daily input.

The daily/daily control has two identical inputs with independent model
dropout draws, matching the paired model's passes and optimizer updates.
The availability RNG is independent of the basin/window sampler.
"""
    if mode != "paired":
        return [(mode, training_di_features(q, mode, rng, warmup=warmup))]
    if paired_second not in ("mixed", "daily"):
        raise ValueError("paired_second must be mixed or daily")
    return [("daily", training_di_features(q, "daily", rng, warmup=warmup)),
            (paired_second, training_di_features(q, paired_second, rng, warmup=warmup))]
