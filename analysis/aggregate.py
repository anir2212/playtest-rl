"""Seed-level bootstrap intervals; no episode-level pseudo-replication."""
from __future__ import annotations

import numpy as np


def mean_ci(values, seed=91, resamples=2000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("empty sample")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(draws, 0.025)),
                                                   float(np.quantile(draws, 0.975))],
            "seed_values": values.tolist()}
