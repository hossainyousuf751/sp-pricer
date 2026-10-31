"""Path simulation models."""

from __future__ import annotations

import numpy as np

from sppricer.market import Market


def simulate_gbm(market: Market, times: np.ndarray, n_paths: int,
                 seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Simulate correlated geometric Brownian motion at the given times.

    The log-Euler step is exact for GBM, so paths are only generated on the
    dates the payoff needs (e.g. discrete observation dates).

    Returns an array of shape (n_paths, n_times, n_assets). With antithetic=True
    the first half of the paths uses Z and the second half uses -Z.
    """
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic=True")

    rng = np.random.default_rng(seed)
    n_draw = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((n_draw, len(times), market.n_assets))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)

    chol = np.linalg.cholesky(market.correlation)
    z = z @ chol.T

    dt = np.diff(np.concatenate([[0.0], times]))[None, :, None]
    vol = market.vols[None, None, :]
    drift = (market.rate - market.div_yields[None, None, :] - 0.5 * vol**2) * dt
    shocks = vol * np.sqrt(dt) * z
    log_paths = np.log(market.spots)[None, None, :] + np.cumsum(drift + shocks, axis=1)
    return np.exp(log_paths)
