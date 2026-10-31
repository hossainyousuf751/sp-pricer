"""Product interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from sppricer.market import Market


class Product(ABC):
    """A product defines which dates to simulate and how to value each path.

    `discounted_payoff` receives paths of shape (n_paths, n_times, n_assets),
    sampled at `simulation_times`, and returns the present value per path.
    """

    @property
    @abstractmethod
    def simulation_times(self) -> np.ndarray: ...

    @abstractmethod
    def discounted_payoff(self, paths: np.ndarray, market: Market) -> np.ndarray: ...


def validate_schedule(times: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(times, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("schedule must be a non-empty sequence of times")
    if arr[0] <= 0 or np.any(np.diff(arr) <= 0):
        raise ValueError("schedule must be positive and strictly increasing")
    return arr


BARRIER_MONITORING = ("european", "continuous", "daily")

# Broadie-Glasserman-Kou: a barrier monitored every dt behaves like a continuous
# barrier shifted away from spot by exp(-beta * vol * sqrt(dt)), beta = -zeta(1/2)/sqrt(2*pi).
_BGK_BETA = 0.5826


def validate_monitoring(monitoring: str) -> None:
    if monitoring not in BARRIER_MONITORING:
        raise ValueError(f"barrier_monitoring must be one of {BARRIER_MONITORING}, got {monitoring!r}")


def knock_in_probability(paths: np.ndarray, times: np.ndarray, initial_levels: tuple[float, ...],
                         barrier: float, market: Market, monitoring: str) -> np.ndarray:
    """Probability, per path, that any asset touched the down barrier over [0, times[-1]].

    Uses the Brownian-bridge crossing probability between simulated dates,
    exp(-2 * a * b / (vol^2 * dt)) with a, b the log-distances above the barrier,
    so paths only need the product's observation dates. "daily" applies the
    Broadie-Glasserman-Kou shift to match daily closing-price monitoring.
    """
    vols = market.vols
    levels = np.asarray(initial_levels, dtype=float) * barrier
    if monitoring == "daily":
        levels = levels * np.exp(-_BGK_BETA * vols * np.sqrt(1 / 252))
    with np.errstate(divide="ignore"):
        log_b = np.log(levels)
    start = np.broadcast_to(np.log(market.spots), (paths.shape[0], 1, paths.shape[2]))
    dist = np.concatenate([start, np.log(paths)], axis=1) - log_b
    a, b = np.clip(dist[:, :-1], 0, None), np.clip(dist[:, 1:], 0, None)
    var = vols**2 * np.diff(np.concatenate([[0.0], times]))[:, None]
    p_cross = np.exp(-2 * a * b / np.maximum(var, 1e-300))
    # ponytail: assets' bridges treated as independent given endpoints; slightly overstates
    # knock-in for positively correlated baskets on coarse grids. Refine the grid if it matters.
    return 1.0 - np.prod(1.0 - p_cross, axis=(1, 2))


def worst_performance(paths: np.ndarray, initial_levels: tuple[float, ...]) -> np.ndarray:
    """Worst-of performance S_t / S_0 across assets, shape (n_paths, n_times)."""
    if paths.shape[2] != len(initial_levels):
        raise ValueError(
            f"product has {len(initial_levels)} initial levels but market has {paths.shape[2]} assets"
        )
    return (paths / np.asarray(initial_levels, dtype=float)).min(axis=2)
