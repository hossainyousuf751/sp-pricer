"""Path simulation models: GBM (exact, on product dates) and Dupire local vol (log-Euler)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from sppricer.market import Curve, Market


def _check(times, n_paths: int, antithetic: bool) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if times.ndim != 1 or len(times) == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if times[0] <= 0 or np.any(np.diff(times) <= 0):
        raise ValueError("times must be positive and strictly increasing")
    if antithetic and n_paths % 2:
        raise ValueError("n_paths must be even when antithetic=True")
    return times


def simulate_gbm(market: Market, times: np.ndarray, n_paths: int,
                 seed: int | None = None, antithetic: bool = True) -> np.ndarray:
    """Simulate correlated geometric Brownian motion at the given times.

    The log-Euler step is exact for GBM, so paths are only generated on the
    dates the payoff needs (e.g. discrete observation dates). With market.curves
    set, the drift over each step is ln F(t2) - ln F(t1) - vol^2 dt / 2.

    Returns an array of shape (n_paths, n_times, n_assets). With antithetic=True
    the first half of the paths uses Z and the second half uses -Z.
    """
    times = _check(times, n_paths, antithetic)

    rng = np.random.default_rng(seed)
    n_draw = n_paths // 2 if antithetic else n_paths
    z = rng.standard_normal((n_draw, len(times), market.n_assets))
    if antithetic:
        z = np.concatenate([z, -z], axis=0)

    chol = np.linalg.cholesky(market.correlation)
    z = z @ chol.T

    dt = np.diff(np.concatenate([[0.0], times]))[None, :, None]
    vol = market.vols[None, None, :]
    if market.curves is None:
        drift = (market.rate - market.div_yields[None, None, :] - 0.5 * vol**2) * dt
    else:
        drift = np.diff(market.log_growth(np.concatenate([[0.0], times])), axis=0)[None] - 0.5 * vol**2 * dt
    shocks = vol * np.sqrt(dt) * z
    log_paths = np.log(market.spots)[None, None, :] + np.cumsum(drift + shocks, axis=1)
    return np.exp(log_paths)


class PathModel(Protocol):
    """Anything that simulates paths at the product's dates for price_mc.

    Must return shape (n_paths, len(times), n_assets) in the antithetic layout
    (first half Z, second half -Z), which the engine's standard error relies on.
    """

    def simulate(self, market: Market, times: np.ndarray, n_paths: int,
                 seed: int | None = None, antithetic: bool = True) -> np.ndarray: ...


@dataclass(frozen=True)
class GBMModel:
    """Flat-vol GBM, the default model: exactly simulate_gbm."""

    def simulate(self, market: Market, times: np.ndarray, n_paths: int,
                 seed: int | None = None, antithetic: bool = True) -> np.ndarray:
        return simulate_gbm(market, times, n_paths, seed, antithetic)


class LocalVolSource(Protocol):
    """A Dupire local-vol surface for one asset and the forward curve it was built on.

    local_vol(K, T) takes absolute strikes (an array) and a time in years and
    returns the local vol at each; it must be finite on the strikes paths reach
    (clamp or extrapolate in the surface, not here).
    """

    curve: Curve

    def local_vol(self, K, T): ...


@dataclass(frozen=True)
class LocalVolModel:
    """Dupire (1994) local volatility, one surface per asset, simulated by log-Euler.

    The grid is every multiple of dt up to the last product date, plus the product
    dates; a date already on the dt grid adds no step, so it does not shift the
    random stream (common random numbers survive a schedule change). Over a step
    [t0, t1] each asset moves by

        d ln S = ln F(t1) - ln F(t0) - sigma^2 dt / 2 + sigma sqrt(dt) Z,
        sigma  = local_vol(S(t0), (t0 + t1) / 2),

    with F the surface's forward curve, so the asset grows like its forward (the
    surface's curve, not market.rate; match market.discount_curve to it). Z is
    correlated with the market's Cholesky factor, in the antithetic layout.
    Euler bias is O(dt): halve dt to check it.
    """

    surfaces: tuple[LocalVolSource, ...]
    dt: float = 1 / 252
    n_strikes: int = 401  # strikes per step at which local vol is evaluated, then interpolated in ln K

    def __post_init__(self) -> None:
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not self.dt > 0:
            raise ValueError("dt must be positive")

    def grid(self, times: np.ndarray) -> np.ndarray:
        """Simulation grid, starting at 0: multiples of dt below the last date, plus the dates."""
        times = np.asarray(times, dtype=float)
        base = self.dt * np.arange(1, int(np.ceil(times[-1] / self.dt - 1e-9)))
        base = base[np.abs(base[:, None] - times[None, :]).min(axis=1) > 1e-9]
        return np.concatenate([[0.0], np.sort(np.concatenate([base, times]))])

    def simulate(self, market: Market, times: np.ndarray, n_paths: int,
                 seed: int | None = None, antithetic: bool = True) -> np.ndarray:
        """Paths sampled at `times`, shape (n_paths, n_times, n_assets), like simulate_gbm."""
        return self._run(market, times, n_paths, seed, antithetic, keep_grid=False)[1]

    def simulate_grid(self, market: Market, times: np.ndarray, n_paths: int,
                      seed: int | None = None, antithetic: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """(grid, paths on the whole grid), e.g. for barrier monitoring on every step.

        Same random numbers as simulate. Memory is n_paths * len(grid) * n_assets floats.
        """
        return self._run(market, times, n_paths, seed, antithetic, keep_grid=True)

    def _run(self, market, times, n_paths, seed, antithetic, keep_grid):
        times = _check(times, n_paths, antithetic)
        if len(self.surfaces) != market.n_assets:
            raise ValueError(f"{len(self.surfaces)} surfaces for {market.n_assets} assets")
        grid = self.grid(times)
        sample = np.searchsorted(grid, times)
        ln_f = np.stack([np.log(s.curve.forward(grid)) for s in self.surfaces], axis=1)
        chol = np.linalg.cholesky(market.correlation)
        rng = np.random.default_rng(seed)
        n_draw = n_paths // 2 if antithetic else n_paths

        log_s = np.tile(np.log(market.spots), (n_paths, 1))
        out = np.empty((n_paths, len(grid) if keep_grid else len(times), market.n_assets))
        if keep_grid:
            out[:, 0] = market.spots
        j = 0
        for i in range(1, len(grid)):
            t0, t1 = grid[i - 1], grid[i]
            z = rng.standard_normal((n_draw, market.n_assets))
            z = (np.concatenate([z, -z]) if antithetic else z) @ chol.T
            sigma = np.column_stack([self._sigma(s, log_s[:, a], 0.5 * (t0 + t1))
                                     for a, s in enumerate(self.surfaces)])
            log_s += ln_f[i] - ln_f[i - 1] - 0.5 * sigma**2 * (t1 - t0) + sigma * np.sqrt(t1 - t0) * z
            if keep_grid:
                out[:, i] = np.exp(log_s)
            elif i == sample[j]:
                out[:, j] = np.exp(log_s)
                j += 1
        return grid, out

    def _sigma(self, surface: LocalVolSource, log_s: np.ndarray, t: float) -> np.ndarray:
        # ponytail: local vol evaluated at n_strikes points spanning today's paths and interpolated
        # linearly in ln K, not per path; raise n_strikes if a surface has sharp features.
        lo, hi = float(log_s.min()), float(log_s.max())
        k = np.linspace(lo, hi, self.n_strikes) if hi > lo else np.array([lo])
        lv = np.asarray(surface.local_vol(np.exp(k), t), dtype=float) * np.ones_like(k)
        if not np.all(np.isfinite(lv)):
            raise ValueError(f"local vol is not finite at t={t:.4f} for strikes the paths reach; "
                             "clamp or extrapolate the surface (e.g. floor/cap)")
        return np.interp(log_s, k, lv)
