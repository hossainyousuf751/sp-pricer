"""Phoenix autocallable on a worst-of basket."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.market import Market
from sppricer.products.base import (
    Product,
    knock_in_probability,
    validate_monitoring,
    validate_schedule,
    worst_performance,
)


@dataclass(frozen=True)
class PhoenixAutocallable(Product):
    """Worst-of Phoenix autocallable with optional memory coupon.

    On each observation date t_i (while the note is alive), with W = worst-of
    performance S_t / S_0:
      * Coupon: if W >= coupon_barrier, pay notional * coupon_rate
        (plus any missed coupons if memory=True).
      * Autocall (from first_autocall_index, before maturity): if
        W >= autocall_barrier, repay notional and terminate.
      * Maturity: with barrier_monitoring="european", repay notional if
        W >= protection_barrier, otherwise notional * W. With "continuous" or
        "daily", the protection barrier is a knock-in watched over the whole
        life: if any asset touched it and W < 1, repay notional * W.

    autocall_indices (optional) lists the observation dates on which the note can be called, for
    notes whose coupon is observed more often than the call (e.g. monthly coupons, quarterly calls).
    When set it replaces first_autocall_index; by default every date from first_autocall_index on is
    callable, as before.

    Barriers are expressed as fractions of the initial level (0.7 = 70%).
    coupon_rate is per observation period (e.g. 0.02 for 2% per quarter).
    """

    initial_levels: tuple[float, ...]
    observation_times: tuple[float, ...]
    coupon_rate: float
    coupon_barrier: float = 0.70
    autocall_barrier: float = 1.00
    protection_barrier: float = 0.60
    memory: bool = True
    first_autocall_index: int = 0
    notional: float = 100.0
    barrier_monitoring: str = "european"
    autocall_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        validate_schedule(self.observation_times)
        validate_monitoring(self.barrier_monitoring)
        for name in ("coupon_barrier", "autocall_barrier", "protection_barrier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.first_autocall_index < len(self.observation_times):
            raise ValueError("first_autocall_index out of range")
        if self.autocall_indices is not None:
            idx = tuple(int(i) for i in self.autocall_indices)
            if list(idx) != sorted(set(idx)) or not all(0 <= i < len(self.observation_times) - 1 for i in idx):
                raise ValueError("autocall_indices must be increasing indices of observation dates before maturity")
            object.__setattr__(self, "autocall_indices", idx)

    @property
    def simulation_times(self) -> np.ndarray:
        return np.asarray(self.observation_times, dtype=float)

    def discounted_payoff(self, paths: np.ndarray, market: Market) -> np.ndarray:
        worst = worst_performance(paths, self.initial_levels)
        n_paths, n_obs = worst.shape
        alive = np.ones(n_paths, dtype=bool)
        callable_ = np.zeros(n_obs, dtype=bool)
        if self.autocall_indices is None:
            callable_[self.first_autocall_index:] = True
        else:
            callable_[list(self.autocall_indices)] = True
        missed = np.zeros(n_paths, dtype=int)
        pv = np.zeros(n_paths)

        for i, t in enumerate(self.simulation_times):
            df = market.discount(t)
            w = worst[:, i]

            pays_coupon = alive & (w >= self.coupon_barrier)
            n_coupons = missed + 1 if self.memory else 1
            pv += np.where(pays_coupon, self.notional * self.coupon_rate * n_coupons, 0.0) * df
            missed = np.where(alive & ~pays_coupon, missed + 1, 0)

            if i == n_obs - 1:
                if self.barrier_monitoring == "european":
                    redemption = np.where(w >= self.protection_barrier, self.notional, self.notional * w)
                else:
                    p_ki = knock_in_probability(paths, self.simulation_times, self.initial_levels,
                                                self.protection_barrier, market, self.barrier_monitoring)
                    redemption = self.notional * (1.0 - p_ki * np.maximum(1.0 - w, 0.0))
                pv += np.where(alive, redemption, 0.0) * df
            elif callable_[i]:
                called = alive & (w >= self.autocall_barrier)
                pv += np.where(called, self.notional, 0.0) * df
                alive &= ~called

        return pv
