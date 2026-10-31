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

    def __post_init__(self) -> None:
        validate_schedule(self.observation_times)
        validate_monitoring(self.barrier_monitoring)
        for name in ("coupon_barrier", "autocall_barrier", "protection_barrier"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0 <= self.first_autocall_index < len(self.observation_times):
            raise ValueError("first_autocall_index out of range")

    @property
    def simulation_times(self) -> np.ndarray:
        return np.asarray(self.observation_times, dtype=float)

    def discounted_payoff(self, paths: np.ndarray, market: Market) -> np.ndarray:
        worst = worst_performance(paths, self.initial_levels)
        n_paths, n_obs = worst.shape
        alive = np.ones(n_paths, dtype=bool)
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
            elif i >= self.first_autocall_index:
                called = alive & (w >= self.autocall_barrier)
                pv += np.where(called, self.notional, 0.0) * df
                alive &= ~called

        return pv
