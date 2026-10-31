"""Barrier reverse convertible on a worst-of basket."""

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
class BarrierReverseConvertible(Product):
    """Guaranteed coupons plus a short down-and-in put at maturity.

    Coupons of notional * coupon_rate are paid on every coupon date. At
    maturity the investor receives notional, unless the put has knocked in
    and worst-of performance W is below 1, in which case they get notional * W.

    barrier_monitoring:
      * "european": knocked in if W < knock_in_barrier at maturity only.
      * "continuous" / "daily": knocked in if any asset touched the barrier at
        any time (daily closes for "daily"), via a Brownian-bridge correction
        on the coupon-date grid.
    """

    initial_levels: tuple[float, ...]
    coupon_times: tuple[float, ...]
    coupon_rate: float
    knock_in_barrier: float = 0.60
    notional: float = 100.0
    barrier_monitoring: str = "european"

    def __post_init__(self) -> None:
        validate_schedule(self.coupon_times)
        validate_monitoring(self.barrier_monitoring)

    @property
    def simulation_times(self) -> np.ndarray:
        if self.barrier_monitoring == "european":
            return np.array([self.coupon_times[-1]])
        return np.asarray(self.coupon_times, dtype=float)

    def discounted_payoff(self, paths: np.ndarray, market: Market) -> np.ndarray:
        times = np.asarray(self.coupon_times, dtype=float)
        coupons_pv = self.notional * self.coupon_rate * np.sum(market.discount(times))
        w = worst_performance(paths, self.initial_levels)[:, -1]
        if self.barrier_monitoring == "european":
            redemption = np.where(w >= self.knock_in_barrier, self.notional, self.notional * w)
        else:
            p_ki = knock_in_probability(paths, self.simulation_times, self.initial_levels,
                                        self.knock_in_barrier, market, self.barrier_monitoring)
            redemption = self.notional * (1.0 - p_ki * np.maximum(1.0 - w, 0.0))
        return coupons_pv + redemption * market.discount(times[-1])
