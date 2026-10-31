"""European vanilla options (benchmark product for engine validation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.market import Market
from sppricer.products.base import Product


@dataclass(frozen=True)
class EuropeanOption(Product):
    strike: float
    maturity: float
    is_call: bool = True
    asset_index: int = 0

    @property
    def simulation_times(self) -> np.ndarray:
        return np.array([self.maturity])

    def discounted_payoff(self, paths: np.ndarray, market: Market) -> np.ndarray:
        s_t = paths[:, -1, self.asset_index]
        intrinsic = s_t - self.strike if self.is_call else self.strike - s_t
        return np.maximum(intrinsic, 0.0) * market.discount(self.maturity)
