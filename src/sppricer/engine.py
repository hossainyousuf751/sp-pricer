"""Monte Carlo pricing engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.market import Market
from sppricer.models import simulate_gbm
from sppricer.products.base import Product


@dataclass(frozen=True)
class PricingResult:
    price: float
    std_error: float
    n_paths: int

    @property
    def ci95(self) -> tuple[float, float]:
        return self.price - 1.96 * self.std_error, self.price + 1.96 * self.std_error

    def __str__(self) -> str:
        lo, hi = self.ci95
        return f"{self.price:.4f} ± {1.96 * self.std_error:.4f} (95% CI [{lo:.4f}, {hi:.4f}], {self.n_paths:,} paths)"


def price_mc(product: Product, market: Market, n_paths: int = 100_000,
             seed: int | None = None, antithetic: bool = True) -> PricingResult:
    """Price a product by Monte Carlo under GBM.

    With antithetic sampling, each (Z, -Z) pair is averaged first so the
    standard error reflects the true (reduced) variance of the estimator.
    """
    paths = simulate_gbm(market, product.simulation_times, n_paths, seed, antithetic)
    pv = product.discounted_payoff(paths, market)
    if antithetic:
        half = n_paths // 2
        pv = 0.5 * (pv[:half] + pv[half:])
    return PricingResult(float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv))), n_paths)
