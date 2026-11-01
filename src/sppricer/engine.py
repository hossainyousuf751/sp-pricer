"""Monte Carlo pricing engine."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.market import Market
from sppricer.models import GBMModel, PathModel, simulate_gbm
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
             seed: int | None = None, antithetic: bool = True, model: PathModel | None = None) -> PricingResult:
    """Price a product by Monte Carlo, under GBM unless another PathModel is given.

    With antithetic sampling, each (Z, -Z) pair is averaged first so the
    standard error reflects the true (reduced) variance of the estimator.

    Barrier monitoring other than "european" uses a Brownian-bridge correction
    that assumes flat GBM vol, so it raises under any other model.
    """
    if model is None:
        paths = simulate_gbm(market, product.simulation_times, n_paths, seed, antithetic)
    else:
        monitoring = getattr(product, "barrier_monitoring", "european")
        if monitoring != "european" and not isinstance(model, GBMModel):
            raise ValueError(f"barrier_monitoring={monitoring!r} assumes flat-vol GBM between dates; "
                             f"use 'european' with {type(model).__name__}")
        paths = model.simulate(market, product.simulation_times, n_paths, seed, antithetic)
    pv = product.discounted_payoff(paths, market)
    if antithetic:
        half = n_paths // 2
        pv = 0.5 * (pv[:half] + pv[half:])
    return PricingResult(float(pv.mean()), float(pv.std(ddof=1) / np.sqrt(len(pv))), n_paths)
