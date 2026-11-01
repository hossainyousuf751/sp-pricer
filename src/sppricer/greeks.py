"""Finite-difference Greeks with common random numbers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.engine import price_mc
from sppricer.market import Market
from sppricer.models import PathModel
from sppricer.products.base import Product


@dataclass(frozen=True)
class Greeks:
    price: float
    delta: np.ndarray   # per underlying, per 1 unit of spot
    gamma: np.ndarray   # per underlying, per 1 unit of spot squared
    vega: np.ndarray    # per underlying, per 1 vol point (0.01)
    rho: float          # per 1 basis point


def bump_greeks(product: Product, market: Market, n_paths: int = 100_000, seed: int = 0,
                spot_bump: float = 0.01, vol_bump: float = 0.01,
                rate_bump: float = 0.0001, model: PathModel | None = None) -> Greeks:
    """Central-difference Greeks, reusing the same seed for every revaluation.

    Common random numbers cancel most Monte Carlo noise between bumped prices.
    Gamma of digital/barrier features is still noisy with bump-and-reprice;
    see the roadmap (pathwise / likelihood-ratio estimators, payoff smoothing).

    `model` is passed to price_mc. Vega bumps the underlyings' flat vols, so under
    a model that ignores them (LocalVolModel) it is zero: bump the surface instead.
    """
    def pv(m: Market) -> float:
        return price_mc(product, m, n_paths=n_paths, seed=seed, model=model).price

    base = pv(market)
    n = market.n_assets
    delta, gamma, vega = np.zeros(n), np.zeros(n), np.zeros(n)

    for i, u in enumerate(market.underlyings):
        h = u.spot * spot_bump
        up, dn = pv(market.with_spot(i, u.spot + h)), pv(market.with_spot(i, u.spot - h))
        delta[i] = (up - dn) / (2 * h)
        gamma[i] = (up - 2 * base + dn) / h**2

        vol_dn = max(u.vol - vol_bump, 0.0)
        v_up = pv(market.with_vol(i, u.vol + vol_bump))
        v_dn = pv(market.with_vol(i, vol_dn))
        vega[i] = (v_up - v_dn) / (u.vol + vol_bump - vol_dn) * 0.01

    r_up = pv(market.with_rate(market.rate + rate_bump))
    r_dn = pv(market.with_rate(market.rate - rate_bump))
    rho = (r_up - r_dn) / 2

    return Greeks(base, delta, gamma, vega, rho)
