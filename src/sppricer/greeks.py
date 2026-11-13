"""Greeks: central differences with common random numbers, and likelihood-ratio estimators."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sppricer.engine import price_mc
from sppricer.market import Market
from sppricer.models import PathModel, simulate_gbm
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


@dataclass(frozen=True)
class LRGreeks:
    """Likelihood-ratio Greeks with their Monte Carlo standard errors (units as in Greeks)."""

    price: float
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    price_se: float
    delta_se: np.ndarray
    gamma_se: np.ndarray
    vega_se: np.ndarray


def lr_greeks(product: Product, market: Market, n_paths: int = 100_000, seed: int = 0,
              antithetic: bool = True, vol_bump: float = 1e-4) -> LRGreeks:
    """Delta, gamma and vega by the likelihood-ratio method (Broadie and Glasserman 1996), under GBM.

    The payoff is never differentiated: each Greek is E[discounted payoff x weight], where the weight
    is the derivative of the log density of the simulated path. So it works unchanged for digital
    coupons, autocall triggers and barriers, where bump-and-reprice gamma is noisy and biased, and
    every Greek comes with an honest standard error.

    Each step's log return is normal. With x the standardised shocks of a step (correlated N(0, rho)),
    y = rho^-1 x and t1 the first observation date:
        delta_i: y_i / (S0_i sigma_i sqrt(t1))                        (first step only: S0 enters there)
        gamma_i: (y_i^2 - (rho^-1)_ii - y_i sigma_i sqrt(t1)) / (S0_i sigma_i sqrt(t1))^2
        vega_i:  sum over steps j of y_ji (x_ji / sigma_i - sqrt(dt_j)) - 1 / sigma_i
    gamma is the diagonal (own-asset) gamma, as in bump_greeks. Continuous or daily barrier
    monitoring prices a Brownian-bridge probability that depends on sigma explicitly; vega adds that
    term by revaluing the same paths at sigma +/- vol_bump.

    For smooth payoffs bump_greeks (common random numbers, so effectively pathwise) has the smaller
    variance; use this for discontinuous ones. Weights grow like 1 / (sigma sqrt(t1)), so a very
    early first observation date makes them noisy.
    """
    # ponytail: pure likelihood ratio; a mixed pathwise-LR gamma (Glasserman 2004, 7.3) has lower
    # variance for smooth payoffs but needs each product's payoff derivative.
    times = np.asarray(product.simulation_times, dtype=float)
    paths = simulate_gbm(market, times, n_paths, seed, antithetic)
    pv = product.discounted_payoff(paths, market)

    vol, s0 = market.vols, market.spots
    dt = np.diff(np.concatenate([[0.0], times]))
    sqdt = np.sqrt(dt)[:, None]                                            # (steps, 1)
    drift = np.diff(market.log_growth(np.concatenate([[0.0], times])), axis=0) - 0.5 * vol**2 * dt[:, None]
    log_s = np.log(np.concatenate([np.broadcast_to(s0, (len(paths), 1, len(s0))), paths], axis=1))
    x = (np.diff(log_s, axis=1) - drift) / (vol * sqdt)                    # (paths, steps, assets)
    rinv = np.linalg.inv(market.correlation)
    y = x @ rinv                                                           # rho^-1 x (rho symmetric)

    a = vol * sqdt[0]                                                      # sigma_i sqrt(t1)
    w_delta = y[:, 0] / (s0 * a)
    w_gamma = (y[:, 0] ** 2 - np.diag(rinv) - y[:, 0] * a) / (s0 * a) ** 2
    w_vega = (y * (x / vol - sqdt)).sum(axis=1) - len(times) / vol

    explicit = np.zeros((len(paths), len(s0)))
    if getattr(product, "barrier_monitoring", "european") != "european":
        for i, u in enumerate(market.underlyings):
            up = product.discounted_payoff(paths, market.with_vol(i, u.vol + vol_bump))
            dn = product.discounted_payoff(paths, market.with_vol(i, u.vol - vol_bump))
            explicit[:, i] = (up - dn) / (2 * vol_bump)

    def mean_se(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if antithetic:                                                     # pair (Z, -Z) first, as price_mc
            half = len(v) // 2
            v = 0.5 * (v[:half] + v[half:])
        return v.mean(axis=0), v.std(axis=0, ddof=1) / np.sqrt(len(v))

    price, price_se = mean_se(pv)
    delta, delta_se = mean_se(pv[:, None] * w_delta)
    gamma, gamma_se = mean_se(pv[:, None] * w_gamma)
    vega, vega_se = mean_se(pv[:, None] * w_vega + explicit)
    return LRGreeks(float(price), delta, gamma, 0.01 * vega, float(price_se), delta_se, gamma_se,
                    0.01 * vega_se)
