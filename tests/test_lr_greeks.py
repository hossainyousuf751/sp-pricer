"""M5: likelihood-ratio Greeks against closed forms, with tolerances in standard errors."""

from dataclasses import dataclass, replace
from math import erf, exp, log, pi, sqrt

import numpy as np

from sppricer import (
    BarrierReverseConvertible,
    EuropeanOption,
    Market,
    Underlying,
    bs_delta,
    bs_vega,
    bump_greeks,
    lr_greeks,
)
from sppricer.products.base import Product

S, K, T, R, VOL, Q = 100.0, 105.0, 1.0, 0.03, 0.25, 0.01
MARKET = Market.single(S, VOL, R, Q)


def _pdf(x):
    return exp(-0.5 * x * x) / sqrt(2 * pi)


def _d(s=S, k=K, t=T, r=R, v=VOL, q=Q):
    d1 = (log(s / k) + (r - q + 0.5 * v * v) * t) / (v * sqrt(t))
    return d1, d1 - v * sqrt(t)


def _within(est, se, exact, n_se=4.0):
    assert abs(est - exact) < n_se * se, f"{est:.6f} vs exact {exact:.6f} ({(est - exact) / se:+.1f} SE)"


@dataclass(frozen=True)
class Digital(Product):
    """Pays 1 at maturity if S_T > strike (cash-or-nothing call), on asset `asset`."""

    strike: float
    maturity: float
    asset: int = 0
    dates: tuple[float, ...] | None = None    # simulate on these dates (the payoff uses the last)

    @property
    def simulation_times(self) -> np.ndarray:
        return np.array(self.dates or (self.maturity,))

    def discounted_payoff(self, paths, market):
        return (paths[:, -1, self.asset] > self.strike) * market.discount(self.maturity)


@dataclass(frozen=True)
class Call(Product):
    """European call on asset `asset`, optionally simulated on intermediate dates."""

    strike: float
    maturity: float
    asset: int = 0
    dates: tuple[float, ...] | None = None

    @property
    def simulation_times(self) -> np.ndarray:
        return np.array(self.dates or (self.maturity,))

    def discounted_payoff(self, paths, market):
        return np.maximum(paths[:, -1, self.asset] - self.strike, 0.0) * market.discount(self.maturity)


def test_smooth_call_matches_black_scholes():
    g = lr_greeks(EuropeanOption(K, T), MARKET, n_paths=400_000, seed=1)
    d1, _ = _d()
    _within(g.delta[0], g.delta_se[0], bs_delta(S, K, T, R, VOL, Q))
    _within(g.gamma[0], g.gamma_se[0], exp(-Q * T) * _pdf(d1) / (S * VOL * sqrt(T)))
    _within(g.vega[0], g.vega_se[0], bs_vega(S, K, T, R, VOL, Q) / 100)  # bs_vega is per 1.00 vol


def test_digital_delta_and_gamma_match_closed_form():
    g = lr_greeks(Digital(K, T), MARKET, n_paths=400_000, seed=2)
    d1, d2 = _d()
    df = exp(-R * T)
    _within(g.price, g.price_se, df * 0.5 * (1 + erf(d2 / sqrt(2))))
    _within(g.delta[0], g.delta_se[0], df * _pdf(d2) / (S * VOL * sqrt(T)))
    _within(g.gamma[0], g.gamma_se[0], -df * _pdf(d2) * d1 / (S**2 * VOL**2 * T))


def test_correlation_puts_delta_on_the_right_asset():
    """A call on asset 1 of two correlated assets: all the delta is asset 1's, none is asset 0's."""
    market = Market((Underlying("A", 50.0, 0.30, 0.0), Underlying("B", S, VOL, Q)), R,
                    np.array([[1.0, 0.7], [0.7, 1.0]]))
    g = lr_greeks(Call(K, T, asset=1), market, n_paths=400_000, seed=3)
    _within(g.delta[1], g.delta_se[1], bs_delta(S, K, T, R, VOL, Q))
    _within(g.delta[0], g.delta_se[0], 0.0)
    _within(g.vega[1], g.vega_se[1], bs_vega(S, K, T, R, VOL, Q) / 100)  # bs_vega is per 1.00 vol
    _within(g.vega[0], g.vega_se[0], 0.0)


def test_vega_sums_over_every_simulation_step():
    """The same call simulated on quarterly dates: the vega score spans all four steps."""
    g = lr_greeks(Call(K, T, dates=(0.25, 0.5, 0.75, 1.0)), MARKET, n_paths=400_000, seed=4)
    _within(g.vega[0], g.vega_se[0], bs_vega(S, K, T, R, VOL, Q) / 100)  # bs_vega is per 1.00 vol
    _within(g.delta[0], g.delta_se[0], bs_delta(S, K, T, R, VOL, Q))


def test_continuous_barrier_vega_includes_the_explicit_bridge_term():
    """Brownian-bridge pricing depends on sigma directly; LR vega must agree with bump vega."""
    brc = BarrierReverseConvertible((100.0,), (0.25, 0.5, 0.75, 1.0), 0.02, knock_in_barrier=0.7,
                                    barrier_monitoring="continuous")
    g = lr_greeks(brc, MARKET, n_paths=400_000, seed=5)
    bumps = [bump_greeks(brc, MARKET, n_paths=40_000, seed=s).vega[0] for s in range(10)]
    bump_mean, bump_se = np.mean(bumps), np.std(bumps, ddof=1) / np.sqrt(len(bumps))
    _within(g.vega[0], np.hypot(g.vega_se[0], bump_se), bump_mean)
    without = lr_greeks(replace(brc, barrier_monitoring="european"), MARKET, n_paths=400_000, seed=5)
    assert abs(g.vega[0] - without.vega[0]) > 4 * np.hypot(g.vega_se[0], without.vega_se[0])


def test_price_matches_price_mc_on_the_same_paths():
    from sppricer import price_mc
    note = BarrierReverseConvertible((100.0,), (0.5, 1.0), 0.03)
    assert lr_greeks(note, MARKET, n_paths=20_000, seed=6).price == price_mc(note, MARKET, 20_000, seed=6).price
