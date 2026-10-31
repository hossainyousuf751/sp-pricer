"""Barrier monitoring: Brownian-bridge estimator vs brute-force daily simulation."""

import numpy as np
import pytest

from sppricer import BarrierReverseConvertible, Market, PhoenixAutocallable, Underlying, price_mc, simulate_gbm

R, Q, N = 0.04, 0.02, 100.0
QUARTERLY_1Y = (0.25, 0.5, 0.75, 1.0)
DAILY = np.arange(1, 253) / 252
SINGLE = Market.single(100.0, 0.25, R, Q)
BASKET = Market(
    (Underlying("A", 100.0, 0.25, Q), Underlying("B", 50.0, 0.30, Q)),
    R,
    np.array([[1.0, 0.6], [0.6, 1.0]]),
)


@pytest.mark.parametrize("market,levels", [(SINGLE, (100.0,)), (BASKET, (100.0, 50.0))])
def test_daily_bridge_matches_brute_force_daily_monitoring(market, levels):
    brc = BarrierReverseConvertible(levels, QUARTERLY_1Y, 0.02, knock_in_barrier=0.75,
                                    barrier_monitoring="daily")
    paths = simulate_gbm(market, DAILY, 20_000, seed=7, antithetic=False)

    # Brute force: knocked in if the worst-of closes below the barrier on any day.
    worst = (paths / np.array(levels)).min(axis=2)
    knocked = worst.min(axis=1) < 0.75
    w_t = worst[:, -1]
    redemption = np.where(knocked & (w_t < 1.0), N * w_t, N)
    coupons = N * 0.02 * np.sum(market.discount(np.array(QUARTERLY_1Y)))
    brute = coupons + redemption * market.discount(1.0)

    idx = np.searchsorted(DAILY, QUARTERLY_1Y)
    bridge = brc.discounted_payoff(paths[:, idx], market)

    diff = bridge - brute
    assert abs(diff.mean()) < 4 * diff.std(ddof=1) / np.sqrt(len(diff))


@pytest.mark.parametrize("product", ["brc", "phoenix"])
def test_more_monitoring_means_cheaper_note(product):
    def make(monitoring):
        if product == "brc":
            return BarrierReverseConvertible((100.0,), QUARTERLY_1Y, 0.02, 0.7, barrier_monitoring=monitoring)
        return PhoenixAutocallable((100.0,), QUARTERLY_1Y, 0.02, barrier_monitoring=monitoring)

    prices = [price_mc(make(m), SINGLE, n_paths=200_000, seed=8).price for m in ("european", "daily", "continuous")]
    assert prices[0] > prices[1] > prices[2]


def test_zero_barrier_never_knocks_in():
    brc = BarrierReverseConvertible((100.0,), QUARTERLY_1Y, 0.02, 0.0, barrier_monitoring="continuous")
    euro = BarrierReverseConvertible((100.0,), QUARTERLY_1Y, 0.02, 0.0)
    assert price_mc(brc, SINGLE, 10_000, seed=0).price == pytest.approx(price_mc(euro, SINGLE, 10_000, seed=0).price)


def test_invalid_monitoring_raises():
    with pytest.raises(ValueError):
        BarrierReverseConvertible((100.0,), QUARTERLY_1Y, 0.02, barrier_monitoring="weekly")
