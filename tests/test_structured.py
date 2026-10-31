"""Limit-case tests: barriers set so the product collapses to something known."""

import math

import numpy as np
import pytest

from sppricer import (
    BarrierReverseConvertible,
    Market,
    PhoenixAutocallable,
    Underlying,
    price_mc,
)

R, Q, N = 0.04, 0.02, 100.0
QUARTERLY = tuple(0.25 * k for k in range(1, 13))
SINGLE = Market.single(100.0, 0.25, R, Q)
BASKET = Market(
    (Underlying("A", 100.0, 0.25, Q), Underlying("B", 50.0, 0.30, Q)),
    R,
    np.array([[1.0, 0.6], [0.6, 1.0]]),
)


def phoenix(**kw):
    defaults = dict(initial_levels=(100.0,), observation_times=QUARTERLY, coupon_rate=0.02)
    return PhoenixAutocallable(**{**defaults, **kw})


def test_unreachable_barriers_give_zero_coupon_bond():
    p = phoenix(coupon_barrier=np.inf, autocall_barrier=np.inf, protection_barrier=0.0)
    res = price_mc(p, SINGLE, n_paths=10_000, seed=0)
    assert res.price == pytest.approx(N * math.exp(-R * 3.0), abs=1e-10)


def test_always_called_at_first_date():
    p = phoenix(coupon_barrier=0.0, autocall_barrier=0.0)
    res = price_mc(p, SINGLE, n_paths=10_000, seed=0)
    assert res.price == pytest.approx(N * 1.02 * math.exp(-R * 0.25), abs=1e-10)


def test_full_downside_no_coupons_is_discounted_forward():
    p = phoenix(coupon_barrier=np.inf, autocall_barrier=np.inf, protection_barrier=np.inf)
    res = price_mc(p, SINGLE, n_paths=400_000, seed=0)
    assert abs(res.price - N * math.exp(-Q * 3.0)) < 4 * res.std_error


def test_memory_coupon_is_worth_more():
    mem = price_mc(phoenix(memory=True), SINGLE, n_paths=200_000, seed=5).price
    no_mem = price_mc(phoenix(memory=False), SINGLE, n_paths=200_000, seed=5).price
    assert mem > no_mem


def test_worst_of_is_cheaper_than_single_asset():
    single = price_mc(phoenix(), SINGLE, n_paths=200_000, seed=6).price
    basket = price_mc(phoenix(initial_levels=(100.0, 50.0)), BASKET, n_paths=200_000, seed=6).price
    assert basket < single


def test_brc_with_zero_barrier_is_coupon_bond():
    brc = BarrierReverseConvertible((100.0,), QUARTERLY, 0.02, knock_in_barrier=0.0)
    res = price_mc(brc, SINGLE, n_paths=10_000, seed=0)
    expected = N * 0.02 * sum(math.exp(-R * t) for t in QUARTERLY) + N * math.exp(-R * 3.0)
    assert res.price == pytest.approx(expected, abs=1e-10)


def test_mismatched_assets_raise():
    with pytest.raises(ValueError):
        price_mc(phoenix(), BASKET, n_paths=1_000, seed=0)


def test_invalid_correlation_raises():
    with pytest.raises(ValueError):
        Market(BASKET.underlyings, R, np.array([[1.0, 1.2], [1.2, 1.0]]))
