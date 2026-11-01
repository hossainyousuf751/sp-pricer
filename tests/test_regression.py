"""Pin the README headline numbers to examples/price_worst_of_phoenix.py (same market, note and seeds)."""

from dataclasses import replace

import numpy as np
import pytest

from sppricer import Market, PhoenixAutocallable, Underlying, bump_greeks, price_mc

MARKET = Market(
    underlyings=(Underlying("STOCK_A", 100.0, 0.25, 0.02), Underlying("STOCK_B", 50.0, 0.30, 0.015)),
    rate=0.04, correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
)
NOTE = PhoenixAutocallable(
    initial_levels=(100.0, 50.0), observation_times=tuple(0.25 * k for k in range(1, 13)), coupon_rate=0.02,
    coupon_barrier=0.70, autocall_barrier=1.00, protection_barrier=0.60, memory=True, first_autocall_index=3,
)


def test_fast_pin_of_readme_note():
    r = price_mc(NOTE, MARKET, n_paths=20_000, seed=42)
    assert (r.price, r.std_error) == pytest.approx((93.83814373477094, 0.1506193346004773), rel=1e-12)


@pytest.mark.slow
def test_readme_numbers():
    r = price_mc(NOTE, MARKET, n_paths=400_000, seed=42)
    assert (r.price, r.std_error) == pytest.approx((93.8766851766323, 0.03365449456597753), rel=1e-12)
    d = price_mc(replace(NOTE, barrier_monitoring="daily"), MARKET, n_paths=400_000, seed=42)
    assert d.price == pytest.approx(90.58335155691613, rel=1e-12)
    g = bump_greeks(NOTE, MARKET, n_paths=200_000, seed=42)
    np.testing.assert_allclose(g.vega, [-0.43585270527092457, -0.5409208815147859], rtol=1e-10)
    np.testing.assert_allclose(g.delta, [0.18175017955223183, 0.48963410780034167], rtol=1e-10)
    assert g.rho == pytest.approx(-0.007058554987075638, rel=1e-10)
