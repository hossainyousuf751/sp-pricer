"""Validate the MC engine against Black-Scholes closed forms."""

import math

import pytest

from sppricer import EuropeanOption, Market, bs_delta, bs_price, bs_vega, bump_greeks, price_mc

S, K, T, R, VOL, Q = 100.0, 105.0, 1.5, 0.04, 0.25, 0.02
MKT = Market.single(S, VOL, R, Q)


@pytest.mark.parametrize("is_call", [True, False])
def test_mc_converges_to_black_scholes(is_call):
    res = price_mc(EuropeanOption(K, T, is_call), MKT, n_paths=400_000, seed=1)
    exact = bs_price(S, K, T, R, VOL, Q, is_call)
    assert abs(res.price - exact) < 4 * res.std_error


def test_put_call_parity():
    call = price_mc(EuropeanOption(K, T, True), MKT, n_paths=400_000, seed=2)
    put = price_mc(EuropeanOption(K, T, False), MKT, n_paths=400_000, seed=2)
    forward_pv = S * math.exp(-Q * T) - K * math.exp(-R * T)
    tol = 4 * math.hypot(call.std_error, put.std_error)
    assert abs((call.price - put.price) - forward_pv) < tol


def test_antithetic_reduces_standard_error():
    opt = EuropeanOption(K, T, True)
    plain = price_mc(opt, MKT, n_paths=200_000, seed=3, antithetic=False)
    anti = price_mc(opt, MKT, n_paths=200_000, seed=3, antithetic=True)
    assert anti.std_error < plain.std_error


def test_bump_delta_and_vega_match_black_scholes():
    g = bump_greeks(EuropeanOption(K, T, True), MKT, n_paths=400_000, seed=4)
    assert g.delta[0] == pytest.approx(bs_delta(S, K, T, R, VOL, Q, True), abs=0.01)
    assert g.vega[0] == pytest.approx(bs_vega(S, K, T, R, VOL, Q) / 100, abs=0.01)
