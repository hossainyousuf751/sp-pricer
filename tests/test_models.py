"""Path models (M6): curves, the PathModel seam and local vol, tested with in-test stubs only."""

from dataclasses import replace

import numpy as np
import pytest

from sppricer import (
    EuropeanOption,
    GBMModel,
    LocalVolModel,
    Market,
    PhoenixAutocallable,
    Underlying,
    bs_price,
    price_mc,
    simulate_gbm,
)


class FlatCurve:
    """F(T) = S exp((r - q) T), DF(T) = exp(-r T)."""

    def __init__(self, spot, r, q=0.0):
        self.spot, self.r, self.q = spot, r, q

    def forward(self, T):
        return self.spot * np.exp((self.r - self.q) * np.asarray(T, dtype=float))

    def discount(self, T):
        return np.exp(-self.r * np.asarray(T, dtype=float))


class SlopedCurve:
    """A non-flat term structure: zero rate 2% + 1%·T, dividend yield 3% - 1%·T."""

    def __init__(self, spot):
        self.spot = spot

    def forward(self, T):
        T = np.asarray(T, dtype=float)
        return self.spot * np.exp((0.02 + 0.01 * T) * T - (0.03 - 0.01 * T) * T)

    def discount(self, T):
        T = np.asarray(T, dtype=float)
        return np.exp(-(0.02 + 0.01 * T) * T)


class Stub:
    """A LocalVolSource: local vol from a function of (K, T), on a given curve."""

    def __init__(self, curve, fn):
        self.curve, self.fn = curve, fn

    def local_vol(self, K, T):
        return self.fn(np.asarray(K, dtype=float), T)


def flat(vol):
    return lambda K, T: np.full(K.shape, vol)


def test_gbm_model_is_the_default_bit_for_bit():
    m = Market.single(100.0, 0.2, 0.03, 0.01)
    put = EuropeanOption(95.0, 1.0, is_call=False)
    assert price_mc(put, m, 10_000, seed=3) == price_mc(put, m, 10_000, seed=3, model=GBMModel())


def test_constant_local_vol_reproduces_gbm():
    """Test 1: constant local vol is GBM, within 3 combined standard errors."""
    m = Market.single(100.0, 0.25, 0.04, 0.02)
    lv = LocalVolModel([Stub(FlatCurve(100.0, 0.04, 0.02), flat(0.25))], dt=1 / 52)
    for opt in (EuropeanOption(100.0, 1.0), EuropeanOption(80.0, 2.0, is_call=False)):
        r = price_mc(opt, m, 100_000, seed=1, model=lv)
        exact = bs_price(100.0, opt.strike, opt.maturity, 0.04, 0.25, 0.02, opt.is_call)
        assert abs(r.price - exact) < 3 * r.std_error

    # the README's two-asset worst-of Phoenix, correlation 0.6
    m2 = Market((Underlying("A", 100.0, 0.25, 0.02), Underlying("B", 50.0, 0.30, 0.015)), 0.04,
                np.array([[1.0, 0.6], [0.6, 1.0]]))
    note = PhoenixAutocallable((100.0, 50.0), tuple(0.25 * k for k in range(1, 13)), 0.02, first_autocall_index=3)
    lv2 = LocalVolModel([Stub(FlatCurve(100.0, 0.04, 0.02), flat(0.25)),
                         Stub(FlatCurve(50.0, 0.04, 0.015), flat(0.30))], dt=1 / 52)
    a, b = price_mc(note, m2, 40_000, seed=5, model=lv2), price_mc(note, m2, 40_000, seed=6)
    assert abs(a.price - b.price) < 3 * np.hypot(a.std_error, b.std_error)


def test_euler_bias_shrinks_when_dt_is_halved():
    """Test 2: with a strongly spot-dependent local vol, the dt bias falls as dt halves."""
    m = Market.single(100.0, 0.3, 0.03)
    stub = Stub(FlatCurve(100.0, 0.03), lambda K, T: np.clip(0.3 * (K / 100.0) ** -1.5, 0.05, 2.0))
    put = EuropeanOption(80.0, 1.0, is_call=False)
    ref = price_mc(put, m, 100_000, seed=1, model=LocalVolModel([stub], dt=1 / 32))
    bias = {}
    for dt in (1 / 2, 1 / 4):
        r = price_mc(put, m, 100_000, seed=2, model=LocalVolModel([stub], dt=dt))
        bias[dt], se = r.price - ref.price, np.hypot(r.std_error, ref.std_error)
    assert abs(bias[1 / 2]) > 6 * se  # the coarse grid is visibly biased...
    assert abs(bias[1 / 4]) < 0.5 * abs(bias[1 / 2])  # ...and halving dt more than halves it


@pytest.mark.parametrize("model", [None, LocalVolModel([Stub(SlopedCurve(100.0), flat(0.2))], dt=1 / 52)])
def test_put_call_parity_under_a_non_flat_curve(model):
    """Test 3: C - P = DF(T) (F(T) - K) path by path in expectation, with curves for forward and discounting."""
    curve = SlopedCurve(100.0)
    m = Market.single(100.0, 0.2, 0.0)
    m = replace(m, curves=(curve,), discount_curve=curve)
    T, K = 2.0, 105.0
    call, put = EuropeanOption(K, T), EuropeanOption(K, T, is_call=False)
    paths = (model or GBMModel()).simulate(m, call.simulation_times, 100_000, seed=4)
    diff = call.discounted_payoff(paths, m) - put.discounted_payoff(paths, m)
    pair = 0.5 * (diff[:50_000] + diff[50_000:])
    se = pair.std(ddof=1) / np.sqrt(len(pair))
    target = curve.discount(T) * (curve.forward(T) - K)
    assert abs(pair.mean() - target) < 3 * max(se, 1e-12)
    assert m.discount(T) == pytest.approx(float(curve.discount(T)), rel=1e-15)


def test_adding_a_date_on_the_grid_keeps_the_random_stream():
    """Common random numbers: a date already on the dt grid must not shift later draws."""
    m = Market.single(100.0, 0.2, 0.02)
    lv = LocalVolModel([Stub(FlatCurve(100.0, 0.02), lambda K, T: 0.2 * (K / 100.0) ** -0.5)], dt=1 / 4)
    a = lv.simulate(m, np.array([0.5, 1.0]), 2_000, seed=9)
    b = lv.simulate(m, np.array([0.25, 0.5, 1.0]), 2_000, seed=9)
    np.testing.assert_array_equal(a, b[:, 1:])
    grid, full = lv.simulate_grid(m, np.array([0.5, 1.0]), 2_000, seed=9)
    np.testing.assert_allclose(grid, [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(full[:, [2, 4]], a)
    assert np.all(full[:, 0] == 100.0)


def test_local_vol_guards():
    m = Market.single(100.0, 0.2, 0.02)
    stub = Stub(FlatCurve(100.0, 0.02), flat(0.2))
    note = PhoenixAutocallable((100.0,), (0.5, 1.0), 0.02, barrier_monitoring="daily")
    with pytest.raises(ValueError, match="flat-vol GBM"):
        price_mc(note, m, 1_000, seed=0, model=LocalVolModel([stub]))
    price_mc(note, m, 1_000, seed=0, model=GBMModel())  # the bridge is valid under GBM
    with pytest.raises(ValueError, match="not finite"):
        price_mc(EuropeanOption(100.0, 0.5), m, 1_000, seed=0,
                 model=LocalVolModel([Stub(stub.curve, lambda K, T: np.where(K > 0, np.nan, 0.2))]))
    with pytest.raises(ValueError, match="2 surfaces"):
        price_mc(EuropeanOption(100.0, 0.5), m, 1_000, seed=0, model=LocalVolModel([stub, stub]))


def test_market_validates_curves():
    with pytest.raises(ValueError, match="one curve per underlying"):
        Market.single(100.0, 0.2, 0.02).__class__(
            (Underlying("S", 100.0, 0.2),), 0.02, curves=(FlatCurve(100.0, 0.02),) * 2)
    m = Market.single(100.0, 0.2, 0.02, 0.01)
    np.testing.assert_allclose(m.log_growth(np.array([1.0, 2.0])), [[0.01], [0.02]], rtol=1e-15)
    paths = simulate_gbm(replace(m, curves=(FlatCurve(100.0, 0.02, 0.01),)), np.array([1.0, 2.0]), 1_000, seed=1)
    np.testing.assert_allclose(paths, simulate_gbm(m, np.array([1.0, 2.0]), 1_000, seed=1), rtol=1e-12)
