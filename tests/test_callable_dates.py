"""autocall_indices: call only on selected observation dates (monthly coupons, quarterly calls)."""

from dataclasses import replace

import numpy as np
import pytest

from sppricer import Market, PhoenixAutocallable, price_mc

MARKET = Market.single(100.0, 0.25, 0.03, 0.01)
MONTHLY = tuple(k / 12 for k in range(1, 25))
NOTE = PhoenixAutocallable((100.0,), MONTHLY, coupon_rate=0.007, memory=False, first_autocall_index=5)


def test_listing_every_date_from_the_first_call_reproduces_the_default():
    every = replace(NOTE, autocall_indices=tuple(range(5, len(MONTHLY) - 1)))
    assert price_mc(every, MARKET, 50_000, seed=3).price == price_mc(NOTE, MARKET, 50_000, seed=3).price


def test_calls_only_on_listed_dates():
    quarterly = replace(NOTE, autocall_indices=(5, 8, 11))          # callable at months 6, 9, 12
    flat = Market.single(100.0, 0.25, 0.0)                          # no discounting: read cash flows directly
    levels = np.full(len(MONTHLY), 90.0)
    levels[6] = 105.0                                               # month 7: above 100% but not callable
    paths = levels[None, :, None]
    assert quarterly.discounted_payoff(paths, flat)[0] == pytest.approx(100 + 0.7 * len(MONTHLY))
    levels[8] = 105.0                                               # month 9: callable, so it is called
    pv = quarterly.discounted_payoff(levels[None, :, None], flat)[0]
    assert pv == pytest.approx(100 + 0.7 * 9)                       # nine coupons, then par
    assert NOTE.discounted_payoff(paths, flat)[0] == pytest.approx(100 + 0.7 * 7)  # default calls at month 7


@pytest.mark.parametrize("bad", [(8, 5), (5, 5), (-1,), (len(MONTHLY) - 1,)])
def test_rejects_unsorted_duplicate_or_out_of_range_indices(bad):
    with pytest.raises(ValueError, match="autocall_indices"):
        replace(NOTE, autocall_indices=bad)
