"""Price a 3-year quarterly worst-of Phoenix autocallable and compute its Greeks."""

from dataclasses import replace

import numpy as np

from sppricer import Market, PhoenixAutocallable, Underlying, bump_greeks, price_mc

market = Market(
    underlyings=(
        Underlying("STOCK_A", spot=100.0, vol=0.25, div_yield=0.02),
        Underlying("STOCK_B", spot=50.0, vol=0.30, div_yield=0.015),
    ),
    rate=0.04,
    correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
)

note = PhoenixAutocallable(
    initial_levels=(100.0, 50.0),
    observation_times=tuple(0.25 * k for k in range(1, 13)),  # quarterly, 3 years
    coupon_rate=0.02,            # 2% per quarter (8% p.a.)
    coupon_barrier=0.70,
    autocall_barrier=1.00,
    protection_barrier=0.60,
    memory=True,
    first_autocall_index=3,      # non-call for the first year
)

result = price_mc(note, market, n_paths=400_000, seed=42)
print(f"Fair value (per 100 notional): {result}")
print(f"Implied issuer margin vs. 100 issue price: {100 - result.price:.2f}")

daily = price_mc(replace(note, barrier_monitoring="daily"), market, n_paths=400_000, seed=42)
print(f"Same note, protection barrier monitored on daily closes: {daily}")

g = bump_greeks(note, market, n_paths=200_000, seed=42)
for i, u in enumerate(market.underlyings):
    print(f"{u.name}: delta={g.delta[i]:+.4f}  gamma={g.gamma[i]:+.5f}  vega/1vol={g.vega[i]:+.4f}")
print(f"rho/1bp = {g.rho:+.5f}")
