# sppricer

![tests](https://github.com/hossainyousuf751/sp-pricer/actions/workflows/ci.yml/badge.svg)

Monte Carlo pricing and risk for retail structured products: worst-of Phoenix
autocallables and barrier reverse convertibles. The goal is a transparent,
tested engine that answers a practical question: **what is a structured note
actually worth compared with its issue price?**

## Quickstart

```bash
pip install -e ".[dev]"
pytest
python examples/price_worst_of_phoenix.py
```

```python
import numpy as np
from sppricer import Market, Underlying, PhoenixAutocallable, price_mc, bump_greeks

market = Market(
    (Underlying("STOCK_A", 100.0, 0.25, 0.02), Underlying("STOCK_B", 50.0, 0.30, 0.015)),
    rate=0.04,
    correlation=np.array([[1.0, 0.6], [0.6, 1.0]]),
)
note = PhoenixAutocallable(
    initial_levels=(100.0, 50.0),
    observation_times=tuple(0.25 * k for k in range(1, 13)),
    coupon_rate=0.02, coupon_barrier=0.70, autocall_barrier=1.00,
    protection_barrier=0.60, memory=True, first_autocall_index=3,
)
print(price_mc(note, market, n_paths=400_000, seed=42))
```

Example output (3-year, quarterly, 8% p.a. memory coupon, 60% European protection):

| Metric | Value |
|---|---|
| Fair value per 100 notional | 93.88 ± 0.07 (95% CI) |
| Implied issuer margin | ~6.1 |
| Fair value, barrier on daily closes | 90.58 ± 0.06 |
| Vega per vol point (A / B) | −0.44 / −0.54 |

The investor is short volatility and short gamma, which is the defining risk of
these products: the coupon is compensation for the down-and-in put the investor sells.

## Barrier monitoring

Both structured products take `barrier_monitoring`:

- `"european"` (default): barrier checked at maturity only.
- `"continuous"`: knock-in if any asset ever touches the barrier.
- `"daily"`: knock-in on daily closes, which is what most term sheets specify.

Paths are still simulated only on observation dates. The chance of crossing
between them comes from the Brownian-bridge formula, and `"daily"` adds the
Broadie-Glasserman-Kou barrier shift. The tests check this against a brute-force
daily simulation using the same random paths. Daily monitoring takes about 3.3
points off the example note, so a maturity-only barrier noticeably overstates its value.

## Design

| Module | Responsibility |
|---|---|
| `market.py` | Immutable market snapshot: spots, flat vols, dividend yields, rate, correlation (validated PSD) |
| `models.py` | Correlated GBM via Cholesky, exact log-step on observation dates only, antithetic variates |
| `products/` | Each product declares its simulation dates and a vectorized `discounted_payoff(paths, market)` |
| `engine.py` | MC pricing with correct standard errors under antithetic sampling |
| `greeks.py` | Central-difference Greeks with common random numbers |
| `analytics.py` | Black-Scholes closed forms used as benchmarks |

Adding a product means implementing two methods; the engine and Greeks work unchanged.

## Validation

The test suite checks the engine against cases with known answers:

- European call/put prices converge to Black-Scholes within 4 standard errors.
- Put-call parity holds; antithetic sampling reduces standard error.
- Bump-and-reprice delta and vega match Black-Scholes analytics.
- Limit cases: unreachable barriers give a zero-coupon bond; zero barriers give an immediate
  call; full downside with no coupons gives the discounted forward; a zero-barrier BRC gives a coupon bond.
- Monotonicity: memory coupons are worth more; worst-of is cheaper than single-asset;
  european > daily > continuous barrier monitoring.
- The daily-monitoring Brownian-bridge estimator matches brute-force daily simulation
  (single asset and basket) within 4 standard errors.

## Roadmap

Done: M1 core engine, M2 Greeks, M4 continuous barriers, T1 tooling. Next: issuer funding spread,
variance reduction. Full plan with acceptance criteria in [ROADMAP.md](ROADMAP.md); how to
contribute in [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

Flat volatility and no skew understate the value of the short down-and-in put,
so fair values here are biased upward relative to a skew-aware model (M6).
Dividends are continuous yields, and there is no issuer credit spread (funding
discounting), which matters materially for real notes. For baskets, the
Brownian-bridge correction treats each asset's path between observation dates as
independent, which slightly overstates knock-in on coarse grids (about 0.03 on the test basket).

## Disclaimer

Educational project. All examples use synthetic or publicly available term sheets.
