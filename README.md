# sppricer

![tests](https://github.com/hossainyousuf751/sp-pricer/actions/workflows/ci.yml/badge.svg)

Monte Carlo pricing and risk for retail structured products: worst-of Phoenix
autocallables and barrier reverse convertibles. The goal is a transparent,
tested engine that answers a practical question: **what is a structured note
actually worth compared with its issue price?**

## Quickstart

```bash
pip install sppricer                       # from PyPI
pip install -e ".[dev]"                    # or, from a clone, for development
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

## Local volatility and term structure (v0.2, M6)

Every product here is short a down-and-in put, so its value depends on the put skew, which flat
vol ignores. `price_mc(product, market, model=...)` takes any `PathModel`: `GBMModel` (the default,
identical numbers) or `LocalVolModel`, a Dupire (1994) local-vol simulation.

```python
from sppricer import LocalVolModel, price_mc

model = LocalVolModel([surface_a, surface_b], dt=1 / 252)  # any objects with local_vol(K, T) and .curve
price_mc(note, market, n_paths=200_000, seed=42, model=model)
```

- **No dependency on a surface library.** A surface is anything with `local_vol(K, T)` (absolute
  strike, years) and a `curve` with `forward(T)` / `discount(T)` (the `LocalVolSource` protocol).
  [quant-stack](https://github.com/hossainyousuf751/quant-stack) wires in volsurface's SSVI surfaces.
- **Scheme.** Log-Euler on a grid of every dt plus every observation date. Each step drifts by
  ln F(t+dt) − ln F(t) − ½σ²dt, so the asset tracks its forward curve, with σ = local vol at the
  current spot and the step's mid-time. Correlation uses the market's Cholesky factor, and
  antithetic sampling is kept, so standard errors and common random numbers work as before.
  The bias is O(dt): the tests check that it shrinks when dt halves.
- **Term structure.** `Market(..., curves=(...), discount_curve=...)` replaces the flat rate and
  dividend yield by forward and discount curves. GBM then drifts along each asset's forward, and
  products discount with `market.discount`.
- **Barriers.** Daily or continuous monitoring uses a Brownian-bridge correction that assumes
  flat GBM vol between dates, so `price_mc` raises under local vol. With local vol, use
  `"european"` monitoring, or `LocalVolModel.simulate_grid` to monitor on every step.

## Likelihood-ratio Greeks (v0.3, M5)

Bump-and-reprice gamma is noisy for these notes: the coupon, the autocall and the barrier are all
digital, so a 1% spot bump moves only the few paths near a trigger. `lr_greeks` uses the
likelihood-ratio method (Broadie and Glasserman 1996). It differentiates the density of the simulated
path rather than the payoff, so it works on every product unchanged, and each Greek comes with a
standard error.

```python
from sppricer import lr_greeks
g = lr_greeks(note, market, n_paths=200_000, seed=42)
g.gamma, g.gamma_se           # also g.delta, g.vega and their _se, in the units of bump_greeks
```

Measured on the worst-of Phoenix above, 200,000 paths per run (`examples/greeks_lr_vs_bump.py`). Bump
standard errors are the spread over 20 seeds, because bump-and-reprice gives none:

| Greek, asset A / B | Bump SE | LR SE | Better |
|---|---|---|---|
| gamma | 0.0083 / 0.0426 | 0.0042 / 0.0115 | **LR, 2.0× / 3.7× smaller** |
| delta | 0.0035 / 0.0073 | 0.0061 / 0.0102 | bump |
| vega per vol point | 0.0047 / 0.0046 | 0.066 / 0.055 | bump, about 10× |

Use `lr_greeks` for gamma and `bump_greeks` for delta and vega. The vega weight sums noise over all
12 observation steps; delta is smooth enough that common random numbers already make bumping
effectively pathwise. An LR run also costs one simulation instead of eleven, so it was about 6×
faster here (0.6 s against 3.5 s). On this note LR's reported gamma SE is conservative: the spread
over seeds is about 0.7 of it. The estimator is validated against closed forms: call and digital
delta and gamma, correlated baskets, multi-step vega, and the Brownian-bridge vega term.

## Design

| Module | Responsibility |
|---|---|
| `market.py` | Immutable market snapshot: spots, flat vols, dividend yields, rate (or forward/discount curves), correlation (validated PSD) |
| `models.py` | Correlated GBM via Cholesky, exact log-step on observation dates only, antithetic variates; Dupire local vol (log-Euler) behind the same `PathModel` interface |
| `products/` | Each product declares its simulation dates and a vectorized `discounted_payoff(paths, market)` |
| `engine.py` | MC pricing with correct standard errors under antithetic sampling |
| `greeks.py` | Central-difference Greeks with common random numbers; likelihood-ratio delta, gamma and vega with standard errors |
| `analytics.py` | Black-Scholes closed forms used as benchmarks |

Adding a product means implementing two methods; the engine and Greeks work unchanged.

## Validation

The test suite checks the engine against cases with known answers:

- European call/put prices converge to Black-Scholes within 4 standard errors.
- Put-call parity holds; antithetic sampling reduces standard error.
- Bump-and-reprice delta and vega match Black-Scholes analytics.
- Likelihood-ratio delta, gamma and vega match Black-Scholes (call) and closed-form digital Greeks
  within 4 standard errors, with z-scores over 10 fresh seeds averaging about 0 with spread about 1.
- Limit cases: unreachable barriers give a zero-coupon bond; zero barriers give an immediate
  call; full downside with no coupons gives the discounted forward; a zero-barrier BRC gives a coupon bond.
- Monotonicity: memory coupons are worth more; worst-of is cheaper than single-asset;
  european > daily > continuous barrier monitoring.
- The daily-monitoring Brownian-bridge estimator matches brute-force daily simulation
  (single asset and basket) within 4 standard errors.

## Roadmap

Done: M1 core engine, M2 Greeks, M4 continuous barriers, M5 likelihood-ratio Greeks, M6 local vol and
curves, T1 tooling.
Next: issuer funding spread, variance reduction. Full plan with acceptance criteria in [ROADMAP.md](https://github.com/hossainyousuf751/sp-pricer/blob/main/ROADMAP.md); how to
contribute in [CONTRIBUTING.md](https://github.com/hossainyousuf751/sp-pricer/blob/main/CONTRIBUTING.md).

## Limitations

Under the default flat volatility, the skew is ignored, which understates the value of the
short down-and-in put. Fair values are therefore biased upward relative to a skew-aware model.
`LocalVolModel` removes that bias given a surface. Under local vol, `bump_greeks` vega is zero,
so bump the surface instead. Barrier monitoring must be European.
Dividends are continuous yields, and there is no issuer credit spread (funding
discounting), which matters materially for real notes. For baskets, the
Brownian-bridge correction treats each asset's path between observation dates as
independent, which slightly overstates knock-in on coarse grids (about 0.03 on the test basket).

## Part of quant-stack

sp-pricer is one of five independent libraries linked in
[quant-stack](https://github.com/hossainyousuf751/quant-stack). There, volsurface surfaces drive
`LocalVolModel` (study S1, skew and the autocallable), and risk-lab revalues sp-pricer notes for
VaR. sp-pricer itself depends only on numpy.

## Disclaimer

Educational project. All examples use synthetic or publicly available term sheets.
