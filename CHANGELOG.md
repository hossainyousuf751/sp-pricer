# Changelog

Every change to default prices must be recorded here (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## 0.3.0 (2026-11-14)

### Added
- M5: `lr_greeks(product, market, ...) -> LRGreeks`, likelihood-ratio delta, gamma and vega under GBM
  with standard errors (Broadie and Glasserman 1996). It works on every product unchanged, including
  digital coupons, autocalls and barriers. For continuous or daily monitoring, vega includes the
  Brownian-bridge term. On the README worst-of note, gamma standard error is 2.0× / 3.7× smaller than
  bump-and-reprice; delta and vega stay better by bumping (measured in `examples/greeks_lr_vs_bump.py`).
- **No change to default prices or to `bump_greeks`.**

## 0.2.1 (2026-11-11)

First release on PyPI (`pip install sppricer`). No code changes; prices are identical to 0.2.0.
- MIT licence (`LICENSE`), declared in the package metadata.
- `publish` workflow: a version bump merged to main tags, releases and uploads to PyPI.
- README links are absolute, so they work on PyPI.

## 0.2.0 (2026-11-01)

### Added
- M6: `Curve` protocol and optional `Market(curves=, discount_curve=)`. GBM drifts along each forward
  curve and `market.discount` uses the discount curve. Default flat rate: **no change to default prices.**
- M6: `PathModel` protocol, `GBMModel` (exactly `simulate_gbm`), and `price_mc(..., model=None)` /
  `bump_greeks(..., model=None)`. Default `None`: **no change to default prices** (pinned in
  `tests/test_regression.py`).
- M6: `LocalVolSource` protocol and `LocalVolModel(surfaces, dt=1/252)`:
  - Dupire local vol by log-Euler, with forward-curve drift, Cholesky correlation and the antithetic layout;
  - `simulate_grid` for full-grid paths;
  - `price_mc` raises for non-European barrier monitoring under a non-GBM model, because the bridge
    correction assumes flat vol.
- Regression tests pinning the README numbers; README-scale runs are marked `slow`.
- `barrier_monitoring` (`european` | `continuous` | `daily`) on `PhoenixAutocallable` and
  `BarrierReverseConvertible`, using a Brownian-bridge crossing correction plus the
  Broadie-Glasserman-Kou shift for daily monitoring. Default `european`: **no change to default prices.**
- Tooling: `ruff` lint and coverage reporting in CI.
- Project guidelines: CONTRIBUTING.md, ROADMAP.md, CLAUDE.md.

### Fixed
- Vega used the wrong denominator when the down-bumped vol was floored at zero.
  Affects only vols below the 1-point bump size.

## 0.1.0

- Core engine (correlated GBM, antithetic MC), Phoenix autocallable, barrier reverse
  convertible, European option, bump-and-reprice Greeks.
