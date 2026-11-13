# Roadmap

The single source of truth for what gets built and in what order. Each milestone lists its goal,
scope and acceptance criteria. A milestone is done when every criterion is ticked and the
[definition of done](CONTRIBUTING.md#definition-of-done) holds.

**Status:** ✅ done · 🔜 next · ⬜ planned

## Priority order

1. ✅ **T1** Repository and tooling
2. ✅ **M6** Realistic dynamics (skew): moved ahead of M8 (2026-11-01) because quant-stack's
   local-vol link needed it, and skew moves the example note by 2.5–5 points, more than funding
3. 🔜 **M8** Issuer funding spread
4. ⬜ **M3** Variance reduction
5. ✅ **M5** Better Greeks
6. ⬜ **M9** Dates and calendars
7. ⬜ **M7** Term-sheet case studies (needs M6, M8 and M9)

Change this order only with a note explaining why.

---

## ✅ M1 Core engine

Correlated GBM, Phoenix autocallable, barrier reverse convertible, Monte Carlo with correct
standard errors under antithetic sampling.

## ✅ M2 Greeks

Central-difference delta, gamma, vega and rho with common random numbers.

## ✅ M4 Continuous barriers

`barrier_monitoring` = `european` | `continuous` | `daily` on both structured products, using a
Brownian-bridge crossing probability plus the Broadie-Glasserman-Kou shift for daily monitoring.

- [x] Daily estimator matches brute-force daily simulation within 4 SE (single asset and basket)
- [x] european > daily > continuous price ordering
- [x] Default (`european`) prices unchanged

Known limit: bridges are treated as independent across assets (about 0.03 bias on the test basket).

---

## ✅ T1 Repository and tooling

**Goal:** every change is versioned and checked automatically.

- [x] `git init`, first commit, push to GitHub; fix the `<your-username>` badge in README
- [x] Add `ruff` (lint and import sorting) to `[dev]` and CI, configured with line length 120
- [x] Add `pytest-cov` to CI and report coverage, without failing the build on it yet (95% at T1)
- [x] Add a `CHANGELOG.md` that records every change to default prices

## 🔜 M8 Issuer funding spread

**Goal:** discount the note's cash flows at the issuer's funding curve, not the risk-free rate.
On real notes this is worth 1–4 points.

**Scope:** add a flat `funding_spread` (continuously compounded) used only for discounting note
cash flows. Simulation drift stays at the risk-free rate.

- [ ] `funding_spread=0` reproduces current prices exactly
- [ ] Zero-coupon limit: unreachable barriers give `N * exp(-(r + s) * T)` to 1e-10
- [ ] Price strictly decreases as the spread increases
- [ ] Vanilla options are unaffected (they are not issuer obligations), or this is documented
- [ ] Rho is unchanged in meaning; add a separate spread sensitivity (per bp) to `Greeks`
- [ ] README example shows fair value at a realistic spread (e.g. 100 bp)

## ⬜ M3 Variance reduction

**Goal:** reach the same accuracy with several times fewer paths.

- [ ] Control variate: the European-barrier version of the same product (its price is cheap
      to get) or a discounted forward; optimal coefficient estimated per run
- [ ] Standard error falls at least 3× on the README example, at the same price within 4 SE
- [ ] Sobol + Brownian bridge path construction as an option, using numpy only unless a written
      reason justifies `scipy.stats.qmc`
- [ ] Convergence plot script in `examples/`; matplotlib stays optional, never a runtime dependency

## ✅ M6 Realistic dynamics

**Goal:** stop undervaluing the down-and-in put caused by flat vol.

- [x] Term structure with flat as the default: any `Curve` (forward, discount) per underlying, and a
      discount curve on `Market`. This is more general than piecewise-flat.
- [x] Local vol behind the same paths interface: `PathModel`, `GBMModel`, `LocalVolModel`, and
      `price_mc(model=)`. Heston is not done.
- [x] Vanilla prices match the calibrated surface within 3 SE. This is tested in
      [quant-stack](https://github.com/hossainyousuf751/quant-stack) against a volsurface SSVI surface,
      because sp-pricer has no surface of its own. Here, a constant local vol matches Black-Scholes.
- [x] Flat-vol limit reproduces GBM prices (`tests/test_models.py`)
- [x] Skew effect on the example note: quant-stack study S1

## ✅ M5 Better Greeks

**Goal:** stable gamma and vega for digital and barrier features.

- [x] Delta and vega for smooth payoffs matching Black-Scholes within 4 SE. Met by `lr_greeks` (tested).
  No separate pathwise estimator was added: for smooth payoffs `bump_greeks` with common random numbers
  is the pathwise estimator in the limit of small bumps, and it measured lower variance than
  likelihood ratio for delta and vega.
- [x] Likelihood-ratio estimator for digital coupons: `lr_greeks`, matching closed-form digital delta
  and gamma within 4 SE.
- [x] Gamma standard error on the README example falls 2.0× / 3.7× versus bump-and-reprice
  (`examples/greeks_lr_vs_bump.py`, 20 seeds).

## ⬜ M9 Dates and calendars

**Goal:** price from actual term-sheet dates.

- [ ] Accept `datetime.date` schedules, converted to year fractions with ACT/365F by default
- [ ] Day-count conventions: ACT/365F, ACT/360, 30/360
- [ ] Business-day adjustment with a simple holiday list (no new dependency)
- [ ] Year-fraction inputs keep working unchanged

## ⬜ M7 Term-sheet case studies

**Goal:** answer the project's question on real notes: what is the issuer margin?

- [ ] At least 3 public term sheets encoded in `examples/` with sources cited
- [ ] Margin analysis versus barrier, coupon and correlation
- [ ] Only synthetic or publicly available term sheets (see the README disclaimer)
