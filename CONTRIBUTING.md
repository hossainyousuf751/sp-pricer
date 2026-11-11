# Contributing to sppricer

These rules apply to every change, whether a person or an AI agent makes it.
What to build next is in [ROADMAP.md](ROADMAP.md). This file covers how to build it.

## Principles

1. **Every number must be checkable.** Each pricing feature needs a test that compares it
   with something known independently: a closed form, a limit case, a brute-force simulation,
   or a monotonicity relationship. Agreeing with itself doesn't count.
2. **Defaults don't move prices.** A new modelling option ships with a default that reproduces
   the old results exactly. For example, `barrier_monitoring="european"` stayed the default when
   daily monitoring was added. Changing a default is its own decision, with its own changelog entry.
3. **Name your approximations.** Any shortcut with a known limit gets a `ponytail:` comment in
   the code that names the limit and how to remove it, plus a line under README "Limitations".
4. **Keep it small and dependency-light.** The only runtime dependency is `numpy`. Adding another
   needs a written reason in the PR, and `scipy` counts. Don't add an abstraction until a second
   use case exists.
5. **One milestone or fix per change.** Don't mix refactors with new features.

## Setup

```bash
pip install -e ".[dev]"
pytest -q                                  # whole suite, should finish in under 30 s
python examples/price_worst_of_phoenix.py
```

Supported Python versions are 3.10 to 3.12, all tested in CI. Don't use syntax newer than 3.10.

## Code conventions

Follow the code that's already there:

- **Layout:** library code lives in `src/sppricer/`, each product gets one module in `products/`,
  tests go in `tests/`, and runnable scripts go in `examples/`.
- **Style:** start every module with a one-line docstring and `from __future__ import annotations`,
  add type hints on public functions, and keep lines to 120 characters at most.
- **Data types:** market data and products are `@dataclass(frozen=True)`. Validate inputs in
  `__post_init__` and raise `ValueError` with a message that names the bad field.
- **Products:** subclass `Product` and implement only `simulation_times` and `discounted_payoff`.
  Payoffs must be vectorised over paths. Looping over observation dates is fine; looping over
  paths in Python is not.
- **Arrays:** paths have shape `(n_paths, n_times, n_assets)`, and `discounted_payoff` returns
  `(n_paths,)`.
- **Reuse before writing:** `validate_schedule`, `worst_performance`, `knock_in_probability`,
  `validate_monitoring` and the `Market.with_*` bump helpers already exist. Use them.
- **Public API:** anything a user is meant to import goes into `sppricer/__init__.py` and `__all__`.

### Units (fixed across the whole library)

| Quantity | Convention |
|---|---|
| Time | Year fractions from valuation date (`0.25` = 3 months) |
| Rates, dividend yields | Continuously compounded, annualised |
| Barriers, strikes in products | Fraction of initial level (`0.70` = 70%) |
| `coupon_rate` | Per observation period, not per annum |
| Notional | `100.0` by default; prices are per 100 notional |
| Vega | Per 1 vol point (0.01) |
| Rho | Per 1 basis point |
| Delta / gamma | Per 1 unit of spot |

A change that needs a different convention must update this table and every module that uses it.

## Testing standard

- **Tolerances are in standard errors:** `abs(mc - exact) < 4 * std_error`. Never add a fixed
  allowance (`+ 0.05`) to make a test pass. If the test fails, find out why.
- **Exact limit cases** (e.g. unreachable barriers give a zero-coupon bond) assert with
  `pytest.approx(..., abs=1e-10)`.
- **Comparing two estimators:** run both on the same simulated paths and test the paired
  difference, as in `tests/test_barriers.py`.
- **Seeds:** always pass a fixed `seed`. Pick paths so each test runs in about 1 s.
- **Each new option needs:** a limit-case test, a brute-force or closed-form comparison where one
  exists, an ordering check where theory gives one, and an invalid-input test.
- **Report the real error:** when adding an approximation, measure its bias against brute force
  and write the number in the PR and the README.

## Definition of done

A change is complete when all of these hold:

- [ ] `pytest -q` passes locally and the suite stays under 30 s.
- [ ] New behaviour has tests that meet the standard above.
- [ ] Existing default prices are unchanged, or the change is called out as deliberate.
- [ ] `python examples/price_worst_of_phoenix.py` runs. If prices moved, the README example
      table has been regenerated from its output.
- [ ] README is updated: Design table (new module), Validation (new checks), Limitations
      (new approximations).
- [ ] ROADMAP.md shows the milestone's status, with acceptance criteria ticked.
- [ ] No new runtime dependency, or a written reason for one.

## Releasing

Versions follow semantic versioning in `pyproject.toml`. A release is one PR: add a `## X.Y.Z (date)`
section to `CHANGELOG.md` and bump `version`. When it merges, `.github/workflows/publish.yml` builds the
package, creates the `vX.Y.Z` tag and GitHub release (notes = that changelog section) and uploads to PyPI.
It does nothing if the version is already tagged, and stops before releasing anything if the changelog
section is missing. PyPI versions cannot be re-uploaded, so a mistake means a new patch version.

One-time PyPI setup (trusted publishing, no token): on pypi.org, *Your account → Publishing → Add a new
pending publisher* with project `sppricer`, owner `hossainyousuf751`, repository `sp-pricer`, workflow
`publish.yml`, environment `pypi`.

## Git workflow

- Branch per change: `m3-control-variates`, `fix-vega-floor`.
- Commit messages: an imperative subject of 72 characters at most (`Add daily barrier monitoring`),
  with the reason in the body.
- PR description: what changed, how it was validated (with measured numbers), and any price moves.
