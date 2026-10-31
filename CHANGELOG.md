# Changelog

Every change to default prices must be recorded here (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## Unreleased

### Added
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
