"""Market data containers: underlyings, rates (flat or curves), and correlation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np


class Curve(Protocol):
    """A forward and discount term structure, e.g. one implied from an option chain.

    forward(T) is the forward price of the asset for delivery at T (so forward(0)
    is its spot); discount(T) is the zero-coupon discount factor to T.
    """

    def forward(self, T): ...

    def discount(self, T): ...


@dataclass(frozen=True)
class Underlying:
    """A single underlying with flat volatility and continuous dividend yield."""

    name: str
    spot: float
    vol: float
    div_yield: float = 0.0

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError(f"{self.name}: spot must be positive")
        if self.vol < 0:
            raise ValueError(f"{self.name}: vol must be non-negative")


@dataclass(frozen=True, eq=False)
class Market:
    """Market snapshot: underlyings, flat continuously compounded rate, correlation.

    Term structure (optional): `curves` gives one Curve per underlying, and the
    asset then grows like its forward, S0 * F(t) / F(0), instead of at r - q; S0
    stays the underlying's spot, so spot bumps work as before. `discount_curve`
    replaces exp(-rate * t) in `discount`. With either set, `rate` and
    `div_yield` no longer enter prices, so `bump_greeks` reports rho = 0.
    """

    underlyings: tuple[Underlying, ...]
    rate: float
    correlation: np.ndarray | None = None
    curves: tuple[Curve, ...] | None = None
    discount_curve: Curve | None = None

    def __post_init__(self) -> None:
        n = len(self.underlyings)
        if n == 0:
            raise ValueError("at least one underlying is required")
        corr = np.eye(n) if self.correlation is None else np.asarray(self.correlation, dtype=float)
        if corr.shape != (n, n):
            raise ValueError(f"correlation must be {n}x{n}, got {corr.shape}")
        if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
            raise ValueError("correlation must be symmetric with unit diagonal")
        try:
            np.linalg.cholesky(corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError("correlation matrix is not positive definite") from exc
        if self.curves is not None:
            if len(self.curves) != n:
                raise ValueError(f"need one curve per underlying ({n}), got {len(self.curves)}")
            object.__setattr__(self, "curves", tuple(self.curves))
        object.__setattr__(self, "underlyings", tuple(self.underlyings))
        object.__setattr__(self, "correlation", corr)

    @classmethod
    def single(cls, spot: float, vol: float, rate: float, div_yield: float = 0.0,
               name: str = "S") -> Market:
        return cls((Underlying(name, spot, vol, div_yield),), rate)

    @property
    def n_assets(self) -> int:
        return len(self.underlyings)

    @property
    def spots(self) -> np.ndarray:
        return np.array([u.spot for u in self.underlyings])

    @property
    def vols(self) -> np.ndarray:
        return np.array([u.vol for u in self.underlyings])

    @property
    def div_yields(self) -> np.ndarray:
        return np.array([u.div_yield for u in self.underlyings])

    def discount(self, t: float | np.ndarray) -> float | np.ndarray:
        if self.discount_curve is not None:
            return self.discount_curve.discount(np.asarray(t, dtype=float))
        return np.exp(-self.rate * np.asarray(t))

    def log_growth(self, t: np.ndarray) -> np.ndarray:
        """ln E[S(t)] / S(0) per asset, shape (len(t), n_assets): ln F(t)/F(0) or (r - q) t."""
        t = np.asarray(t, dtype=float)
        if self.curves is None:
            return (self.rate - self.div_yields)[None, :] * t[:, None]
        return np.stack([np.log(c.forward(t) / c.forward(0.0)) for c in self.curves], axis=1)

    # Bump helpers (used for finite-difference Greeks)
    def _with_underlying(self, i: int, **changes: float) -> Market:
        unds = list(self.underlyings)
        unds[i] = replace(unds[i], **changes)
        return replace(self, underlyings=tuple(unds))

    def with_spot(self, i: int, spot: float) -> Market:
        return self._with_underlying(i, spot=spot)

    def with_vol(self, i: int, vol: float) -> Market:
        return self._with_underlying(i, vol=vol)

    def with_rate(self, rate: float) -> Market:
        return replace(self, rate=rate)
