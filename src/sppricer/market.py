"""Market data containers: underlyings, flat rate, and correlation."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


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
    """Market snapshot: underlyings, flat continuously compounded rate, correlation."""

    underlyings: tuple[Underlying, ...]
    rate: float
    correlation: np.ndarray | None = None

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
        return np.exp(-self.rate * np.asarray(t))

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
