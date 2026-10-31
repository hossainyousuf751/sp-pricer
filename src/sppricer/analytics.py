"""Closed-form Black-Scholes formulas, used as benchmarks for the MC engine."""

from __future__ import annotations

import math


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _d1_d2(spot, strike, maturity, rate, vol, div_yield):
    sqrt_t = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - div_yield + 0.5 * vol**2) * maturity) / (vol * sqrt_t)
    return d1, d1 - vol * sqrt_t


def bs_price(spot: float, strike: float, maturity: float, rate: float, vol: float,
             div_yield: float = 0.0, is_call: bool = True) -> float:
    d1, d2 = _d1_d2(spot, strike, maturity, rate, vol, div_yield)
    fwd_s = spot * math.exp(-div_yield * maturity)
    pv_k = strike * math.exp(-rate * maturity)
    if is_call:
        return fwd_s * _norm_cdf(d1) - pv_k * _norm_cdf(d2)
    return pv_k * _norm_cdf(-d2) - fwd_s * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, maturity: float, rate: float, vol: float,
             div_yield: float = 0.0, is_call: bool = True) -> float:
    d1, _ = _d1_d2(spot, strike, maturity, rate, vol, div_yield)
    disc_q = math.exp(-div_yield * maturity)
    return disc_q * _norm_cdf(d1) if is_call else disc_q * (_norm_cdf(d1) - 1.0)


def bs_vega(spot: float, strike: float, maturity: float, rate: float, vol: float,
            div_yield: float = 0.0) -> float:
    """Vega per 1.00 change in vol (divide by 100 for per-vol-point)."""
    d1, _ = _d1_d2(spot, strike, maturity, rate, vol, div_yield)
    return spot * math.exp(-div_yield * maturity) * _norm_pdf(d1) * math.sqrt(maturity)
