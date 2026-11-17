"""Seeded synthetic BTC/ETH hourly OHLC feed used when real parquet files are absent.

GARCH(1,1) volatility clustering, fat-tailed t(4) innovations, slow AR(1) trend
regimes, rare jumps, correlated cross-market shocks, realistic intrabar wicks.
"""
import numpy as np
import pandas as pd


def _garch(z, omega, alpha, beta, uncond):
    n = len(z)
    s2 = np.empty(n)
    s2[0] = uncond
    for i in range(1, n):
        s2[i] = omega + alpha * (z[i - 1] ** 2) * s2[i - 1] + beta * s2[i - 1]
    return np.sqrt(s2)


def _ohlc_from_rets(index, rets, sigma, s0, gaps, wick_u, wick_d):
    close = s0 * np.exp(np.cumsum(rets))
    prev = np.concatenate(([s0], close[:-1]))
    op = prev * np.exp(gaps)
    hi = np.maximum(op, close) * np.exp(wick_u * sigma)
    lo = np.minimum(op, close) * np.exp(-wick_d * sigma)
    return pd.DataFrame(
        {"open": op, "high": hi, "low": lo, "close": close}, index=index
    )


def make_synthetic_pair(seed: int = 7):
    index = pd.date_range("2016-01-01", "2021-01-01", freq="h", inclusive="left")
    index = index.astype("datetime64[s]")
    n = len(index)
    rng = np.random.default_rng(seed)

    z_b = rng.standard_t(4, n) / np.sqrt(2.0)
    rho = 0.62
    z_e = rho * z_b + np.sqrt(1 - rho**2) * (rng.standard_t(4, n) / np.sqrt(2.0))

    uncond_b = 0.006**2
    alpha, beta = 0.08, 0.90
    sig_b = _garch(z_b, uncond_b * (1 - alpha - beta), alpha, beta, uncond_b)
    uncond_e = (0.006 * 1.35) ** 2
    sig_e = _garch(z_e, uncond_e * (1 - alpha - beta), alpha, beta, uncond_e)

    trend = np.empty(n)
    t = 0.0
    shocks = rng.normal(0, 0.02, n)
    for i in range(n):
        t = 0.9995 * t + shocks[i]
        trend[i] = t

    jump_b = (rng.random(n) < 0.0015) * rng.normal(0, 0.035, n)
    jump_e = (rng.random(n) < 0.0018) * rng.normal(0, 0.045, n)

    rets_b = 0.000012 + 0.00045 * trend + sig_b * z_b + jump_b
    rets_e = 0.000010 + 0.00060 * trend + sig_e * z_e + jump_e

    gaps_b = rng.normal(0, 0.0004, n)
    gaps_e = rng.normal(0, 0.0005, n)
    wu_b, wd_b = np.abs(rng.normal(0, 0.55, n)), np.abs(rng.normal(0, 0.55, n))
    wu_e, wd_e = np.abs(rng.normal(0, 0.60, n)), np.abs(rng.normal(0, 0.60, n))

    btc = _ohlc_from_rets(index, rets_b, sig_b, 430.0, gaps_b, wu_b, wd_b)
    eth = _ohlc_from_rets(index, rets_e, sig_e, 8.5, gaps_e, wu_e, wd_e)
    return btc, eth
