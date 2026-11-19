"""Two simple, transparent trading rules. Both are causal: the position for
bar t uses only information available at the close of bar t-1 or earlier.
Output is a target position series: +1 long, -1 short, 0 flat.
"""
import numpy as np
import pandas as pd


def donchian_signal(close: pd.Series, lookback: int) -> pd.Series:
    """Channel breakout ("Turtle" rule).

    Channel = highest / lowest close of the last `lookback` COMPLETED bars.
    Close above the channel -> long; below -> short; otherwise hold the
    previous position. Flat until the first breakout.
    """
    upper = close.rolling(lookback).max().shift(1)
    lower = close.rolling(lookback).min().shift(1)
    event = pd.Series(np.where(close > upper, 1.0, np.where(close < lower, -1.0, np.nan)),
                      index=close.index)
    return event.ffill().fillna(0.0)


def ma_signal(close: pd.Series, fast: int, slow: int, allow_short: bool = False) -> pd.Series:
    """Moving-average crossover. Long when the fast average is above the slow
    average; flat (or short) otherwise. Fixed parameters -- nothing optimized,
    which makes it the clean control case for the permutation test.
    """
    f, s = close.rolling(fast).mean(), close.rolling(slow).mean()
    if allow_short:
        pos = np.where(f > s, 1.0, np.where(f < s, -1.0, 0.0))
    else:
        pos = np.where(f > s, 1.0, 0.0)
    return pd.Series(pos, index=close.index).where(s.notna(), 0.0)
