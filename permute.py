"""Build control datasets by shuffling price bars ("bar permutation").

Why: a backtest result only means something if the same strategy does clearly
worse on data that resembles the real market but contains no exploitable
pattern. We create that data by cutting each bar into reusable pieces and
dealing them out in random order.

Each bar t (in log prices) is split into four pieces:
  gap = log(open_t) - log(close_{t-1})   overnight move
  h   = log(high_t) - log(open_t)        upper wick
  l   = log(low_t)  - log(open_t)        lower wick
  c   = log(close_t)- log(open_t)        intraday move

The (h, l, c, volume) group is shuffled together, so every synthetic bar keeps
a realistic shape and its original volume. Gaps are shuffled separately.
Bars up to `start` are left untouched (walk-forward keeps its first training
window real). Rebuilding is a cumulative sum, so it is fully vectorized.

What survives: the distribution of returns, bar shapes, gaps, and the
return-volume link within a bar. What is destroyed: time-order patterns --
trends, momentum, volatility clustering -- exactly what a trend rule claims
to exploit.

Reference: Aronson (2007), Evidence-Based Technical Analysis; Masters (2011),
Permutation and Randomization Tests for Trading System Development.
"""
import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def permute_bars(df: pd.DataFrame, start: int = 0, seed: int | None = None) -> pd.DataFrame:
    """Return one shuffled copy of `df`: same index/columns, new price path.
    Shuffling begins at bar start+1; earlier bars are kept as-is.
    """
    rng = np.random.RandomState(seed)
    logp = np.log(df[OHLC].to_numpy(dtype=float))
    o, h, l, c = logp.T
    n = len(df)
    if not 0 <= start < n - 1:
        raise ValueError("start must leave at least one bar to shuffle")
    i0 = start + 1

    gap = (o - np.concatenate(([np.nan], c[:-1])))[i0:]  # overnight pieces
    rel_h, rel_l, rel_c = (h - o)[i0:], (l - o)[i0:], (c - o)[i0:]
    has_vol = "volume" in df.columns
    vol = df["volume"].to_numpy(dtype=float) if has_vol else None

    order_bars = rng.permutation(n - i0)   # one order for the whole bar body
    order_gaps = rng.permutation(n - i0)   # independent order for gaps
    rel_h, rel_l, rel_c = rel_h[order_bars], rel_l[order_bars], rel_c[order_bars]
    gap = gap[order_gaps]

    # Rebuild: each close = previous close + gap + intraday move (log space).
    new_c = c[start] + np.cumsum(gap + rel_c)
    new_o = np.concatenate(([c[start]], new_c[:-1])) + gap
    out = np.empty_like(logp)
    out[:i0] = logp[:i0]                   # untouched history
    out[i0:] = np.column_stack([new_o, new_o + rel_h, new_o + rel_l, new_c])

    res = pd.DataFrame(np.exp(out), index=df.index, columns=OHLC)
    if has_vol:
        res["volume"] = vol.copy()
        res.iloc[i0:, res.columns.get_loc("volume")] = vol[i0:][order_bars]
    return res[df.columns]
