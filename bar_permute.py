"""Bar permutation — the MCPT null-hypothesis generator.

Preserves the marginal distribution of returns (mean/std/skew/kurt), intrabar
geometry and the gap distribution; destroys serial dependence (volatility
clustering, long memory, self-similarity, spectral structure). Joint permutation
of a list of markets preserves contemporaneous cross-market correlation.

Vectorized rebuild — mathematically identical to the reference bar-by-bar loop:
    close[i] = close[i-1] + rel_open[k] + rel_close[k]   (in log space)
"""
from typing import List, Union

import numpy as np
import pandas as pd

OHLC = ["open", "high", "low", "close"]


def get_permutation(
    ohlc: Union[pd.DataFrame, List[pd.DataFrame]], start_index: int = 0, seed=None
):
    assert start_index >= 0
    rs = np.random.RandomState(seed)  # same stream as np.random.seed(seed)

    if isinstance(ohlc, list):
        time_index = ohlc[0].index
        for mkt in ohlc:
            assert np.all(time_index == mkt.index), "Indexes do not match"
        markets = ohlc
    else:
        time_index = ohlc.index
        markets = [ohlc]

    n_markets = len(markets)
    n_bars = len(markets[0])
    assert start_index < n_bars - 1, "start_index leaves no bars to permute"

    perm_index = start_index + 1
    perm_n = n_bars - perm_index

    start_bar = np.empty((n_markets, 4))
    rel_o = np.empty((n_markets, perm_n))
    rel_h = np.empty((n_markets, perm_n))
    rel_l = np.empty((n_markets, perm_n))
    rel_c = np.empty((n_markets, perm_n))

    for m, bars in enumerate(markets):
        lb = np.log(bars[OHLC].to_numpy(dtype=float))
        o, h, l, c = lb[:, 0], lb[:, 1], lb[:, 2], lb[:, 3]
        start_bar[m] = lb[start_index]
        r_o = o - np.concatenate(([np.nan], c[:-1]))  # gap: open vs last close
        rel_o[m] = r_o[perm_index:]
        rel_h[m] = (h - o)[perm_index:]
        rel_l[m] = (l - o)[perm_index:]
        rel_c[m] = (c - o)[perm_index:]

    idx = np.arange(perm_n)
    perm1 = rs.permutation(idx)  # intrabar shape shuffled jointly
    rel_h = rel_h[:, perm1]
    rel_l = rel_l[:, perm1]
    rel_c = rel_c[:, perm1]
    perm2 = rs.permutation(idx)  # gaps shuffled separately
    rel_o = rel_o[:, perm2]

    out = []
    for m, bars in enumerate(markets):
        lb = np.log(bars[OHLC].to_numpy(dtype=float))
        pb = np.empty((n_bars, 4))
        pb[:start_index] = lb[:start_index]  # real history kept intact
        pb[start_index] = start_bar[m]

        closes = start_bar[m, 3] + np.cumsum(rel_o[m] + rel_c[m])
        prev_c = np.concatenate(([start_bar[m, 3]], closes[:-1]))
        opens = prev_c + rel_o[m]
        pb[perm_index:, 0] = opens
        pb[perm_index:, 1] = opens + rel_h[m]
        pb[perm_index:, 2] = opens + rel_l[m]
        pb[perm_index:, 3] = closes

        out.append(
            pd.DataFrame(np.exp(pb), index=time_index, columns=OHLC)
        )

    return out if n_markets > 1 else out[0]
