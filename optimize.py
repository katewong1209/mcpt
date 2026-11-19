"""Parameter selection for the Donchian rule, in-sample and walk-forward.

The sweep scores every lookback on NET performance (after costs), because
that is what would actually be deployed. Selecting the best of many
candidates on the same data inflates the result -- the permutation test
measures exactly that inflation, so the sweep is repeated inside every
permutation.
"""
import numpy as np
import pandas as pd

from .backtest import CostModel, evaluate, prepare
from .strategies import donchian_signal


def sweep_donchian(df: pd.DataFrame, grid: tuple[int, int, int], fill: str,
                   costs: CostModel, stat: str = "pf"):
    """Try every lookback in range(lb_min, lb_max+1, step); keep the best.
    Returns (best_lookback, table of candidates, backtest of the winner).
    Ties go to the smallest lookback (simpler model)."""
    lb_min, lb_max, lb_step = grid
    ctx = prepare(df, fill, costs)          # shared inputs, computed once
    rows, best = [], (None, -np.inf, None)
    for lb in range(lb_min, lb_max + 1, lb_step):
        bt = evaluate(donchian_signal(df["close"], lb), ctx)
        score = bt.stats[f"{stat}_net"]
        score = -np.inf if np.isnan(score) else score   # never traded = worst
        rows.append({"lookback": lb, f"{stat}_net": score,
                     "sharpe_net": bt.stats["sharpe_net"], "n_trades": bt.stats["n_trades_net"]})
        if score > best[1]:
            best = (lb, score, bt)
    if best[0] is None:
        raise ValueError("No lookback produced any trades on this sample.")
    return best[0], pd.DataFrame(rows), best[2]


def walkforward_donchian(df: pd.DataFrame, train: int, step: int, grid: tuple[int, int, int],
                         fill: str, costs: CostModel, stat: str = "pf"):
    """Rolling re-optimization: every `step` bars, pick the lookback using only
    the trailing `train` bars, then trade the next `step` unseen bars with it.
    Returns the stitched out-of-sample signal (NaN before the first training
    window ends) and the (date, lookback) choices for inspection."""
    n = len(df)
    if train + 2 * step > n:
        raise ValueError(f"Need more than {train + 2 * step} bars for this train/step split; got {n}.")
    oos = pd.Series(np.nan, index=df.index)
    choices = []
    for t in range(train, n, step):
        lb, _, _ = sweep_donchian(df.iloc[t - train:t], grid, fill, costs, stat)
        sig = donchian_signal(df["close"], lb)
        oos.iloc[t:min(t + step, n)] = sig.iloc[t:min(t + step, n)]
        choices.append((df.index[t], lb))
    return oos, choices
