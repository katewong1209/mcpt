"""Fast/slow MA crossover — long/flat (optionally long/short)."""
import numpy as np
import pandas as pd


def ma_crossover(ohlc: pd.DataFrame, fast=10, slow=30, allow_short=False):
    c = ohlc["close"]
    f = c.rolling(fast).mean()
    s = c.rolling(slow).mean()
    if allow_short:
        arr = np.where(f > s, 1.0, np.where(f < s, -1.0, 0.0))
    else:
        arr = np.where(f > s, 1.0, 0.0)
    signal = pd.Series(arr, index=ohlc.index)
    r = np.log(c).diff().shift(-1)
    rets = signal * r
    pos = rets[rets > 0].sum()
    neg = rets[rets < 0].abs().sum()
    pf = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else float("nan"))
    return signal, pf


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    if Path("BTCUSD3600.pq").exists():
        df = pd.read_parquet("BTCUSD3600.pq")
    else:
        from market_data import make_synthetic_pair

        df, _ = make_synthetic_pair()
    df.index = df.index.astype("datetime64[s]")
    sig, pf = ma_crossover(df, 10, 30)
    r = np.log(df["close"]).diff().shift(-1)
    strat_r = sig * r
    print("Profit factor:", pf, "Sharpe (hourly):", strat_r.mean() / strat_r.std())
    plt.style.use("dark_background")
    strat_r.cumsum().plot(color="#f7931a")
    plt.title("MA Crossover — Cumulative Log Return")
    plt.show()
