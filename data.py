"""Load and validate daily OHLCV data.

Conventions used everywhere in this package:
  - one asset per DataFrame, one row per trading day, oldest row first
  - columns: open, high, low, close, volume (volume = shares traded)
  - a DatetimeIndex with no duplicates
"""
import numpy as np
import pandas as pd

REQUIRED = ["open", "high", "low", "close", "volume"]
BARS_PER_YEAR = 252  # US equities: ~252 trading days per year


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check the schema and basic sanity; return a cleaned copy.

    Raises ValueError with a plain message if the data is unusable, so the
    UI can show the problem instead of a stack trace.
    """
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Need {REQUIRED}.")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Index must be dates (parse the timestamp column first).")
    df = df[REQUIRED].apply(pd.to_numeric, errors="coerce")
    df = df[~df.index.duplicated(keep="first")].sort_index().dropna()
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be positive.")
    if (df["volume"] < 0).any():
        raise ValueError("Volume cannot be negative.")
    bad = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | (
           df["low"]  > df[["open", "close"]].min(axis=1) + 1e-9)
    if bad.any():
        raise ValueError(f"{int(bad.sum())} bars have high/low inconsistent with open/close.")
    if len(df) < 60:
        raise ValueError(f"Only {len(df)} usable rows; need at least 60 daily bars.")
    return df


def load_csv(path_or_buffer) -> pd.DataFrame:
    """Read a CSV with a date column (named date/timestamp, or the first column)."""
    df = pd.read_csv(path_or_buffer)
    date_col = next((c for c in df.columns
                     if c.lower() in ("date", "timestamp", "datetime")), df.columns[0])
    df[date_col] = pd.to_datetime(df[date_col], utc=True).dt.tz_localize(None)
    return validate(df.set_index(date_col))


def load_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download split/dividend-adjusted daily bars. Requires `pip install yfinance`."""
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return validate(df)


def make_demo(seed: int = 42, years: int = 10, liquid: bool = True) -> pd.DataFrame:
    """Synthetic daily bars for offline demos. NOT market data.

    Mimics three features of real equity returns that matter for this test:
    volatility clusters, returns have fat tails, and volume rises on big moves.
    `liquid=False` produces a thin small-cap-like name so cost effects are visible.
    """
    rng = np.random.default_rng(seed)
    n = BARS_PER_YEAR * years
    idx = pd.bdate_range(end="2025-12-31", periods=n)

    base_vol = 0.011 if liquid else 0.022
    drift = (0.06 if liquid else 0.04) / BARS_PER_YEAR
    adv_dollars = 5e9 if liquid else 3e6      # typical daily dollar volume
    s0 = 100.0 if liquid else 25.0

    # Volatility regime: an AR(1) in log-volatility makes calm and stormy periods.
    lv = np.zeros(n)
    for t in range(1, n):
        lv[t] = 0.94 * lv[t - 1] + 0.25 * rng.standard_normal()
    sigma = base_vol * np.exp(lv - 0.5 * 0.25**2 / (1 - 0.94**2))

    z = rng.standard_t(5, n) / np.sqrt(5 / 3)          # fat-tailed shocks
    jumps = (rng.random(n) < 0.01) * rng.normal(0, 3 * base_vol, n)
    r = drift + sigma * z + jumps
    r[1:] += 0.06 * r[:-1]                             # slight short-term momentum

    close = s0 * np.exp(np.cumsum(r))
    prev = np.concatenate(([s0], close[:-1]))
    open_ = prev * np.exp(rng.normal(0, 0.25 * base_vol, n))   # overnight gaps
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.5 * sigma)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.5 * sigma)))
    # Volume tracks volatility and the size of today's move (both realistic).
    shares = adv_dollars / s0 * np.exp(lv + 0.7 * np.abs(z) - 0.5)
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": shares}, index=idx)
