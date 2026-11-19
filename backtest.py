"""Turn a position series into returns, with and without transaction costs.

Timing convention (no look-ahead):
  signal[t] is decided at the close of bar t.
  fill="close":     assume a fill at that close (optimistic paper backtest);
                    the position earns the next bar's close-to-close return.
  fill="next_open": trade at the NEXT bar's open (realistic); the position
                    earns that bar's open-to-close return and misses the
                    overnight gap. The difference vs "close" is the delay cost.

Cost model, charged only when the position CHANGES (turnover), never while holding:
  cost = turnover * (spread + fees) + turnover * Y * sigma * sqrt(participation)
  participation = order notional / trailing average daily dollar volume
The square-root term is the standard market-impact rule of thumb
(Almgren et al. 2005); Y is of order 1 and should be calibrated on real fills.
All returns are log returns on capital, so costs subtract directly.

Structure: `prepare` computes shared inputs once (returns, volatility, volume);
`evaluate` scores one signal against them. The parameter sweep calls `prepare`
once and `evaluate` per candidate, which keeps permutations fast.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

BARS_PER_YEAR = 252


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 2.0       # half-spread paid per unit traded, basis points
    fee_bps: float = 0.5          # commission + fees per unit traded, basis points
    impact_y: float = 1.0         # impact multiplier Y in the square-root rule
    capital: float = 1_000_000    # strategy size in dollars (sets participation)
    window: int = 20              # trailing bars for volatility / volume estimates


@dataclass
class Backtest:
    gross: np.ndarray             # per-bar log returns before costs
    net: np.ndarray               # per-bar log returns after costs
    turnover: np.ndarray          # |position change| per bar, fraction of capital
    cost: np.ndarray              # per-bar cost, log-return units
    stats: dict


def prepare(df: pd.DataFrame, fill: str, costs: CostModel) -> dict:
    """Inputs shared by every candidate signal on this data. Trailing
    estimates are shifted one bar so they are known at decision time."""
    close = df["close"]
    r_cc = np.log(close).diff().fillna(0.0)                       # close-to-close
    ret = (r_cc if fill == "close" else np.log(close / df["open"])).to_numpy()
    w = costs.window
    sigma = r_cc.rolling(w, min_periods=5).std().shift(1).to_numpy()
    adv = (close * df["volume"]).rolling(w, min_periods=5).mean().shift(1).to_numpy()
    return {"ret": ret, "sigma": np.nan_to_num(sigma), "adv": adv,
            "base": (costs.spread_bps + costs.fee_bps) / 1e4,
            "y": costs.impact_y, "capital": costs.capital}


def metrics(r: np.ndarray, turnover: np.ndarray, cost: np.ndarray, gross: np.ndarray) -> dict:
    """Summary statistics for one return series."""
    wins, losses = r[r > 0].sum(), -r[r < 0].sum()
    pf = wins / losses if losses > 0 else (np.inf if wins > 0 else np.nan)
    sd = r.std(ddof=1)
    sharpe = r.mean() / sd * np.sqrt(BARS_PER_YEAR) if sd > 0 else np.nan
    eq = np.cumsum(r)
    maxdd = float((eq - np.maximum.accumulate(eq)).min())
    traded = turnover.sum()
    return {
        "pf": float(pf), "sharpe": float(sharpe), "total_return": float(eq[-1]),
        "max_drawdown": maxdd,
        "n_trades": int((turnover > 1e-12).sum()),
        "turnover_per_year": float(traded / (len(r) / BARS_PER_YEAR)),
        "cost_total": float(cost.sum()),
        "cost_bps_per_trade": float(cost.sum() / traded * 1e4) if traded > 0 else 0.0,
        # Constant cost per unit traded that would erase the gross edge.
        "breakeven_bps": float(gross.sum() / traded * 1e4) if traded > 0 else np.nan,
    }


def evaluate(signal, ctx: dict, eval_start: int | None = None) -> Backtest:
    """Score one position series against prepared inputs. Undefined signal
    (NaN) means flat. `eval_start` restricts scoring to bars >= eval_start
    (walk-forward scores only out-of-sample bars); costs still use the full
    history, as they would in production."""
    sig = np.nan_to_num(np.asarray(signal, dtype=float))
    held = np.concatenate(([0.0], sig[:-1]))          # position held during bar t
    gross = held * ctx["ret"]
    turnover = np.abs(np.diff(held, prepend=0.0))
    part = np.divide(turnover * ctx["capital"], ctx["adv"],
                     out=np.zeros_like(turnover), where=ctx["adv"] > 0)
    impact = ctx["y"] * ctx["sigma"] * np.sqrt(np.maximum(part, 0.0))
    cost = turnover * ctx["base"] + turnover * impact
    net = gross - cost
    s = slice(eval_start or 0, None)
    sg = metrics(gross[s], turnover[s], np.zeros_like(cost[s]), gross[s])
    sn = metrics(net[s], turnover[s], cost[s], gross[s])
    stats = {f"{k}_gross": v for k, v in sg.items()} | {f"{k}_net": v for k, v in sn.items()}
    return Backtest(gross, net, turnover, cost, stats)


def backtest(df: pd.DataFrame, signal, fill: str = "next_open",
             costs: CostModel = CostModel(), eval_start: int | None = None) -> Backtest:
    """Convenience wrapper: prepare inputs and evaluate one signal."""
    return evaluate(signal, prepare(df, fill, costs), eval_start)
