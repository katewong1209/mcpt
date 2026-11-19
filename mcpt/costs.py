# costs.py
import numpy as np, pandas as pd

def cost_per_turn(close, volume, order_qty, fee_bps=5.0, half_spread_bps=2.0, Y=1.0, vol_span=24):
    """Fractional cost of ONE unit of position turnover. Causal: uses info up to bar close."""
    c, v = np.asarray(close, float), np.asarray(volume, float)
    r = np.diff(np.log(c), prepend=np.log(c[0]))
    sigma = pd.Series(r).ewm(span=vol_span, min_periods=2).std().bfill().to_numpy()
    part = order_qty / np.maximum(v, 1e-12)               # participation = Q / bar volume
    impact = Y * sigma * np.sqrt(np.minimum(part, 1.0))   # square-root law
    return (fee_bps + half_spread_bps) / 1e4 + impact

def net_returns(signal, close, cost):
    c = np.asarray(close, float)
    r = np.empty(len(c)); r[:-1] = np.diff(np.log(c)); r[-1] = np.nan  # your engine's convention
    pos = pd.Series(signal).ffill().fillna(0.0).to_numpy()
    turn = np.abs(np.diff(pos, prepend=0.0))              # entry=1, flip=2, hold=0
    return pos * r - turn * cost, pos * r, turn
