"""Donchian breakout strategy: reference implementation + vectorized Turbo engine.

Reference functions mirror the original research code exactly. The Turbo engine
computes identical profit factors via suffix-extreme matrices (all lookbacks'
rolling max/min in two passes) and segment cumulative sums (PF without building
per-bar return arrays).
"""
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def profit_factor_np(rets) -> float:
    r = np.asarray(rets, dtype=float)
    pos = r[r > 0].sum()
    neg = -r[r < 0].sum()
    if neg > 0:
        return float(pos / neg)
    return float("inf") if pos > 0 else float("nan")


def _log_fwd_returns(close):
    lc = np.log(np.asarray(close, dtype=float))
    r = np.empty(len(lc))
    r[:-1] = np.diff(lc)
    r[-1] = np.nan
    return r


def donchian_breakout(ohlc: pd.DataFrame, lookback: int):
    upper = ohlc["close"].rolling(lookback - 1).max().shift(1)
    lower = ohlc["close"].rolling(lookback - 1).min().shift(1)
    signal = pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index)
    signal.loc[ohlc["close"] > upper] = 1
    signal.loc[ohlc["close"] < lower] = -1
    return signal.ffill()


def breakout_signal_np(close, lookback: int):
    c = np.asarray(close, dtype=float)
    s = pd.Series(c)
    up = s.rolling(lookback - 1).max().shift(1).to_numpy()
    dn = s.rolling(lookback - 1).min().shift(1).to_numpy()
    sig = np.where(c > up, 1.0, np.where(c < dn, -1.0, np.nan))
    return pd.Series(sig).ffill().to_numpy()


def optimize_donchian(ohlc: pd.DataFrame, lb_min=12, lb_max=168, lb_step=1):
    """Reference optimizer — literal pandas loop over the lookback grid."""
    best_pf, best_lookback = 0.0, -1
    r = np.log(ohlc["close"]).diff().shift(-1)
    for lookback in range(lb_min, lb_max + 1, lb_step):
        signal = donchian_breakout(ohlc, lookback)
        sig_rets = signal * r
        pf = profit_factor_np(sig_rets.to_numpy())
        if pf > best_pf:
            best_pf, best_lookback = pf, lookback
    return best_lookback, best_pf


def optimize_donchian_fast(close, lb_min=12, lb_max=168, lb_step=1):
    """Turbo optimizer. Returns (best_lookback, best_pf, lookbacks, pf_curve)."""
    c = np.asarray(close, dtype=float)
    n = len(c)
    lookbacks = np.arange(lb_min, lb_max + 1, lb_step, dtype=int)
    W = lb_max - 1

    r = np.diff(np.log(c))
    P = np.concatenate(([0.0], np.cumsum(np.maximum(r, 0.0))))
    N = np.concatenate(([0.0], np.cumsum(np.maximum(-r, 0.0))))

    curve = np.full(len(lookbacks), np.nan)

    if n < W + 3:  # tiny window fallback — reference math
        rr = _log_fwd_returns(c)
        for k, L in enumerate(lookbacks):
            curve[k] = profit_factor_np(breakout_signal_np(c, int(L)) * rr)
    else:
        win = sliding_window_view(c, W)
        smax = np.maximum.accumulate(win[:, ::-1], axis=1)[:, ::-1]
        smin = np.minimum.accumulate(win[:, ::-1], axis=1)[:, ::-1]
        head = pd.Series(c[:W])

        for k, L in enumerate(lookbacks):
            w = int(L) - 1
            up = np.empty(n)
            dn = np.empty(n)
            up[:W] = head.rolling(w).max().shift(1).to_numpy()
            dn[:W] = head.rolling(w).min().shift(1).to_numpy()
            p = W - w
            up[W:] = smax[: n - W, p]
            dn[W:] = smin[: n - W, p]

            up_mask = c > up
            dn_mask = c < dn
            ev = np.flatnonzero(up_mask | dn_mask)
            if ev.size == 0:
                continue
            vals = np.where(up_mask[ev], 1.0, -1.0)
            bounds = np.concatenate((ev, [n - 1]))
            seg_p = P[bounds[1:]] - P[bounds[:-1]]
            seg_n = N[bounds[1:]] - N[bounds[:-1]]
            lng = vals > 0
            tot_pos = seg_p[lng].sum() + seg_n[~lng].sum()
            tot_neg = seg_n[lng].sum() + seg_p[~lng].sum()
            curve[k] = (
                tot_pos / tot_neg
                if tot_neg > 0
                else (np.inf if tot_pos > 0 else np.nan)
            )

    best_pf, best_lookback = 0.0, -1
    for L, pf in zip(lookbacks, curve):
        if pf > best_pf:
            best_pf, best_lookback = float(pf), int(L)
    return best_lookback, best_pf, lookbacks, curve


def walkforward_donch(
    ohlc: pd.DataFrame,
    train_lookback: int = 24 * 365 * 4,
    train_step: int = 24 * 30,
    lb_min=12,
    lb_max=168,
    lb_step=1,
):
    """Reference walk-forward — re-optimize every train_step bars."""
    n = len(ohlc)
    wf_signal = np.full(n, np.nan)
    tmp_signal = None
    next_train = train_lookback
    for i in range(next_train, n):
        if i == next_train:
            best_lookback, _ = optimize_donchian(
                ohlc.iloc[i - train_lookback : i], lb_min, lb_max, lb_step
            )
            tmp_signal = donchian_breakout(ohlc, best_lookback)
            next_train += train_step
        wf_signal[i] = tmp_signal.iloc[i]
    return wf_signal


def walkforward_donch_fast(
    close,
    train_lookback: int,
    train_step: int,
    lb_min=12,
    lb_max=168,
    lb_step=1,
    return_choices=False,
):
    c = np.asarray(close, dtype=float)
    n = len(c)
    wf = np.full(n, np.nan)
    choices = []
    t = train_lookback
    while t < n:
        best_L, _, _, _ = optimize_donchian_fast(
            c[t - train_lookback : t], lb_min, lb_max, lb_step
        )
        choices.append((t, best_L))
        if best_L > 0:
            sig = breakout_signal_np(c, best_L)
            end = min(t + train_step, n)
            wf[t:end] = sig[t:end]
        t += train_step
    return (wf, choices) if return_choices else wf
