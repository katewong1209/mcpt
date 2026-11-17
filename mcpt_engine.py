"""MCPT worker — runs in a background thread, streams results into a plain dict.

Timeframe-aware: the app resamples the feed once (hourly / daily / monthly /
yearly) and every window, lookback and feature span below is counted in BARS of
that timeframe. Nothing is hard-coded to hours and there is no fixed minimum bar
count — windows auto-fit the data that is actually there.

Full-N permutation loop: N actual shuffles are generated (default 1000).
The real run is counted as one extra observation, so
    p = (1 + #{perm_PF >= real_PF}) / (N + 1)
which is the standard correction that keeps p strictly positive.
"""
import time
import traceback

import numpy as np
import pandas as pd

from bar_permute import get_permutation
from donchian import (
    breakout_signal_np,
    optimize_donchian,
    optimize_donchian_fast,
    profit_factor_np,
    walkforward_donch,
    walkforward_donch_fast,
)
from moving_average import ma_crossover
from tree_strat import train_tree, tree_strategy

STRAT_LABELS = {
    "donchian_is": "Donchian Breakout",
    "donchian_wf": "Donchian Walk-Forward",
    "tree": "Decision Tree",
    "ma": "MA Crossover",
}

MAX_SAMPLE_CURVES = 12
CURVE_PTS = 1400

# ---------------- timeframes ----------------
TIMEFRAMES = ("1h", "1D", "1M", "1Y")
TF_LABEL = {"1h": "Hourly", "1D": "Daily", "1M": "Monthly", "1Y": "Yearly"}
TF_UNIT = {"1h": "hour", "1D": "day", "1M": "month", "1Y": "year"}
TF_PER_YEAR = {"1h": 8760, "1D": 365, "1M": 12, "1Y": 1}
TF_DEFAULTS = {
    "1h": dict(lb_min=12, lb_max=168, lb_step=1, train=35040, step=720,
               ma_fast=10, ma_slow=30, spans=(6, 24, 168), horizon=24, lags=168, anat=17520),
    "1D": dict(lb_min=5, lb_max=60, lb_step=1, train=365, step=30,
               ma_fast=5, ma_slow=20, spans=(1, 5, 21), horizon=5, lags=60, anat=1500),
    "1M": dict(lb_min=3, lb_max=24, lb_step=1, train=36, step=3,
               ma_fast=3, ma_slow=12, spans=(1, 3, 12), horizon=3, lags=24, anat=120),
    "1Y": dict(lb_min=2, lb_max=5, lb_step=1, train=5, step=1,
               ma_fast=2, ma_slow=3, spans=(1, 2, 3), horizon=1, lags=5, anat=60),
}
_RULE_CACHE = {}


def tf_rule(tf):
    """pandas resample rule for a timeframe (version-safe month/year aliases)."""
    if tf in _RULE_CACHE:
        return _RULE_CACHE[tf]
    if tf == "1h":
        rule = None
    elif tf == "1D":
        rule = "1D"
    else:
        probe = pd.Series([1.0], index=pd.date_range("2020-01-01", periods=1, freq="h"))
        rule = None
        for cand in ({"1M": ("1ME", "1M"), "1Y": ("1YE", "1Y")}[tf]):
            try:
                probe.resample(cand).last()
                rule = cand
                break
            except ValueError:
                continue
    _RULE_CACHE[tf] = rule
    return rule


def resample_ohlc(df, tf):
    """Aggregate the feed into the selected timeframe. Hourly is returned as-is."""
    rule = tf_rule(tf)
    if rule is None:
        return df
    return df.resample(rule).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def bars_per_year(tf):
    return TF_PER_YEAR.get(tf, 1)


def tf_defaults(tf):
    return TF_DEFAULTS.get(tf, TF_DEFAULTS["1h"])


# ---------------- window geometry (auto-fit, no fixed minimum) ----------------
def clamp_grid(lb_min, lb_max, lb_step, n_bars):
    """Fit the lookback grid to the bars available. Never returns an empty grid."""
    n = int(n_bars)
    hi = int(max(2, min(int(lb_max), max(2, n - 2))))
    lo = int(max(2, min(int(lb_min), hi)))
    return lo, hi, int(max(1, int(lb_step)))


def wf_train_max(n_bars):
    return int(max(2, int(n_bars) - 2))


def wf_step_max(n_bars):
    return int(max(1, int(n_bars) // 2))


def wf_plan(n_bars, train_bars, step_bars):
    """Auto-fit the walk-forward windows to the data — no minimum bar count.

    Returns (train_bars, step_bars, oos_bars). At least two out-of-sample bars are
    always left, and two re-optimization steps whenever the sample allows it.
    The lookback grid is clamped to the training window afterwards (clamp_grid).
    """
    n = int(n_bars)
    if n < 4:
        raise ValueError(
            f"Only {n} bar(s) at this timeframe — pick a finer timeframe or a wider date range."
        )
    step = int(max(1, min(int(step_bars), max(1, n // 4))))
    oos_min = int(min(2 * step, max(1, n // 2)))
    train = int(max(2, min(int(train_bars), n - oos_min - 1)))
    train = int(max(train, min(3, n - 2)))      # the optimizer needs a couple of bars
    train = int(min(train, n - 2))              # always leave >= 2 OOS bars
    return train, step, int(n - train)


def _years(df, y0, y1):
    return df[(df.index.year >= y0) & (df.index.year < y1)]


def _rets(close):
    lc = np.log(np.asarray(close, dtype=float))
    r = np.empty(len(lc))
    r[:-1] = np.diff(lc)
    r[-1] = np.nan
    return r


def _downsample(n):
    return max(1, n // CURVE_PTS)


def _equity(rets):
    return np.nancumsum(np.asarray(rets, dtype=float))


def run_mcpt(job, btc, cfg):
    """`btc` must already be resampled to cfg['tf']."""
    try:
        strat = cfg["strategy"]
        tf = cfg.get("tf", "1h")
        d = tf_defaults(tf)
        n = int(cfg["n_perms"])
        base = int(cfg["seed"])
        turbo = bool(cfg["turbo"])
        spans = tuple(cfg.get("tree_spans", d["spans"]))
        horizon = int(cfg.get("tree_horizon", d["horizon"]))
        job["label"] = STRAT_LABELS[strat]
        job["tf"] = tf
        job["n_target"] = n

        # ---------------- real baseline ----------------
        job["phase"] = "Optimizing real baseline…"
        if strat in ("donchian_is", "tree"):
            work_df = _years(btc, cfg["is_start"], cfg["is_end"])
            if len(work_df) < 4:
                raise ValueError(
                    f"Only {len(work_df)} bar(s) in that date range — widen it or pick a finer timeframe."
                )
            lb = clamp_grid(cfg["lb_min"], cfg["lb_max"], cfg["lb_step"], len(work_df))
            job["grid"] = lb
            job["window_desc"] = (
                f"{cfg['is_start']}–{cfg['is_end']} · {len(work_df):,} {TF_UNIT[tf]} bars"
            )
            r = _rets(work_df["close"])
            step = _downsample(len(work_df))
            job["curve_x"] = work_df.index[::step]

            if strat == "donchian_is":
                if turbo:
                    bL, real_pf, lookbacks, curve = optimize_donchian_fast(
                        work_df["close"].to_numpy(), *lb
                    )
                    job["lookbacks"] = lookbacks
                    job["pf_curve"] = curve
                    job["env_cnt"] = np.zeros(len(lookbacks))
                    job["env_sum"] = np.zeros(len(lookbacks))
                    job["env_ssq"] = np.zeros(len(lookbacks))
                else:
                    bL, real_pf = optimize_donchian(work_df, *lb)
                job["best_lookback"] = int(bL)
                sig = (
                    breakout_signal_np(work_df["close"].to_numpy(), bL)
                    if bL > 1
                    else np.full(len(work_df), np.nan)
                )
                real_rets = sig * r
            else:
                model = train_tree(work_df, cfg["tree_leaf"], cfg["tree_depth"],
                                   spans=spans, horizon=horizon)
                sig_s, real_pf = tree_strategy(work_df, model, spans=spans)
                job["importances"] = model.feature_importances_.tolist()
                job["spans"] = spans
                real_rets = (sig_s * pd.Series(r, index=work_df.index)).to_numpy()

        elif strat == "donchian_wf":
            work_df = _years(btc, cfg["wf_start"], cfg["wf_end"])
            lb = clamp_grid(cfg["lb_min"], cfg["lb_max"], cfg["lb_step"], len(work_df))
            tw, tstep, n_oos = wf_plan(len(work_df), cfg["wf_train_bars"], cfg["wf_step_bars"])
            lb = clamp_grid(lb[0], lb[1], lb[2], tw)
            job["grid"] = lb
            job["wf_train"] = tw
            job["wf_step"] = tstep
            job["oos_start"] = tw
            job["window_desc"] = (
                f"train {tw:,} · OOS {n_oos:,} · {max(1, n_oos // tstep)} steps"
            )
            c = work_df["close"].to_numpy()
            if turbo:
                sig, choices = walkforward_donch_fast(c, tw, tstep, *lb, return_choices=True)
                job["wf_choices"] = [
                    (work_df.index[min(t, len(work_df) - 1)], int(b)) for t, b in choices
                ]
            else:
                sig = walkforward_donch(work_df, tw, tstep, *lb)
            # OUT-OF-SAMPLE ONLY — the training block is never scored, counted or plotted.
            real_rets = (sig * _rets(c))[tw:]
            real_pf = profit_factor_np(real_rets)
            oos_x = work_df.index[tw:]
            step = _downsample(len(oos_x))
            job["curve_x"] = oos_x[::step]

        else:  # ma
            work_df = btc
            if len(work_df) < 4:
                raise ValueError("Not enough bars at this timeframe — pick a finer timeframe.")
            job["window_desc"] = f"full feed · {len(work_df):,} {TF_UNIT[tf]} bars"
            fast, slow = int(cfg["ma_fast"]), int(cfg["ma_slow"])
            if fast >= slow:
                fast, slow = max(1, slow - 1), slow
            job["ma_pair"] = (fast, slow)
            sig_s, real_pf = ma_crossover(work_df, fast, slow, cfg["ma_short"])
            real_rets = (sig_s * np.log(work_df["close"]).diff().shift(-1)).to_numpy()
            step = _downsample(len(work_df))
            job["curve_x"] = work_df.index[::step]

        if np.isnan(real_pf):
            raise ValueError(
                "The strategy never traded on this sample — pick a finer timeframe, "
                "widen the date range, lower the lookback grid or allow shorts."
            )
        if np.isinf(real_pf):
            job["note"] = "No losing bars in this sample — PF is unbounded; treat the p-value with care."
        job["real_pf"] = float(real_pf)
        job["real_curve"] = _equity(real_rets)[::step]

        # ---------------- permutation loop: FULL N shuffles ----------------
        job["phase"] = "Permuting universes…"
        wf_mode = strat == "donchian_wf"
        for i in range(n):
            if job["stop"].is_set():
                job["status"] = "stopped"
                break
            seed = base + i + 1

            if wf_mode:
                perm = get_permutation(work_df, start_index=tw, seed=seed)
                pc = perm["close"].to_numpy()
                if turbo:
                    psig = walkforward_donch_fast(pc, tw, tstep, *lb)
                else:
                    psig = walkforward_donch(perm, tw, tstep, *lb)
                pr = (psig * _rets(pc))[tw:]          # OOS only — same slice as the real run
                ppf = profit_factor_np(pr)
                if i < MAX_SAMPLE_CURVES:
                    job["sample_curves"].append(_equity(pr)[::step])

            elif strat == "donchian_is":
                perm = get_permutation(work_df, seed=seed)
                if turbo:
                    pL, ppf, _, pcurve = optimize_donchian_fast(perm["close"].to_numpy(), *lb)
                    m = np.isfinite(pcurve)
                    job["env_cnt"][m] += 1
                    job["env_sum"][m] += pcurve[m]
                    job["env_ssq"][m] += pcurve[m] ** 2
                else:
                    pL, ppf = optimize_donchian(perm, *lb)
                if i < MAX_SAMPLE_CURVES:
                    psig = (
                        breakout_signal_np(perm["close"].to_numpy(), pL)
                        if pL > 1
                        else np.full(len(perm), np.nan)
                    )
                    job["sample_curves"].append(_equity(psig * _rets(perm["close"]))[::step])

            elif strat == "tree":
                perm = get_permutation(work_df, seed=seed)
                pm = train_tree(perm, cfg["tree_leaf"], cfg["tree_depth"],
                                spans=spans, horizon=horizon)
                psig, ppf = tree_strategy(perm, pm, spans=spans)
                if i < MAX_SAMPLE_CURVES:
                    pr = pd.Series(_rets(perm["close"]), index=perm.index)
                    job["sample_curves"].append(_equity((psig * pr).to_numpy())[::step])

            else:  # ma
                perm = get_permutation(work_df, seed=seed)
                psig, ppf = ma_crossover(perm, fast, slow, cfg["ma_short"])
                if i < MAX_SAMPLE_CURVES:
                    pr = np.log(perm["close"]).diff().shift(-1)
                    job["sample_curves"].append(_equity((psig * pr).to_numpy())[::step])

            ppf = float(ppf)
            if np.isnan(ppf):        # undefined = no edge, not a missing observation
                ppf = 0.0
            job["perm_pfs"].append(ppf)
            if ppf >= job["real_pf"]:
                job["better"] += 1
            job["done"] = i + 1
            job["p"] = job["better"] / (job["done"] + 1)

        if job["status"] == "running":
            job["status"] = "done"
        job["elapsed"] = time.time() - job["start"]
        job["p"] = job["better"] / (job["done"] + 1)

    except Exception:
        job["error"] = traceback.format_exc(limit=4)
        job["status"] = "error"
