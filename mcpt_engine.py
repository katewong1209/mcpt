"""MCPT worker — runs in a background thread, streams results into a plain dict.

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
    "donchian_is": "Donchian · In-Sample (optimized)",
    "donchian_wf": "Donchian · Walk-Forward",
    "tree": "Decision Tree · In-Sample",
    "ma": "MA Crossover",
}

MAX_SAMPLE_CURVES = 12
CURVE_PTS = 1400


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


# ---- walk-forward geometry (single source of truth for engine + UI) ----
BARS_PER_YEAR = 24 * 365          # hourly bars in a non-leap year
MIN_OOS_STEPS = 2                 # at least two re-optimizations, otherwise the test is meaningless


def wf_min_bars(step_days=30):
    """Smallest window that can host 1 training year + MIN_OOS_STEPS OOS steps."""
    tstep = max(int(step_days) * 24, 24)
    return BARS_PER_YEAR + MIN_OOS_STEPS * tstep + 2


def wf_max_train_years(n_bars, step_days=30):
    """Largest whole training window (years) that still leaves MIN_OOS_STEPS OOS steps."""
    tstep = max(int(step_days) * 24, 24)
    return int(max(0, (int(n_bars) - MIN_OOS_STEPS * tstep - 2) // BARS_PER_YEAR))


def wf_plan(n_bars, train_years, step_days=30):
    """Clamp the walk-forward windows to the data instead of crashing.

    Returns (train_bars, step_bars, oos_bars). Raises ValueError only when even a
    one-year training window cannot fit — with an actionable message.
    """
    n_bars = int(n_bars)
    tstep = max(int(step_days) * 24, 24)
    max_years = wf_max_train_years(n_bars, step_days)
    if max_years < 1:
        raise ValueError(
            f"Walk-forward needs at least {wf_min_bars(step_days):,} hourly bars in the selected "
            f"years (found {n_bars:,}). Widen the walk-forward years, shorten the training window "
            f"or reduce the re-optimization step."
        )
    years = int(min(max(int(train_years), 1), max_years))
    tw = min(years * BARS_PER_YEAR, n_bars - MIN_OOS_STEPS * tstep - 2)
    return int(tw), int(tstep), int(n_bars - tw)


def run_mcpt(job, btc, cfg):
    try:
        strat = cfg["strategy"]
        n = int(cfg["n_perms"])
        base = int(cfg["seed"])
        turbo = bool(cfg["turbo"])
        lb = (cfg["lb_min"], cfg["lb_max"], cfg["lb_step"])
        job["label"] = STRAT_LABELS[strat]
        job["n_target"] = n

        # ---------------- real baseline ----------------
        job["phase"] = "Optimizing real baseline…"
        if strat in ("donchian_is", "tree"):
            train = _years(btc, cfg["is_start"], cfg["is_end"])
            if len(train) < max(lb[1] + 10, 400):
                raise ValueError("Training window too short — widen the year range.")
            job["window_desc"] = f"{cfg['is_start']}–{cfg['is_end']} · {len(train):,} bars"
            r = _rets(train["close"])
            step = _downsample(len(train))
            job["curve_x"] = train.index[::step]

            if strat == "donchian_is":
                if turbo:
                    bL, real_pf, lookbacks, curve = optimize_donchian_fast(
                        train["close"].to_numpy(), *lb
                    )
                    job["lookbacks"] = lookbacks
                    job["pf_curve"] = curve
                    job["env_cnt"] = np.zeros(len(lookbacks))
                    job["env_sum"] = np.zeros(len(lookbacks))
                    job["env_ssq"] = np.zeros(len(lookbacks))
                else:
                    bL, real_pf = optimize_donchian(train, *lb)
                job["best_lookback"] = int(bL)
                sig = (
                    breakout_signal_np(train["close"].to_numpy(), bL)
                    if bL > 0
                    else np.full(len(train), np.nan)
                )
                real_rets = sig * r
            else:
                model = train_tree(train, cfg["tree_leaf"], cfg["tree_depth"])
                sig_s, real_pf = tree_strategy(train, model)
                job["importances"] = model.feature_importances_.tolist()
                real_rets = (sig_s * pd.Series(r, index=train.index)).to_numpy()
            work_df = train

        elif strat == "donchian_wf":
            work_df = _years(btc, cfg["wf_start"], cfg["wf_end"])
            tw, tstep, n_oos = wf_plan(len(work_df), cfg["wf_train_years"], cfg["wf_step_days"])
            job["wf_train"] = tw
            job["oos_start"] = tw
            job["window_desc"] = (
                f"{cfg['wf_start']}–{cfg['wf_end']} · train {tw:,} bars · OOS {n_oos:,} bars "
                f"· {max(1, n_oos // tstep)} re-optimizations"
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
            if not np.isfinite(real_pf):
                raise ValueError(
                    "No out-of-sample trades were produced — widen the walk-forward years "
                    "or shorten the training window."
                )
            oos_x = work_df.index[tw:]
            step = _downsample(len(oos_x))
            job["curve_x"] = oos_x[::step]

        else:  # ma
            work_df = btc
            job["window_desc"] = f"full feed · {len(work_df):,} bars"
            sig_s, real_pf = ma_crossover(
                work_df, cfg["ma_fast"], cfg["ma_slow"], cfg["ma_short"]
            )
            real_rets = (sig_s * np.log(work_df["close"]).diff().shift(-1)).to_numpy()
            step = _downsample(len(work_df))
            job["curve_x"] = work_df.index[::step]

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
                        if pL > 0
                        else np.full(len(perm), np.nan)
                    )
                    job["sample_curves"].append(_equity(psig * _rets(perm["close"]))[::step])

            elif strat == "tree":
                perm = get_permutation(work_df, seed=seed)
                pm = train_tree(perm, cfg["tree_leaf"], cfg["tree_depth"])
                psig, ppf = tree_strategy(perm, pm)
                if i < MAX_SAMPLE_CURVES:
                    pr = pd.Series(_rets(perm["close"]), index=perm.index)
                    job["sample_curves"].append(_equity((psig * pr).to_numpy())[::step])

            else:  # ma
                perm = get_permutation(work_df, seed=seed)
                psig, ppf = ma_crossover(perm, cfg["ma_fast"], cfg["ma_slow"], cfg["ma_short"])
                if i < MAX_SAMPLE_CURVES:
                    pr = np.log(perm["close"]).diff().shift(-1)
                    job["sample_curves"].append(_equity((psig * pr).to_numpy())[::step])

            ppf = float(ppf)
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
