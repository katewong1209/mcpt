"""The permutation test itself.

Procedure (identical for the real data and every shuffled copy):
  1. Run the full pipeline -- including parameter optimization -- and score
     it after transaction costs.
  2. Shuffle the bars, run the SAME pipeline again, record the score.
  3. Repeat N times. Those scores form the "noise" distribution: what this
     pipeline achieves when there is nothing to find.
  4. p-value = (1 + number of shuffled scores >= real score) / (N + 1).
     The +1 counts the real run as one observation and keeps p > 0.

A small p (say < 0.05) means the real result is hard to reproduce on
patternless data -- evidence the edge is real, NOT a promise of profit.
Costs are charged in the shuffled runs too, so the test judges the net result.
"""
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

from .backtest import CostModel, backtest
from .optimize import sweep_donchian, walkforward_donchian
from .permute import permute_bars
from .strategies import donchian_signal, ma_signal


@dataclass
class Config:
    strategy: str = "donchian_wf"      # donchian_is | donchian_wf | ma
    grid: tuple = (10, 120, 10)        # Donchian lookback sweep (bars)
    ma_fast: int = 10
    ma_slow: int = 40
    train: int = 756                   # walk-forward training window (~3 years)
    step: int = 63                     # re-optimize every quarter
    fill: str = "next_open"            # next_open (realistic) | close (paper)
    costs: CostModel = field(default_factory=CostModel)
    n_perms: int = 500
    seed: int = 42
    stat: str = "pf"                   # test statistic: pf | sharpe


def _run_pipeline(df: pd.DataFrame, cfg: Config):
    """One full run of the selected strategy. Returns (backtest, signal, extra)."""
    if cfg.strategy == "donchian_is":
        lb, table, bt = sweep_donchian(df, cfg.grid, cfg.fill, cfg.costs, cfg.stat)
        return bt, donchian_signal(df["close"], lb), {"best_lookback": lb, "sweep": table}
    if cfg.strategy == "donchian_wf":
        sig, choices = walkforward_donchian(df, cfg.train, cfg.step, cfg.grid,
                                            cfg.fill, cfg.costs, cfg.stat)
        bt = backtest(df, sig, cfg.fill, cfg.costs, eval_start=cfg.train)
        return bt, sig, {"choices": choices}
    if cfg.strategy == "ma":
        sig = ma_signal(df["close"], cfg.ma_fast, cfg.ma_slow)
        return backtest(df, sig, cfg.fill, cfg.costs), sig, {}
    raise ValueError(f"Unknown strategy: {cfg.strategy}")


@dataclass
class Result:
    p_value: float
    real_stat: float
    null_stats: np.ndarray             # one score per permutation
    real_curve: np.ndarray             # cumulative net log return, scored region
    null_curves: np.ndarray            # same, per permutation (for the envelope)
    real_stats: dict                   # full gross/net metrics of the real run
    real_signal: np.ndarray            # position series of the real run (for cost charts)
    extra: dict                        # best lookback / walk-forward choices
    config: dict
    n_no_trade: int                    # permutations where the strategy never traded
    elapsed_s: float

    def to_dict(self) -> dict:
        skip = ("null_curves", "extra", "null_stats", "real_curve", "real_signal")
        d = {k: v for k, v in asdict(self).items() if k not in skip}
        d["null_stats"] = self.null_stats.tolist()
        d["real_curve"] = self.real_curve.tolist()
        d["extra"] = {"best_lookback": self.extra.get("best_lookback"),
                      "choices": [(str(t), int(lb)) for t, lb in self.extra.get("choices", [])]}
        return d


def permutation_test(df: pd.DataFrame, cfg: Config, progress=None) -> Result:
    """Run the real pipeline, then N permuted copies. `progress(fraction, text)`
    is an optional callback so the UI can show a progress bar."""
    t0 = time.time()
    real_bt, real_sig, extra = _run_pipeline(df, cfg)
    key = f"{cfg.stat}_net"
    real_stat = real_bt.stats[key]
    if np.isnan(real_stat):
        raise ValueError("The strategy never traded on this sample; widen the data or parameters.")

    start = cfg.train if cfg.strategy == "donchian_wf" else 0
    s = slice(start, None)
    real_curve = np.cumsum(real_bt.net[s])
    null_stats = np.empty(cfg.n_perms)
    null_curves = np.empty((cfg.n_perms, len(real_curve)), dtype=np.float32)
    n_no_trade = 0

    for i in range(cfg.n_perms):
        perm = permute_bars(df, start=start, seed=cfg.seed + i)
        bt, _, _ = _run_pipeline(perm, cfg)
        stat = bt.stats[key]
        if np.isnan(stat):             # no trades on this shuffle: cannot beat the real run
            stat, n_no_trade = -np.inf, n_no_trade + 1
        null_stats[i] = stat
        null_curves[i] = np.cumsum(bt.net[s]).astype(np.float32)
        if progress and (i % max(1, cfg.n_perms // 100) == 0 or i == cfg.n_perms - 1):
            progress((i + 1) / cfg.n_perms, f"Permutation {i + 1:,} / {cfg.n_perms:,}")

    beats = int(np.sum(null_stats >= real_stat))
    return Result(p_value=(1 + beats) / (cfg.n_perms + 1), real_stat=float(real_stat),
                  null_stats=null_stats, real_curve=real_curve, null_curves=null_curves,
                  real_stats=real_bt.stats, real_signal=np.nan_to_num(np.asarray(real_sig, float)),
                  extra=extra, config=asdict(cfg), n_no_trade=n_no_trade, elapsed_s=time.time() - t0)
