import numpy as np
import pandas as pd
from mcpt.backtest import CostModel, backtest
from mcpt.data import make_demo
from mcpt.strategies import donchian_signal, ma_signal

df = make_demo(years=3)
sig = donchian_signal(df["close"], 20)

def test_no_cost_means_net_equals_gross():
    bt = backtest(df, sig, costs=CostModel(spread_bps=0, fee_bps=0, impact_y=0))
    assert np.allclose(bt.net, bt.gross)

def test_costs_only_on_turnover():
    bt = backtest(df, sig)
    assert (bt.cost[bt.turnover == 0] == 0).all()      # holding is free
    assert (bt.cost[bt.turnover > 0] > 0).all()        # trading is not
    assert bt.stats["pf_net"] <= bt.stats["pf_gross"]

def test_signal_is_causal():
    # Changing the future must not change today's position.
    df2 = df.copy(); df2.iloc[400:, df2.columns.get_loc("close")] *= 1.5
    assert np.allclose(ma_signal(df["close"], 10, 40).iloc[:400],
                       ma_signal(df2["close"], 10, 40).iloc[:400])

def test_bigger_size_costs_more():
    small = backtest(df, sig, costs=CostModel(capital=1e5))
    big = backtest(df, sig, costs=CostModel(capital=1e8))
    assert big.stats["cost_total_net"] > small.stats["cost_total_net"]

def test_flat_strategy_earns_nothing():
    bt = backtest(df, pd.Series(0.0, index=df.index))
    assert bt.net.sum() == 0 and bt.cost.sum() == 0

def test_delay_cost_of_next_open_fill():
    # Close-fill minus next-open-fill = exactly the overnight gaps we would miss.
    z = CostModel(spread_bps=0, fee_bps=0, impact_y=0)
    c = backtest(df, sig, fill="close", costs=z)
    o = backtest(df, sig, fill="next_open", costs=z)
    gaps = np.log(df["open"] / df["close"].shift(1)).fillna(0.0).to_numpy()
    held = np.concatenate(([0.0], sig.to_numpy()[:-1]))
    assert np.allclose(c.gross - o.gross, held * gaps)
