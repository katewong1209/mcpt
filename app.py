"""Streamlit UI for the permutation-test framework.

Run with:  streamlit run app.py
All research logic lives in the mcpt package; this file only wires widgets
to it and draws the results.
"""
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mcpt.backtest import CostModel, backtest
from mcpt.data import load_csv, load_yfinance, make_demo
from mcpt.mcpt import Config, permutation_test

st.set_page_config(page_title="Strategy Validation Lab", layout="wide")
st.title("Strategy Validation Lab")
st.caption("Does a trading rule beat shuffled data, and does the edge survive trading costs? "
           "Daily US-equity/ETF bars; all results are after costs unless labelled gross.")

# ------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Data")
    source = st.selectbox("Source", ["Demo: liquid ETF (synthetic)",
                                     "Demo: illiquid ETF (synthetic)", "Upload CSV", "Yahoo Finance"])
    if source == "Upload CSV":
        up = st.file_uploader("OHLCV CSV (date, open, high, low, close, volume)", type="csv")
    elif source == "Yahoo Finance":
        ticker = st.text_input("Ticker", "SPY")
        start = st.text_input("Start", "2015-01-01")
        end = st.text_input("End", "2025-12-31")

    st.header("Strategy")
    strat = st.selectbox("Rule", ["Donchian breakout — walk-forward",
                                  "Donchian breakout — in-sample sweep",
                                  "MA crossover — fixed parameters"])
    strat_key = {"Donchian breakout — walk-forward": "donchian_wf",
                 "Donchian breakout — in-sample sweep": "donchian_is",
                 "MA crossover — fixed parameters": "ma"}[strat]
    if strat_key != "ma":
        lb_min, lb_max = st.slider("Lookback range (bars)", 5, 250, (10, 120))
        lb_step = st.number_input("Lookback step", 1, 50, 10)
    if strat_key == "donchian_wf":
        train = st.number_input("Training window (bars)", 252, 2520, 756, step=63)
        step = st.number_input("Re-optimize every (bars)", 21, 252, 63, step=21)
    if strat_key == "ma":
        ma_fast = st.number_input("Fast MA", 2, 100, 10)
        ma_slow = st.number_input("Slow MA", 5, 250, 40)

    st.header("Execution assumptions")
    fill = st.radio("Fill timing", ["next_open", "close"],
                    format_func=lambda x: "Next bar's open (realistic)" if x == "next_open"
                    else "Signal bar's close (paper)")
    spread = st.number_input("Half-spread (bps per unit traded)", 0.0, 100.0, 2.0, 0.5)
    fee = st.number_input("Fees (bps per unit traded)", 0.0, 50.0, 0.5, 0.5)
    impact_y = st.number_input("Impact multiplier Y", 0.0, 3.0, 1.0, 0.1)
    capital = st.number_input("Strategy capital ($M)", 0.1, 500.0, 1.0) * 1e6

    st.header("Test")
    stat = st.radio("Test statistic", ["pf", "sharpe"],
                    format_func=lambda x: "Profit factor" if x == "pf" else "Sharpe ratio")
    n_perms = st.number_input("Permutations", 50, 5000, 300, 50)
    seed = st.number_input("Random seed", 0, 100000, 42)
    run = st.button("Run test", type="primary", use_container_width=True)

# ------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def demo(liquid):
    return make_demo(liquid=liquid)

try:
    if source.startswith("Demo"):
        df = demo("liquid" in source)
        st.info("Synthetic demo data — for exercising the tool, not market conclusions.")
    elif source == "Upload CSV":
        if up is None:
            st.stop()
        df = load_csv(up)
    else:
        df = load_yfinance(ticker, start, end)
except ValueError as e:
    st.error(str(e)); st.stop()
except Exception as e:
    st.error(f"Could not load data: {e}"); st.stop()

st.caption(f"{len(df):,} daily bars · {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d} · "
           f"median dollar volume ${np.median(df.close * df.volume) / 1e6:,.0f}M/day")

with st.expander("Method, in one paragraph"):
    st.write(
        "We run the full strategy pipeline — including parameter optimization — on the real data "
        "and score it after costs. Then we shuffle the price bars (keeping each bar's shape, gaps "
        "and volume, but removing any time-order pattern) and run the identical pipeline again, "
        "N times. The p-value is the share of shuffled runs that match or beat the real result: "
        "p = (1 + count) / (N + 1). Small p means the edge is unlikely to be data-mining luck. "
        "It is not a profit forecast: regime change and costs beyond the model are still your risk.")

# ------------------------------------------------------------- run
cfg = Config(strategy=strat_key, grid=(lb_min, lb_max, lb_step) if strat_key != "ma" else (10, 120, 10),
             ma_fast=ma_fast if strat_key == "ma" else 10,
             ma_slow=ma_slow if strat_key == "ma" else 40,
             train=train if strat_key == "donchian_wf" else 756,
             step=step if strat_key == "donchian_wf" else 63,
             fill=fill, costs=CostModel(spread, fee, impact_y, capital),
             n_perms=int(n_perms), seed=int(seed), stat=stat)

if run:
    bar = st.progress(0.0, text="Running real baseline…")
    try:
        st.session_state["res"] = permutation_test(df, cfg, lambda f, t: bar.progress(f, text=t))
        st.session_state["df"] = df
    except ValueError as e:
        st.error(str(e))
    bar.empty()

if "res" not in st.session_state:
    st.stop()
res, df = st.session_state["res"], st.session_state["df"]
cfg_used = Config(**{**res.config, "costs": CostModel(**res.config["costs"])})
start = cfg_used.train if cfg_used.strategy == "donchian_wf" else 0
x = df.index[start:]

# ------------------------------------------------------------- results
p = res.p_value
if p <= 0.05:
    st.success(f"**p = {p:.3f}** — the net result is hard to reproduce on shuffled data. "
               "Evidence of a real pattern; check the cost charts before sizing it.")
elif p <= 0.20:
    st.warning(f"**p = {p:.3f}** — inconclusive. The result is within reach of data-mining luck.")
else:
    st.error(f"**p = {p:.3f}** — indistinguishable from noise. This pipeline, on this sample, "
             "shows no reliable edge.")

s = res.real_stats
rows = ["pf", "sharpe", "total_return", "max_drawdown", "n_trades",
        "turnover_per_year", "cost_bps_per_trade", "breakeven_bps"]
table = pd.DataFrame({"Gross": [s[f"{k}_gross"] for k in rows],
                      "Net": [s[f"{k}_net"] for k in rows]},
                     index=["Profit factor", "Sharpe (ann.)", "Total log return", "Max drawdown (log)",
                            "Position changes", "Turnover / year (× capital)",
                            "Cost per trade (bps)", "Breakeven cost (bps)"]).round(3)
c1, c2 = st.columns([1, 2])
c1.dataframe(table, use_container_width=True)

finite = res.null_stats[np.isfinite(res.null_stats)]
hist = go.Figure(go.Histogram(x=finite, nbinsx=40, marker_color="#4c78a8"))
hist.add_vline(x=res.real_stat, line_color="#e45756", line_width=2, annotation_text="real (net)")
hist.update_layout(title=f"{cfg_used.stat.upper()} on {cfg_used.n_perms} shuffled datasets",
                   height=320, margin=dict(t=40, b=30), showlegend=False)
c2.plotly_chart(hist, use_container_width=True)

lo, med, hi = np.percentile(res.null_curves, [5, 50, 95], axis=0)
eq = go.Figure()
eq.add_trace(go.Scatter(x=x, y=hi, mode="lines", line_width=0, showlegend=False))
eq.add_trace(go.Scatter(x=x, y=lo, mode="lines", line_width=0, fill="tonexty",
                        fillcolor="rgba(76,120,168,0.2)", name="Shuffled 5–95%"))
eq.add_trace(go.Scatter(x=x, y=med, line=dict(color="#4c78a8", dash="dot"), name="Shuffled median"))
eq.add_trace(go.Scatter(x=x, y=res.real_curve, line=dict(color="#e45756", width=2), name="Real (net)"))
eq.update_layout(title="Cumulative net log return — real vs shuffled", height=340,
                 margin=dict(t=40, b=30), legend=dict(orientation="h"))
st.plotly_chart(eq, use_container_width=True)

# Cost sensitivity and capacity: re-score the REAL signal under other assumptions.
c3, c4 = st.columns(2)
bps_grid = [0, 1, 2, 5, 10, 20, 30, 50]
sens = [backtest(df, res.real_signal, cfg_used.fill,
                 replace(cfg_used.costs, spread_bps=b), eval_start=start or None).stats
        for b in bps_grid]
f1 = go.Figure()
f1.add_trace(go.Scatter(x=bps_grid, y=[q["sharpe_net"] for q in sens], name="Net Sharpe"))
f1.add_trace(go.Scatter(x=bps_grid, y=[q["pf_net"] for q in sens], name="Net PF", yaxis="y2"))
f1.update_layout(title="Cost sensitivity (real signal)", height=320, margin=dict(t=40, b=30),
                 xaxis_title="Half-spread (bps)", legend=dict(orientation="h"),
                 yaxis2=dict(overlaying="y", side="right"))
c3.plotly_chart(f1, use_container_width=True)

cap_grid = np.array([0.1, 0.3, 1, 3, 10, 30, 100]) * 1e6
cap = [backtest(df, res.real_signal, cfg_used.fill,
                replace(cfg_used.costs, capital=float(c)), eval_start=start or None).stats["sharpe_net"]
       for c in cap_grid]
f2 = go.Figure(go.Scatter(x=cap_grid / 1e6, y=cap, mode="lines+markers"))
f2.update_layout(title="Capacity: net Sharpe vs strategy size", height=320,
                 margin=dict(t=40, b=30), xaxis_title="Capital ($M)", xaxis_type="log")
c4.plotly_chart(f2, use_container_width=True)

if cfg_used.strategy == "donchian_is":
    st.caption(f"Selected lookback: {res.extra['best_lookback']} bars "
               "(chosen on this same sample — the p-value accounts for that search).")
st.download_button("Download results (JSON)", json.dumps(res.to_dict(), indent=2),
                   file_name="mcpt_results.json", mime="application/json")
