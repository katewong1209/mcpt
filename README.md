# Strategy Validation Lab

Does a trading rule's backtest reflect a real pattern, or data-mining luck —
and does the edge survive transaction costs? This project answers both with
Monte Carlo permutation tests and an execution-aware backtest, on daily US
equity/ETF data. Streamlit app for exploration; CLI script for reproducible studies.

## Method
1. **Backtest with realistic timing.** The signal is decided at bar t's close;
   the default fill is the *next bar's open*. Filling at the signal close is
   available as a paper benchmark; the difference is the delay cost.
2. **Charge costs on turnover, not on holding.**
   `cost = turnover·(spread + fees) + turnover·Y·σ·√(order $ / trailing ADV $)`
   — the square-root impact rule (Almgren et al., 2005), with trailing,
   causal volatility and volume estimates.
3. **Shuffle the bars.** Each bar is split into gap + intraday pieces; pieces
   are dealt out in random order (volume travels with its bar). Distributions
   survive; trends, momentum and volatility clustering do not.
4. **Re-run the whole pipeline on each shuffle — optimizer included, costs
   included.** p = (1 + #shuffled scores ≥ real score) / (N + 1).

Strategies: Donchian breakout (in-sample sweep and walk-forward) and a
fixed-parameter MA crossover as the no-optimization control.

## Example result (synthetic demo data, N = 500, seed 42, $1M, next-open fills)
MA 10/40 on a liquid vs. an illiquid demo asset:

| | Liquid (2 bp spread) | Illiquid (15 bp spread) |
|---|---|---|
| PF gross → net | 1.04 → 1.04 | 0.99 → 0.91 |
| Sharpe gross → net | 0.16 → 0.14 | −0.03 → −0.34 |
| Cost per trade | 4 bp | 105 bp |
| Breakeven cost | 31 bp | −10 bp |
| MCPT p-value | 0.409 | 0.353 |

Neither run shows a reliable edge on synthetic data (as expected). The point
is the second column: the illiquid asset's result is a mid-price artifact —
execution costs exceed the gross edge, and the framework says so quantitatively.

## Quickstart
    pip install -r requirements.txt
    streamlit run app.py                      # interactive
    python -m pytest tests/                   # 14 tests
    python scripts/run_study.py --data SPY.csv --strategy donchian_wf \
        --perms 1000 --out results/spy_wf.json

CSV schema: `date,open,high,low,close,volume` (daily, ascending). The app can
also download adjusted daily bars via yfinance. Demo data is synthetic and
labelled as such; use real data for any market conclusion.

## Limitations (stated up front)
- Daily closes only; no intraday session effects, auctions, or borrow costs.
- Square-root impact is a rule of thumb; calibrate Y on real fills.
- The test prices the pipeline you ran. Testing several strategies and
  reporting the best requires a multiple-testing correction (e.g., White's
  Reality Check) — out of scope here, flagged in the UI.
- A small p-value is evidence against luck, not a profit forecast.

## References
Aronson (2007), *Evidence-Based Technical Analysis* · Masters (2011),
*Permutation and Randomization Tests for Trading System Development* ·
Liu (2023), *Quantitative Trading Strategies Using Python* · Almgren et al.
(2005)
