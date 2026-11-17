# Permutation Tests · Trading Strategies
Built by Kate Wong.

Optimization always induces data-mining bias. A good strategy exploits a real,
repeatable market pattern. MCPT asks: does the fully-optimized pipeline perform
this well on structureless noise too? If yes, it is overfitting.

## Quickstart
    pip install -r requirements.txt
    streamlit run app.py

## Data
The sidebar defaults to **Upload files**. Drop in `Currency 1 hourly
(.pq / .parquet / .csv)`; `Currency 2 hourly (optional for correlation)` lives in
its own collapsed section so only one upload is on screen at a time. With no file
uploaded the app runs on a seeded synthetic BTC/ETH hourly feed (GARCH volatility
clustering, trend regimes, fat tails) so every workflow works end to end. A local
`BTCUSD3600.pq` is offered automatically when present.

Seeds are fixed internally for reproducibility and are not user-editable.

## Method
- Permute bars: intrabar geometry (H/L/C relative to open) shuffled jointly,
  close-to-open gaps shuffled separately. Marginal distribution preserved;
  serial dependence destroyed.
- Re-run the ENTIRE pipeline, optimization included, on each permutation.
- p = (1 + #{perm_PF >= real_PF}) / N_permutations. Aim for 1000 permutations.
- Walk-forward permutes only the out-of-sample region
  (start_index = train_window) and re-optimizes inside every permutation.

Preserved: mean / std / skew / kurtosis, intrabar shape, gap distribution,
contemporaneous cross-market correlation under joint permutation.
Destroyed: volatility clustering, long memory, self-similarity, frequency-domain
structure, anything involving dependence.

## Pipelines
Donchian in-sample (optimized) · Donchian walk-forward · decision tree ·
MA crossover. Each is explained with mechanics, strengths and weaknesses in the
app's Method & Guide tab.

## Files
app.py · mcpt_engine.py · donchian.py · bar_permute.py · tree_strat.py ·
moving_average.py · market_data.py · insample_donchian_mcpt.py ·
walkforward_donchian_mcpt.py · insample_tree_mcpt.py

The Turbo engine is a vectorized, mathematically identical implementation of the
reference pandas loops (suffix-extreme matrices + segment cumulative sums).
Toggle it off for the literal reference implementation.
