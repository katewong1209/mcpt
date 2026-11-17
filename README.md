# PERMUTLAB · Monte Carlo Permutation Testing for Trading Strategies

Optimization always induces data-mining bias. A good strategy exploits a real,
repeatable market pattern. MCPT asks: *does the fully-optimized pipeline perform
this well on structureless noise too?* If yes — it is overfitting.

## Quickstart
    pip install -r requirements.txt
    streamlit run app.py

## Data
Drop `BTCUSD3600.pq` (and optionally `ETHUSD3600.pq`) next to `app.py`, or upload
them in the sidebar. With no files present the app boots on a seeded synthetic
BTC/ETH hourly feed (GARCH volatility clustering + trend regimes + fat tails) so
every workflow runs end-to-end out of the box.

## Method
- Permute bars: intrabar geometry (H/L/C relative to open) shuffled jointly,
  close→open gaps shuffled separately. Marginal distribution preserved;
  serial dependence destroyed.
- Re-run the ENTIRE strategy pipeline (including optimization) on each perm.
- p = (1 + #{perm_PF >= real_PF}) / N_permutations. Aim for 1000 permutations.
- Walk-forward variant permutes only the out-of-sample region
  (`start_index = train_window`) and re-trains inside every permutation.

Preserved: mean / std / skew / kurtosis, intrabar shape, gap distribution,
contemporaneous cross-market correlation (joint permutation).
Destroyed: volatility clustering, long memory, self-similarity, frequency-domain
structure — anything involving dependence. Strategies that unknowingly exploit
those properties interact with this null hypothesis; interpret accordingly.

## Files
app.py · mcpt_engine.py · donchian.py · bar_permute.py · tree_strat.py ·
moving_average.py · market_data.py · insample_donchian_mcpt.py ·
walkforward_donchian_mcpt.py · insample_tree_mcpt.py

The Turbo engine is a vectorized, mathematically identical re-implementation of
the reference pandas loops (suffix-extreme matrices + segment cumulative sums).
Toggle it off in the sidebar for the literal reference implementation.
