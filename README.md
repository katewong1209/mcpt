# Strategy Validation Lab

A research framework for answering two questions about a trading rule, in order:

1. **Is the backtest real, or data-mining luck?** — Monte Carlo hypothesis tests.
2. **If it's real, does it survive execution?** — a transaction-cost model with
   realistic fill timing, spread, fees, and market impact.

Most backtest projects answer neither. This one is built around the idea that a
profit factor is meaningless until you've stated the null hypothesis it's being
tested against, and meaningless again until you've subtracted the cost of
capturing it. Daily US equity/ETF bars; Streamlit app for exploration, CLI for
reproducible studies.

---

## 1. The problem

A backtest is an estimate produced by a *procedure* applied to *one realized
path* of history. Two things inflate it:

- **Selection bias.** If you sweep 50 lookbacks and report the best, you've
  reported a maximum of 50 noisy estimates, not an expected value. The more you
  search, the better the winner looks *even on pure noise*.
- **Execution idealization.** Filling at the signal bar's close, ignoring the
  spread, and pretending your order doesn't move the price can turn a negative-
  expectancy rule into a positive one on paper.

This framework prices both, separately and explicitly.

---

## 2. Statistical method

### 2.1 The test

Run the **entire pipeline — optimizer included — after costs** on the real data
to get a score. Then generate N synthetic "no-skill" datasets, run the *identical*
pipeline on each, and collect N scores. The p-value is

    p = (1 + #{synthetic scores >= real score}) / (N + 1)

The `+1` counts the real run as one observation under the null and keeps p > 0.
Re-running the optimizer inside every resample is essential: it means the test
prices the *search you actually did*, not a fixed rule you pretended to pick in
advance (Aronson 2007; Masters 2011).

### 2.2 The null hypothesis is a design choice, not a default

This is the most important idea in the repo. "Shuffle the data" is not one
hypothesis — it's a family, and which one you pick determines what a small
p-value actually proves.

**Null A — bar permutation (`permute_bars`).** Each bar is cut into an overnight
gap and intraday pieces; the pieces are dealt out in random order. This preserves
the marginal distribution of returns, bar shapes, and gaps, but destroys **all**
serial dependence — trends, momentum, *and volatility clustering*.

The catch: real markets cluster volatility, and clustered volatility resembles
trends. A trend rule can beat Null A simply because the real data has persistent
volatility, not because it forecasts direction. Rejecting Null A can therefore be
rejecting "no serial structure of any kind" rather than "no directional skill."

**Null B — stationary block bootstrap (`stationary_bootstrap`).** Instead of
single bars, resample **blocks of consecutive bars** with geometrically
distributed lengths (mean L), wrapping circularly (Politis & Romano 1994).
Dependence *within* a block survives; dependence *across* block boundaries is
broken. Volatility clustering on scales shorter than L is preserved.

Null B asks the sharper question: *is there structure beyond the observed
short-range clustering?* An edge that survives Null B is not a volatility-
clustering artifact. It is the more conservative, more credible test.

**Practice: report both.** Significance under Null A but not Null B is a warning
that the "edge" may be clustered volatility in disguise. Significance under both
is robust. The app exposes the choice and the mean block length so the reader can
see the sensitivity rather than take a single number on faith.

| | Null A: bar shuffle | Null B: block bootstrap |
|---|---|---|
| Return distribution | preserved exactly | preserved approximately (bootstrap variance) |
| Volatility clustering | destroyed | preserved within ~L bars |
| Short-range autocorrelation | destroyed | preserved within ~L bars |
| Long-range structure | destroyed | destroyed |
| Resampling | without replacement | with replacement |
| Conservatism | weaker null (easier to beat) | stronger null (harder to beat) |

**A third option, discussed not implemented:** *sign randomization* — flip the
sign of each return while keeping |returns| intact. This preserves the volatility
path *exactly* and destroys only directional predictability, making it the most
targeted null for a directional rule. It is clean on a return series but fiddly on
OHLC (high/low and gaps must be reflected consistently), so it is left as a
documented extension rather than shipped half-correct.

### 2.3 Block length: the bias–variance dial

L controls what the bootstrap preserves:

- **Too small** → blocks break the clustering you wanted to keep → the null
  collapses toward Null A (too easy to beat).
- **Too large** → blocks preserve so much that little is randomized → the test
  loses power (too hard to beat), and you have fewer effective independent blocks.

Rules of thumb: L should exceed the decay horizon of the dependence you want to
preserve (for daily equity |returns|, clustering is material out to ~20–60 bars),
and optimal-length selectors exist (Politis & White 2004; Patton et al. 2009),
typically scaling like n^(1/3). The app defaults to L = 20 and lets you sweep it;
**a result that flips sign as L varies is not a result.** Report the sensitivity.

### 2.4 Test statistic

Profit factor (gross wins / gross losses) is the default for continuity with the
literature, but it ignores path and drawdown and is unstable with few trades.
Sharpe is available and is generally the better statistic for a return-series
test. Both are computed **net of costs**, and the choice is exposed. Whatever the
statistic, it is recomputed identically on every resample.

### 2.5 Monte Carlo error and multiple testing

- The smallest attainable p is 1/(N+1); Monte Carlo error on p shrinks ~1/√N.
  N = 1000 is the point where a 0.05 verdict is stable. Use the CLI for large N.
- The resampling test prices the search *within one pipeline*. If you try several
  strategies and report the best, you need a family-wise correction: White's
  Reality Check (2000), Hansen's SPA (2005), a Harvey–Liu–Zhu haircut (2016), or a
  deflated Sharpe (Bailey & López de Prado 2014). This is flagged, not silently
  ignored.

---

## 3. Execution model

A signal is not a fill. The backtest separates them:

- **Timing.** The position is decided at bar t's close. Default fill is the **next
  bar's open**; the position earns that bar's open-to-close return and misses the
  overnight gap. A close-fill "paper" mode is available; the difference between
  the two is the **delay cost**, and a unit test asserts it equals exactly the
  missed gaps.
- **Costs are charged on turnover, never on holding:**

      cost = turnover·(spread + fees) + turnover·Y·σ·√(order $ / trailing ADV $)

  The square-root term is the standard impact rule (Almgren et al. 2005); σ and
  ADV are trailing, causal estimates. Y is of order 1 and must be calibrated on
  real fills — it is a shape, not a quote.
- **Outputs:** gross→net waterfall, cost per trade (bps), **breakeven cost** (the
  constant bps that erases the gross edge), and a **capacity curve** (net Sharpe
  vs capital). Costs are charged inside the resamples too, so the test judges the
  net result.

---

## 4. Walk-forward

Every `step` bars, re-optimize on the trailing `train` bars only, then trade the
next `step` unseen bars. Only out-of-sample bars are scored. In the resampling
test, the first training window stays real and resampling begins at its end; the
re-optimization runs inside every resample. This tests the *adaptive procedure
you would deploy*, not a frozen parameter set. (Purged/embargoed CV à la López de
Prado 2018 is the natural next step for overlapping labels.)

---

## 5. What this does NOT prove

Stated up front, because a validation tool that overclaims is worse than none:

- A small p-value is evidence against the stated null, **not a profit forecast**.
  Regime change is outside the test.
- The bar-shuffle null preserves the mean return, so the test isolates **timing
  skill** from passive drift exposure. An always-long rule riding positive drift
  will not look significant — correctly.
- Daily closes only: no intraday session effects, auctions, borrow costs, or
  queue position. The impact model is a rule of thumb, not a calibrated estimate.
- The block bootstrap preserves short-range *directional* autocorrelation too, so
  for a rule that exploits exactly that, Null B can be overly conservative. Choose
  L relative to the strategy's horizon and say so.

---

## 6. Example result (synthetic demo, N = 500, seed 42, $1M, next-open fills)

MA 10/40, liquid vs illiquid demo asset:

| | Liquid (2 bp spread) | Illiquid (15 bp spread) |
|---|---|---|
| PF gross → net | 1.04 → 1.04 | 0.99 → 0.91 |
| Sharpe gross → net | 0.16 → 0.14 | −0.03 → −0.34 |
| Cost per trade | 4 bp | 105 bp |
| Breakeven cost | 31 bp | −10 bp |

Neither shows a reliable edge on synthetic data (expected). The point is the
second column: the illiquid result is a mid-price artifact — costs exceed the
gross edge, and the framework says so quantitatively. Replace the demo with a
committed SPY/QQQ/IWM study (`results/`) for a real-data artifact.

---

## 7. Layout & quickstart

    mcpt/
    ├── app.py                  # thin Streamlit UI
    ├── mcpt/
    │   ├── data.py             # load / validate / demo
    │   ├── permute.py          # BOTH null generators (shuffle + block bootstrap)
    │   ├── strategies.py       # Donchian + MA rules
    │   ├── backtest.py         # fill timing, cost model, metrics
    │   ├── optimize.py         # grid sweep + walk-forward
    │   └── mcpt.py             # test runner, p-value, null selection
    ├── scripts/run_study.py    # CLI → results JSON
    └── tests/                  # invariants, causality, equivalence, clustering

    pip install -r requirements.txt
    streamlit run app.py
    python -m pytest tests/ -q
    python scripts/run_study.py --data SPY.csv --strategy donchian_wf \
        --perms 1000 --null stationary_bootstrap --mean-block 20 --out results/spy.json

CSV schema: `date,open,high,low,close,volume` (daily, ascending). Demo data is
synthetic and labelled as such.

**Reproducibility:** fixed seeds; the results JSON carries the full config
(including null method and block length) and every resampled score, so any
reported run can be regenerated exactly.

---

## 8. References

Aronson (2007), *Evidence-Based Technical Analysis* · Masters (2011),
*Permutation and Randomization Tests for Trading System Development* ·
Politis & Romano (1994), *The Stationary Bootstrap* · Künsch (1989), *The
Jackknife and the Bootstrap for General Stationary Observations* · Politis &
White (2004) / Patton, Politis & White (2009), *Automatic Block-Length Selection* ·
White (2000), *A Reality Check for Data Snooping* · Hansen (2005), *A Test for
Superior Predictive Ability* · Harvey, Liu & Zhu (2016), *…and the Cross-Section
of Expected Returns* · Bailey & López de Prado (2014), *The Deflated Sharpe
Ratio* · Almgren, Thum, Hauptmann & Li (2005), *Direct Estimation of Equity
Market Impact* · López de Prado (2018), *Advances in Financial Machine Learning*.
