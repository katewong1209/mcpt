"""Decision-tree momentum classifier.

Feature spans and the label horizon are counted in BARS of whatever timeframe the
caller supplies (hourly / daily / monthly / yearly) — no hard-coded hour counts.
"""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DEFAULT_SPANS = (6, 24, 168)
DEFAULT_HORIZON = 24


def feature_names(spans=DEFAULT_SPANS):
    return [f"diff{int(s)}" for s in spans]


def _features(log_c, spans):
    f = pd.concat([log_c.diff(int(s)) for s in spans], axis=1)
    f.columns = feature_names(spans)
    return f


def train_tree(ohlc: pd.DataFrame, min_samples_leaf=5, max_depth=None, random_state=69,
               spans=DEFAULT_SPANS, horizon=DEFAULT_HORIZON):
    spans = tuple(int(s) for s in spans)
    horizon = int(max(1, horizon))
    names = feature_names(spans)
    log_c = np.log(ohlc["close"])
    target = (np.sign(log_c.diff(horizon).shift(-horizon)) + 1) / 2
    ds = pd.concat([_features(log_c, spans), target.rename("target")], axis=1).dropna()
    if len(ds) < 3:
        raise ValueError(
            f"Only {len(ds)} usable {'row' if len(ds) == 1 else 'rows'} for the tree "
            "at this timeframe — pick a finer timeframe or a wider date range."
        )
    leaf = int(max(1, min(int(min_samples_leaf), max(1, len(ds) // 2))))
    model = DecisionTreeClassifier(
        min_samples_leaf=leaf, max_depth=max_depth, random_state=random_state
    )
    model.fit(ds[names].to_numpy(), ds["target"].astype(int).to_numpy())
    return model


def tree_strategy(ohlc: pd.DataFrame, model, spans=DEFAULT_SPANS):
    log_c = np.log(ohlc["close"])
    feats = _features(log_c, spans).dropna()
    if len(feats) == 0:
        return pd.Series(np.full(len(ohlc), np.nan), index=ohlc.index), float("nan")
    pred = pd.Series(model.predict(feats.to_numpy()), index=feats.index).reindex(ohlc.index)
    # NaN — not "short" — wherever the features are still warming up
    signal = pd.Series(np.where(pred > 0, 1.0, -1.0), index=ohlc.index).where(pred.notna())
    r = log_c.diff().shift(-1)
    rets = signal * r
    pos = rets[rets > 0].sum()
    neg = rets[rets < 0].abs().sum()
    pf = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else float("nan"))
    return signal, pf
