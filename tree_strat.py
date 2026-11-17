"""Decision-tree momentum classifier (6h / 24h / 168h log-return features)."""
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

FEATURES = ["diff6", "diff24", "diff168"]


def train_tree(ohlc: pd.DataFrame, min_samples_leaf=5, max_depth=None, random_state=69):
    log_c = np.log(ohlc["close"])
    feats = pd.concat([log_c.diff(6), log_c.diff(24), log_c.diff(168)], axis=1)
    feats.columns = FEATURES
    # 0/1 label: does the next 24h go down/up?
    target = (np.sign(log_c.diff(24).shift(-24)) + 1) / 2
    dataset = pd.concat([feats, target], axis=1)
    dataset.columns = FEATURES + ["target"]
    dataset = dataset.dropna()
    model = DecisionTreeClassifier(
        min_samples_leaf=min_samples_leaf, max_depth=max_depth, random_state=random_state
    )
    model.fit(
        dataset[FEATURES].to_numpy(), dataset["target"].astype(int).to_numpy()
    )
    return model


def tree_strategy(ohlc: pd.DataFrame, model):
    log_c = np.log(ohlc["close"])
    dataset = pd.concat([log_c.diff(6), log_c.diff(24), log_c.diff(168)], axis=1)
    dataset.columns = FEATURES
    dataset = dataset.dropna()
    pred = pd.Series(model.predict(dataset.to_numpy()), index=dataset.index)
    pred = pred.reindex(ohlc.index)
    signal = pd.Series(np.where(pred > 0, 1, -1), index=ohlc.index)
    r = log_c.diff().shift(-1)
    rets = signal * r
    pos = rets[rets > 0].sum()
    neg = rets[rets < 0].abs().sum()
    pf = float(pos / neg) if neg > 0 else (float("inf") if pos > 0 else float("nan"))
    return signal, pf
