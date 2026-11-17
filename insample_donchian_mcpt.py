"""Standalone in-sample Donchian MCPT — reference workflow (tqdm + matplotlib)."""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from donchian import optimize_donchian
from bar_permute import get_permutation

if Path("BTCUSD3600.pq").exists():
    df = pd.read_parquet("BTCUSD3600.pq")
else:
    from market_data import make_synthetic_pair
    print("BTCUSD3600.pq not found — using synthetic demo feed.")
    df, _ = make_synthetic_pair()

df.index = df.index.astype("datetime64[s]")
train_df = df[(df.index.year >= 2016) & (df.index.year < 2020)]

best_lookback, best_real_pf = optimize_donchian(train_df)
print("In-sample PF", best_real_pf, "Best Lookback", best_lookback)

n_permutations = 1000
perm_better_count = 1
permuted_pfs = []

print("In-Sample MCPT")
for perm_i in tqdm(range(1, n_permutations)):
    train_perm = get_permutation(train_df)
    _, best_perm_pf = optimize_donchian(train_perm)
    if best_perm_pf >= best_real_pf:
        perm_better_count += 1
    permuted_pfs.append(best_perm_pf)

insample_mcpt_pval = perm_better_count / n_permutations
print(f"In-sample MCPT P-Value: {insample_mcpt_pval}")

plt.style.use("dark_background")
pd.Series(permuted_pfs).hist(color="blue", label="Permutations")
plt.axvline(best_real_pf, color="red", label="Real")
plt.xlabel("Profit Factor")
plt.title(f"In-sample MCPT. P-Value: {insample_mcpt_pval}")
plt.grid(False)
plt.legend()
plt.show()
