"""Reproducible study runner: same test as the app, from the command line.

Example:
    python scripts/run_study.py --demo liquid --strategy donchian_wf --perms 1000 --out results/wf_liquid.json
The JSON contains the full configuration, every permuted score, and the real
metrics -- attach it to any result you report so the run can be checked.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcpt.backtest import CostModel
from mcpt.data import load_csv, make_demo
from mcpt.mcpt import Config, permutation_test


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="CSV file; omit to use synthetic demo data")
    ap.add_argument("--demo", choices=["liquid", "illiquid"], default="liquid")
    ap.add_argument("--strategy", choices=["donchian_is", "donchian_wf", "ma"], default="donchian_wf")
    ap.add_argument("--perms", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stat", choices=["pf", "sharpe"], default="pf")
    ap.add_argument("--fill", choices=["next_open", "close"], default="next_open")
    ap.add_argument("--spread-bps", type=float, default=2.0)
    ap.add_argument("--fee-bps", type=float, default=0.5)
    ap.add_argument("--impact-y", type=float, default=1.0)
    ap.add_argument("--capital", type=float, default=1e6)
    ap.add_argument("--out", default="results/study.json")
    a = ap.parse_args()

    df = load_csv(a.data) if a.data else make_demo(liquid=(a.demo == "liquid"))
    cfg = Config(strategy=a.strategy, n_perms=a.perms, seed=a.seed, stat=a.stat, fill=a.fill,
                 costs=CostModel(a.spread_bps, a.fee_bps, a.impact_y, a.capital))
    res = permutation_test(df, cfg, progress=lambda f, t: print(f"\r{t}", end=""))
    print(f"\np = {res.p_value:.4f} | real {cfg.stat} (net) = {res.real_stat:.3f} "
          f"| elapsed {res.elapsed_s:.0f}s")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res.to_dict(), indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
