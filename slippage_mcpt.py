import numpy as np
import pandas as pd

def apply_microstructure_slippage(rets, volumes, prices, base_bps=5.0, volume_impact=0.15):
    """
    Applies a realistic slippage model to returns.
    Slippage scales with a base fee plus a market impact component proportional 
    to the inverse of relative volume (capturing illiquidity).
    """
    base_slip = base_bps / 10000.0
    # Market impact: higher when volume is low relative to its rolling median
    rel_vol = volumes / volumes.rolling(20).median()
    rel_vol = rel_vol.fillna(1.0).clip(lower=0.2, upper=5.0)
    impact = volume_impact * (1.0 / rel_vol)
    
    # Slippage is always a cost, applied against the trade direction
    slip_cost = base_slip + impact.values
    net_rets = rets - np.sign(rets) * slip_cost
    return net_rets

def evaluate_slippage_sensitivity(close, signal, volumes, base_bps_range):
    """Evaluates how Profit Factor degrades as slippage increases."""
    rets = np.diff(np.log(close))
    rets = np.append(rets, np.nan)
    results = []
    
    for bps in base_bps_range:
        gross_rets = signal * rets
        net_rets = apply_microstructure_slippage(gross_rets, volumes, close, base_bps=bps)
        
        pos = np.nansum(net_rets[net_rets > 0])
        neg = -np.nansum(net_rets[net_rets < 0])
        pf = pos / neg if neg > 0 else np.inf
        
        results.append({'slippage_bps': bps, 'profit_factor': float(pf) if np.isfinite(pf) else 0.0, 'type': 'Net'})
        
    return pd.DataFrame(results)
