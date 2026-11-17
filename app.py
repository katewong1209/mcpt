"""PERMUTLAB — Monte Carlo Permutation Testing terminal for trading strategies."""
import io
import json
import threading
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from bar_permute import get_permutation
from donchian import (
    breakout_signal_np,
    optimize_donchian,
    optimize_donchian_fast,
    profit_factor_np,
    walkforward_donch,
    walkforward_donch_fast,
)
from market_data import make_synthetic_pair
from mcpt_engine import STRAT_LABELS, run_mcpt
from moving_average import ma_crossover
from tree_strat import train_tree, tree_strategy

st.set_page_config(
    page_title="PERMUTLAB · MCPT Terminal",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- palette
C = dict(
    ink="#05070d", panel="#0b101c", panel2="#101728",
    btc="#f7931a", btc2="#ffb347", cyan="#22d3ee", magenta="#ff2ea6",
    green="#2ce5a7", red="#ff4d5e", violet="#8b7bff",
    text="#e8ecf6", muted="#8b94ab",
)
PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root{
  --ink:#05070d; --panel:#0b101c; --panel2:#101728;
  --btc:#f7931a; --btc2:#ffb347; --cyan:#22d3ee; --magenta:#ff2ea6;
  --green:#2ce5a7; --red:#ff4d5e; --violet:#8b7bff;
  --text:#e8ecf6; --muted:#8b94ab; --line:rgba(247,147,26,.16);
}
html,body,[class*="st-"]{font-family:'Space Grotesk',system-ui,sans-serif;}
code,pre,.mono{font-family:'IBM Plex Mono',monospace !important;}

[data-testid="stAppViewContainer"]{
  background:
    radial-gradient(1100px 700px at 88% -12%, rgba(34,211,238,.11), transparent 60%),
    radial-gradient(950px 640px at -12% 12%, rgba(247,147,26,.13), transparent 55%),
    radial-gradient(820px 820px at 52% 112%, rgba(255,46,166,.08), transparent 62%),
    linear-gradient(180deg,#05070d 0%,#070b14 100%);
  color:var(--text);
}
[data-testid="stAppViewContainer"]::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    repeating-linear-gradient(0deg, rgba(139,148,171,.045) 0 1px, transparent 1px 52px),
    repeating-linear-gradient(90deg, rgba(139,148,171,.045) 0 1px, transparent 1px 52px);
}
[data-testid="stAppViewContainer"]::after{
  content:""; position:fixed; inset:0; z-index:9999; pointer-events:none; opacity:.05;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E");
}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stDecoration"]{visibility:hidden; height:0;}
.block-container{padding-top:1.2rem; padding-bottom:3rem; max-width:1500px;}

/* scrollbars — artifact palette */
*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:#0a0f1a}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--btc),var(--magenta));border-radius:8px;border:2px solid #0a0f1a}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--cyan),var(--violet))}
*{scrollbar-width:thin; scrollbar-color:#f7931a #0a0f1a;}

/* sidebar */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a0f1a 0%,#0b1120 100%);
  border-right:1px solid var(--line); min-width:332px; max-width:332px;
}
[data-testid="stSidebar"] h1,[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3{color:var(--text);}
.sb-sec{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.22em;
  color:var(--btc);text-transform:uppercase;margin:1.15rem 0 .35rem;
  display:flex;align-items:center;gap:.5rem;}
.sb-sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.sb-logo{font-weight:700;font-size:1.5rem;letter-spacing:-.02em;line-height:1;}
.sb-logo em{font-style:normal;background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.sb-tag{font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:var(--muted);letter-spacing:.18em;margin-top:.2rem;}

/* hero */
.hero{position:relative;padding:1.1rem 0 .4rem;}
.hero .tape{font-family:'IBM Plex Mono',monospace;font-size:.68rem;letter-spacing:.28em;color:var(--cyan);}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 10px var(--green);margin-right:.55rem;animation:pulse 1.8s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.hero h1{font-size:clamp(2.3rem,4.6vw,3.7rem);font-weight:700;letter-spacing:-.03em;
  margin:.25rem 0 .1rem;line-height:1.02;color:var(--text);}
.hero h1 .grad{background:linear-gradient(92deg,var(--btc) 5%,var(--magenta) 50%,var(--cyan) 95%);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:var(--muted);max-width:820px;font-size:.98rem;margin:.2rem 0 .7rem;}
.hero .sub b{color:var(--btc2);font-weight:600;}
.hairline{height:3px;border-radius:3px;margin:.55rem 0 1rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan),var(--btc));
  background-size:300% 100%;animation:shimmer 7s linear infinite;opacity:.85;}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:300% 0}}
.chips{display:flex;flex-wrap:wrap;gap:.45rem;margin-bottom:.4rem;}
.chip{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.08em;
  padding:.28rem .6rem;border-radius:999px;border:1px solid var(--line);
  background:rgba(16,23,40,.75);color:var(--muted);}
.chip b{color:var(--text);font-weight:600;}
.chip.hot{border-color:rgba(247,147,26,.5);color:var(--btc2);}
.chip.cool{border-color:rgba(34,211,238,.4);color:var(--cyan);}

/* cards */
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(158px,1fr));gap:.7rem;margin:.6rem 0 .9rem;}
.card{position:relative;overflow:hidden;background:linear-gradient(160deg,#0d1322,#0a0f1a);
  border:1px solid var(--line);border-radius:14px;padding:.8rem .95rem .7rem;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--btc),var(--magenta));opacity:.8;}
.card.cyan::before{background:linear-gradient(90deg,var(--cyan),var(--violet));}
.card.green::before{background:linear-gradient(90deg,var(--green),var(--cyan));}
.card.red::before{background:linear-gradient(90deg,var(--red),var(--magenta));}
.card .k{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--muted);}
.card .v{font-family:'IBM Plex Mono',monospace;font-size:1.55rem;font-weight:600;color:var(--text);margin-top:.15rem;}
.card .d{font-size:.7rem;color:var(--muted);margin-top:.1rem;}
.card .d.up{color:var(--green);} .card .d.dn{color:var(--red);} .card .d.warn{color:var(--btc2);}

/* verdict */
.verdict{border-radius:16px;padding:1rem 1.2rem;margin:.4rem 0 1rem;position:relative;overflow:hidden;
  border:1px solid;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
.verdict .big{font-weight:700;font-size:1.25rem;letter-spacing:.02em;}
.verdict .txt{color:var(--muted);font-size:.86rem;flex:1;min-width:260px;}
.verdict.sig{border-color:rgba(44,229,167,.5);background:linear-gradient(120deg,rgba(44,229,167,.12),rgba(34,211,238,.06));}
.verdict.sig .big{color:var(--green);text-shadow:0 0 18px rgba(44,229,167,.5);}
.verdict.weak{border-color:rgba(247,147,26,.5);background:linear-gradient(120deg,rgba(247,147,26,.13),rgba(255,46,166,.05));}
.verdict.weak .big{color:var(--btc2);text-shadow:0 0 18px rgba(247,147,26,.5);}
.verdict.noise{border-color:rgba(255,77,94,.5);background:linear-gradient(120deg,rgba(255,77,94,.13),rgba(255,46,166,.06));}
.verdict.noise .big{color:var(--red);text-shadow:0 0 18px rgba(255,77,94,.5);}

/* sections */
.sec{display:flex;align-items:center;gap:.65rem;margin:1.3rem 0 .5rem;}
.sec .idx{font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:var(--ink);
  background:linear-gradient(135deg,var(--btc),var(--magenta));border-radius:6px;padding:.15rem .4rem;font-weight:600;}
.sec .t{font-weight:600;font-size:1.06rem;color:var(--text);letter-spacing:.01em;}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}

/* onboarding / method cards */
.panel{background:linear-gradient(150deg,#0d1424,#0a0f1a);border:1px solid var(--line);
  border-radius:16px;padding:1.15rem 1.3rem;margin:.5rem 0;}
.panel h3{margin:.1rem 0 .5rem;font-size:1.15rem;color:var(--text);}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.8rem;margin-top:.7rem;}
.step{border:1px solid rgba(139,148,171,.18);border-radius:12px;padding:.8rem .9rem;background:rgba(11,16,28,.6);}
.step .n{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:600;
  background:linear-gradient(90deg,var(--btc),var(--magenta));-webkit-background-clip:text;background-clip:text;color:transparent;}
.step .h{font-weight:600;color:var(--text);margin:.2rem 0;}
.step .b{font-size:.8rem;color:var(--muted);line-height:1.45;}
.pd-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.pd{border-radius:14px;padding:1rem 1.1rem;border:1px solid;}
.pd.keep{border-color:rgba(44,229,167,.35);background:rgba(44,229,167,.06);}
.pd.kill{border-color:rgba(255,77,94,.35);background:rgba(255,77,94,.06);}
.pd h4{margin:0 0 .5rem;font-size:.95rem;}
.pd.keep h4{color:var(--green);} .pd.kill h4{color:var(--red);}
.pd ul{margin:0;padding-left:1.05rem;color:var(--muted);font-size:.83rem;line-height:1.65;}
.pd li b{color:var(--text);font-weight:600;}
.callout{border-left:3px solid var(--btc);background:rgba(247,147,26,.07);
  border-radius:0 12px 12px 0;padding:.8rem 1rem;color:var(--muted);font-size:.86rem;margin:.8rem 0;}
.callout b{color:var(--btc2);}

/* widgets */
button[data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.78rem;letter-spacing:.06em;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--btc2);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--btc),var(--magenta));height:3px;}
div[data-baseweb="tab-border"]{background:var(--line);}
[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--btc),var(--magenta));color:#0a0710;border:none;
  font-weight:700;letter-spacing:.04em;box-shadow:0 4px 22px rgba(247,147,26,.35);
  transition:transform .15s ease, box-shadow .15s ease;}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 28px rgba(255,46,166,.45);color:#0a0710;}
[data-testid="stBaseButton-secondary"]{background:var(--panel2);border:1px solid rgba(34,211,238,.35);color:var(--text);}
[data-testid="stBaseButton-secondary"]:hover{border-color:var(--btc);color:var(--btc2);}
[data-baseweb="slider"] div[role="slider"]{background:var(--btc);box-shadow:0 0 10px rgba(247,147,26,.6);}
[data-baseweb="select"]>div{background:var(--panel2);border-color:var(--line);}
[data-testid="stNumberInput"] input,[data-testid="stTextInput"] input{background:var(--panel2);color:var(--text);border-color:var(--line);font-family:'IBM Plex Mono',monospace;}
[data-baseweb="checkbox"] [aria-checked="true"]{background-color:var(--btc) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel2);border:1px dashed rgba(247,147,26,.4);border-radius:12px;}
[data-testid="stProgress"] div[role="progressbar"]{background:rgba(139,148,171,.15);}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line);border-radius:12px;}
[data-testid="stCodeBlock"] pre{max-height:460px;overflow:auto;background:#0a0f1a !important;
  border:1px solid var(--line);border-radius:12px;}
[data-testid="stTable"] table{background:transparent;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:.8rem;}
[data-testid="stTable"] th{color:var(--btc2) !important;border-bottom:1px solid var(--line) !important;}
[data-testid="stTable"] td{border-bottom:1px solid rgba(139,148,171,.12) !important;}
[data-testid="stAlert"]{border-radius:12px;}
.hist-row{display:flex;align-items:center;gap:.6rem;padding:.45rem .6rem;border:1px solid rgba(139,148,171,.14);
  border-radius:10px;margin-bottom:.4rem;background:rgba(11,16,28,.55);font-family:'IBM Plex Mono',monospace;font-size:.72rem;}
.hist-row .lbl{flex:1;color:var(--text);}
.hist-row .m{color:var(--muted);}
.badge{padding:.12rem .5rem;border-radius:999px;font-size:.62rem;font-weight:600;}
.badge.sig{background:rgba(44,229,167,.15);color:var(--green);}
.badge.weak{background:rgba(247,147,26,.15);color:var(--btc2);}
.badge.noise{background:rgba(255,77,94,.15);color:var(--red);}
.foot{margin-top:2.2rem;padding-top:.9rem;border-top:1px solid var(--line);
  font-family:'IBM Plex Mono',monospace;font-size:.64rem;letter-spacing:.14em;color:var(--muted);}
@media (max-width:760px){
  [data-testid="stSidebar"]{min-width:0;max-width:none;}
  .pd-grid{grid-template-columns:1fr;}
  .card .v{font-size:1.25rem;}
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------- helpers
def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CFG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key, config=PLOTLY_CFG)


def base_fig(height=360, title=None):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,26,.55)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#aab3c8"),
        margin=dict(l=48, r=18, t=44 if title else 18, b=42),
        title=dict(text=title, x=0.01, font=dict(size=13, color=C["text"], family="Space Grotesk")),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["btc"], font=dict(family="IBM Plex Mono", size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    return fig


def hist_fig(perm_pfs, real_pf, p, live=False):
    vals = np.asarray([v for v in perm_pfs if np.isfinite(v)], dtype=float)
    fig = base_fig(380, "Optimized PF on permuted noise vs real market" + (" · LIVE" if live else ""))
    if len(vals) == 0:
        fig.add_annotation(text="collecting permutations…", xref="paper", yref="paper", x=.5, y=.5, showarrow=False, font=dict(color=C["muted"]))
        return fig
    lo = min(vals.min(), real_pf if np.isfinite(real_pf) else vals.min())
    hi = max(vals.max(), real_pf if np.isfinite(real_pf) else vals.max())
    pad = max((hi - lo) * 0.05, 0.01)
    counts, edges = np.histogram(vals, bins=42, range=(lo - pad, hi + pad))
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * .92,
                         marker_color=C["cyan"], opacity=.78, name="Permuted best PF",
                         hovertemplate="PF %{x:.3f}<br>count %{y}<extra></extra>"))
    if np.isfinite(real_pf):
        fig.add_vrect(x0=real_pf, x1=hi + pad, fillcolor=C["red"], opacity=.08, line_width=0)
        fig.add_vline(x=real_pf, line_color=C["btc"], line_width=2.5, line_dash="dash",
                      annotation_text=f"REAL {real_pf:.3f}", annotation_position="top",
                      annotation_font=dict(color=C["btc2"], size=11))
    med = float(np.median(vals))
    fig.add_vline(x=med, line_color=C["violet"], line_width=1.4, line_dash="dot",
                  annotation_text=f"noise median {med:.3f}", annotation_position="bottom",
                  annotation_font=dict(color=C["violet"], size=10))
    vcol = C["green"] if p <= .05 else (C["btc2"] if p <= .2 else C["red"])
    fig.add_annotation(xref="paper", yref="paper", x=.99, y=.99, xanchor="right",
                       text=f"<b>p = {p:.3f}</b>", showarrow=False,
                       font=dict(color=vcol, size=17, family="Space Grotesk"))
    fig.update_xaxes(title="Profit Factor")
    fig.update_yaxes(title="permutations")
    return fig


def equity_fig(x, real_eq, samples, title="Equity — real vs permuted universes (cumulative log return)"):
    fig = base_fig(380, title)
    for i, s in enumerate(samples):
        fig.add_trace(go.Scatter(x=x[: len(s)], y=s, mode="lines",
                                 line=dict(color=C["cyan"], width=1), opacity=.26,
                                 showlegend=(i == 0), name="Permuted strategies", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[: len(real_eq)], y=real_eq, mode="lines",
                             line=dict(color=C["btc"], width=2.6), name="Real strategy"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.35)", line_width=1)
    fig.update_yaxes(title="Σ log return")
    return fig


def optcurve_fig(lookbacks, curve, best_L, mean=None, std=None):
    fig = base_fig(360, "Data-mining surface — PF vs lookback · real vs noise envelope")
    if mean is not None:
        m = np.isfinite(mean) & np.isfinite(std)
        xl = np.asarray(lookbacks)[m]
        fig.add_trace(go.Scatter(x=np.concatenate([xl, xl[::-1]]),
                                 y=np.concatenate([(mean + std)[m], (mean - std)[m][::-1]]),
                                 fill="toself", fillcolor="rgba(139,123,255,.16)",
                                 line=dict(width=0), name="Noise ±1σ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xl, y=mean[m], mode="lines",
                                 line=dict(color=C["cyan"], width=1.6, dash="dot"), name="Noise mean best-PF"))
    fig.add_trace(go.Scatter(x=lookbacks, y=curve, mode="lines",
                             line=dict(color=C["btc"], width=2.2), name="Real PF(lookback)"))
    if best_L and best_L > 0:
        k = int(np.argmin(np.abs(np.asarray(lookbacks) - best_L)))
        fig.add_trace(go.Scatter(x=[lookbacks[k]], y=[curve[k]], mode="markers",
                                 marker=dict(symbol="star", size=14, color=C["magenta"],
                                             line=dict(color=C["text"], width=1)),
                                 name=f"Selected L={best_L}"))
    fig.add_hline(y=1.0, line_color="rgba(139,148,171,.4)", line_dash="dash", line_width=1)
    fig.update_xaxes(title="Donchian lookback (bars)")
    fig.update_yaxes(title="Profit Factor")
    return fig


def candle_fig(daily, overlays, markers, title):
    fig = base_fig(440, title)
    fig.add_trace(go.Candlestick(
        x=daily.index, open=daily["open"], high=daily["high"], low=daily["low"], close=daily["close"],
        increasing_line_color=C["green"], increasing_fillcolor="rgba(44,229,167,.55)",
        decreasing_line_color=C["red"], decreasing_fillcolor="rgba(255,77,94,.55)",
        name="OHLC · 1D", whiskerwidth=.4))
    for name, (series, color, dash) in overlays.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.to_numpy(), mode="lines",
                                 line=dict(color=color, width=1.5, dash=dash), name=name))
    mt, mp, md = markers
    up = md > 0
    fig.add_trace(go.Scatter(x=mt[up], y=mp[up], mode="markers", name="Turn long",
                             marker=dict(symbol="triangle-up", size=9, color=C["green"], line=dict(color=C["ink"], width=1))))
    fig.add_trace(go.Scatter(x=mt[~up], y=mp[~up], mode="markers", name="Turn short/flat",
                             marker=dict(symbol="triangle-down", size=9, color=C["magenta"], line=dict(color=C["ink"], width=1))))
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    return fig


def dd_fig(x, dd):
    fig = base_fig(210, "Drawdown (log-equity units)")
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color=C["red"], width=1.2),
                             fillcolor="rgba(255,77,94,.25)", name="Drawdown"))
    return fig


def acf_fig(lags, real, perm, n, title):
    fig = base_fig(300, title)
    fig.add_trace(go.Bar(x=lags, y=real, marker_color=C["btc"], opacity=.85, width=.42, name="Real"))
    fig.add_trace(go.Bar(x=lags, y=perm, marker_color=C["cyan"], opacity=.6, width=.42, name="Permuted"))
    b = 1.96 / np.sqrt(n)
    fig.add_hline(y=b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.add_hline(y=-b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.update_layout(barmode="group", bargap=.15)
    fig.update_xaxes(title="lag (hours)")
    return fig


def spec_fig(fx, px, fy, py):
    fig = base_fig(320, "Power spectrum of |returns| — long memory lives here (log-log)")
    fig.add_trace(go.Scatter(x=fx, y=px, mode="lines", line=dict(color=C["btc"], width=2), name="Real |r|"))
    fig.add_trace(go.Scatter(x=fy, y=py, mode="lines", line=dict(color=C["cyan"], width=1.6, dash="dot"), name="Permuted |r|"))
    fig.update_xaxes(type="log", title="frequency (cycles / bar)")
    fig.update_yaxes(type="log", title="power")
    return fig


def corr_fig(rb, re, pb, pe, rho_r, rho_p):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Real · ρ = {rho_r:.3f}", f"Joint permutation · ρ = {rho_p:.3f}"))
    fig.add_trace(go.Scattergl(x=rb, y=re, mode="markers", marker=dict(size=3, color=C["btc"], opacity=.45), name="Real"), 1, 1)
    fig.add_trace(go.Scattergl(x=pb, y=pe, mode="markers", marker=dict(size=3, color=C["cyan"], opacity=.45), name="Permuted"), 1, 2)
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,.55)",
                      font=dict(family="IBM Plex Mono", size=11, color="#aab3c8"),
                      margin=dict(l=48, r=18, t=48, b=42), showlegend=False,
                      title=dict(text="Cross-market correlation survives joint permutation", x=.01,
                                 font=dict(size=13, color=C["text"], family="Space Grotesk")))
    fig.update_xaxes(gridcolor="rgba(139,148,171,.1)", title="BTC log return")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.1)", title="ETH log return")
    for a in fig.layout.annotations:
        a.font = dict(family="IBM Plex Mono", size=11, color=C["muted"])
    return fig


def imp_fig(importances):
    names = ["diff6 (6h momentum)", "diff24 (1d momentum)", "diff168 (1w momentum)"]
    fig = base_fig(230, "Tree feature importances — what the classifier actually uses")
    fig.add_trace(go.Bar(y=names, x=importances, orientation="h",
                         marker_color=[C["btc"], C["cyan"], C["magenta"]], opacity=.85,
                         text=[f"{v:.1%}" for v in importances], textposition="outside",
                         textfont=dict(color=C["text"])))
    fig.update_xaxes(title="Gini importance", range=[0, max(importances) * 1.25 + 1e-6])
    return fig


def verdict(p):
    if p <= .05:
        return "EDGE DETECTED", "sig", C["green"]
    if p <= .20:
        return "WEAK / INCONCLUSIVE", "weak", C["btc2"]
    return "CONSISTENT WITH NOISE", "noise", C["red"]


def cards_html(items):
    h = '<div class="cards">'
    for it in items:
        cls = it.get("cls", "")
        d = it.get("d", "")
        dc = it.get("dc", "")
        h += f'<div class="card {cls}"><div class="k">{it["k"]}</div><div class="v">{it["v"]}</div>'
        if d:
            h += f'<div class="d {dc}">{d}</div>'
        h += "</div>"
    return h + "</div>"


def sec(idx, title):
    st.markdown(f'<div class="sec"><span class="idx">{idx}</span><span class="t">{title}</span></div>', unsafe_allow_html=True)


def fmt_t(s):
    s = int(max(s, 0))
    return f"{s//60:02d}:{s%60:02d}"


def acf(x, nlags):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 ** int(np.ceil(np.log2(2 * n))))
    ac = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    return ac / ac[0], n


def logbin_spectrum(x, nbins=46):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    n = len(x)
    p = (np.abs(np.fft.rfft(x * np.hanning(n))) ** 2)[1:]
    fr = np.fft.rfftfreq(n)[1:]
    edges = np.unique(np.logspace(0, np.log10(len(p)), nbins + 1).astype(int))
    fx, px = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            fx.append(float(np.exp(np.mean(np.log(fr[a:b])))))
            px.append(float(np.mean(p[a:b])))
    return np.asarray(fx), np.asarray(px)


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Synthesizing BTC/ETH hourly feed…")
def synth_pair(seed=7):
    return make_synthetic_pair(seed)


@st.cache_data(show_spinner=False)
def parse_bytes(name, data):
    bio = io.BytesIO(data)
    if name.lower().endswith(".csv"):
        df = pd.read_csv(bio)
        c0 = df.columns[0]
        df[c0] = pd.to_datetime(df[c0])
        df = df.set_index(c0)
    else:
        df = pd.read_parquet(bio)
    df.columns = [str(c).lower() for c in df.columns]
    df.index = pd.to_datetime(df.index).astype("datetime64[s]")
    df = df.sort_index()
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"{name}: needs columns open/high/low/close")
    return df[["open", "high", "low", "close"]].dropna()


@st.cache_data(show_spinner=False)
def load_path(path, mtime):
    return parse_bytes(path, Path(path).read_bytes())


def resolve_data(source, up_btc, up_eth):
    eth = None
    if source == "Upload files":
        if up_btc is None:
            b, e = synth_pair()
            return b, e, "SYNTHETIC (awaiting upload)"
        btc = parse_bytes(up_btc.name, up_btc.getvalue())
        if up_eth is not None:
            eth = parse_bytes(up_eth.name, up_eth.getvalue())
        return btc, eth, f"UPLOAD · {up_btc.name}"
    if source == "Auto-detect files":
        p = Path("BTCUSD3600.pq")
        if p.exists():
            btc = load_path(str(p), p.stat().st_mtime)
            pe = Path("ETHUSD3600.pq")
            if pe.exists():
                eth = load_path(str(pe), pe.stat().st_mtime)
            return btc, eth, "BTCUSD3600.pq" + (" + ETH" if eth is not None else "")
    b, e = synth_pair()
    return b, e, "SYNTHETIC DEMO FEED"


# ---------------------------------------------------------------- callbacks
def collect_cfg():
    g = st.session_state.get
    depth = g("w_depth", "Unlimited")
    lbmin = int(g("w_lbmin", 12))
    lbmax = int(g("w_lbmax", 168))
    if lbmin >= lbmax:
        lbmin, lbmax = min(lbmin, lbmax - 1), max(lbmax, lbmin + 1)
    return dict(
        strategy=g("w_strategy", "donchian_is"),
        n_perms=int(g("w_perms", 1000)),
        seed=int(g("w_seed", 69)),
        turbo=bool(g("w_turbo", True)),
        lb_min=lbmin, lb_max=lbmax, lb_step=int(g("w_lbstep", 1)),
        is_start=int(g("w_is_years", (2016, 2020))[0]),
        is_end=int(g("w_is_years", (2016, 2020))[1]),
        wf_start=int(g("w_wf_years", (2016, 2021))[0]),
        wf_end=int(g("w_wf_years", (2016, 2021))[1]),
        wf_train_years=int(g("w_wf_train", 4)),
        wf_step_days=int(g("w_wf_step", 30)),
        tree_leaf=int(g("w_leaf", 5)),
        tree_depth=None if depth == "Unlimited" else int(depth),
        ma_fast=int(g("w_maf", 10)), ma_slow=int(g("w_mas", 30)),
        ma_short=bool(g("w_mashort", False)),
    )


def start_run():
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        return
    btc = st.session_state.get("btc_df")
    if btc is None or len(btc) < 500:
        st.session_state["run_error"] = "Not enough data loaded to test."
        return
    if active and "stop" in active:
        active["stop"].set()
    cfg = collect_cfg()
    job = dict(
        id=uuid.uuid4().hex[:10], status="running", phase="Initializing…",
        config=cfg, start=time.time(), done=0, better=1, p=1.0,
        perm_pfs=[], sample_curves=[], real_pf=float("nan"),
        best_lookback=None, lookbacks=None, pf_curve=None,
        env_cnt=None, env_sum=None, env_ssq=None,
        real_curve=None, curve_x=None, importances=None, wf_choices=None,
        window_desc="", label=STRAT_LABELS[cfg["strategy"]],
        stop=threading.Event(), elapsed=0.0, error=None, _finalized=False,
    )
    th = threading.Thread(target=run_mcpt, args=(job, btc.copy(), cfg), daemon=True)
    job["thread"] = th
    st.session_state["active"] = job
    st.session_state["run_error"] = None
    th.start()


def request_stop():
    a = st.session_state.get("active")
    if a and "stop" in a:
        a["stop"].set()


def clear_active():
    a = st.session_state.get("active")
    if a and a.get("status") == "running":
        return
    st.session_state["active"] = None


def randomize_seed():
    st.session_state["w_seed"] = int(np.random.randint(0, 2**31 - 4096))


def preset_quick():
    st.session_state["w_perms"] = 200


def preset_full():
    st.session_state["w_perms"] = 1000


def finalize(job):
    job["p"] = job["better"] / (job["done"] + 1)
    job["verdict"] = verdict(job["p"])
    job["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    job["_finalized"] = True
    saved = {k: v for k, v in job.items() if k not in ("stop", "thread")}
    saved["perm_pfs"] = list(job["perm_pfs"])
    hist = st.session_state.setdefault("history", [])
    hist.insert(0, saved)
    del hist[16:]


def load_run(run_id):
    for r in st.session_state.get("history", []):
        if r["id"] == run_id:
            st.session_state["active"] = r
            return


def delete_run(run_id):
    st.session_state["history"] = [r for r in st.session_state.get("history", []) if r["id"] != run_id]


def reshuffle():
    st.session_state["w_anat_seed"] = int(st.session_state.get("w_anat_seed", 42)) + 1


# ---------------------------------------------------------------- strategy details
@st.cache_data(show_spinner="Computing strategy…")
def compute_details(btc, cfg):
    try:
        strat = cfg["strategy"]
        out = {"error": None}
        if strat in ("donchian_is", "tree"):
            win = btc[(btc.index.year >= cfg["is_start"]) & (btc.index.year < cfg["is_end"])]
        elif strat == "donchian_wf":
            win = btc[(btc.index.year >= cfg["wf_start"]) & (btc.index.year < cfg["wf_end"])]
        else:
            win = btc
        if len(win) < 400:
            return {"error": "Window too short — widen the year range."}
        c = win["close"].to_numpy()
        lc = np.log(c)
        r = np.empty(len(c)); r[:-1] = np.diff(lc); r[-1] = np.nan
        overlays = {}
        extra = {}

        if strat == "donchian_is":
            if cfg["turbo"]:
                bL, pf, lbs, curve = optimize_donchian_fast(c, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"])
                extra["opt"] = (lbs, curve, bL)
            else:
                bL, pf = optimize_donchian(win, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"])
            sig = breakout_signal_np(c, bL) if bL > 0 else np.full(len(c), np.nan)
            s = win["close"]
            overlays[f"Upper · L={bL}"] = (s.rolling(bL - 1).max().shift(1), C["btc2"], "dot")
            overlays[f"Lower · L={bL}"] = (s.rolling(bL - 1).min().shift(1), C["violet"], "dot")
            extra["best_L"] = bL
        elif strat == "donchian_wf":
            tw = cfg["wf_train_years"] * 24 * 365
            tstep = cfg["wf_step_days"] * 24
            if len(win) <= tw + 10:
                return {"error": "Walk-forward window must exceed the training window."}
            if cfg["turbo"]:
                sig, choices = walkforward_donch_fast(c, tw, tstep, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"], return_choices=True)
                extra["wf_choices"] = choices
            else:
                sig = walkforward_donch(win, tw, tstep, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"])
        elif strat == "tree":
            model = train_tree(win, cfg["tree_leaf"], cfg["tree_depth"])
            sig_s, pf = tree_strategy(win, model)
            sig = sig_s.to_numpy()
            extra["importances"] = model.feature_importances_.tolist()
        else:
            sig_s, pf = ma_crossover(win, cfg["ma_fast"], cfg["ma_slow"], cfg["ma_short"])
            sig = sig_s.to_numpy()
            overlays[f"MA {cfg['ma_fast']}"] = (win["close"].rolling(cfg["ma_fast"]).mean(), C["cyan"], "solid")
            overlays[f"MA {cfg['ma_slow']}"] = (win["close"].rolling(cfg["ma_slow"]).mean(), C["magenta"], "solid")

        rets = sig * r
        pf = profit_factor_np(rets)
        rs = pd.Series(rets)
        eq = np.nancumsum(rets)
        step = max(1, len(eq) // 2600)
        dd = eq - np.maximum.accumulate(eq)
        chg = np.flatnonzero((sig[1:] != sig[:-1]) & np.isfinite(sig[1:]) & np.isfinite(sig[:-1])) + 1
        if len(chg) > 350:
            chg = chg[:: max(1, len(chg) // 350)]
        dirs = np.where(sig[chg] > sig[chg - 1], 1.0, -1.0) if len(chg) else np.array([])
        daily = win.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        dov = {}
        for name, (ser, col, dash) in overlays.items():
            dov[name] = (ser.resample("1D").last(), col, dash)
        nz = rs.dropna()
        nz = nz[nz != 0]
        out.update(
            daily=daily, overlays=dov,
            markers=(win.index[chg], c[chg], dirs),
            eq_x=win.index[::step], eq=eq[::step], dd=dd[::step],
            stats=dict(
                pf=pf, sharpe=float(rs.mean() / rs.std()) if rs.std() > 0 else float("nan"),
                total=float(np.nansum(rets)),
                exposure=float(np.nanmean(np.abs(sig))) if np.isfinite(sig).any() else 0.0,
                trades=int(len(chg)),
                winrate=float((nz > 0).mean()) if len(nz) else float("nan"),
                bars=len(win),
            ),
            extra=extra,
        )
        return out
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- anatomy
@st.cache_data(show_spinner="Shuffling the universe…")
def anatomy(btc, eth, seed, nlags):
    yrs = btc.index.year
    sl = btc[(yrs >= 2018) & (yrs < 2020)]
    if len(sl) < 3000:
        sl = btc.iloc[-min(len(btc), 17520):]
    perm = get_permutation(sl, seed=seed)
    rr = np.log(sl["close"]).diff().dropna().to_numpy()
    pr = np.log(perm["close"]).diff().dropna().to_numpy()
    moments = pd.DataFrame(
        {
            "REAL": [rr.mean(), rr.std(), pd.Series(rr).skew(), pd.Series(rr).kurt()],
            "PERMUTED": [pr.mean(), pr.std(), pd.Series(pr).skew(), pd.Series(pr).kurt()],
        },
        index=["Mean", "Std Dev", "Skew", "Kurtosis (excess)"],
    )
    ar, n = acf(rr, nlags)
    ap, _ = acf(pr, nlags)
    aar, _ = acf(np.abs(rr), nlags)
    aap, _ = acf(np.abs(pr), nlags)
    fxr, pxr = logbin_spectrum(np.abs(rr))
    fxp, pxp = logbin_spectrum(np.abs(pr))
    lcr = np.log(sl["close"].to_numpy()); lcr -= lcr[0]
    lcp = np.log(perm["close"].to_numpy()); lcp -= lcp[0]
    step = max(1, len(sl) // 2200)
    out = dict(
        x=sl.index[::step], lcr=lcr[::step], lcp=lcp[::step],
        moments=moments, lags=np.arange(nlags + 1),
        ar=ar, ap=ap, aar=aar, aap=aap, n=n,
        fxr=fxr, pxr=pxr, fxp=fxp, pxp=pxp, corr=None,
    )
    if eth is not None:
        e = eth.reindex(sl.index).dropna()
        common = sl.index.intersection(e.index)
        if len(common) > 1000:
            b2 = sl.loc[common]; e2 = e.loc[common]
            pb, pe = get_permutation([b2, e2], seed=seed)
            rb = np.log(b2["close"]).diff().dropna()
            re = np.log(e2["close"]).diff().dropna()
            qb = np.log(pb["close"]).diff().dropna()
            qe = np.log(pe["close"]).diff().dropna()
            k = min(2600, len(rb))
            sel = np.linspace(0, len(rb) - 1, k).astype(int)
            out["corr"] = dict(
                rb=rb.to_numpy()[sel], re=re.to_numpy()[sel],
                pb=qb.to_numpy()[sel], pe=qe.to_numpy()[sel],
                rho_r=float(rb.corr(re)), rho_p=float(qb.corr(qe)),
            )
    return out


# ---------------------------------------------------------------- results rendering
def result_body(job, live=False):
    cfg = job["config"]
    done, better = job["done"], job["better"]
    p = better / (done + 1)
    real_pf = job.get("real_pf", float("nan"))
    n_target = cfg["n_perms"]

    if not live:
        vlabel, vcls, _ = job.get("verdict", verdict(p))
        finite = [v for v in job["perm_pfs"] if np.isfinite(v)]
        med = float(np.median(finite)) if finite else float("nan")
        lift = real_pf - med if finite else float("nan")
        if vcls == "sig":
            txt = f"Only {better-1} of {done} permuted universes matched or beat the real optimized PF. Selection bias alone rarely explains this — evidence of a genuine, exploitable pattern."
        elif vcls == "weak":
            txt = f"{better-1} of {done} noise universes beat the real result. Suggestive, not conclusive. More permutations, out-of-sample validation, and economic rationale are warranted."
        else:
            txt = f"{better-1} of {done} structureless universes produced an equally good 'edge'. The optimizer is mining noise — classic data-mining bias. Treat this strategy as unproven."
        st.markdown(
            f'<div class="verdict {vcls}"><span class="big">{"◆ " if vcls=="sig" else "◇ "}{vlabel}</span>'
            f'<span class="txt">{txt}</span></div>', unsafe_allow_html=True)
        items = [
            dict(k="Real Profit Factor", v=f"{real_pf:.4f}", cls=""),
            dict(k="MCPT p-value", v=f"{p:.3f}", cls="green" if vcls == "sig" else ("red" if vcls == "noise" else ""), d=f"{better} / {done+1} ≥ real", dc="warn"),
            dict(k="Permutations", v=f"{done:,}", cls="cyan", d=f"target {n_target:,}" + (" · stopped early" if job["status"] == "stopped" else "")),
        ]
        if job.get("best_lookback"):
            items.append(dict(k="Best Lookback", v=str(job["best_lookback"]), cls="cyan", d="selected on real bars"))
        if finite:
            items.append(dict(k="Edge vs noise median", v=f"{lift:+.4f}", cls="green" if lift > 0 else "red", d=f"noise median {med:.4f}", dc="up" if lift > 0 else "dn"))
        items.append(dict(k="Elapsed", v=fmt_t(job.get("elapsed", 0)), cls="cyan", d=job.get("finished_at", "")))
        st.markdown(cards_html(items), unsafe_allow_html=True)

    colA, colB = st.columns([1.35, 1])
    with colA:
        show_fig(hist_fig(job["perm_pfs"], real_pf, p, live), key=f"hist_{job['id']}_{live}")
    with colB:
        if job.get("real_curve") is not None:
            show_fig(equity_fig(job["curve_x"], job["real_curve"], job["sample_curves"]), key=f"eq_{job['id']}_{live}")
        else:
            st.info("Equity curves appear once the baseline finishes.")

    if job.get("pf_curve") is not None and job.get("env_cnt") is not None:
        cnt = job["env_cnt"]
        mean = np.where(cnt > 0, job["env_sum"] / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 0, job["env_ssq"] / np.maximum(cnt, 1) - mean**2, np.nan)
        std = np.sqrt(np.maximum(var, 0))
        show_fig(optcurve_fig(job["lookbacks"], job["pf_curve"], job.get("best_lookback"), mean, std), key=f"opt_{job['id']}_{live}")
        st.caption("The amber curve is the real optimization surface; the cyan/violet envelope is what the same optimizer achieves on pure noise. The gap between them — not the peak itself — is the evidence.")

    if job.get("wf_choices"):
        ch = job["wf_choices"]
        fig = base_fig(240, "Walk-forward — lookback re-selected at each training step")
        fig.add_trace(go.Scatter(x=[job["curve_x"][0]] + [job["curve_x"][min(t, len(job['curve_x'])-1)] for t, _ in ch],
                                 y=[ch[0][1]] + [b for _, b in ch], mode="lines",
                                 line=dict(color=C["magenta"], width=2, shape="hv"), name="Chosen lookback"))
        show_fig(fig, key=f"wfch_{job['id']}")

    if not live:
        with st.expander("⚠ What this null hypothesis does — and does not — rule out"):
            st.markdown(
                "Bar permutation preserves the **marginal distribution** (mean, std, skew, kurtosis), intrabar geometry and gaps, "
                "but destroys **volatility clustering**, **long memory**, **self-similarity** and spectral structure — anything involving dependence. "
                "A strategy that unknowingly harvests volatility clustering is being tested against a null where that property is absent; "
                "a significant p-value here means the edge needs *serial structure*, not merely fat tails. "
                "Conversely, a non-significant result does not prove absence of every conceivable edge — only that this pipeline cannot distinguish itself from shuffled noise."
            )
        e1, e2, e3 = st.columns(3)
        payload = dict(
            id=job["id"], label=job["label"], config=cfg, real_pf=real_pf,
            p_value=p, better=better, completed=done,
            best_lookback=job.get("best_lookback"),
            perm_pfs=job["perm_pfs"], finished_at=job.get("finished_at"),
        )
        e1.download_button("⬇ Results (.json)", json.dumps(payload, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=2),
                           file_name=f"mcpt_{job['id']}.json", mime="application/json", key=f"js_{job['id']}")
        e2.download_button("⬇ Permuted PFs (.csv)", "perm_pf\n" + "\n".join(f"{v:.6f}" for v in job["perm_pfs"]),
                           file_name=f"perm_pfs_{job['id']}.csv", mime="text/csv", key=f"csv_{job['id']}")
        e3.button("↻ Re-run with current settings", on_click=start_run, key=f"re_{job['id']}")


def onboarding():
    st.markdown(
        """
<div class="panel">
  <h3>Arm the test — is your optimized edge real, or the best-looking lie in 157 tries?</h3>
  <div style="color:var(--muted);font-size:.88rem;max-width:860px">
    Optimization <b style="color:var(--btc2)">always</b> induces data-mining bias. PERMUTLAB rebuilds the market as
    structureless noise — same distribution, zero memory — re-runs your <i>entire</i> pipeline on each shuffled universe,
    and asks how often noise alone beats your real result. Aim for <b style="color:var(--cyan)">1000 permutations</b>.
  </div>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Optimize on real bars</div><div class="b">Donchian lookback sweep, walk-forward re-training, decision tree, or MA crossover — the full selection procedure, bias included.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Shuffle 1000 universes</div><div class="b">Intrabar shape and gaps permuted separately. Mean/std/skew/kurt survive; volatility clustering and long memory die.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Compare &amp; verdict</div><div class="b">p = (1 + #noise-PF ≥ real-PF) / N. If noise matches your edge, the edge is a mirage. Watch the histogram build live.</div></div>
  </div>
  <div class="callout" style="margin-top:.9rem"><b>→ Pick a strategy and press RUN MCPT in the sidebar.</b> The Turbo engine streams results as they compute; Stop anytime.</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.6)
def live_fragment():
    job = st.session_state.get("active")
    if job is None or job.get("status") != "running":
        st.rerun(scope="app")
        return
    if job["status"] == "error":
        finalize(job)
        st.rerun(scope="app")
        return
    cfg = job["config"]
    n_target = cfg["n_perms"]
    done = job["done"]
    elapsed = time.time() - job["start"]
    eta = (elapsed / done) * (n_target - 1 - done) if done > 0 else None
    prog = done / max(n_target - 1, 1)
    st.progress(min(prog, 1.0))
    p = job["better"] / (done + 1)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(cards_html([
            dict(k="Phase", v="⚙ LIVE", cls="cyan", d=job.get("phase", "")),
            dict(k="Completed", v=f"{done:,}", d=f"of {n_target-1:,} permutations"),
            dict(k="Noise beats real", v=str(job["better"] - 1), cls="red" if job["better"] - 1 > done * .2 else "green", d="count ≥ real PF", dc="warn"),
            dict(k="Live p-value", v=f"{p:.3f}", cls="cyan"),
            dict(k="Elapsed / ETA", v=fmt_t(elapsed), cls="cyan", d=f"≈ {fmt_t(eta)} left" if eta is not None else "measuring…"),
        ]), unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)
        st.button("■ STOP", on_click=request_stop, key="stop_btn", use_container_width=True)
        st.caption(f"seed {cfg['seed']} · {'turbo' if cfg['turbo'] else 'reference'}")
    if np.isfinite(job.get("real_pf", float("nan"))):
        result_body(job, live=True)
    else:
        st.info(f"{job.get('phase','Working')} — the real-data baseline (full optimization) runs first, then permutations stream in.")

    if job["status"] != "running":
        finalize(job)
        st.rerun(scope="app")


def history_block():
    hist = st.session_state.get("history", [])
    if not hist:
        return
    sec("H", "Run history")
    for r in hist:
        vlabel, vcls, _ = r.get("verdict", verdict(r["p"]))
        c1, c2, c3 = st.columns([6, 1, 1])
        with c1:
            st.markdown(
                f'<div class="hist-row"><span class="lbl">{r["label"]}</span>'
                f'<span class="m">n={r["done"]+1} · PF {r["real_pf"]:.3f} · p {r["p"]:.3f}</span>'
                f'<span class="badge {vcls}">{vlabel}</span>'
                f'<span class="m">{r.get("finished_at","")}</span></div>', unsafe_allow_html=True)
        c2.button("view", key=f"v_{r['id']}", on_click=load_run, args=(r["id"],))
        c3.button("✕", key=f"d_{r['id']}", on_click=delete_run, args=(r["id"],))
    if st.button("Clear history", key="clr_hist"):
        st.session_state["history"] = []
        st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="sb-logo">PERMUT<em>LAB</em></div><div class="sb-tag">MCPT RESEARCH TERMINAL</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">01 · Market feed</div>', unsafe_allow_html=True)
    source = st.selectbox("Data source", ["Auto-detect files", "Synthetic demo feed", "Upload files"], key="w_source")
    up_btc = up_eth = None
    if source == "Upload files":
        up_btc = st.file_uploader("BTCUSD hourly (.pq / .parquet / .csv)", type=["pq", "parquet", "csv"], key="up_btc")
        up_eth = st.file_uploader("ETHUSD hourly (optional, for correlation demo)", type=["pq", "parquet", "csv"], key="up_eth")

    try:
        btc_df, eth_df, src_label = resolve_data(source, up_btc, up_eth)
        data_err = None
    except Exception as e:
        btc_df, eth_df = synth_pair()
        src_label, data_err = "SYNTHETIC (parse failed)", str(e)
    st.session_state["btc_df"] = btc_df
    st.session_state["eth_df"] = eth_df
    if data_err:
        st.error(data_err)
    ymin = int(btc_df.index.year.min())
    ymax = int(btc_df.index.year.max()) + 1
    st.caption(f"{src_label} · {len(btc_df):,} hourly bars · {ymin}–{ymax-1}")

    st.markdown('<div class="sb-sec">02 · Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Pipeline under test", list(STRAT_LABELS.keys()), format_func=lambda k: STRAT_LABELS[k], key="w_strategy")

    if strategy in ("donchian_is", "donchian_wf"):
        ca, cb = st.columns(2)
        ca.slider("Lookback min", 4, 60, 12, key="w_lbmin")
        cb.slider("Lookback max", 60, 240, 168, key="w_lbmax")
        st.slider("Grid step", 1, 12, 1, key="w_lbstep")
    if strategy in ("donchian_is", "tree"):
        d0 = (max(ymin, min(2016, ymax - 1)), min(ymax, max(2020, ymin + 1)))
        if d0[0] >= d0[1]:
            d0 = (ymin, ymax)
        st.slider("Training years (end exclusive)", ymin, ymax, d0, key="w_is_years")
    if strategy == "donchian_wf":
        d1 = (ymin, ymax)
        st.slider("Walk-forward years", ymin, ymax, d1, key="w_wf_years")
        st.slider("Training window (years)", 1, 4, 4, key="w_wf_train")
        st.slider("Re-optimize every (days)", 5, 90, 30, key="w_wf_step")
    if strategy == "tree":
        st.slider("min_samples_leaf", 1, 100, 5, key="w_leaf")
        st.selectbox("max_depth", ["Unlimited", 4, 6, 8, 12], key="w_depth")
    if strategy == "ma":
        ca, cb = st.columns(2)
        ca.slider("Fast MA", 2, 60, 10, key="w_maf")
        cb.slider("Slow MA", 10, 240, 30, key="w_mas")
        st.toggle("Allow shorts (else long/flat)", key="w_mashort")

    st.markdown('<div class="sb-sec">03 · Permutation engine</div>', unsafe_allow_html=True)
    st.slider("Permutations — aim for 1000", 50, 2000, 1000, step=50, key="w_perms")
    pa, pb = st.columns(2)
    pa.button("⚡ Quick · 200", on_click=preset_quick, use_container_width=True)
    pb.button("🎯 Research · 1000", on_click=preset_full, use_container_width=True)
    sa, sb = st.columns([3, 1])
    sa.number_input("Master seed", 0, 2**31 - 4096, 69, key="w_seed")
    sb.markdown("<div style='height:1.62rem'></div>", unsafe_allow_html=True)
    sb.button("🎲", on_click=randomize_seed, help="Randomize seed", use_container_width=True)
    st.toggle("Turbo engine (vectorized, identical math)", value=True, key="w_turbo")
    if strategy == "donchian_wf":
        st.caption("Walk-forward re-optimizes inside every permutation — the heaviest test. Turbo strongly recommended; 200 perms is a sane first pass.")
    elif not st.session_state.get("w_turbo", True):
        st.caption("Reference engine runs the literal pandas loops — expect long runtimes at 1000 permutations.")

    st.markdown('<div class="sb-sec">04 · Execute</div>', unsafe_allow_html=True)
    active = st.session_state.get("active")
    running = bool(active and active.get("status") == "running")
    st.button("🧪 RUN MCPT", type="primary", on_click=start_run, disabled=running, use_container_width=True)
    st.button("Clear results", on_click=clear_active, disabled=running, use_container_width=True)
    if running:
        st.caption(f"● running — {active['done']:,} / {active['config']['n_perms']-1:,} permutations")
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])


# ---------------------------------------------------------------- header
cfg_now = collect_cfg()
st.markdown(
    f"""
<div class="hero">
  <div><span class="pulse"></span><span class="tape">BTCUSD · 1H · MONTE CARLO PERMUTATION TESTING</span></div>
  <h1>PERMUT<span class="grad">LAB</span></h1>
  <div class="sub">Optimization always induces <b>data-mining bias</b>. A good strategy exploits a <b>real market pattern</b>.
  This terminal re-runs your fully-optimized pipeline on <b>1000 shuffled universes</b> — same distribution, zero memory — and measures how often noise alone beats you.</div>
  <div class="chips">
    <span class="chip hot">FEED <b>{src_label}</b></span>
    <span class="chip">BARS <b>{len(btc_df):,}</b></span>
    <span class="chip">PIPELINE <b>{STRAT_LABELS[cfg_now['strategy']]}</b></span>
    <span class="chip cool">N <b>{cfg_now['n_perms']:,}</b></span>
    <span class="chip">SEED <b>{cfg_now['seed']}</b></span>
    <span class="chip cool">ENGINE <b>{'TURBO' if cfg_now['turbo'] else 'REFERENCE'}</b></span>
  </div>
  <div class="hairline"></div>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_strat, tab_anat, tab_method = st.tabs(
    ["🧪 PERMUTATION LAB", "📈 STRATEGY", "🔬 PERMUTATION ANATOMY", "📚 METHOD & CODE"]
)

# ---------------------------------------------------------------- LAB
with tab_lab:
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        sec("L", f"Live run · {active['label']} · {active.get('window_desc','')}")
        live_fragment()
    elif active is None:
        onboarding()
        history_block()
    elif active.get("status") == "error":
        sec("E", "Run failed")
        st.error(active.get("error", "Unknown error"))
        st.code(active.get("error", ""), language="text")
        history_block()
    else:
        sec("R", f"Results · {active['label']} · {active.get('window_desc','')}")
        st.markdown(
            f'<div class="chips"><span class="chip hot">{active["label"]}</span>'
            f'<span class="chip">seed <b>{active["config"]["seed"]}</b></span>'
            f'<span class="chip">engine <b>{"turbo" if active["config"]["turbo"] else "reference"}</b></span>'
            f'<span class="chip cool">{active.get("window_desc","")}</span>'
            f'<span class="chip">status <b>{active["status"].upper()}</b></span></div>',
            unsafe_allow_html=True)
        result_body(active, live=False)
        history_block()

# ---------------------------------------------------------------- STRATEGY
with tab_strat:
    sec("S", f"Strategy anatomy · {STRAT_LABELS[cfg_now['strategy']]}")
    det = compute_details(btc_df, cfg_now)
    if det.get("error"):
        st.warning(det["error"])
    else:
        s = det["stats"]
        st.markdown(cards_html([
            dict(k="Profit Factor", v=f"{s['pf']:.4f}", cls="" if s['pf'] >= 1 else "red"),
            dict(k="Sharpe (hourly)", v=f"{s['sharpe']:.4f}", cls="cyan"),
            dict(k="Σ Log Return", v=f"{s['total']:+.3f}", cls="green" if s['total'] > 0 else "red"),
            dict(k="Exposure", v=f"{s['exposure']:.0%}", cls="cyan"),
            dict(k="Position changes", v=f"{s['trades']:,}", cls="cyan"),
            dict(k="Bar win-rate", v=f"{s['winrate']:.1%}", cls=""),
        ]), unsafe_allow_html=True)
        title = "Daily candles · channel · position turns" if cfg_now["strategy"] == "donchian_is" else \
                "Daily candles · position turns" if cfg_now["strategy"] != "ma" else \
                "Daily candles · moving averages · position turns"
        show_fig(candle_fig(det["daily"], det["overlays"], det["markers"], title), key="candles")
        ce, cd = st.columns([1.6, 1])
        with ce:
            show_fig(equity_fig(det["eq_x"], det["eq"], [], "Strategy equity — cumulative log return (hourly)"), key="seq")
        with cd:
            show_fig(dd_fig(det["eq_x"], det["dd"]), key="sdd")
        ex = det["extra"]
        if "opt" in ex:
            lbs, curve, bL = ex["opt"]
            show_fig(optcurve_fig(lbs, curve, bL), key="sopt")
            st.caption(f"Best lookback on real bars: **{bL}**. This peak is exactly what MCPT must justify — the Strategy tab shows the claim; the Lab renders the verdict.")
        if "wf_choices" in ex:
            ch = ex["wf_choices"]
            fig = base_fig(240, "Walk-forward — lookback chosen at each re-training")
            xs = [btc_df.index[min(t, len(btc_df)-1)] for t, _ in ch]
            fig.add_trace(go.Scatter(x=xs, y=[b for _, b in ch], mode="lines",
                                     line=dict(color=C["magenta"], width=2, shape="hv"), name="Chosen lookback"))
            show_fig(fig, key="swf")
        if "importances" in ex:
            show_fig(imp_fig(ex["importances"]), key="simp")

# ---------------------------------------------------------------- ANATOMY
with tab_anat:
    sec("A", "What permutation preserves — and what it kills")
    ca, cb, cc = st.columns([1, 1, 2])
    ca.number_input("Anatomy seed", 0, 2**31 - 1, 42, key="w_anat_seed")
    cb.markdown("<div style='height:1.62rem'></div>", unsafe_allow_html=True)
    cb.button("🔀 Reshuffle", on_click=reshuffle, use_container_width=True)
    nlags = cc.slider("ACF lags (hours)", 48, 336, 168, key="w_lags")
    an = anatomy(btc_df, eth_df, int(st.session_state.get("w_anat_seed", 42)), nlags)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = base_fig(360, "One real market vs one permuted universe (log price, rebased)")
        fig.add_trace(go.Scatter(x=an["x"], y=an["lcr"], mode="lines", line=dict(color=C["btc"], width=2), name="Real"))
        fig.add_trace(go.Scatter(x=an["x"], y=an["lcp"], mode="lines", line=dict(color=C["cyan"], width=1.6), name="Permuted"))
        show_fig(fig, key="an_price")
    with col2:
        st.markdown('<div class="panel"><h3>Marginal distribution — preserved</h3><div style="color:var(--muted);font-size:.8rem">Shuffling rearranges bars; it cannot change the histogram of returns. Moments match to sampling noise.</div></div>', unsafe_allow_html=True)
        st.table(an["moments"].style.format("{:.6f}"))

    c1, c2 = st.columns(2)
    with c1:
        show_fig(acf_fig(an["lags"], an["ar"], an["ap"], an["n"], "ACF of returns — linear dependence"), key="an_acf_r")
    with c2:
        show_fig(acf_fig(an["lags"], an["aar"], an["aap"], an["n"], "ACF of |returns| — volatility clustering"), key="an_acf_a")
    st.caption("Dashed lines: ±1.96/√n white-noise bounds. Real |returns| stay correlated for days (volatility clustering); the permuted series collapses to the bounds immediately. Any strategy silently leaning on that clustering interacts with this null.")

    show_fig(spec_fig(an["fxr"], an["pxr"], an["fxp"], an["pxp"]), key="an_spec")
    st.caption("Frequency domain: real |returns| show a slow low-frequency decay — the spectral signature of long memory / self-similarity. Permutation flattens it. Distribution survives; dependence does not.")

    if an["corr"]:
        cr = an["corr"]
        show_fig(corr_fig(cr["rb"], cr["re"], cr["pb"], cr["pe"], cr["rho_r"], cr["rho_p"]), key="an_corr")
        st.caption("Joint permutation applies the *same* shuffle to both markets, so contemporaneous BTC–ETH correlation survives while each market's serial structure dies — the correct null for multi-asset pipelines.")
    else:
        st.info("Load ETHUSD3600.pq (or upload it) to see the joint-permutation correlation demo.")

# ---------------------------------------------------------------- METHOD & CODE
with tab_method:
    sec("M", "The logic of the test")
    st.markdown(
        """
<div class="panel">
<h3>Why permutation testing?</h3>
<div style="color:var(--muted);font-size:.9rem;line-height:1.7">
Using the best version of a strategy <b style="color:var(--text)">requires optimization</b> — and optimization
<b style="color:var(--btc2)">always induces data-mining bias</b>: pick the best of 157 lookbacks and the winner's PF is inflated by selection alone.
A good strategy finds and exploits a <b style="color:var(--text)">real, repeatable pattern</b> in the market. The question is whether your pipeline's result
is distinguishable from what the <i>same pipeline</i> achieves on data with no pattern at all.<br><br>
MCPT builds that null by shuffling bars, then re-runs the <b style="color:var(--cyan)">entire procedure — optimization included —</b> on every permuted universe.
If the optimized strategy performs just as well on noise, it is likely overfitting the real data. <b style="color:var(--cyan)">Aim for 1000 permutations.</b>
</div>
</div>
""", unsafe_allow_html=True)
    st.latex(r"p \;=\; \frac{1 \;+\; \#\{\,PF^{\,perm} \;\ge\; PF^{\,real}\,\}}{N_{perm}}")
    st.caption("The real run counts as one observation (hence the +1). p ≤ 0.05 → evidence of a real pattern; p large → the optimizer is mining noise.")

    st.markdown(
        """
<div class="pd-grid">
  <div class="pd keep"><h4>✓ PRESERVED by bar permutation</h4><ul>
    <li><b>Marginal distribution</b> of returns — mean, std, skew, kurtosis</li>
    <li><b>Intrabar geometry</b> — high/low/close relative to open, shuffled jointly</li>
    <li><b>Gap distribution</b> — close→open jumps, shuffled separately</li>
    <li><b>Cross-market correlation</b> — when markets are permuted jointly with the same indices</li>
    <li><b>Real history before start_index</b> — walk-forward permutes only the OOS region</li>
  </ul></div>
  <div class="pd kill"><h4>✕ DESTROYED by bar permutation</h4><ul>
    <li><b>Volatility clustering</b> — significant serial correlation of absolute returns</li>
    <li><b>Long memory</b> — slow decay of dependency over longer lags</li>
    <li><b>Self-similarity</b> and fractal scaling across horizons</li>
    <li><b>Frequency-domain structure</b> — the spectrum of |returns| flattens</li>
    <li><b>Anything involving dependence</b> — trends, mean-reversion, calendar effects</li>
  </ul></div>
</div>
<div class="callout"><b>Know your null.</b> Your strategy may interact with these properties even if you are unaware of it.
A vol-harvesting rule tested against shuffled bars is tested against a world where vol clustering does not exist —
significance then means the edge requires <i>serial structure</i>, not just fat tails. Interpret p-values with the destroyed properties in mind.</div>
<div class="callout" style="border-left-color:var(--cyan)"><b>Walk-forward variant.</b> Permutation starts at <code>start_index = train_window</code>, keeping the first training block real,
and the optimizer re-trains inside every permutation — so the test prices in the full re-selection procedure, not a frozen parameter set.</div>
""", unsafe_allow_html=True)

    sec("C", "Complete source — every file, runnable")
    FILES = [
        "app.py", "mcpt_engine.py", "donchian.py", "bar_permute.py", "tree_strat.py",
        "moving_average.py", "market_data.py", "insample_donchian_mcpt.py",
        "walkforward_donchian_mcpt.py", "insample_tree_mcpt.py", "requirements.txt", "README.md",
    ]

    @st.cache_data(show_spinner=False)
    def project_zip():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for f in FILES:
                p = Path(f)
                if p.exists():
                    z.write(p, f)
        return buf.getvalue()

    pick = st.selectbox("Source file", FILES, key="w_file")
    p = Path(pick)
    if p.exists():
        src = p.read_text(encoding="utf-8")
        st.code(src, language="python" if pick.endswith(".py") else "text", line_numbers=True)
        d1, d2 = st.columns(2)
        d1.download_button(f"⬇ {pick}", src, file_name=pick, key="dl_file")
        d2.download_button("⬇ Full project (.zip)", project_zip(), file_name="permutlab.zip", mime="application/zip", key="dl_zip")
    else:
        st.warning(f"{pick} not found on disk.")
    st.markdown(
        """
<div class="callout"><b>Run it:</b> <code>pip install -r requirements.txt</code> → <code>streamlit run app.py</code>.
Standalone research scripts (<code>insample_donchian_mcpt.py</code>, <code>walkforward_donchian_mcpt.py</code>, <code>insample_tree_mcpt.py</code>)
run the classic tqdm + matplotlib workflow against <code>BTCUSD3600.pq</code>, falling back to the synthetic feed when the file is absent.</div>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="foot">PERMUTLAB · MCPT IN THE TRADITION OF ARONSON\'S EVIDENCE-BASED TA &amp; MASTERS\' PERMUTATION TESTS · '
    'DISTRIBUTION SURVIVES THE SHUFFLE — DEPENDENCE DOES NOT · BUILT WITH STREAMLIT + PLOTLY</div>',
    unsafe_allow_html=True,
)
