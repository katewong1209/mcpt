"""Permutation Tests · Trading Strategies — built by Kate Wong.

Monte Carlo permutation testing terminal. Reference workflow after
neurotrader888/mcpt (MIT); adds live streaming, a vectorized Turbo engine,
and a full-N permutation loop with p = (1 + #noise >= real) / (N + 1).
"""
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
    page_title="Permutation Tests Trading Strategies",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

FIXED_SEED = 69  # seed is fixed by design — not user-adjustable

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
    radial-gradient(1100px 700px at 88% -12%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(950px 640px at -12% 12%, rgba(247,147,26,.12), transparent 55%),
    radial-gradient(820px 820px at 52% 112%, rgba(255,46,166,.07), transparent 62%),
    linear-gradient(180deg,#05070d 0%,#070b14 100%);
  color:var(--text);
}
[data-testid="stAppViewContainer"]::before{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background:
    repeating-linear-gradient(0deg, rgba(139,148,171,.04) 0 1px, transparent 1px 52px),
    repeating-linear-gradient(90deg, rgba(139,148,171,.04) 0 1px, transparent 1px 52px);
}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stDecoration"], [data-testid="stToolbar"]{visibility:hidden; height:0;}
.block-container{padding-top:1.4rem; padding-bottom:3rem; max-width:1400px;}

/* fix broken material-icon ligature ("keyboard_double_arrow_right") on sidebar toggles */
[data-testid="stSidebarCollapsedControl"] *, [data-testid="stSidebarCollapseButton"] *,
[data-testid="collapsedControl"] *{font-size:0 !important;}
[data-testid="stSidebarCollapsedControl"] button, [data-testid="collapsedControl"] button,
[data-testid="stSidebarCollapseButton"] button{display:flex;align-items:center;justify-content:center;}
[data-testid="stSidebarCollapsedControl"] button::after, [data-testid="collapsedControl"] button::after{
  content:"☰"; font-size:1.05rem; color:var(--btc);}
[data-testid="stSidebarCollapseButton"] button::after{content:"✕"; font-size:.95rem; color:var(--muted);}

*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:#0a0f1a}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--btc),var(--magenta));border-radius:8px;border:2px solid #0a0f1a}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--cyan),var(--violet))}
*{scrollbar-width:thin; scrollbar-color:#f7931a #0a0f1a;}

[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a0f1a 0%,#0b1120 100%);
  border-right:1px solid var(--line); min-width:320px; max-width:320px;
}
.sb-sec{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.22em;
  color:var(--btc);text-transform:uppercase;margin:1.1rem 0 .35rem;
  display:flex;align-items:center;gap:.5rem;}
.sb-sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.sb-logo{font-weight:700;font-size:1.32rem;letter-spacing:-.02em;line-height:1.1;color:var(--text);}
.sb-logo em{font-style:normal;background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.sb-tag{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--muted);letter-spacing:.16em;margin-top:.25rem;}

.hero{padding:.6rem 0 .2rem;}
.hero h1{font-size:clamp(1.9rem,3.6vw,2.9rem);font-weight:700;letter-spacing:-.02em;
  margin:.1rem 0 .2rem;line-height:1.08;color:var(--text);}
.hero h1 .grad{background:linear-gradient(92deg,var(--btc) 5%,var(--magenta) 55%,var(--cyan) 95%);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:var(--muted);max-width:760px;font-size:.92rem;margin:.15rem 0 .6rem;}
.hairline{height:3px;border-radius:3px;margin:.4rem 0 1.1rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan),var(--btc));
  background-size:300% 100%;animation:shimmer 7s linear infinite;opacity:.8;}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:300% 0}}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.7rem;margin:.5rem 0 .9rem;}
.card{position:relative;overflow:hidden;background:linear-gradient(160deg,#0d1322,#0a0f1a);
  border:1px solid var(--line);border-radius:14px;padding:.75rem .95rem .65rem;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--btc),var(--magenta));opacity:.8;}
.card.cyan::before{background:linear-gradient(90deg,var(--cyan),var(--violet));}
.card.green::before{background:linear-gradient(90deg,var(--green),var(--cyan));}
.card.red::before{background:linear-gradient(90deg,var(--red),var(--magenta));}
.card .k{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.18em;text-transform:uppercase;color:var(--muted);}
.card .v{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:var(--text);margin-top:.12rem;}
.card .d{font-size:.7rem;color:var(--muted);margin-top:.08rem;}
.card .d.up{color:var(--green);} .card .d.dn{color:var(--red);} .card .d.warn{color:var(--btc2);}

.verdict{border-radius:14px;padding:.85rem 1.1rem;margin:.3rem 0 .9rem;border:1px solid;
  display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
.verdict .big{font-weight:700;font-size:1.1rem;letter-spacing:.02em;white-space:nowrap;}
.verdict .txt{color:var(--muted);font-size:.84rem;flex:1;min-width:240px;}
.verdict.sig{border-color:rgba(44,229,167,.5);background:linear-gradient(120deg,rgba(44,229,167,.12),rgba(34,211,238,.05));}
.verdict.sig .big{color:var(--green);}
.verdict.weak{border-color:rgba(247,147,26,.5);background:linear-gradient(120deg,rgba(247,147,26,.13),rgba(255,46,166,.05));}
.verdict.weak .big{color:var(--btc2);}
.verdict.noise{border-color:rgba(255,77,94,.5);background:linear-gradient(120deg,rgba(255,77,94,.13),rgba(255,46,166,.05));}
.verdict.noise .big{color:var(--red);}

.sec{display:flex;align-items:center;gap:.65rem;margin:1.2rem 0 .5rem;}
.sec .idx{font-family:'IBM Plex Mono',monospace;font-size:.62rem;color:var(--ink);
  background:linear-gradient(135deg,var(--btc),var(--magenta));border-radius:6px;padding:.15rem .4rem;font-weight:600;}
.sec .t{font-weight:600;font-size:1.02rem;color:var(--text);}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}

.panel{background:linear-gradient(150deg,#0d1424,#0a0f1a);border:1px solid var(--line);
  border-radius:16px;padding:1.1rem 1.25rem;margin:.5rem 0;}
.panel h3{margin:.1rem 0 .5rem;font-size:1.1rem;color:var(--text);}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:.8rem;margin-top:.7rem;}
.step{border:1px solid rgba(139,148,171,.18);border-radius:12px;padding:.8rem .9rem;background:rgba(11,16,28,.6);}
.step .n{font-family:'IBM Plex Mono',monospace;font-size:1.25rem;font-weight:600;
  background:linear-gradient(90deg,var(--btc),var(--magenta));-webkit-background-clip:text;background-clip:text;color:transparent;}
.step .h{font-weight:600;color:var(--text);margin:.2rem 0;}
.step .b{font-size:.8rem;color:var(--muted);line-height:1.45;}
.pd-grid{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;}
.pd{border-radius:14px;padding:.9rem 1.05rem;border:1px solid;}
.pd.keep{border-color:rgba(44,229,167,.35);background:rgba(44,229,167,.06);}
.pd.kill{border-color:rgba(255,77,94,.35);background:rgba(255,77,94,.06);}
.pd h4{margin:0 0 .45rem;font-size:.92rem;}
.pd.keep h4{color:var(--green);} .pd.kill h4{color:var(--red);}
.pd ul{margin:0;padding-left:1.05rem;color:var(--muted);font-size:.82rem;line-height:1.6;}
.pd li b{color:var(--text);font-weight:600;}
.callout{border-left:3px solid var(--btc);background:rgba(247,147,26,.07);
  border-radius:0 12px 12px 0;padding:.75rem 1rem;color:var(--muted);font-size:.85rem;margin:.8rem 0;}
.callout b{color:var(--btc2);}
.sg{display:grid;grid-template-columns:1fr 1fr;gap:.8rem;margin:.5rem 0;}
.sgc{border:1px solid rgba(139,148,171,.18);border-radius:12px;padding:.85rem 1rem;background:rgba(11,16,28,.6);}
.sgc h4{margin:0 0 .45rem;color:var(--btc2);font-size:.95rem;}
.sgc p{margin:.3rem 0;font-size:.8rem;color:var(--muted);line-height:1.55;}
.sgc b{color:var(--text);}
.sgc .pro b{color:var(--green);} .sgc .con b{color:var(--red);}

button[data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.76rem;letter-spacing:.05em;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--btc2);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--btc),var(--magenta));height:3px;}
div[data-baseweb="tab-border"]{background:var(--line);}
[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--btc),var(--magenta));color:#0a0710;border:none;
  font-weight:700;letter-spacing:.04em;box-shadow:0 4px 22px rgba(247,147,26,.3);
  transition:transform .15s ease, box-shadow .15s ease;}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 28px rgba(255,46,166,.4);color:#0a0710;}
[data-testid="stBaseButton-secondary"]{background:var(--panel2);border:1px solid rgba(34,211,238,.35);color:var(--text);}
[data-testid="stBaseButton-secondary"]:hover{border-color:var(--btc);color:var(--btc2);}
[data-baseweb="slider"] div[role="slider"]{background:var(--btc);box-shadow:0 0 10px rgba(247,147,26,.6);}
[data-baseweb="select"]>div{background:var(--panel2);border-color:var(--line);}
[data-testid="stNumberInput"] input{background:var(--panel2);color:var(--text);border-color:var(--line);font-family:'IBM Plex Mono',monospace;}
[data-baseweb="checkbox"] [aria-checked="true"]{background-color:var(--btc) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel2);border:1px dashed rgba(247,147,26,.4);border-radius:12px;}
[data-testid="stProgress"] div[role="progressbar"]{background:rgba(139,148,171,.15);}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line);border-radius:12px;}
[data-testid="stCodeBlock"] pre{max-height:440px;overflow:auto;background:#0a0f1a !important;
  border:1px solid var(--line);border-radius:12px;}
[data-testid="stTable"] table{background:transparent;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:.8rem;}
[data-testid="stTable"] th{color:var(--btc2) !important;border-bottom:1px solid var(--line) !important;}
[data-testid="stTable"] td{border-bottom:1px solid rgba(139,148,171,.12) !important;}
.hist-row{display:flex;align-items:center;gap:.6rem;padding:.4rem .6rem;border:1px solid rgba(139,148,171,.14);
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
  .pd-grid,.sg{grid-template-columns:1fr;}
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


def base_fig(height=340, title=None):
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,26,.5)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#aab3c8"),
        margin=dict(l=46, r=16, t=40 if title else 14, b=38),
        title=dict(text=title, x=0.01, font=dict(size=12.5, color=C["text"], family="Space Grotesk")),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="closest",
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["btc"], font=dict(family="IBM Plex Mono", size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(139,148,171,.08)", zerolinecolor="rgba(139,148,171,.2)", linecolor="rgba(139,148,171,.2)")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.08)", zerolinecolor="rgba(139,148,171,.2)", linecolor="rgba(139,148,171,.2)")
    return fig


def hist_fig(perm_pfs, real_pf, p):
    vals = np.asarray([v for v in perm_pfs if np.isfinite(v)], dtype=float)
    fig = base_fig(350, "Shuffled noise vs real result")
    fig.update_layout(showlegend=False)
    if len(vals) == 0:
        fig.add_annotation(text="collecting permutations…", xref="paper", yref="paper",
                           x=.5, y=.5, showarrow=False, font=dict(color=C["muted"]))
        return fig
    lo = min(vals.min(), real_pf if np.isfinite(real_pf) else vals.min())
    hi = max(vals.max(), real_pf if np.isfinite(real_pf) else vals.max())
    pad = max((hi - lo) * 0.05, 0.01)
    counts, edges = np.histogram(vals, bins=40, range=(lo - pad, hi + pad))
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * .92,
                         marker_color=C["cyan"], opacity=.75,
                         hovertemplate="PF %{x:.3f}<br>count %{y}<extra></extra>"))
    if np.isfinite(real_pf):
        fig.add_vrect(x0=real_pf, x1=hi + pad, fillcolor=C["red"], opacity=.06, line_width=0)
        fig.add_vline(x=real_pf, line_color=C["btc"], line_width=2.5, line_dash="dash",
                      annotation_text=f"REAL {real_pf:.3f}", annotation_position="top",
                      annotation_font=dict(color=C["btc2"], size=11))
    vcol = C["green"] if p <= .05 else (C["btc2"] if p <= .2 else C["red"])
    fig.add_annotation(xref="paper", yref="paper", x=.99, y=.99, xanchor="right",
                       text=f"<b>p = {p:.3f}</b>", showarrow=False,
                       font=dict(color=vcol, size=16, family="Space Grotesk"))
    fig.update_xaxes(title="Profit factor")
    return fig


def equity_fig(x, real_eq, samples, title="Equity — real vs shuffled"):
    fig = base_fig(350, title)
    for i, s in enumerate(samples):
        fig.add_trace(go.Scatter(x=x[: len(s)], y=s, mode="lines",
                                 line=dict(color=C["cyan"], width=1), opacity=.18,
                                 showlegend=(i == 0), name="Shuffled", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[: len(real_eq)], y=real_eq, mode="lines",
                             line=dict(color=C["btc"], width=2.4), name="Real"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.3)", line_width=1)
    return fig


def optcurve_fig(lookbacks, curve, best_L, mean=None, std=None):
    fig = base_fig(330, "Optimization surface — real vs noise")
    if mean is not None:
        m = np.isfinite(mean) & np.isfinite(std)
        xl = np.asarray(lookbacks)[m]
        fig.add_trace(go.Scatter(x=np.concatenate([xl, xl[::-1]]),
                                 y=np.concatenate([(mean + std)[m], (mean - std)[m][::-1]]),
                                 fill="toself", fillcolor="rgba(139,123,255,.15)",
                                 line=dict(width=0), name="Noise ±1σ", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=xl, y=mean[m], mode="lines",
                                 line=dict(color=C["cyan"], width=1.5, dash="dot"), name="Noise mean"))
    fig.add_trace(go.Scatter(x=lookbacks, y=curve, mode="lines",
                             line=dict(color=C["btc"], width=2.2), name="Real"))
    if best_L and best_L > 0:
        k = int(np.argmin(np.abs(np.asarray(lookbacks) - best_L)))
        fig.add_trace(go.Scatter(x=[lookbacks[k]], y=[curve[k]], mode="markers",
                                 marker=dict(symbol="star", size=12, color=C["magenta"],
                                             line=dict(color=C["text"], width=1)),
                                 name=f"L={best_L}"))
    fig.add_hline(y=1.0, line_color="rgba(139,148,171,.35)", line_dash="dash", line_width=1)
    fig.update_xaxes(title="Lookback (bars)")
    fig.update_yaxes(title="Profit factor")
    return fig


def candle_fig(daily, overlays, title="Price · daily candles"):
    fig = base_fig(400, title)
    fig.add_trace(go.Candlestick(
        x=daily.index, open=daily["open"], high=daily["high"], low=daily["low"], close=daily["close"],
        increasing_line_color=C["green"], increasing_fillcolor="rgba(44,229,167,.55)",
        decreasing_line_color=C["red"], decreasing_fillcolor="rgba(255,77,94,.55)",
        name="OHLC", whiskerwidth=.4))
    for name, (series, color, dash) in overlays.items():
        fig.add_trace(go.Scatter(x=series.index, y=series.to_numpy(), mode="lines",
                                 line=dict(color=color, width=1.4, dash=dash), name=name))
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    return fig


def position_fig(x, sig):
    fig = base_fig(170, "Position")
    fig.update_layout(showlegend=False)
    fig.add_trace(go.Scatter(x=x, y=sig, mode="lines",
                             line=dict(color=C["magenta"], width=1.6, shape="hv"),
                             hovertemplate="%{y:+.0f}<extra></extra>"))
    fig.update_yaxes(tickvals=[-1, 0, 1], range=[-1.25, 1.25])
    return fig


def dd_fig(x, dd):
    fig = base_fig(200, "Drawdown")
    fig.update_layout(showlegend=False)
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color=C["red"], width=1.2),
                             fillcolor="rgba(255,77,94,.22)"))
    return fig


def acf_fig(lags, real, perm, n, title):
    fig = base_fig(280, title)
    fig.add_trace(go.Bar(x=lags, y=real, marker_color=C["btc"], opacity=.85, width=.42, name="Real"))
    fig.add_trace(go.Bar(x=lags, y=perm, marker_color=C["cyan"], opacity=.6, width=.42, name="Shuffled"))
    b = 1.96 / np.sqrt(n)
    fig.add_hline(y=b, line_color="rgba(139,148,171,.45)", line_dash="dash", line_width=1)
    fig.add_hline(y=-b, line_color="rgba(139,148,171,.45)", line_dash="dash", line_width=1)
    fig.update_layout(barmode="group", bargap=.15)
    fig.update_xaxes(title="lag (hours)")
    return fig


def spec_fig(fx, px, fy, py):
    fig = base_fig(300, "|Returns| spectrum — long memory (log-log)")
    fig.add_trace(go.Scatter(x=fx, y=px, mode="lines", line=dict(color=C["btc"], width=2), name="Real"))
    fig.add_trace(go.Scatter(x=fy, y=py, mode="lines", line=dict(color=C["cyan"], width=1.5, dash="dot"), name="Shuffled"))
    fig.update_xaxes(type="log", title="frequency (cycles / bar)")
    fig.update_yaxes(type="log", title="power")
    return fig


def corr_fig(rb, re, pb, pe, rho_r, rho_p):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Real · ρ = {rho_r:.3f}", f"Shuffled · ρ = {rho_p:.3f}"))
    fig.add_trace(go.Scattergl(x=rb, y=re, mode="markers", marker=dict(size=3, color=C["btc"], opacity=.45)), 1, 1)
    fig.add_trace(go.Scattergl(x=pb, y=pe, mode="markers", marker=dict(size=3, color=C["cyan"], opacity=.45)), 1, 2)
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,.5)",
                      font=dict(family="IBM Plex Mono", size=11, color="#aab3c8"),
                      margin=dict(l=46, r=16, t=44, b=38), showlegend=False,
                      title=dict(text="Cross-market correlation survives joint shuffling", x=.01,
                                 font=dict(size=12.5, color=C["text"], family="Space Grotesk")))
    fig.update_xaxes(gridcolor="rgba(139,148,171,.08)", title="Currency 1 log return")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.08)", title="Currency 2 log return")
    for a in fig.layout.annotations:
        a.font = dict(family="IBM Plex Mono", size=11, color=C["muted"])
    return fig


def imp_fig(importances):
    names = ["diff6 · 6h momentum", "diff24 · 1d momentum", "diff168 · 1w momentum"]
    fig = base_fig(220, "Tree feature importances")
    fig.update_layout(showlegend=False)
    fig.add_trace(go.Bar(y=names, x=importances, orientation="h",
                         marker_color=[C["btc"], C["cyan"], C["magenta"]], opacity=.85,
                         text=[f"{v:.1%}" for v in importances], textposition="outside",
                         textfont=dict(color=C["text"])))
    fig.update_xaxes(range=[0, max(importances) * 1.25 + 1e-6])
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
        h += f'<div class="card {it.get("cls","")}"><div class="k">{it["k"]}</div><div class="v">{it["v"]}</div>'
        if it.get("d"):
            h += f'<div class="d {it.get("dc","")}">{it["d"]}</div>'
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
@st.cache_data(show_spinner="Synthesizing demo feed…")
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
    if not {"open", "high", "low", "close"}.issubset(df.columns):
        raise ValueError(f"{name}: needs columns open/high/low/close")
    return df[["open", "high", "low", "close"]].dropna()


@st.cache_data(show_spinner=False)
def load_path(path, mtime):
    return parse_bytes(path, Path(path).read_bytes())


def resolve_data(uploads):
    """Single picker: first file = Currency 1, second (optional) = Currency 2."""
    if uploads:
        try:
            btc = parse_bytes(uploads[0].name, uploads[0].getvalue())
            eth = parse_bytes(uploads[1].name, uploads[1].getvalue()) if len(uploads) > 1 else None
            label = f"UPLOAD · {uploads[0].name}" + (" + Currency 2" if eth is not None else "")
            return btc, eth, label, None
        except Exception as e:
            b, e2 = synth_pair()
            return b, e2, "SYNTHETIC (upload failed)", str(e)
    p = Path("BTCUSD3600.pq")
    if p.exists():
        try:
            btc = load_path(str(p), p.stat().st_mtime)
            pe = Path("ETHUSD3600.pq")
            eth = load_path(str(pe), pe.stat().st_mtime) if pe.exists() else None
            return btc, eth, "BTCUSD3600.pq" + (" + ETH" if eth is not None else ""), None
        except Exception as e:
            b, e2 = synth_pair()
            return b, e2, "SYNTHETIC (file error)", str(e)
    b, e2 = synth_pair()
    return b, e2, "SYNTHETIC DEMO FEED", None


# ---------------------------------------------------------------- callbacks
def collect_cfg():
    g = st.session_state.get
    depth = g("w_depth", "Unlimited")
    lbmin, lbmax = int(g("w_lbmin", 12)), int(g("w_lbmax", 168))
    if lbmin >= lbmax:
        lbmin, lbmax = min(lbmin, lbmax - 1), max(lbmax, lbmin + 1)
    return dict(
        strategy=g("w_strategy", "donchian_is"),
        n_perms=int(g("w_perms", 1000)),
        seed=FIXED_SEED,
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
        window_desc="", label=STRAT_LABELS[cfg["strategy"]], n_target=cfg["n_perms"],
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


def new_shuffle():
    st.session_state["anat_seed"] = int(np.random.randint(0, 2**31 - 1))


# ---------------------------------------------------------------- strategy details
@st.cache_data(show_spinner="Computing strategy…")
def compute_details(btc, cfg):
    try:
        strat = cfg["strategy"]
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
        overlays, extra = {}, {}

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
        pstep = max(1, len(sig) // 1500)
        daily = win.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        dov = {name: (ser.resample("1D").last(), col, dash) for name, (ser, col, dash) in overlays.items()}
        return dict(
            error=None, daily=daily, overlays=dov,
            pos_x=win.index[::pstep], pos_sig=sig[::pstep],
            eq_x=win.index[::step], eq=eq[::step], dd=dd[::step],
            stats=dict(
                pf=pf, sharpe=float(rs.mean() / rs.std()) if rs.std() > 0 else float("nan"),
                total=float(np.nansum(rets)), trades=int(len(chg)),
                exposure=float(np.nanmean(np.abs(sig))) if np.isfinite(sig).any() else 0.0,
            ),
            extra=extra,
        )
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
            "SHUFFLED": [pr.mean(), pr.std(), pd.Series(pr).skew(), pd.Series(pr).kurt()],
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
            b2, e2 = sl.loc[common], e.loc[common]
            pb, pe = get_permutation([b2, e2], seed=seed)
            rb = np.log(b2["close"]).diff().dropna()
            re = np.log(e2["close"]).diff().dropna()
            qb = np.log(pb["close"]).diff().dropna()
            qe = np.log(pe["close"]).diff().dropna()
            sel = np.linspace(0, len(rb) - 1, min(2600, len(rb))).astype(int)
            out["corr"] = dict(
                rb=rb.to_numpy()[sel], re=re.to_numpy()[sel],
                pb=qb.to_numpy()[sel], pe=qe.to_numpy()[sel],
                rho_r=float(rb.corr(re)), rho_p=float(qb.corr(qe)),
            )
    return out


# ---------------------------------------------------------------- results
def result_body(job, live=False):
    cfg = job["config"]
    done, better = job["done"], job["better"]
    p = better / (done + 1)
    real_pf = job.get("real_pf", float("nan"))
    n = cfg["n_perms"]

    if not live:
        vlabel, vcls, _ = job.get("verdict", verdict(p))
        b = better - 1
        if vcls == "sig":
            txt = f"Only {b} of {done:,} shuffled markets beat the real result — evidence of a genuine pattern."
        elif vcls == "weak":
            txt = f"{b} of {done:,} shuffled markets beat the real result. Suggestive, not conclusive."
        else:
            txt = f"{b} of {done:,} shuffled markets matched the real result — the optimizer is mining noise."
        st.markdown(
            f'<div class="verdict {vcls}"><span class="big">{vlabel}</span><span class="txt">{txt}</span></div>',
            unsafe_allow_html=True)
        items = [
            dict(k="Real Profit Factor", v=f"{real_pf:.4f}"),
            dict(k="MCPT p-value", v=f"{p:.3f}",
                 cls="green" if vcls == "sig" else ("red" if vcls == "noise" else ""),
                 d=f"{b} of {done:,} noise runs ≥ real", dc="warn"),
            dict(k="Permutations", v=f"{done:,}", cls="cyan",
                 d=f"target {n:,}" + (" · stopped early" if job["status"] == "stopped" else "")),
        ]
        if job.get("best_lookback"):
            items.append(dict(k="Best Lookback", v=str(job["best_lookback"]), cls="cyan", d="selected on real bars"))
        else:
            items.append(dict(k="Elapsed", v=fmt_t(job.get("elapsed", 0)), cls="cyan"))
        st.markdown(cards_html(items), unsafe_allow_html=True)

    colA, colB = st.columns([1.3, 1])
    with colA:
        show_fig(hist_fig(job["perm_pfs"], real_pf, p), key=f"hist_{job['id']}_{live}")
    with colB:
        if job.get("real_curve") is not None:
            show_fig(equity_fig(job["curve_x"], job["real_curve"], job["sample_curves"]), key=f"eq_{job['id']}_{live}")

    if not live:
        if job.get("pf_curve") is not None and job.get("env_cnt") is not None:
            with st.expander("Optimization surface — real vs noise envelope"):
                cnt = job["env_cnt"]
                mean = np.where(cnt > 0, job["env_sum"] / np.maximum(cnt, 1), np.nan)
                var = np.where(cnt > 0, job["env_ssq"] / np.maximum(cnt, 1) - mean**2, np.nan)
                show_fig(optcurve_fig(job["lookbacks"], job["pf_curve"], job.get("best_lookback"),
                                      mean, np.sqrt(np.maximum(var, 0))), key=f"opt_{job['id']}")
        if job.get("wf_choices"):
            with st.expander("Walk-forward — lookback chosen at each re-training"):
                ch = job["wf_choices"]
                fig = base_fig(230)
                fig.update_layout(showlegend=False)
                xs = [job["curve_x"][min(t, len(job["curve_x"]) - 1)] for t, _ in ch]
                fig.add_trace(go.Scatter(x=xs, y=[b for _, b in ch], mode="lines",
                                         line=dict(color=C["magenta"], width=2, shape="hv")))
                show_fig(fig, key=f"wfch_{job['id']}")

        st.caption(f"p = (1 + #noise ≥ real) / (N + 1) · {fmt_t(job.get('elapsed', 0))} · "
                   f"{'turbo' if cfg['turbo'] else 'reference'} engine · interpretation → Method & Guide")
        e1, e2, e3 = st.columns(3)
        payload = dict(
            id=job["id"], label=job["label"], config=cfg, real_pf=real_pf,
            p_value=p, better=better, completed=done,
            best_lookback=job.get("best_lookback"),
            perm_pfs=job["perm_pfs"], finished_at=job.get("finished_at"),
        )
        e1.download_button("⬇ Results (.json)",
                           json.dumps(payload, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=2),
                           file_name=f"mcpt_{job['id']}.json", mime="application/json", key=f"js_{job['id']}")
        e2.download_button("⬇ Permuted PFs (.csv)", "perm_pf\n" + "\n".join(f"{v:.6f}" for v in job["perm_pfs"]),
                           file_name=f"perm_pfs_{job['id']}.csv", mime="text/csv", key=f"csv_{job['id']}")
        e3.button("↻ Re-run with current settings", on_click=start_run, key=f"re_{job['id']}")


def onboarding():
    st.markdown(
        """
<div class="panel">
  <h3>Is your optimized edge real — or the best-looking lie in 157 tries?</h3>
  <div style="color:var(--muted);font-size:.88rem;max-width:820px">
    Optimization <b style="color:var(--btc2)">always</b> induces data-mining bias. This terminal rebuilds the market as
    structureless noise — same distribution, zero memory — re-runs your <i>entire</i> pipeline on each shuffled universe,
    and asks how often noise alone beats your real result. Default: <b style="color:var(--cyan)">1000 permutations</b>.
  </div>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Optimize on real bars</div><div class="b">Donchian sweep, walk-forward re-training, decision tree, or MA crossover — the full selection procedure, bias included.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Shuffle 1000 universes</div><div class="b">Intrabar shape and gaps permuted separately. Mean/std/skew/kurt survive; volatility clustering and long memory die.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Compare &amp; verdict</div><div class="b">p = (1 + #noise-PF ≥ real-PF) / (N + 1). If noise matches your edge, the edge is a mirage. Watch it build live.</div></div>
  </div>
  <div class="callout" style="margin-top:.9rem"><b>→ Pick a strategy and press RUN MCPT in the sidebar.</b> Results stream as they compute; stop anytime.</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.6)
def live_fragment():
    job = st.session_state.get("active")
    if job is None or job.get("status") != "running":
        st.rerun(scope="app")
        return
    cfg = job["config"]
    n = cfg["n_perms"]
    done = job["done"]
    elapsed = time.time() - job["start"]
    eta = (elapsed / done) * (n - done) if done > 0 else None
    st.progress(min(done / max(n, 1), 1.0))
    st.caption(job.get("phase", "Working…"))
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(cards_html([
            dict(k="Completed", v=f"{done:,}", cls="cyan", d=f"of {n:,} permutations"),
            dict(k="Live p-value", v=f"{job['better'] / (done + 1):.3f}",
                 d=f"{job['better'] - 1} noise runs ≥ real", dc="warn"),
            dict(k="ETA", v=fmt_t(eta) if eta is not None else "—", cls="cyan", d=f"elapsed {fmt_t(elapsed)}"),
        ]), unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.button("■ STOP", on_click=request_stop, key="stop_btn", use_container_width=True)
    if np.isfinite(job.get("real_pf", float("nan"))):
        result_body(job, live=True)
    else:
        st.info("Running the real-data baseline (full optimization) first — permutations stream in next.")


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
                f'<span class="m">n={r["done"]:,} · PF {r["real_pf"]:.3f} · p {r["p"]:.3f}</span>'
                f'<span class="badge {vcls}">{vlabel}</span>'
                f'<span class="m">{r.get("finished_at","")}</span></div>', unsafe_allow_html=True)
        c2.button("view", key=f"v_{r['id']}", on_click=load_run, args=(r["id"],))
        c3.button("✕", key=f"d_{r['id']}", on_click=delete_run, args=(r["id"],))
    if st.button("Clear history", key="clr_hist"):
        st.session_state["history"] = []
        st.rerun()


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="sb-logo">Permutation <em>Tests</em></div>'
                '<div class="sb-tag">TRADING STRATEGIES · BUILT BY KATE WONG</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">01 · Market feed</div>', unsafe_allow_html=True)
    uploads = st.file_uploader(
        "Currency 1 hourly (.pq / .parquet / .csv)",
        type=["pq", "parquet", "csv"],
        accept_multiple_files=True,
        key="up_files",
        help="Currency 2 hourly (optional for correlation): select a second file in the same picker.",
    )
    st.caption("Currency 2 hourly (optional for correlation): add a second file above. First file = Currency 1.")
    btc_df, eth_df, src_label, data_err = resolve_data(uploads)
    st.session_state["btc_df"] = btc_df
    st.session_state["eth_df"] = eth_df
    if data_err:
        st.error(data_err)
    ymin = int(btc_df.index.year.min())
    ymax = int(btc_df.index.year.max()) + 1
    st.caption(f"{src_label} · {len(btc_df):,} hourly bars · {ymin}–{ymax-1}")

    st.markdown('<div class="sb-sec">02 · Strategy</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Pipeline under test", list(STRAT_LABELS.keys()),
                            format_func=lambda k: STRAT_LABELS[k], key="w_strategy")
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
        st.slider("Walk-forward years", ymin, ymax, (ymin, ymax), key="w_wf_years")
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
    st.slider("Permutations", 50, 2000, 1000, step=50, key="w_perms")
    st.toggle("Turbo engine (vectorized, identical math)", value=True, key="w_turbo")
    if strategy == "donchian_wf":
        st.caption("Walk-forward re-optimizes inside every permutation — the heaviest test. Turbo recommended.")

    st.markdown('<div class="sb-sec">04 · Execute</div>', unsafe_allow_html=True)
    active = st.session_state.get("active")
    running = bool(active and active.get("status") == "running")
    st.button("🧪 RUN MCPT", type="primary", on_click=start_run, disabled=running, use_container_width=True)
    st.button("Clear results", on_click=clear_active, disabled=running, use_container_width=True)
    if running:
        st.caption(f"● running — {active['done']:,} / {active['config']['n_perms']:,}")
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])


# ---------------------------------------------------------------- header
cfg_now = collect_cfg()
st.markdown(
    """
<div class="hero">
  <h1>Permutation Tests <span class="grad">Trading Strategies</span></h1>
  <div class="sub">Does your optimized strategy beat shuffled noise? Same distribution, zero memory — 1000 permutations, live verdict.</div>
  <div class="hairline"></div>
</div>
""", unsafe_allow_html=True)

tab_test, tab_strat, tab_anat, tab_method = st.tabs(
    ["🧪 PERMUTATION TEST", "📈 STRATEGY", "🔬 PERMUTATION ANATOMY", "📚 METHOD & GUIDE"]
)

# ---------------------------------------------------------------- PERMUTATION TEST
with tab_test:
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        live_fragment()
    elif active is None:
        onboarding()
        history_block()
    elif active.get("status") == "error":
        st.error("Run failed — details below.")
        st.code(active.get("error", "Unknown error"), language="text")
        history_block()
    else:
        if not active.get("_finalized"):
            finalize(active)
        st.caption(f"{active['label']} · {active.get('window_desc','')} · "
                   f"{'turbo' if active['config']['turbo'] else 'reference'} · {active['status'].upper()}")
        result_body(active, live=False)
        history_block()

# ---------------------------------------------------------------- STRATEGY
with tab_strat:
    det = compute_details(btc_df, cfg_now)
    if det.get("error"):
        st.warning(det["error"])
    else:
        s = det["stats"]
        st.markdown(cards_html([
            dict(k="Profit Factor", v=f"{s['pf']:.4f}", cls="" if s['pf'] >= 1 else "red"),
            dict(k="Sharpe (hourly)", v=f"{s['sharpe']:.4f}", cls="cyan"),
            dict(k="Σ Log Return", v=f"{s['total']:+.3f}", cls="green" if s['total'] > 0 else "red"),
            dict(k="Position changes", v=f"{s['trades']:,}", cls="cyan"),
        ]), unsafe_allow_html=True)
        show_fig(candle_fig(det["daily"], det["overlays"]), key="candles")
        show_fig(position_fig(det["pos_x"], det["pos_sig"]), key="position")
        show_fig(equity_fig(det["eq_x"], det["eq"], [], "Equity — cumulative log return"), key="seq")
        with st.expander("Drawdown"):
            show_fig(dd_fig(det["eq_x"], det["dd"]), key="sdd")
        ex = det["extra"]
        if "opt" in ex:
            with st.expander(f"Optimization surface · best lookback L={ex['opt'][2]}"):
                lbs, curve, bL = ex["opt"]
                show_fig(optcurve_fig(lbs, curve, bL), key="sopt")
        if "wf_choices" in ex:
            with st.expander("Walk-forward — lookback chosen at each re-training"):
                ch = ex["wf_choices"]
                fig = base_fig(230)
                fig.update_layout(showlegend=False)
                xs = [btc_df.index[min(t, len(btc_df) - 1)] for t, _ in ch]
                fig.add_trace(go.Scatter(x=xs, y=[b for _, b in ch], mode="lines",
                                         line=dict(color=C["magenta"], width=2, shape="hv")))
                show_fig(fig, key="swf")
        if "importances" in ex:
            with st.expander("Tree feature importances"):
                show_fig(imp_fig(ex["importances"]), key="simp")

# ---------------------------------------------------------------- ANATOMY
with tab_anat:
    ca, cb = st.columns([1, 3])
    ca.button("🔀 New shuffle", on_click=new_shuffle, use_container_width=True)
    nlags = cb.slider("ACF lags (hours)", 48, 336, 168, key="w_lags")
    an = anatomy(btc_df, eth_df, int(st.session_state.get("anat_seed", 42)), nlags)

    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig = base_fig(330, "Real vs shuffled (log price, rebased)")
        fig.add_trace(go.Scatter(x=an["x"], y=an["lcr"], mode="lines", line=dict(color=C["btc"], width=2), name="Real"))
        fig.add_trace(go.Scatter(x=an["x"], y=an["lcp"], mode="lines", line=dict(color=C["cyan"], width=1.5), name="Shuffled"))
        show_fig(fig, key="an_price")
    with col2:
        st.table(an["moments"].style.format("{:.6f}"))
        st.caption("Moments match — the distribution survives the shuffle.")

    c1, c2 = st.columns(2)
    with c1:
        show_fig(acf_fig(an["lags"], an["ar"], an["ap"], an["n"], "Returns — ACF"), key="an_acf_r")
    with c2:
        show_fig(acf_fig(an["lags"], an["aar"], an["aap"], an["n"], "|Returns| — ACF · volatility clustering"), key="an_acf_a")
    st.caption("Dashed: white-noise bounds. Real |returns| stay correlated for days; the shuffle removes it.")

    show_fig(spec_fig(an["fxr"], an["pxr"], an["fxp"], an["pxp"]), key="an_spec")
    if an["corr"]:
        cr = an["corr"]
        show_fig(corr_fig(cr["rb"], cr["re"], cr["pb"], cr["pe"], cr["rho_r"], cr["rho_p"]), key="an_corr")
    else:
        st.caption("Add Currency 2 in the sidebar picker to see the joint-shuffle correlation demo.")

# ---------------------------------------------------------------- METHOD & GUIDE
with tab_method:
    sec("M", "The logic of the test")
    st.markdown(
        """
<div class="panel">
<h3>Why permutation testing?</h3>
<div style="color:var(--muted);font-size:.88rem;line-height:1.7">
Using the best version of a strategy <b style="color:var(--text)">requires optimization</b> — and optimization
<b style="color:var(--btc2)">always induces data-mining bias</b>: pick the best of 157 lookbacks and the winner's profit factor is inflated by selection alone.
A good strategy exploits a <b style="color:var(--text)">real, repeatable market pattern</b>. MCPT shuffles the bars into structureless noise,
re-runs the <b style="color:var(--cyan)">entire pipeline — optimization included —</b> on every shuffled universe, and measures how often noise alone beats you.
If the optimized strategy performs just as well on noise, it is likely overfitting. <b style="color:var(--cyan)">Aim for 1000 permutations.</b>
</div>
</div>
""", unsafe_allow_html=True)
    st.latex(r"p \;=\; \frac{1 \;+\; \#\{\,PF^{\,perm} \;\ge\; PF^{\,real}\,\}}{N \;+\; 1}")
    st.caption("The app runs N full shuffles (default 1000). The +1 counts the real run and keeps p strictly positive. "
               "The original reference scripts loop range(1, N) — the real run occupies one of the N slots, hence 999 shuffles there.")

    sec("S", "Strategy guide — how each pipeline works, pros & cons")
    st.markdown(
        """
<div class="sg">
  <div class="sgc"><h4>Donchian · In-Sample (optimized)</h4>
    <p><b>How:</b> go long when close breaks above the highest close of the prior L−1 bars, short below the lowest; hold until the opposite break. Sweeps L = 12…168 and keeps the highest profit factor.</p>
    <p class="pro"><b>Pros:</b> classic trend-capture; one transparent parameter; rides fat-tailed moves.</p>
    <p class="con"><b>Cons:</b> the sweep itself inflates PF (selection bias); whipsaws in ranges; the “best” L may be pure noise — exactly what MCPT checks.</p>
  </div>
  <div class="sgc"><h4>Donchian · Walk-Forward</h4>
    <p><b>How:</b> every 30 days, re-optimize L on the trailing 4-year window, then trade the next month out-of-sample with the fresh choice.</p>
    <p class="pro"><b>Pros:</b> adapts to regime change; honest out-of-sample framing; close to live deployment.</p>
    <p class="con"><b>Cons:</b> re-optimization re-introduces mining bias at every step; MCPT must retrain inside each shuffle (slow); parameter churn.</p>
  </div>
  <div class="sgc"><h4>Decision Tree · In-Sample</h4>
    <p><b>How:</b> features are 6h / 24h / 168h log momentum; label is the sign of the next 24h return. A CART classifier (min leaf 5) predicts up/down; the strategy is always long or short.</p>
    <p class="pro"><b>Pros:</b> captures nonlinear interactions; feature importances are inspectable.</p>
    <p class="con"><b>Cons:</b> deep trees memorize noise; labels are noisy; markets are non-stationary; heavy retrain cost per permutation.</p>
  </div>
  <div class="sgc"><h4>MA Crossover</h4>
    <p><b>How:</b> long when the 10-bar mean is above the 30-bar mean; flat otherwise (optionally short). No optimization by default.</p>
    <p class="pro"><b>Pros:</b> robust, few parameters; a clean baseline trend filter.</p>
    <p class="con"><b>Cons:</b> lagging; whipsaws sideways; even unoptimized rules deserve a noise benchmark.</p>
  </div>
</div>
""", unsafe_allow_html=True)

    sec("T", "Turbo engine — what it means")
    st.markdown(
        """
<div class="callout"><b>Same math, vectorized.</b> The reference implementation loops over lookbacks with pandas
<code>rolling().max()/min()</code> and builds a per-bar return array for every candidate. The Turbo engine computes identical numbers three ways faster:
<b>(1)</b> a sliding-window view plus reversed cumulative max/min produces every lookback's rolling channel in two passes;
<b>(2)</b> profit factors are assembled from cumulative sums of positive/negative returns between breakout events — no per-bar signal arrays;
<b>(3)</b> permutations are rebuilt with <code>cumsum</code> instead of a Python bar loop. Typically 5–10× faster.
Toggle it off in the sidebar to run the literal reference loops.</div>
""", unsafe_allow_html=True)

    sec("L", "Lineage — neurotrader888/mcpt")
    st.markdown(
        """
<div class="callout" style="border-left-color:var(--cyan)"><b>Yes — same workflow.</b> The bundled modules
(<code>bar_permute.py</code>, <code>donchian.py</code>, <code>tree_strat.py</code>, <code>moving_average.py</code> and the three MCPT scripts)
replicate <a href="https://github.com/neurotrader888/mcpt" style="color:var(--cyan)">github.com/neurotrader888/mcpt</a> (MIT):
joint intrabar shuffles + separate gap shuffles, Donchian optimization, walk-forward re-training, and tree retraining per permutation.
This app adds live streaming, the Turbo engine, and a full-N permutation count with p = (1 + #≥real)/(N+1) — so 1000 means 1000 actual shuffles.</div>
""", unsafe_allow_html=True)

    sec("P", "What the shuffle preserves — and destroys")
    st.markdown(
        """
<div class="pd-grid">
  <div class="pd keep"><h4>✓ PRESERVED</h4><ul>
    <li><b>Marginal distribution</b> — mean, std, skew, kurtosis</li>
    <li><b>Intrabar geometry</b> — high/low/close vs open, shuffled jointly</li>
    <li><b>Gap distribution</b> — close→open jumps, shuffled separately</li>
    <li><b>Cross-market correlation</b> — joint permutation, same indices</li>
    <li><b>Real history before start_index</b> — walk-forward permutes only OOS</li>
  </ul></div>
  <div class="pd kill"><h4>✕ DESTROYED</h4><ul>
    <li><b>Volatility clustering</b> — serial correlation of |returns|</li>
    <li><b>Long memory</b> — slow decay of dependency over long lags</li>
    <li><b>Self-similarity</b> and fractal scaling</li>
    <li><b>Frequency-domain structure</b> — |returns| spectrum flattens</li>
    <li><b>Anything involving dependence</b> — trends, mean-reversion</li>
  </ul></div>
</div>
<div class="callout"><b>Know your null.</b> Your strategy may interact with these properties even if you are unaware of it.
A vol-harvesting rule tested against shuffled bars faces a world without vol clustering — significance then means the edge needs
<i>serial structure</i>, not just fat tails. p ≤ 0.05 → evidence of a real pattern; large p → the optimizer is mining noise.</div>
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
        d2.download_button("⬇ Full project (.zip)", project_zip(), file_name="permutation_tests.zip",
                           mime="application/zip", key="dl_zip")
    else:
        st.warning(f"{pick} not found on disk.")
    st.caption("Run: pip install -r requirements.txt → streamlit run app.py. "
               "Standalone scripts replicate the original neurotrader888 workflow (tqdm + matplotlib).")

st.markdown(
    '<div class="foot">PERMUTATION TESTS · TRADING STRATEGIES — BUILT BY KATE WONG · '
    'WORKFLOW AFTER NEUROTRADER888/MCPT (MIT) · STREAMLIT + PLOTLY</div>',
    unsafe_allow_html=True,
)
