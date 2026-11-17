"""Permutation Tests · Trading Strategies — MCPT terminal. Built by Kate Wong."""
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
from mcpt_engine import STRAT_LABELS, run_mcpt, wf_max_train_years, wf_min_bars, wf_plan
from moving_average import ma_crossover
from tree_strat import train_tree, tree_strategy

st.set_page_config(
    page_title="Permutation Tests · Trading Strategies",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Permutation Tests Trading Strategies"
AUTHOR = "Kate Wong"
BASE_SEED = 69          # fixed, not user-editable
ANATOMY_SEED = 4242     # fixed, not user-editable

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
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"]{
  font-family:'Space Grotesk',system-ui,sans-serif;
}
p, li, label, .stMarkdown, [data-testid="stWidgetLabel"]{font-family:'Space Grotesk',system-ui,sans-serif;}
code, pre, .mono{font-family:'IBM Plex Mono',monospace !important;}

/* --- icon fonts must never inherit the display face (ligature leak fix) --- */
[data-testid="stIconMaterial"], span[data-testid="stIconMaterial"],
.material-symbols-rounded, .material-symbols-outlined, .material-icons,
[data-baseweb="icon"] > svg, [data-testid="stExpander"] summary span{
  font-family:'Material Symbols Rounded','Material Symbols Outlined','Material Icons' !important;
  font-feature-settings:'liga' !important;
  -webkit-font-feature-settings:'liga' !important;
  -webkit-font-smoothing:antialiased;
}
/* sidebar collapse / expand: draw our own chevron so no raw ligature can show */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stExpandSidebarButton"] span{font-size:0 !important;line-height:0 !important;}
[data-testid="stSidebarCollapseButton"] button::after{content:"20392039";}
[data-testid="stSidebarCollapsedControl"] button::after{content:"203A203A";}
[data-testid="stExpandSidebarButton"]::after{content:"203A203A";}
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="stExpandSidebarButton"]::after{
  font-family:'IBM Plex Mono',monospace;font-size:1rem;color:var(--muted);
  letter-spacing:-1px;line-height:1;transition:color .15s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after,
[data-testid="stExpandSidebarButton"]:hover::after{color:var(--btc2);}

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
    repeating-linear-gradient(0deg, rgba(139,148,171,.04) 0 1px, transparent 1px 56px),
    repeating-linear-gradient(90deg, rgba(139,148,171,.04) 0 1px, transparent 1px 56px);
}
[data-testid="stAppViewContainer"]::after{
  content:""; position:fixed; inset:0; z-index:9999; pointer-events:none; opacity:.045;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='160' height='160' filter='url(%23n)' opacity='0.6'/%3E%3C/svg%3E");
}
[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer, [data-testid="stDecoration"]{visibility:hidden; height:0;}
.block-container{padding-top:1.1rem; padding-bottom:3rem; max-width:1480px;}

*::-webkit-scrollbar{width:10px;height:10px}
*::-webkit-scrollbar-track{background:#0a0f1a}
*::-webkit-scrollbar-thumb{background:linear-gradient(180deg,var(--btc),var(--magenta));border-radius:8px;border:2px solid #0a0f1a}
*::-webkit-scrollbar-thumb:hover{background:linear-gradient(180deg,var(--cyan),var(--violet))}
*{scrollbar-width:thin; scrollbar-color:#f7931a #0a0f1a;}

[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0a0f1a 0%,#0b1120 100%);
  border-right:1px solid var(--line); min-width:326px; max-width:326px;
}
.sb-sec{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.2em;
  color:var(--btc);text-transform:uppercase;margin:1.1rem 0 .45rem;
  display:flex;align-items:center;gap:.5rem;}
.sb-sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}
.sb-logo{font-weight:700;font-size:1.06rem;letter-spacing:-.01em;line-height:1.2;color:var(--text);}
.sb-logo em{display:block;font-style:normal;font-size:1.22rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.sb-tag{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--muted);letter-spacing:.16em;margin-top:.35rem;}

.hero{position:relative;padding:.6rem 0 .2rem;}
.hero .tape{font-family:'IBM Plex Mono',monospace;font-size:.66rem;letter-spacing:.26em;color:var(--cyan);}
.pulse{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--green);
  box-shadow:0 0 10px var(--green);margin-right:.55rem;animation:pulse 1.9s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.hero h1{font-size:clamp(1.9rem,3.6vw,2.9rem);font-weight:700;letter-spacing:-.025em;
  margin:.3rem 0 .1rem;line-height:1.06;color:var(--text);}
.hero h1 .grad{display:block;background:linear-gradient(92deg,var(--btc) 5%,var(--magenta) 52%,var(--cyan) 95%);
  -webkit-background-clip:text;background-clip:text;color:transparent;}
.hero .sub{color:var(--muted);max-width:830px;font-size:.94rem;margin:.35rem 0 .6rem;}
.hero .sub b{color:var(--btc2);font-weight:600;}
.hairline{height:3px;border-radius:3px;margin:.5rem 0 .9rem;
  background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan),var(--btc));
  background-size:300% 100%;animation:shimmer 8s linear infinite;opacity:.8;}
@keyframes shimmer{0%{background-position:0% 0}100%{background-position:300% 0}}
.chips{display:flex;flex-wrap:wrap;gap:.4rem;margin-bottom:.3rem;}
.chip{font-family:'IBM Plex Mono',monospace;font-size:.65rem;letter-spacing:.07em;
  padding:.26rem .58rem;border-radius:999px;border:1px solid var(--line);
  background:rgba(16,23,40,.75);color:var(--muted);}
.chip b{color:var(--text);font-weight:600;}
.chip.hot{border-color:rgba(247,147,26,.5);color:var(--btc2);}
.chip.cool{border-color:rgba(34,211,238,.4);color:var(--cyan);}

.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(168px,1fr));gap:.7rem;margin:.5rem 0 .9rem;}
.card{position:relative;overflow:hidden;background:linear-gradient(160deg,#0d1322,#0a0f1a);
  border:1px solid var(--line);border-radius:14px;padding:.75rem .95rem .65rem;}
.card::before{content:"";position:absolute;top:0;left:0;right:0;height:2px;
  background:linear-gradient(90deg,var(--btc),var(--magenta));opacity:.8;}
.card.cyan::before{background:linear-gradient(90deg,var(--cyan),var(--violet));}
.card.green::before{background:linear-gradient(90deg,var(--green),var(--cyan));}
.card.red::before{background:linear-gradient(90deg,var(--red),var(--magenta));}
.card .k{font-family:'IBM Plex Mono',monospace;font-size:.6rem;letter-spacing:.17em;
  text-transform:uppercase;color:var(--muted);}
.card .v{font-family:'IBM Plex Mono',monospace;font-size:1.5rem;font-weight:600;color:var(--text);margin-top:.12rem;}
.card .d{font-size:.7rem;color:var(--muted);margin-top:.08rem;}
.card .d.up{color:var(--green);} .card .d.dn{color:var(--red);} .card .d.warn{color:var(--btc2);}

.verdict{border-radius:16px;padding:.95rem 1.15rem;margin:.3rem 0 .9rem;
  border:1px solid;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;}
.verdict .big{font-weight:700;font-size:1.2rem;letter-spacing:.02em;}
.verdict .txt{color:var(--muted);font-size:.85rem;flex:1;min-width:260px;line-height:1.55;}
.verdict.sig{border-color:rgba(44,229,167,.5);background:linear-gradient(120deg,rgba(44,229,167,.12),rgba(34,211,238,.06));}
.verdict.sig .big{color:var(--green);text-shadow:0 0 18px rgba(44,229,167,.45);}
.verdict.weak{border-color:rgba(247,147,26,.5);background:linear-gradient(120deg,rgba(247,147,26,.13),rgba(255,46,166,.05));}
.verdict.weak .big{color:var(--btc2);text-shadow:0 0 18px rgba(247,147,26,.45);}
.verdict.noise{border-color:rgba(255,77,94,.5);background:linear-gradient(120deg,rgba(255,77,94,.13),rgba(255,46,166,.06));}
.verdict.noise .big{color:var(--red);text-shadow:0 0 18px rgba(255,77,94,.45);}

.sec{display:flex;align-items:center;gap:.6rem;margin:1.15rem 0 .5rem;}
.sec .idx{font-family:'IBM Plex Mono',monospace;font-size:.6rem;color:var(--ink);
  background:linear-gradient(135deg,var(--btc),var(--magenta));border-radius:6px;padding:.14rem .4rem;font-weight:600;}
.sec .t{font-weight:600;font-size:1.02rem;color:var(--text);}
.sec::after{content:"";flex:1;height:1px;background:linear-gradient(90deg,var(--line),transparent);}

.panel{background:linear-gradient(150deg,#0d1424,#0a0f1a);border:1px solid var(--line);
  border-radius:16px;padding:1.05rem 1.2rem;margin:.5rem 0;}
.panel h3{margin:.1rem 0 .45rem;font-size:1.1rem;color:var(--text);}
.panel p{color:var(--muted);font-size:.87rem;line-height:1.65;margin:.2rem 0;}
.steps{display:grid;grid-template-columns:repeat(auto-fit,minmax(215px,1fr));gap:.75rem;margin-top:.7rem;}
.step{border:1px solid rgba(139,148,171,.18);border-radius:12px;padding:.75rem .85rem;background:rgba(11,16,28,.6);}
.step .n{font-family:'IBM Plex Mono',monospace;font-size:1.25rem;font-weight:600;
  background:linear-gradient(90deg,var(--btc),var(--magenta));-webkit-background-clip:text;background-clip:text;color:transparent;}
.step .h{font-weight:600;color:var(--text);margin:.15rem 0;}
.step .b{font-size:.79rem;color:var(--muted);line-height:1.5;}
.pd-grid{display:grid;grid-template-columns:1fr 1fr;gap:.75rem;}
.pd{border-radius:14px;padding:.9rem 1rem;border:1px solid;}
.pd.keep{border-color:rgba(44,229,167,.35);background:rgba(44,229,167,.06);}
.pd.kill{border-color:rgba(255,77,94,.35);background:rgba(255,77,94,.06);}
.pd h4{margin:0 0 .45rem;font-size:.92rem;}
.pd.keep h4{color:var(--green);} .pd.kill h4{color:var(--red);}
.pd ul{margin:0;padding-left:1.05rem;color:var(--muted);font-size:.82rem;line-height:1.6;}
.pd li b{color:var(--text);font-weight:600;}
.callout{border-left:3px solid var(--btc);background:rgba(247,147,26,.07);
  border-radius:0 12px 12px 0;padding:.75rem .95rem;color:var(--muted);font-size:.85rem;margin:.75rem 0;line-height:1.6;}
.callout b{color:var(--btc2);}
.strat-block{border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;margin:.7rem 0;
  background:linear-gradient(155deg,#0d1424,#0a0f1a);}
.strat-block .hd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin-bottom:.45rem;}
.strat-block .nm{font-weight:700;font-size:1.05rem;color:var(--text);}
.strat-block .tagline{font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:var(--cyan);letter-spacing:.08em;}
.strat-block ol{margin:.25rem 0 .7rem;padding-left:1.15rem;color:var(--muted);font-size:.84rem;line-height:1.65;}
.strat-block ol b{color:var(--text);font-weight:600;}
.strat-block .ask{font-size:.8rem;color:var(--btc2);border-top:1px dashed rgba(247,147,26,.25);padding-top:.55rem;margin-top:.6rem;}

button[data-baseweb="tab"]{font-family:'IBM Plex Mono',monospace;font-size:.76rem;letter-spacing:.05em;color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--btc2);}
div[data-baseweb="tab-highlight"]{background:linear-gradient(90deg,var(--btc),var(--magenta));height:3px;}
div[data-baseweb="tab-border"]{background:var(--line);}
[data-testid="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--btc),var(--magenta));color:#0a0710;border:none;
  font-weight:700;letter-spacing:.04em;box-shadow:0 4px 20px rgba(247,147,26,.32);
  transition:transform .15s ease, box-shadow .15s ease;}
[data-testid="stBaseButton-primary"]:hover{transform:translateY(-1px);box-shadow:0 6px 26px rgba(255,46,166,.42);color:#0a0710;}
[data-testid="stBaseButton-secondary"]{background:var(--panel2);border:1px solid rgba(34,211,238,.32);color:var(--text);}
[data-testid="stBaseButton-secondary"]:hover{border-color:var(--btc);color:var(--btc2);}
[data-baseweb="slider"] div[role="slider"]{background:var(--btc);box-shadow:0 0 10px rgba(247,147,26,.55);}
[data-baseweb="select"]>div{background:var(--panel2);border-color:var(--line);}
[data-testid="stNumberInput"] input{background:var(--panel2);color:var(--text);border-color:var(--line);font-family:'IBM Plex Mono',monospace;}
[data-baseweb="checkbox"] [aria-checked="true"]{background-color:var(--btc) !important;}
[data-testid="stFileUploaderDropzone"]{background:var(--panel2);border:1px dashed rgba(247,147,26,.4);border-radius:12px;}
[data-testid="stFileUploaderDropzoneInstructions"] span{color:var(--muted);}
[data-testid="stProgress"] div[role="progressbar"]{background:rgba(139,148,171,.15);}
[data-testid="stProgress"] div[role="progressbar"]>div{background:linear-gradient(90deg,var(--btc),var(--magenta),var(--cyan));}
[data-testid="stExpander"] details{background:var(--panel);border:1px solid var(--line);border-radius:12px;}
[data-testid="stCodeBlock"] pre{max-height:440px;overflow:auto;background:#0a0f1a !important;
  border:1px solid var(--line);border-radius:12px;}
[data-testid="stTable"] table{background:transparent;color:var(--text);font-family:'IBM Plex Mono',monospace;font-size:.78rem;}
[data-testid="stTable"] th{color:var(--btc2) !important;border-bottom:1px solid var(--line) !important;}
[data-testid="stTable"] td{border-bottom:1px solid rgba(139,148,171,.12) !important;}
[data-testid="stAlert"]{border-radius:12px;}
.hist-row{display:flex;align-items:center;gap:.6rem;padding:.42rem .6rem;border:1px solid rgba(139,148,171,.14);
  border-radius:10px;margin-bottom:.35rem;background:rgba(11,16,28,.55);font-family:'IBM Plex Mono',monospace;font-size:.71rem;flex-wrap:wrap;}
.hist-row .lbl{flex:1;color:var(--text);min-width:180px;}
.hist-row .m{color:var(--muted);}
.badge{padding:.12rem .5rem;border-radius:999px;font-size:.61rem;font-weight:600;}
.badge.sig{background:rgba(44,229,167,.15);color:var(--green);}
.badge.weak{background:rgba(247,147,26,.15);color:var(--btc2);}
.badge.noise{background:rgba(255,77,94,.15);color:var(--red);}
.foot{margin-top:2rem;padding-top:.85rem;border-top:1px solid var(--line);
  font-family:'IBM Plex Mono',monospace;font-size:.63rem;letter-spacing:.13em;color:var(--muted);}
.foot b{color:var(--btc2);font-weight:600;}
@media (max-width:760px){
  [data-testid="stSidebar"]{min-width:0;max-width:none;}
  .pd-grid{grid-template-columns:1fr;}
  .card .v{font-size:1.22rem;}
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
        margin=dict(l=48, r=18, t=42 if title else 16, b=40),
        title=dict(text=title, x=0.01, font=dict(size=13, color=C["text"], family="Space Grotesk")),
        legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["btc"], font=dict(family="IBM Plex Mono", size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    return fig


def finite_only(y):
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y), y, np.nan)


def hist_fig(perm_pfs, real_pf, p, live=False):
    vals = np.asarray([v for v in perm_pfs if np.isfinite(v)], dtype=float)
    fig = base_fig(370, "Optimized PF on permuted noise vs real market" + (" · LIVE" if live else ""))
    if len(vals) == 0:
        fig.add_annotation(text="collecting permutations…", xref="paper", yref="paper",
                           x=.5, y=.5, showarrow=False, font=dict(color=C["muted"]))
        return fig
    rf = real_pf if np.isfinite(real_pf) else None
    lo = min(vals.min(), rf if rf is not None else vals.min())
    hi = max(vals.max(), rf if rf is not None else vals.max())
    pad = max((hi - lo) * 0.05, 0.01)
    counts, edges = np.histogram(vals, bins=42, range=(lo - pad, hi + pad))
    centers = (edges[:-1] + edges[1:]) / 2
    fig.add_trace(go.Bar(x=centers, y=counts, width=(edges[1] - edges[0]) * .92,
                         marker_color=C["cyan"], opacity=.78, name="Permuted best PF",
                         hovertemplate="PF %{x:.3f}<br>count %{y}<extra></extra>"))
    if rf is not None:
        fig.add_vrect(x0=rf, x1=hi + pad, fillcolor=C["red"], opacity=.08, line_width=0)
        fig.add_vline(x=rf, line_color=C["btc"], line_width=2.5, line_dash="dash",
                      annotation_text=f"REAL {rf:.3f}", annotation_position="top",
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


def equity_fig(x, real_eq, samples, title="Equity — real vs permuted universes (Σ log return)"):
    fig = base_fig(390, title)
    x = np.asarray(x)
    real = finite_only(real_eq)
    curves = [finite_only(s) for s in samples if len(s)]

    if curves:
        m = min(len(c) for c in curves)
        mat = np.vstack([c[:m] for c in curves])
        xx = x[:m]
        for cy in mat:
            fig.add_trace(go.Scatter(x=xx, y=cy, mode="lines",
                                     line=dict(color=C["cyan"], width=1.1), opacity=.45,
                                     showlegend=False, hoverinfo="skip", name="Permuted universe"))
        lo10, hi10 = np.nanpercentile(mat, 10, axis=0), np.nanpercentile(mat, 90, axis=0)
        fig.add_trace(go.Scatter(x=np.concatenate([xx, xx[::-1]]),
                                 y=np.concatenate([hi10, lo10[::-1]]),
                                 fill="toself", fillcolor="rgba(34,211,238,.14)", line=dict(width=0),
                                 hoverinfo="skip", name=f"Permuted 10–90% · {len(mat)} paths"))
        fig.add_trace(go.Scatter(x=xx, y=np.nanmedian(mat, axis=0), mode="lines",
                                 line=dict(color=C["cyan"], width=1.7, dash="dot"),
                                 hoverinfo="skip", name="Permuted median"))

    # real path last => always on top of the noise bundle (halo + crisp line)
    fig.add_trace(go.Scatter(x=x[:len(real)], y=real, mode="lines",
                             line=dict(color=C["btc"], width=8), opacity=.18,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[:len(real)], y=real, mode="lines",
                             line=dict(color=C["btc"], width=2.8), name="Real strategy",
                             hovertemplate="real ΣlogR %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.14, x=0))
    fig.update_yaxes(title="Σ log return")
    return fig


def rebased_fig(x, lcr, lcp, title="One real market vs one permuted universe (log price, rebased)"):
    fig = base_fig(370, title)
    x = np.asarray(x); yr = finite_only(lcr); yp = finite_only(lcp)
    n = min(len(x), len(yr), len(yp)); x, yr, yp = x[:n], yr[:n], yp[:n]
    fig.add_trace(go.Scatter(x=x, y=yr, mode="lines", name="Real market",
                             line=dict(color=C["btc"], width=2.8),
                             hovertemplate="real %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=yp, mode="lines", name="Permuted universe",
                             line=dict(color=C["cyan"], width=1.8),
                             fill="tonexty", fillcolor="rgba(139,123,255,.14)",
                             hovertemplate="permuted %{y:+.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=x, y=yr, mode="lines", line=dict(color=C["btc"], width=2.8),
                             showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    if n and np.isfinite(yr[-1]) and np.isfinite(yp[-1]):
        fig.add_trace(go.Scatter(x=[x[-1]], y=[yr[-1]], mode="markers", showlegend=False,
                                 hoverinfo="skip",
                                 marker=dict(size=8, color=C["btc"], line=dict(color=C["text"], width=1))))
        fig.add_annotation(x=x[-1], y=float(yr[-1]), xanchor="right", yanchor="bottom",
                           showarrow=False, font=dict(color=C["muted"], size=10),
                           text=f"same bars, same total return ({yr[-1]:+.2f}) — different path")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title="log price · rebased to 0")
    return fig


def optcurve_fig(lookbacks, curve, best_L, mean=None, std=None):
    fig = base_fig(340, "Data-mining surface — PF vs lookback · real vs noise envelope")
    if mean is not None and std is not None:
        m = np.isfinite(mean) & np.isfinite(std)
        xl = np.asarray(lookbacks)[m]
        if len(xl):
            fig.add_trace(go.Scatter(x=np.concatenate([xl, xl[::-1]]),
                                     y=np.concatenate([(mean + std)[m], (mean - std)[m][::-1]]),
                                     fill="toself", fillcolor="rgba(139,123,255,.16)",
                                     line=dict(width=0), name="Noise ±1σ", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xl, y=mean[m], mode="lines",
                                     line=dict(color=C["cyan"], width=1.6, dash="dot"),
                                     name="Noise mean best-PF"))
    yc = finite_only(curve)
    fig.add_trace(go.Scatter(x=lookbacks, y=yc, mode="lines",
                             line=dict(color=C["btc"], width=2.2), name="Real PF(lookback)"))
    if best_L and best_L > 0:
        k = int(np.argmin(np.abs(np.asarray(lookbacks) - best_L)))
        if np.isfinite(yc[k]):
            fig.add_trace(go.Scatter(x=[lookbacks[k]], y=[yc[k]], mode="markers",
                                     marker=dict(symbol="star", size=13, color=C["magenta"],
                                                 line=dict(color=C["text"], width=1)),
                                     name=f"Selected L={best_L}"))
    fig.add_hline(y=1.0, line_color="rgba(139,148,171,.4)", line_dash="dash", line_width=1)
    fig.update_xaxes(title="Donchian lookback (bars)")
    fig.update_yaxes(title="Profit Factor")
    return fig


def price_fig(det, style, show_overlays, title):
    fig = base_fig(420, title)
    if style == "Candles":
        d = det["daily"]
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            increasing_line_color=C["green"], increasing_fillcolor="rgba(44,229,167,.55)",
            decreasing_line_color=C["red"], decreasing_fillcolor="rgba(255,77,94,.55)",
            name="Price · 1D", whiskerwidth=.4))
    else:
        fig.add_trace(go.Scatter(x=det["eq_x"], y=det["close_ds"], mode="lines",
                                 line=dict(color=C["btc"], width=1.8), name="Close · 1H"))
    if show_overlays:
        for name, (series, color, dash) in det["overlays"].items():
            fig.add_trace(go.Scatter(x=series.index, y=finite_only(series.to_numpy()),
                                     mode="lines", line=dict(color=color, width=1.4, dash=dash),
                                     name=name))
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    fig.update_yaxes(title="price")
    return fig


def position_fig(x, pos):
    fig = base_fig(165, "Position lane — separate from price")
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines",
                             line=dict(color=C["cyan"], width=1.4, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(34,211,238,.18)", name="Position"))
    fig.update_yaxes(range=[-1.3, 1.3], tickvals=[-1, 0, 1],
                     ticktext=["−1 short", "0 flat", "+1 long"], title=None)
    fig.update_layout(margin=dict(l=70, r=18, t=36, b=28), showlegend=False)
    return fig


def dd_fig(x, dd):
    fig = base_fig(240, "Drawdown (log-equity units)")
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color=C["red"], width=1.2),
                             fillcolor="rgba(255,77,94,.25)", name="Drawdown"))
    return fig


def choices_fig(choices, title="Walk-forward — lookback re-selected at each training step"):
    fig = base_fig(240, title)
    fig.add_trace(go.Scatter(x=[t for t, _ in choices], y=[b for _, b in choices],
                             mode="lines", line=dict(color=C["magenta"], width=2, shape="hv"),
                             name="Chosen lookback"))
    fig.update_yaxes(title="lookback (bars)")
    return fig


def acf_fig(lags, real, perm, n, title):
    fig = base_fig(290, title)
    fig.add_trace(go.Bar(x=lags, y=real, marker_color=C["btc"], opacity=.85, width=.42, name="Real"))
    fig.add_trace(go.Bar(x=lags, y=perm, marker_color=C["cyan"], opacity=.6, width=.42, name="Permuted"))
    b = 1.96 / np.sqrt(max(n, 2))
    fig.add_hline(y=b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.add_hline(y=-b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.update_layout(barmode="group", bargap=.15)
    fig.update_xaxes(title="lag (hours)")
    return fig


def spec_fig(fx, px, fy, py):
    fig = base_fig(310, "Power spectrum of |returns| — long memory lives here (log-log)")
    fig.add_trace(go.Scatter(x=fx, y=px, mode="lines", line=dict(color=C["btc"], width=2), name="Real |r|"))
    fig.add_trace(go.Scatter(x=fy, y=py, mode="lines", line=dict(color=C["cyan"], width=1.6, dash="dot"), name="Permuted |r|"))
    fig.update_xaxes(type="log", title="frequency (cycles / bar)")
    fig.update_yaxes(type="log", title="power")
    return fig


def corr_fig(rb, re, pb, pe, rho_r, rho_p):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"Real · ρ = {rho_r:.3f}", f"Joint permutation · ρ = {rho_p:.3f}"))
    fig.add_trace(go.Scattergl(x=rb, y=re, mode="markers", marker=dict(size=3, color=C["btc"], opacity=.45), name="Real"), 1, 1)
    fig.add_trace(go.Scattergl(x=pb, y=pe, mode="markers", marker=dict(size=3, color=C["cyan"], opacity=.45), name="Permuted"), 1, 2)
    fig.update_layout(height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,.55)",
                      font=dict(family="IBM Plex Mono", size=11, color="#aab3c8"),
                      margin=dict(l=48, r=18, t=48, b=42), showlegend=False,
                      title=dict(text="Cross-market correlation survives joint permutation", x=.01,
                                 font=dict(size=13, color=C["text"], family="Space Grotesk")))
    fig.update_xaxes(gridcolor="rgba(139,148,171,.1)", title="Currency 1 log return")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.1)", title="Currency 2 log return")
    for a in fig.layout.annotations:
        a.font = dict(family="IBM Plex Mono", size=11, color=C["muted"])
    return fig


def imp_fig(importances):
    names = ["diff6 · 6h momentum", "diff24 · 1d momentum", "diff168 · 1w momentum"]
    fig = base_fig(220, "Tree feature importances — what the classifier actually uses")
    fig.add_trace(go.Bar(y=names, x=importances, orientation="h",
                         marker_color=[C["btc"], C["cyan"], C["magenta"]], opacity=.85,
                         text=[f"{v:.1%}" for v in importances], textposition="outside",
                         textfont=dict(color=C["text"])))
    fig.update_xaxes(title="Gini importance", range=[0, max(importances) * 1.3 + 1e-6])
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
    if len(x) < 8:
        return np.zeros(nlags + 1), max(len(x), 1)
    x = x - x.mean()
    n = len(x)
    f = np.fft.rfft(x, n=2 ** int(np.ceil(np.log2(2 * n))))
    ac = np.fft.irfft(f * np.conj(f))[: nlags + 1]
    if ac[0] <= 0:
        return np.zeros(nlags + 1), n
    return ac / ac[0], n


def logbin_spectrum(x, nbins=46):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 64:
        return np.array([1.0]), np.array([1.0])
    x = x - x.mean()
    n = len(x)
    p = (np.abs(np.fft.rfft(x * np.hanning(n))) ** 2)[1:]
    fr = np.fft.rfftfreq(n)[1:]
    edges = np.unique(np.logspace(0, np.log10(len(p)), nbins + 1).astype(int))
    fx, px = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b > a:
            fx.append(float(np.exp(np.mean(np.log(fr[a:b])))))
            px.append(float(max(np.mean(p[a:b]), 1e-18)))
    if not fx:
        return np.array([1.0]), np.array([1.0])
    return np.asarray(fx), np.asarray(px)


def sanitize_range(key, lo, hi):
    v = st.session_state.get(key)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = int(v[0]), int(v[1])
        if not (lo <= a < b <= hi):
            st.session_state.pop(key, None)
    elif v is not None:
        st.session_state.pop(key, None)


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
    df = df[~df.index.duplicated(keep="first")]
    need = {"open", "high", "low", "close"}
    if not need.issubset(df.columns):
        raise ValueError(f"{name}: needs columns open / high / low / close")
    df = df[["open", "high", "low", "close"]].astype(float).dropna()
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    if len(df) < 500:
        raise ValueError(f"{name}: needs at least 500 hourly bars")
    return df


@st.cache_data(show_spinner=False)
def load_path(path, mtime):
    return parse_bytes(path, Path(path).read_bytes())


def resolve_data(source, up1, up2):
    """Returns (currency1, currency2 or None, label, notice)."""
    if source == "Upload files":
        if up1 is None:
            b, e = synth_pair()
            return b, e, "DEMO FEED", "No file uploaded yet — running on a seeded synthetic demo feed."
        c1 = parse_bytes(up1.name, up1.getvalue())
        c2 = parse_bytes(up2.name, up2.getvalue()) if up2 is not None else None
        return c1, c2, up1.name, None
    if source.startswith("Local"):
        p = Path("BTCUSD3600.pq")
        c1 = load_path(str(p), p.stat().st_mtime)
        pe = Path("ETHUSD3600.pq")
        c2 = load_path(str(pe), pe.stat().st_mtime) if pe.exists() else None
        return c1, c2, p.name + (" + ETH" if c2 is not None else ""), None
    b, e = synth_pair()
    return b, e, "DEMO FEED", None


# ---------------------------------------------------------------- config & callbacks
def collect_cfg():
    g = st.session_state.get
    depth = g("w_depth", "Unlimited")
    lbmin, lbmax = int(g("w_lbmin", 12)), int(g("w_lbmax", 168))
    if lbmin >= lbmax:
        lbmin, lbmax = min(lbmin, lbmax - 1), max(lbmax, lbmin + 1)
    is_years = g("w_is_years", (2016, 2020))
    wf_years = g("w_wf_years", (2016, 2021))
    wf_step = int(g("w_wf_step", 30))
    wf_train = int(g("w_wf_train", 4))
    _btc = st.session_state.get("btc_df")
    if _btc is not None:                      # never let a stale slider value break the run
        _y = _btc.index.year
        _n = int(((_y >= int(wf_years[0])) & (_y < int(wf_years[1]))).sum())
        wf_train = int(min(max(wf_train, 1), max(wf_max_train_years(_n, wf_step), 1)))
    return dict(
        strategy=g("w_strategy", "donchian_is"),
        n_perms=int(g("w_perms", 1000)),
        seed=BASE_SEED,
        turbo=bool(g("w_turbo", True)),
        lb_min=lbmin, lb_max=lbmax, lb_step=int(g("w_lbstep", 1)),
        is_start=int(is_years[0]), is_end=int(is_years[1]),
        wf_start=int(wf_years[0]), wf_end=int(wf_years[1]),
        wf_train_years=wf_train,
        wf_step_days=wf_step,
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
    if cfg["strategy"] == "ma" and cfg["ma_fast"] >= cfg["ma_slow"]:
        st.session_state["run_error"] = "Fast MA must be shorter than slow MA."
        return
    if cfg["strategy"] == "donchian_wf":
        y = btc.index.year
        n_win = int(((y >= cfg["wf_start"]) & (y < cfg["wf_end"])).sum())
        try:
            wf_plan(n_win, cfg["wf_train_years"], cfg["wf_step_days"])
        except ValueError as ex:
            st.session_state["run_error"] = str(ex)
            return
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


def new_permutation_draw():
    st.session_state["anat_draw"] = int(st.session_state.get("anat_draw", 0)) + 1


def json_safe(o):
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


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
            if bL > 1:
                sig = breakout_signal_np(c, bL)
                s = win["close"]
                overlays[f"Channel upper · L={bL}"] = (s.rolling(bL - 1).max().shift(1), C["btc2"], "dot")
                overlays[f"Channel lower · L={bL}"] = (s.rolling(bL - 1).min().shift(1), C["violet"], "dot")
            else:
                sig = np.full(len(c), np.nan)
            extra["best_L"] = bL

        elif strat == "donchian_wf":
            try:
                tw, tstep, n_oos = wf_plan(len(win), cfg["wf_train_years"], cfg["wf_step_days"])
            except ValueError as ex:
                return {"error": str(ex)}
            if cfg["turbo"]:
                sig, choices = walkforward_donch_fast(c, tw, tstep, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"], return_choices=True)
                extra["wf_choices"] = [(win.index[min(t, len(win) - 1)], int(b)) for t, b in choices]
            else:
                sig = walkforward_donch(win, tw, tstep, cfg["lb_min"], cfg["lb_max"], cfg["lb_step"])
            # keep the out-of-sample segment only — no in-sample bar is scored or drawn
            sig = sig[tw:]
            win = win.iloc[tw:]
            c = win["close"].to_numpy()
            r = np.empty(len(c)); r[:-1] = np.diff(np.log(c)); r[-1] = np.nan
            extra["oos_bars"] = len(win)

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
        valid = np.isfinite(sig)
        chg = np.flatnonzero(valid[1:] & valid[:-1] & (sig[1:] != sig[:-1])) + 1
        daily = win.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        dov = {name: (ser.resample("1D").last(), col, dash) for name, (ser, col, dash) in overlays.items()}
        nz = rs.dropna()
        nz = nz[nz != 0]
        std = rs.std()

        return dict(
            error=None, daily=daily, overlays=dov,
            eq_x=win.index[::step], eq=eq[::step], dd=dd[::step],
            close_ds=c[::step], pos_ds=np.nan_to_num(sig[::step], nan=0.0),
            stats=dict(
                pf=pf,
                sharpe=float(rs.mean() / std) if std and std > 0 else float("nan"),
                total=float(np.nansum(rets)),
                exposure=float(np.nanmean(np.abs(sig))) if valid.any() else 0.0,
                trades=int(len(chg)),
                winrate=float((nz > 0).mean()) if len(nz) else float("nan"),
                bars=len(win),
            ),
            extra=extra,
        )
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------- anatomy
@st.cache_data(show_spinner="Shuffling the universe…")
def anatomy(btc, eth, draw, nlags):
    seed = ANATOMY_SEED + int(draw)
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
        fxr=fxr, pxr=pxr, fxp=fxp, pxp=pxp, corr=None, span=f"{sl.index[0]:%Y-%m-%d} → {sl.index[-1]:%Y-%m-%d}",
    )
    if eth is not None:
        common = sl.index.intersection(eth.index)
        if len(common) > 1000:
            b2 = sl.loc[common]
            e2 = eth.loc[common]
            pb, pe = get_permutation([b2, e2], seed=seed)
            rb = np.log(b2["close"]).diff().dropna()
            re = np.log(e2["close"]).diff().dropna()
            qb = np.log(pb["close"]).diff().dropna()
            qe = np.log(pe["close"]).diff().dropna()
            k = min(2600, len(rb), len(qb))
            sel = np.linspace(0, min(len(rb), len(qb)) - 1, k).astype(int)
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
    finite = [v for v in job["perm_pfs"] if np.isfinite(v)]
    med = float(np.median(finite)) if finite else float("nan")

    if not live:
        vlabel, vcls, _ = job.get("verdict", verdict(p))
        if vcls == "sig":
            txt = f"Only {better-1} of {done} permuted universes matched or beat the real optimized PF. Selection bias alone rarely explains this — evidence of a genuine, exploitable pattern."
        elif vcls == "weak":
            txt = f"{better-1} of {done} noise universes beat the real result. Suggestive, not conclusive. More permutations and out-of-sample validation are warranted."
        else:
            txt = f"{better-1} of {done} structureless universes produced an equally good 'edge'. The optimizer is mining noise — classic data-mining bias. Treat this strategy as unproven."
        st.markdown(
            f'<div class="verdict {vcls}"><span class="big">{"◆ " if vcls=="sig" else "◇ "}{vlabel}</span>'
            f'<span class="txt">{txt}</span></div>', unsafe_allow_html=True)

        items = [
            dict(k="Real Profit Factor", v=f"{real_pf:.4f}" if np.isfinite(real_pf) else "—"),
            dict(k="MCPT p-value", v=f"{p:.3f}",
                 cls="green" if vcls == "sig" else ("red" if vcls == "noise" else ""),
                 d=f"{better} / {done+1} ≥ real", dc="warn"),
            dict(k="Permutations", v=f"{done:,}", cls="cyan",
                 d=f"target {cfg['n_perms']:,}" + (" · stopped early" if job["status"] == "stopped" else "")),
        ]
        if finite:
            lift = real_pf - med
            items.append(dict(k="Edge vs noise median", v=f"{lift:+.4f}",
                              cls="green" if lift > 0 else "red",
                              d=f"noise median {med:.4f}", dc="up" if lift > 0 else "dn"))
        st.markdown(cards_html(items), unsafe_allow_html=True)
        meta = [f"elapsed {fmt_t(job.get('elapsed',0))}"]
        if job.get("best_lookback") and job["best_lookback"] > 0:
            meta.append(f"selected lookback {job['best_lookback']}")
        if job.get("finished_at"):
            meta.append(job["finished_at"])
        st.caption(" · ".join(meta))

    colA, colB = st.columns([1.35, 1])
    with colA:
        show_fig(hist_fig(job["perm_pfs"], real_pf, p, live), key=f"hist_{job['id']}_{live}")
    with colB:
        if job.get("real_curve") is not None:
            eq_title = ("Equity — out-of-sample only · real vs permuted universes (Σ log return)"
                        if job.get("oos_start") else
                        "Equity — real vs permuted universes (Σ log return)")
            show_fig(equity_fig(job["curve_x"], job["real_curve"], job["sample_curves"], eq_title),
                     key=f"eq_{job['id']}_{live}")
        else:
            st.info("Equity curves appear once the baseline finishes.")

    if job.get("pf_curve") is not None and job.get("env_cnt") is not None:
        cnt = job["env_cnt"]
        mean = np.where(cnt > 0, job["env_sum"] / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 0, job["env_ssq"] / np.maximum(cnt, 1) - mean**2, np.nan)
        std = np.sqrt(np.maximum(var, 0))
        show_fig(optcurve_fig(job["lookbacks"], job["pf_curve"], job.get("best_lookback"), mean, std),
                 key=f"opt_{job['id']}_{live}")
        st.caption("Amber = the real optimization surface. Cyan/violet = what the same optimizer achieves on pure noise. The gap between them — not the peak itself — is the evidence.")

    if job.get("wf_choices"):
        show_fig(choices_fig(job["wf_choices"]), key=f"wfch_{job['id']}_{live}")

    if job.get("importances"):
        show_fig(imp_fig(job["importances"]), key=f"imp_{job['id']}_{live}")

    if not live:
        with st.expander("What this null hypothesis does — and does not — rule out"):
            st.markdown(
                "Bar permutation preserves the **marginal distribution** (mean, std, skew, kurtosis), intrabar geometry and gaps, "
                "but destroys **volatility clustering**, **long memory**, **self-similarity** and spectral structure — anything involving dependence. "
                "A strategy that unknowingly harvests volatility clustering is tested against a null where that property is absent; "
                "a significant p-value then means the edge needs *serial structure*, not merely fat tails. "
                "A non-significant result does not disprove every conceivable edge — only that this pipeline cannot distinguish itself from shuffled noise. "
                "All results are gross of fees, slippage and funding."
            )
        e1, e2, e3 = st.columns(3)
        payload = dict(
            id=job["id"], label=job["label"], config=cfg, real_pf=real_pf,
            p_value=p, better=better, completed=done,
            best_lookback=job.get("best_lookback"),
            perm_pfs=[v if np.isfinite(v) else None for v in job["perm_pfs"]],
            finished_at=job.get("finished_at"), built_by=AUTHOR,
        )
        e1.download_button("⬇ Results (.json)", json.dumps(payload, default=json_safe, indent=2),
                           file_name=f"mcpt_{job['id']}.json", mime="application/json", key=f"js_{job['id']}")
        e2.download_button("⬇ Permuted PFs (.csv)",
                           "perm_pf\n" + "\n".join(f"{v:.6f}" if np.isfinite(v) else "" for v in job["perm_pfs"]),
                           file_name=f"perm_pfs_{job['id']}.csv", mime="text/csv", key=f"csv_{job['id']}")
        e3.button("↻ Re-run current settings", on_click=start_run, key=f"re_{job['id']}")


def onboarding():
    st.markdown(
        """
<div class="panel">
  <h3>Is your optimized edge real — or the best-looking lie in 157 tries?</h3>
  <p>Optimization <b style="color:var(--btc2)">always</b> induces data-mining bias. This terminal rebuilds the market as
  structureless noise — same distribution, zero memory — re-runs your <i>entire</i> pipeline on each shuffled universe,
  and asks how often noise alone beats your real result. Aim for <b style="color:var(--cyan)">1000 permutations</b>.</p>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Optimize on real bars</div><div class="b">Donchian lookback sweep, walk-forward re-training, decision tree, or MA crossover — the full selection procedure, bias included.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Shuffle 1000 universes</div><div class="b">Intrabar shape and gaps permuted separately. Mean/std/skew/kurt survive; volatility clustering and long memory die.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Compare &amp; verdict</div><div class="b">p = (1 + #noise-PF ≥ real-PF) / (N + 1). If noise matches your edge, the edge is a mirage. Watch the histogram build live.</div></div>
  </div>
  <div class="callout" style="margin-top:.85rem"><b>→ Upload Currency 1 in the sidebar, pick a pipeline, press RUN PERMUTATION TEST.</b> Results stream as they compute; stop anytime. Each pipeline's mechanics, strengths and weaknesses are in Method &amp; Guide.</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.6)
def live_fragment():
    job = st.session_state.get("active")
    if job is None or job.get("status") != "running":
        if job is not None and not job.get("_finalized"):
            finalize(job)
        st.rerun(scope="app")
        return
    cfg = job["config"]
    n_target = cfg["n_perms"]
    done = job["done"]
    elapsed = time.time() - job["start"]
    eta = (elapsed / done) * (n_target - done) if done > 0 else None
    st.progress(min(done / max(n_target, 1), 1.0))
    p = job["better"] / (done + 1)
    c1, c2 = st.columns([4, 1])
    with c1:
        st.markdown(cards_html([
            dict(k="Phase", v="⚙ LIVE", cls="cyan", d=job.get("phase", "")),
            dict(k="Completed", v=f"{done:,}", d=f"of {n_target:,} permutations"),
            dict(k="Noise beats real", v=str(job["better"] - 1),
                 cls="red" if job["better"] - 1 > max(done, 1) * .2 else "green", d="count ≥ real PF", dc="warn"),
            dict(k="Live p-value", v=f"{p:.3f}", cls="cyan",
                 d=f"{fmt_t(elapsed)} elapsed" + (f" · ≈{fmt_t(eta)} left" if eta is not None else "")),
        ]), unsafe_allow_html=True)
    with c2:
        st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
        st.button("■ STOP", on_click=request_stop, key="stop_btn", use_container_width=True)
    if np.isfinite(job.get("real_pf", float("nan"))):
        result_body(job, live=True)
    else:
        st.info(f"{job.get('phase','Working')} — the real-data baseline (full optimization) runs first, then permutations stream in.")

    if job["status"] != "running":
        if not job.get("_finalized"):
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
    st.markdown(
        '<div class="sb-logo">Permutation Tests<em>Trading Strategies</em></div>'
        f'<div class="sb-tag">BUILT BY {AUTHOR.upper()}</div>', unsafe_allow_html=True)

    st.markdown('<div class="sb-sec">01 · Data</div>', unsafe_allow_html=True)
    src_options = ["Upload files"]
    if Path("BTCUSD3600.pq").exists():
        src_options.append("Local BTCUSD3600.pq")
    src_options.append("Synthetic demo feed")
    source = st.selectbox("Source", src_options, index=0, key="w_source")

    up1 = up2 = None
    if source == "Upload files":
        up1 = st.file_uploader("Currency 1 hourly (.pq / .parquet / .csv)",
                               type=["pq", "parquet", "csv"], key="up_1")
        with st.expander("Optional · second currency"):
            up2 = st.file_uploader("Currency 2 hourly (optional for correlation)",
                                   type=["pq", "parquet", "csv"], key="up_2")

    notice = None
    try:
        btc_df, eth_df, src_label, notice = resolve_data(source, up1, up2)
        data_err = None
    except Exception as e:
        btc_df, eth_df = synth_pair()
        src_label, data_err, notice = "DEMO FEED", str(e), None
    st.session_state["btc_df"] = btc_df
    st.session_state["eth_df"] = eth_df
    if data_err:
        st.error(data_err)
    if notice:
        st.caption(notice)
    ymin = int(btc_df.index.year.min())
    ymax = int(btc_df.index.year.max()) + 1
    st.caption(f"{src_label} · {len(btc_df):,} bars · {ymin}–{ymax-1}" +
               (" · currency 2 loaded" if eth_df is not None else ""))

    st.markdown('<div class="sb-sec">02 · Pipeline</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Strategy under test", list(STRAT_LABELS.keys()),
                            format_func=lambda k: STRAT_LABELS[k], key="w_strategy")

    if strategy in ("donchian_is", "donchian_wf"):
        ca, cb = st.columns(2)
        ca.slider("Lookback min", 4, 60, 12, key="w_lbmin")
        cb.slider("Lookback max", 60, 240, 168, key="w_lbmax")
        st.slider("Grid step", 1, 12, 1, key="w_lbstep")
    if strategy in ("donchian_is", "tree"):
        sanitize_range("w_is_years", ymin, ymax)
        d0 = (max(ymin, min(2016, ymax - 1)), min(ymax, max(2020, ymin + 1)))
        st.slider("Training years (end exclusive)", ymin, ymax, d0, key="w_is_years")
    wf_block = None
    if strategy == "donchian_wf":
        sanitize_range("w_wf_years", ymin, ymax)
        st.slider("Walk-forward years", ymin, ymax, (ymin, ymax), key="w_wf_years")
        st.slider("Re-optimize every (days)", 5, 90, 30, key="w_wf_step")
        y0, y1 = st.session_state["w_wf_years"]
        step_days = int(st.session_state.get("w_wf_step", 30))
        n_win = int(((btc_df.index.year >= y0) & (btc_df.index.year < y1)).sum())
        max_train = wf_max_train_years(n_win, step_days)
        if max_train < 1:
            wf_block = (f"Walk-forward needs at least {wf_min_bars(step_days):,} hourly bars — "
                        f"this window has {n_win:,}. Widen the years or shorten the step.")
            st.error(wf_block)
        else:
            opts = list(range(1, min(4, max_train) + 1))
            if st.session_state.get("w_wf_train") not in opts:
                st.session_state.pop("w_wf_train", None)
            st.select_slider("Training window (years)", options=opts,
                             value=min(4, max_train), key="w_wf_train")
            tw, tstep, n_oos = wf_plan(n_win, st.session_state["w_wf_train"], step_days)
            st.caption(f"train {tw:,} bars → out-of-sample {n_oos:,} bars · "
                       f"{max(1, n_oos // tstep)} re-optimizations · training bars are never scored")
    if strategy == "tree":
        st.slider("min_samples_leaf", 1, 100, 5, key="w_leaf")
        st.selectbox("max_depth", ["Unlimited", 4, 6, 8, 12], format_func=str, key="w_depth")
    if strategy == "ma":
        ca, cb = st.columns(2)
        ca.slider("Fast MA", 2, 60, 10, key="w_maf")
        cb.slider("Slow MA", 10, 240, 30, key="w_mas")
        st.toggle("Allow shorts (else long/flat)", key="w_mashort")
        if int(st.session_state.get("w_maf", 10)) >= int(st.session_state.get("w_mas", 30)):
            st.caption("⚠ Fast MA must be shorter than slow MA.")

    st.markdown('<div class="sb-sec">03 · Permutation engine</div>', unsafe_allow_html=True)
    st.slider("Permutations — aim for 1000", 50, 2000, 1000, step=50, key="w_perms")
    st.toggle("Turbo engine (vectorized, identical math)", value=True, key="w_turbo")
    if strategy == "donchian_wf":
        st.caption("Walk-forward re-optimizes inside every permutation — the heaviest test. Turbo recommended.")

    st.markdown('<div class="sb-sec">04 · Execute</div>', unsafe_allow_html=True)
    active = st.session_state.get("active")
    running = bool(active and active.get("status") == "running")
    st.button("🧪 RUN PERMUTATION TEST", type="primary", on_click=start_run,
              disabled=running or bool(wf_block), use_container_width=True)
    st.button("Clear results", on_click=clear_active, disabled=running, use_container_width=True)
    if running:
        st.caption(f"● running — {active['done']:,} / {active['config']['n_perms']:,}")
    if st.session_state.get("run_error"):
        st.error(st.session_state["run_error"])

# ---------------------------------------------------------------- header
cfg_now = collect_cfg()
st.markdown(
    f"""
<div class="hero">
  <div><span class="pulse"></span><span class="tape">MONTE CARLO PERMUTATION TESTING · HOURLY OHLC</span></div>
  <h1>Permutation Tests<span class="grad">Trading Strategies</span></h1>
  <div class="sub">Optimization always induces <b>data-mining bias</b>. A good strategy exploits a <b>real market pattern</b>.
  This terminal re-runs your fully-optimized pipeline on <b>1000 shuffled universes</b> — same distribution, zero memory — and measures how often noise alone beats you.</div>
  <div class="chips">
    <span class="chip hot">FEED <b>{src_label}</b></span>
    <span class="chip">BARS <b>{len(btc_df):,}</b></span>
    <span class="chip cool">N <b>{cfg_now['n_perms']:,}</b></span>
    <span class="chip">BUILT BY <b>{AUTHOR}</b></span>
  </div>
  <div class="hairline"></div>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_strat, tab_anat, tab_method = st.tabs(
    ["PERMUTATION LAB", "STRATEGY", "PERMUTATION ANATOMY", "METHOD & GUIDE"]
)

# ---------------------------------------------------------------- LAB
with tab_lab:
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        sec("L", f"Live · {active['label']}")
        live_fragment()
    elif active is None:
        onboarding()
        history_block()
    elif active.get("status") == "error":
        sec("E", "Run failed")
        st.error(active.get("error", "Unknown error").strip().splitlines()[-1])
        with st.expander("Traceback"):
            st.code(active.get("error", ""), language="text")
        history_block()
    else:
        sec("R", f"Results · {active['label']}")
        if not active.get("_finalized"):
            finalize(active)
        st.markdown(
            f'<div class="chips"><span class="chip hot">{active["label"]}</span>'
            f'<span class="chip">engine <b>{"turbo" if active["config"]["turbo"] else "reference"}</b></span>'
            f'<span class="chip cool">{active.get("window_desc","")}</span>'
            f'<span class="chip">status <b>{active["status"].upper()}</b></span></div>',
            unsafe_allow_html=True)
        result_body(active, live=False)
        history_block()

# ---------------------------------------------------------------- STRATEGY
with tab_strat:
    sec("S", f"Strategy · {STRAT_LABELS[cfg_now['strategy']]}")
    c1, c2, c3 = st.columns([1, 1, 2.2])
    style = c1.selectbox("Price panel", ["Candles", "Line"], key="w_price_style")
    show_ov = c2.toggle("Show indicator overlay", value=True, key="w_show_ov")
    c3.markdown(
        '<div class="callout" style="margin:.2rem 0">Price, position, equity and drawdown are drawn as '
        '<b>separate panels</b> — indicators stay on price, turns stay in the position lane.</div>',
        unsafe_allow_html=True)

    det = compute_details(btc_df, cfg_now)
    if det.get("error"):
        st.warning(det["error"])
    else:
        s = det["stats"]
        st.markdown(cards_html([
            dict(k="Profit Factor", v=f"{s['pf']:.4f}" if np.isfinite(s['pf']) else "—",
                 cls="" if s['pf'] >= 1 else "red"),
            dict(k="Sharpe (hourly)", v=f"{s['sharpe']:.4f}" if np.isfinite(s['sharpe']) else "—", cls="cyan"),
            dict(k="Σ Log Return", v=f"{s['total']:+.3f}", cls="green" if s['total'] > 0 else "red"),
            dict(k="Exposure", v=f"{s['exposure']:.0%}", cls="cyan"),
            dict(k="Position changes", v=f"{s['trades']:,}", cls="cyan", d=f"{s['bars']:,} bars"),
        ]), unsafe_allow_html=True)
        if det["extra"].get("oos_bars"):
            st.caption(f"Walk-forward: {det['extra']['oos_bars']:,} out-of-sample bars only — "
                       "the training block is used for re-optimization and is never scored or plotted.")

        ov_names = list(det["overlays"].keys())
        price_title = "Price panel" + (" · " + " / ".join(ov_names) if (show_ov and ov_names) else "")
        show_fig(price_fig(det, style, show_ov, price_title), key="s_price")
        show_fig(position_fig(det["eq_x"], det["pos_ds"]), key="s_pos")

        ce, cd = st.columns([1.5, 1])
        with ce:
            show_fig(equity_fig(det["eq_x"], det["eq"], [], "Equity — cumulative log return"), key="s_eq")
        with cd:
            show_fig(dd_fig(det["eq_x"], det["dd"]), key="s_dd")

        ex = det["extra"]
        if "opt" in ex:
            lbs, curve, bL = ex["opt"]
            show_fig(optcurve_fig(lbs, curve, bL), key="s_opt")
            st.caption(f"Best lookback on real bars: **{bL}**. This peak is the claim; the Permutation Lab renders the verdict.")
        if "wf_choices" in ex:
            show_fig(choices_fig(ex["wf_choices"]), key="s_wf")
        if "importances" in ex:
            show_fig(imp_fig(ex["importances"]), key="s_imp")

# ---------------------------------------------------------------- ANATOMY
with tab_anat:
    sec("A", "What permutation preserves — and what it kills")
    ca, cb = st.columns([1, 3])
    cb.button("🔀 Draw a new permutation", on_click=new_permutation_draw)
    nlags = ca.slider("ACF lags (hours)", 48, 336, 168, key="w_lags")
    an = anatomy(btc_df, eth_df, int(st.session_state.get("anat_draw", 0)), nlags)
    st.caption(f"Sample window {an['span']} · seeds are fixed internally for reproducibility.")

    col1, col2 = st.columns([1.5, 1])
    with col1:
        show_fig(rebased_fig(an["x"], an["lcr"], an["lcp"]), key="an_price")
    with col2:
        st.markdown('<div class="panel"><h3>Marginal distribution — preserved</h3>'
                    '<p>Shuffling rearranges bars; it cannot change the histogram of returns. Moments match to sampling noise.</p></div>',
                    unsafe_allow_html=True)
        st.table(an["moments"].style.format("{:.6f}"))

    c1, c2 = st.columns(2)
    with c1:
        show_fig(acf_fig(an["lags"], an["ar"], an["ap"], an["n"], "ACF of returns — linear dependence"), key="an_acf_r")
    with c2:
        show_fig(acf_fig(an["lags"], an["aar"], an["aap"], an["n"], "ACF of |returns| — volatility clustering"), key="an_acf_a")
    st.caption("Dashed lines: ±1.96/√n white-noise bounds. Real |returns| stay correlated for days; the permuted series collapses to the bounds immediately.")

    show_fig(spec_fig(an["fxr"], an["pxr"], an["fxp"], an["pxp"]), key="an_spec")
    st.caption("Real |returns| show a slow low-frequency decay — the spectral signature of long memory / self-similarity. Permutation flattens it.")

    if an["corr"]:
        cr = an["corr"]
        show_fig(corr_fig(cr["rb"], cr["re"], cr["pb"], cr["pe"], cr["rho_r"], cr["rho_p"]), key="an_corr")
        st.caption("Joint permutation applies the same shuffle to both currencies, so contemporaneous correlation survives while each market's serial structure dies.")
    else:
        st.info("Load Currency 2 to see the joint-permutation correlation demo.")

# ---------------------------------------------------------------- METHOD & GUIDE
with tab_method:
    sec("M", "The logic of the test")
    st.markdown(
        """
<div class="panel">
<h3>Why permutation testing?</h3>
<p>Using the best version of a strategy <b style="color:var(--text)">requires optimization</b> — and optimization
<b style="color:var(--btc2)">always induces data-mining bias</b>: pick the best of 157 lookbacks and the winner's profit factor is inflated by selection alone.
A good strategy finds and exploits a <b style="color:var(--text)">real, repeatable pattern</b>. The question is whether your pipeline's result is
distinguishable from what the <i>same pipeline</i> achieves on data with no pattern at all.</p>
<p>MCPT builds that null by shuffling bars, then re-runs the <b style="color:var(--cyan)">entire procedure — optimization included —</b> on every permuted universe.
If the optimized strategy performs just as well on noise, it is likely overfitting the real data. <b style="color:var(--cyan)">Aim for 1000 permutations.</b></p>
</div>
""", unsafe_allow_html=True)
    st.latex(r"p \;=\; \frac{1 \;+\; \#\{\,PF^{\,perm} \;\ge\; PF^{\,real}\,\}}{N_{perm} \;+\; 1}")
    st.caption("The real run counts as one observation (hence the +1). p ≤ 0.05 → evidence of a real pattern; p large → the optimizer is mining noise.")

    st.markdown(
        """
<div class="pd-grid">
  <div class="pd keep"><h4>✓ PRESERVED by bar permutation</h4><ul>
    <li><b>Marginal distribution</b> of returns — mean, std, skew, kurtosis</li>
    <li><b>Intrabar geometry</b> — high/low/close relative to open, shuffled jointly</li>
    <li><b>Gap distribution</b> — close-to-open jumps, shuffled separately</li>
    <li><b>Cross-market correlation</b> — when currencies are permuted jointly</li>
    <li><b>Real history before start_index</b> — walk-forward permutes only the OOS region</li>
  </ul></div>
  <div class="pd kill"><h4>✕ DESTROYED by bar permutation</h4><ul>
    <li><b>Volatility clustering</b> — serial correlation of absolute returns</li>
    <li><b>Long memory</b> — slow decay of dependency over longer lags</li>
    <li><b>Self-similarity</b> and fractal scaling across horizons</li>
    <li><b>Frequency-domain structure</b> — the spectrum of |returns| flattens</li>
    <li><b>Anything involving dependence</b> — trends, mean-reversion, calendar effects</li>
  </ul></div>
</div>
<div class="callout"><b>Know your null.</b> Your strategy may interact with these properties even if you are unaware of it.
A volatility-harvesting rule tested against shuffled bars is tested against a world where volatility clustering does not exist —
significance then means the edge requires <i>serial structure</i>, not just fat tails.</div>
""", unsafe_allow_html=True)

    sec("P", "Every pipeline — how it works, strengths, weaknesses")
    st.markdown(
        """
<div class="strat-block">
  <div class="hd"><span class="nm">1 · Donchian Breakout — In-Sample (optimized)</span><span class="tagline">TREND / BREAKOUT · 1 PARAMETER SWEPT</span></div>
  <ol>
    <li>Build the channel from the <b>prior L−1 closes</b>: upper = rolling max, lower = rolling min, both <b>shifted one bar</b> so the current close never enters its own channel.</li>
    <li>Close above upper → position <b>+1</b>. Close below lower → position <b>−1</b>. Otherwise <b>hold</b> the previous position (forward-fill) until the opposite break.</li>
    <li>Earn the next bar's log return: <code>signal × (log close[t+1] − log close[t])</code>.</li>
    <li>Sweep every lookback in the grid (default 12…168) and keep the <b>highest profit factor</b> on that same data. That selection step is the bias MCPT must price.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><h4>Strengths</h4><ul>
      <li>Captures the fat-tailed trending regimes crypto actually produces</li>
      <li>One transparent parameter; no fitting of thresholds or weights</li>
      <li>Always in the market, so it cannot miss a violent breakout</li>
      <li>Channel is causal — no look-ahead in the signal construction</li>
    </ul></div>
    <div class="pd kill"><h4>Weaknesses</h4><ul>
      <li><b>Maximum selection bias</b>: the best of ~157 tries is an upper bound, not an estimate</li>
      <li>PF-vs-lookback surface is spiky; the winner is often unstable</li>
      <li>Whipsaws badly in ranges; no stop, no sizing, no volatility scaling</li>
      <li>Gross of fees, slippage and funding — turnover is high</li>
    </ul></div>
  </div>
  <div class="ask">MCPT asks: does the <i>whole sweep</i> find lookbacks this good on shuffled bars? The noise envelope under the amber curve is the answer.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">2 · Donchian Breakout — Walk-Forward</span><span class="tagline">SAME RULE · RE-OPTIMIZED OUT OF SAMPLE</span></div>
  <ol>
    <li>Reserve a training window (default <b>4 years = 35,040 hourly bars</b>, auto-clamped to what the selected years can hold). Nothing before it is traded.</li>
    <li>At each step, run the full lookback sweep <b>only on the trailing training window</b>, then apply that lookback to the next <b>30 days</b> of unseen bars.</li>
    <li>Roll forward by the step and repeat; stitch the out-of-sample segments into one equity curve. <b>Only OOS bars are scored, counted and plotted.</b></li>
    <li>For MCPT, permutation starts at <code>start_index = train_window</code> — the first training block stays real, and the <b>re-optimization runs inside every permutation</b>.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><h4>Strengths</h4><ul>
      <li>Honest out-of-sample estimate: parameters never see the bars they trade</li>
      <li>Adapts as regimes change instead of freezing one lucky lookback</li>
      <li>Tests the <b>procedure</b>, which is what you would actually deploy</li>
      <li>Selection bias is reduced and, crucially, still measured</li>
    </ul></div>
    <div class="pd kill"><h4>Weaknesses</h4><ul>
      <li>Bias is reduced, not removed — each window still picks a winner</li>
      <li>Parameter churn adds turnover and instability</li>
      <li>Far more compute: one full sweep per step, per permutation</li>
      <li>Shorter traded sample → wider uncertainty; a dead training regime misleads the next step</li>
    </ul></div>
  </div>
  <div class="ask">MCPT asks: does the adaptive procedure beat noise when the noise is re-optimized the same way? This is the strongest of the four tests.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">3 · Decision Tree Classifier</span><span class="tagline">NON-LINEAR MOMENTUM · SUPERVISED</span></div>
  <ol>
    <li>Features per bar: <b>6h, 24h and 168h log-return differences</b> — short, daily and weekly momentum.</li>
    <li>Label: sign of the <b>next 24 hours</b>, mapped from {−1, +1} to {0, 1}. Rows with incomplete history or future are dropped.</li>
    <li>Fit <code>DecisionTreeClassifier(min_samples_leaf=5)</code>, unlimited depth by default — it can carve the feature space until leaves are nearly pure.</li>
    <li>Predict 1 → long, 0 → short; always in the market; scored on the next <b>1-hour</b> return.</li>
    <li>MCPT <b>retrains a fresh tree on every permuted universe</b>, so the model's capacity to memorize is priced in.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><h4>Strengths</h4><ul>
      <li>Learns interactions and thresholds a linear rule cannot express</li>
      <li>Feature importances make the learned logic inspectable</li>
      <li>No distributional assumptions; handles skewed, fat-tailed inputs</li>
      <li>Depth and leaf-size caps give a direct overfitting dial</li>
    </ul></div>
    <div class="pd kill"><h4>Weaknesses</h4><ul>
      <li><b>Enormous overfit capacity</b> — an unlimited-depth tree memorizes noise</li>
      <li>24h labels sampled hourly <b>overlap</b>, inflating apparent skill</li>
      <li>Label horizon (24h) ≠ scoring horizon (1h) — a structural mismatch</li>
      <li>Very unstable: small data changes reshape the whole tree</li>
      <li>Drift makes classes imbalanced; the tree can learn "always up"</li>
    </ul></div>
  </div>
  <div class="ask">MCPT asks: can the tree find equally convincing structure in bars with no memory at all? Usually it can — which is exactly the lesson.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">4 · Moving-Average Crossover</span><span class="tagline">TREND FILTER · LONG/FLAT (OPTIONAL SHORT)</span></div>
  <ol>
    <li>Compute a fast SMA (default 10) and a slow SMA (default 30) of close.</li>
    <li>Fast above slow → <b>long (+1)</b>; otherwise <b>flat (0)</b>, or <b>short (−1)</b> if shorts are enabled.</li>
    <li>No optimization is performed — the windows are fixed, so MCPT tests the <b>raw rule</b> rather than a selected winner.</li>
    <li>Position changes only at crossovers, so turnover is low compared with the breakout rule.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><h4>Strengths</h4><ul>
      <li>Simplest trend filter; few degrees of freedom → little mining bias</li>
      <li>Low turnover makes it far more robust to fees and slippage</li>
      <li>Long/flat sidesteps shorting cost and borrow risk</li>
      <li>Behaviour is easy to reason about and to explain to a risk committee</li>
    </ul></div>
    <div class="pd kill"><h4>Weaknesses</h4><ul>
      <li>Lags every turn — gives back the first leg of each reversal</li>
      <li>Whipsaws in chop; flat periods earn nothing</li>
      <li>Long/flat is implicitly a bet on positive drift, not pure timing skill</li>
      <li>10/30 is still a choice; results are sensitive to that pair</li>
    </ul></div>
  </div>
  <div class="ask">MCPT asks: does this fixed rule beat shuffled bars? With no optimizer in the loop, a significant p-value here is the cleanest evidence of a real trend pattern.</div>
</div>
""", unsafe_allow_html=True)

    sec("G", "Reading the lab")
    st.markdown(
        """
<div class="panel">
<h3>Workflow</h3>
<p><b style="color:var(--text)">1.</b> Upload Currency 1 (Currency 2 only if you want the correlation demo). <b style="color:var(--text)">2.</b> Choose a pipeline and its parameters.
<b style="color:var(--text)">3.</b> Set permutations — 1000 is the target; 200 is a reasonable first pass for walk-forward. <b style="color:var(--text)">4.</b> Run and watch the histogram build.</p>
<p><b style="color:var(--btc2)">Histogram</b> — the distribution of the <i>best</i> profit factor the pipeline achieves on noise. The amber line is your real result; the red tail is everything that beat it.
<b style="color:var(--btc2)">Equity</b> — your real curve (thick amber, always on top) against the permuted bundle, its median and its 10–90% band; if your curve sits inside the band, it is indistinguishable from noise. Walk-forward plots out-of-sample bars only.
<b style="color:var(--btc2)">Optimization surface</b> — the real PF curve against the noise mean ±1σ envelope; the vertical gap is the evidence, not the peak height.</p>
<p><b style="color:var(--green)">p ≤ 0.05</b> edge detected · <b style="color:var(--btc2)">0.05 &lt; p ≤ 0.20</b> inconclusive · <b style="color:var(--red)">p &gt; 0.20</b> consistent with noise.
A permutation p-value can never be smaller than 1/(N+1) — with 1000 permutations the floor is ≈0.001.</p>
</div>
<div class="callout"><b>Limitations to keep in view.</b> Every result is gross of fees, slippage and funding. Profit factor is scale-free but ignores drawdown and path risk.
MCPT prices the selection procedure you actually ran — if you also tried other pipelines and only tested the winner, the honest N is larger than what the app reports.
Seeds are fixed internally so runs are reproducible; the Turbo engine is a vectorized implementation of the same mathematics as the reference loops.</div>
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
        d2.download_button("⬇ Full project (.zip)", project_zip(),
                           file_name="permutation_tests_trading_strategies.zip",
                           mime="application/zip", key="dl_zip")
    else:
        st.warning(f"{pick} not found on disk.")
    st.markdown(
        '<div class="callout"><b>Run it:</b> <code>pip install -r requirements.txt</code> → <code>streamlit run app.py</code>. '
        'The standalone scripts run the classic tqdm + matplotlib workflow against a local parquet file, falling back to the synthetic feed when it is absent.</div>',
        unsafe_allow_html=True)

st.markdown(
    f'<div class="foot">PERMUTATION TESTS · TRADING STRATEGIES · BUILT BY <b>{AUTHOR.upper()}</b> · '
    'DISTRIBUTION SURVIVES THE SHUFFLE — DEPENDENCE DOES NOT</div>',
    unsafe_allow_html=True,
)
