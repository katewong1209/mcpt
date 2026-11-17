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
from mcpt_engine import (
    STRAT_LABELS,
    TIMEFRAMES,
    TF_LABEL,
    TF_UNIT,
    bars_per_year,
    clamp_grid,
    resample_ohlc,
    run_mcpt,
    tf_defaults,
    wf_plan,
    wf_step_max,
    wf_train_max,
)
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
REF_SOURCE = "Reference BTC/ETH pair (seeded)"
REF_SEED = 7            # seed behind the downloadable reference CSVs

C = dict(
    ink="#05070d", panel="#0b101c", panel2="#101728",
    btc="#f7931a", btc2="#ffb347", cyan="#22d3ee", magenta="#ff2ea6",
    green="#2ce5a7", red="#ff4d5e", violet="#8b7bff",
    text="#e8ecf6", muted="#8b94ab",
)
PLOTLY_CFG = {"displayModeBar": False, "responsive": True}

CSS = r"""
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
/* sidebar collapse / expand: literal glyphs, never CSS unicode escapes */
[data-testid="stSidebarCollapseButton"] button span,
[data-testid="stSidebarCollapsedControl"] button span,
[data-testid="stExpandSidebarButton"] span{font-size:0 !important;line-height:0 !important;}
[data-testid="stSidebarCollapseButton"] button::after{content:"‹‹";}
[data-testid="stSidebarCollapsedControl"] button::after{content:"››";}
[data-testid="stExpandSidebarButton"]::after{content:"››";}
[data-testid="stSidebarCollapseButton"] button::after,
[data-testid="stSidebarCollapsedControl"] button::after,
[data-testid="stExpandSidebarButton"]::after{
  font-family:'IBM Plex Mono',monospace;font-size:1rem;color:var(--muted);
  letter-spacing:-1px;line-height:1;transition:color .15s ease;
}
[data-testid="stSidebarCollapseButton"] button:hover::after,
[data-testid="stSidebarCollapsedControl"] button:hover::after,
[data-testid="stExpandSidebarButton"]:hover::after{color:var(--btc2);}

/* no heading anchors: the URL never picks up a #fragment, everything stays in-app */
[data-testid="stHeaderActionElements"],
.stMarkdown a.anchor-link,
.stMarkdown h1 > a, .stMarkdown h2 > a, .stMarkdown h3 > a,
.stMarkdown h4 > a, .stMarkdown h5 > a, .stMarkdown h6 > a{display:none !important;}

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
.panel h3,.panel .h3{margin:.1rem 0 .45rem;font-size:1.1rem;font-weight:600;color:var(--text);}
.panel h4,.panel .h4{margin:.7rem 0 .3rem;font-size:.95rem;font-weight:600;color:var(--btc2);}
.panel p{color:var(--muted);font-size:.87rem;line-height:1.65;margin:.2rem 0;}
.panel ul,.panel ol{color:var(--muted);font-size:.85rem;line-height:1.7;margin:.25rem 0 .2rem;padding-left:1.15rem;}
.panel li b{color:var(--text);font-weight:600;}
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
.pd h4,.pd .h4{margin:0 0 .45rem;font-size:.92rem;font-weight:600;}
.pd.keep h4,.pd.keep .h4{color:var(--green);} .pd.kill h4,.pd.kill .h4{color:var(--red);}
.pd ul{margin:0;padding-left:1.05rem;color:var(--muted);font-size:.82rem;line-height:1.6;}
.pd li b{color:var(--text);font-weight:600;}
.callout{border-left:3px solid var(--btc);background:rgba(247,147,26,.07);
  border-radius:0 12px 12px 0;padding:.75rem .95rem;color:var(--muted);font-size:.85rem;margin:.75rem 0;line-height:1.6;}
.callout b{color:var(--btc2);}
.callout.cool{border-left-color:var(--cyan);background:rgba(34,211,238,.06);}
.callout.cool b{color:var(--cyan);}
.strat-block{border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;margin:.7rem 0;
  background:linear-gradient(155deg,#0d1424,#0a0f1a);}
.strat-block .hd{display:flex;align-items:baseline;gap:.6rem;flex-wrap:wrap;margin-bottom:.45rem;}
.strat-block .nm{font-weight:700;font-size:1.05rem;color:var(--text);}
.strat-block .tagline{font-family:'IBM Plex Mono',monospace;font-size:.66rem;color:var(--cyan);letter-spacing:.08em;}
.strat-block ol{margin:.25rem 0 .7rem;padding-left:1.15rem;color:var(--muted);font-size:.84rem;line-height:1.65;}
.strat-block ol b{color:var(--text);font-weight:600;}
.strat-block .ask{font-size:.8rem;color:var(--btc2);border-top:1px dashed rgba(247,147,26,.25);padding-top:.55rem;margin-top:.6rem;}
.note-sec{border:1px solid var(--line);border-radius:16px;padding:1rem 1.15rem;margin:.7rem 0;
  background:linear-gradient(155deg,#0d1424,#0a0f1a);}
.note-sec h4,.note-sec .h4{margin:.1rem 0 .5rem;font-size:1rem;font-weight:600;color:var(--text);}
.note-sec p{color:var(--muted);font-size:.86rem;line-height:1.7;margin:.35rem 0;}
.note-sec p b{color:var(--btc2);font-weight:600;}
.note-sec .fig{font-family:'IBM Plex Mono',monospace;color:var(--text);}
.note-sec ul{margin:.3rem 0;padding-left:1.15rem;color:var(--muted);font-size:.84rem;line-height:1.65;}
.note-sec li b{color:var(--text);}
.kv{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--muted);letter-spacing:.04em;}

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
/* upload cap wording — update the text if [server] maxUploadSize is changed */
[data-testid="stFileUploaderDropzoneInstructions"] small{
  font-size:0 !important;line-height:0 !important;display:block;}
[data-testid="stFileUploaderDropzoneInstructions"] small::after{
  content:"max 200 MB per file · CSV, PQ, PARQUET";
  display:block;font-size:.72rem;line-height:1.45;color:var(--muted);
  font-family:'IBM Plex Mono',monospace;}
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


# ---------------------------------------------------------------- notices
def notify(text, hints=None, technical=None):
    """Queue one dismissible popup. Nothing is rendered inline; details go to the log."""
    sig = hash((text, tuple(hints or ())))
    seen = st.session_state.setdefault("_notices_seen", set())
    if sig in seen:
        return
    seen.add(sig)
    if technical:
        print("[MCPT]", technical)          # server-side only, never shown to the user
    st.session_state["_popup"] = {"text": text, "hints": list(hints or ())}


def _popup_body():
    p = st.session_state.get("_popup") or {}
    st.markdown(f'<div class="panel"><div class="h3">{p.get("text", "")}</div></div>',
                unsafe_allow_html=True)
    hints = p.get("hints") or []
    if hints:
        st.markdown(
            '<div class="panel"><div class="h4">What to try</div><ul>'
            + "".join(f"<li>{h}</li>" for h in hints) + "</ul></div>",
            unsafe_allow_html=True)
    if st.button("Dismiss", type="primary", use_container_width=True, key="popup_dismiss"):
        st.session_state.pop("_popup", None)
        st.rerun()


_popup_dialog = st.dialog("Notice")(_popup_body) if hasattr(st, "dialog") else None


def flush_popup():
    """Open the queued popup once, at the end of the run."""
    p = st.session_state.get("_popup")
    if not p:
        return
    if _popup_dialog is not None:
        try:
            _popup_dialog()
            return
        except Exception:
            pass
    try:
        st.toast(p["text"], icon="⚠️")
    except Exception:
        pass
    st.session_state.pop("_popup", None)


def reset_notices():
    st.session_state.pop("_notices_seen", None)
    st.session_state.pop("_popup", None)


# ---------------------------------------------------------------- helpers
def show_fig(fig, key=None):
    try:
        st.plotly_chart(fig, width="stretch", key=key, config=PLOTLY_CFG)
    except Exception:
        st.plotly_chart(fig, use_container_width=True, key=key, config=PLOTLY_CFG)


def base_fig(height=340, title=None, top=None):
    """Short title on its own row, legend on the row below it — never overlapping."""
    fig = go.Figure()
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(10,15,26,.55)",
        font=dict(family="IBM Plex Mono, monospace", size=11, color="#aab3c8"),
        margin=dict(l=54, r=16, t=top or (62 if title else 26), b=38),
        title=dict(text=title, x=0.0, xanchor="left", y=.98, yanchor="top",
                   font=dict(size=12.5, color=C["text"], family="Space Grotesk")),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["btc"], font=dict(family="IBM Plex Mono", size=11)),
    )
    fig.update_xaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.10)", zerolinecolor="rgba(139,148,171,.22)", linecolor="rgba(139,148,171,.25)")
    return fig


def finite_only(y):
    y = np.asarray(y, dtype=float)
    return np.where(np.isfinite(y), y, np.nan)


def num_slider(label, lo, hi, default, key):
    """Slider whose bounds always make sense for the data; stale values are dropped."""
    lo = int(lo)
    hi = int(max(int(hi), lo + 1))
    v = st.session_state.get(key)
    if v is not None and not (lo <= int(v) <= hi):
        st.session_state.pop(key, None)
    return int(st.slider(label, lo, hi, int(min(max(int(default), lo), hi)), key=key))


def year_bounds(df):
    """(first selectable year, last selectable year) — INCLUSIVE on both ends.
    The last year is the later of the feed's final year and the current calendar
    year, so the slider always stops on the current year and never needs editing."""
    now = int(pd.Timestamp.now(tz="UTC").year)
    if df is None or len(df) == 0:
        return now - 1, now
    ymin = int(df.index.year.min())
    last = int(df.index.year.max())
    return ymin, int(max(last, now, ymin))


def year_defaults(df):
    """(in-sample default, walk-forward default) — both span first year → last
    selectable year, i.e. through the current year."""
    lo, hi = year_bounds(df)
    return (lo, hi), (lo, hi)


def clamp_years(v, lo, hi, fallback):
    """Keep a stored year range inside the current inclusive bounds (start <= end)."""
    try:
        a, b = int(v[0]), int(v[1])
    except Exception:
        return fallback
    a = max(lo, min(a, hi))
    b = max(a, min(b, hi))
    return a, b


def sanitize_years(key, lo, hi):
    v = st.session_state.get(key)
    if isinstance(v, (tuple, list)) and len(v) == 2:
        a, b = int(v[0]), int(v[1])
        if not (lo <= a <= b <= hi):
            st.session_state.pop(key, None)
    elif v is not None:
        st.session_state.pop(key, None)


def year_range(label, lo, hi, default, key):
    """Inclusive year-range slider; degrades to a caption when one year exists."""
    lo, hi = int(lo), int(hi)
    if hi <= lo:
        st.caption(f"{label}: {lo}")
        return lo, lo
    sanitize_years(key, lo, hi)
    a, b = st.slider(label, lo, hi, (int(default[0]), int(default[1])), key=key)
    return int(a), int(b)


def fmt_n(x, spec="{:.3f}", dash="—"):
    try:
        return spec.format(x) if np.isfinite(x) else dash
    except Exception:
        return dash


def hist_fig(perm_pfs, real_pf, p, live=False):
    vals = np.asarray([v for v in perm_pfs if np.isfinite(v)], dtype=float)
    fig = base_fig(360, "PF · real vs noise" + (" · live" if live else ""))
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
                         marker_color=C["cyan"], opacity=.78, name="Noise",
                         hovertemplate="PF %{x:.3f}<br>count %{y}<extra></extra>"))
    if rf is not None:
        fig.add_vrect(x0=rf, x1=hi + pad, fillcolor=C["red"], opacity=.08, line_width=0)
        fig.add_vline(x=rf, line_color=C["btc"], line_width=2.5, line_dash="dash",
                      annotation_text=f"real {rf:.3f}", annotation_position="top",
                      annotation_font=dict(color=C["btc2"], size=10))
    fig.add_vline(x=float(np.median(vals)), line_color=C["violet"], line_width=1.3, line_dash="dot")
    vcol = C["green"] if p <= .05 else (C["btc2"] if p <= .2 else C["red"])
    fig.add_annotation(xref="paper", yref="paper", x=.99, y=.97, xanchor="right",
                       text=f"<b>p = {p:.3f}</b>", showarrow=False,
                       font=dict(color=vcol, size=16, family="Space Grotesk"))
    fig.update_xaxes(title="profit factor")
    fig.update_yaxes(title="permutations")
    return fig


def equity_fig(x, real_eq, samples, title="Equity"):
    fig = base_fig(360, title)
    x = np.asarray(x)
    real = finite_only(real_eq)
    curves = [finite_only(s) for s in samples if len(s)]

    if curves:
        m = min(len(c) for c in curves)
        mat = np.vstack([c[:m] for c in curves])
        xx = x[:m]
        for cy in mat:
            fig.add_trace(go.Scatter(x=xx, y=cy, mode="lines",
                                     line=dict(color=C["cyan"], width=1), opacity=.40,
                                     showlegend=False, hoverinfo="skip"))
        lo10, hi10 = np.nanpercentile(mat, 10, axis=0), np.nanpercentile(mat, 90, axis=0)
        fig.add_trace(go.Scatter(x=np.concatenate([xx, xx[::-1]]),
                                 y=np.concatenate([hi10, lo10[::-1]]),
                                 fill="toself", fillcolor="rgba(34,211,238,.16)", line=dict(width=0),
                                 hoverinfo="skip", name=f"Noise 10–90% ({len(mat)})"))
        fig.add_trace(go.Scatter(x=xx, y=np.nanmedian(mat, axis=0), mode="lines",
                                 line=dict(color=C["cyan"], width=1.5, dash="dot"),
                                 showlegend=False, hoverinfo="skip"))

    # real path last => always on top of the noise bundle (halo + crisp line)
    fig.add_trace(go.Scatter(x=x[:len(real)], y=real, mode="lines",
                             line=dict(color=C["btc"], width=7), opacity=.18,
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=x[:len(real)], y=real, mode="lines",
                             line=dict(color=C["btc"], width=2.6), name="Real",
                             hovertemplate="%{y:+.3f}<extra>real</extra>"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    fig.update_yaxes(title="Σ log return")
    return fig


def rebased_fig(x, lcr, lcp, title="Real vs permuted · log price", names=("Real", "Permuted")):
    fig = base_fig(360, title)
    x = np.asarray(x); yr = finite_only(lcr); yp = finite_only(lcp)
    n = min(len(x), len(yr), len(yp)); x, yr, yp = x[:n], yr[:n], yp[:n]
    fig.add_trace(go.Scatter(x=x, y=yr, mode="lines", name=names[0],
                             line=dict(color=C["btc"], width=2.6),
                             hovertemplate="%{y:+.3f}<extra>" + names[0] + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=yp, mode="lines", name=names[1],
                             line=dict(color=C["cyan"], width=1.7),
                             fill="tonexty", fillcolor="rgba(139,123,255,.14)",
                             hovertemplate="%{y:+.3f}<extra>" + names[1] + "</extra>"))
    fig.add_trace(go.Scatter(x=x, y=yr, mode="lines", line=dict(color=C["btc"], width=2.6),
                             showlegend=False, hoverinfo="skip"))
    fig.add_hline(y=0, line_color="rgba(139,148,171,.45)", line_width=1)
    fig.update_yaxes(title="log price · rebased to 0")
    return fig


def optcurve_fig(lookbacks, curve, best_L, mean=None, std=None):
    fig = base_fig(330, "PF vs lookback")
    if mean is not None and std is not None:
        m = np.isfinite(mean) & np.isfinite(std)
        xl = np.asarray(lookbacks)[m]
        if len(xl):
            fig.add_trace(go.Scatter(x=np.concatenate([xl, xl[::-1]]),
                                     y=np.concatenate([(mean + std)[m], (mean - std)[m][::-1]]),
                                     fill="toself", fillcolor="rgba(139,123,255,.16)",
                                     line=dict(width=0), name="Noise ±1σ", hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=xl, y=mean[m], mode="lines",
                                     line=dict(color=C["cyan"], width=1.5, dash="dot"),
                                     showlegend=False, hoverinfo="skip"))
    yc = finite_only(curve)
    fig.add_trace(go.Scatter(x=lookbacks, y=yc, mode="lines",
                             line=dict(color=C["btc"], width=2.2), name="Real"))
    if best_L and best_L > 0:
        k = int(np.argmin(np.abs(np.asarray(lookbacks) - best_L)))
        if np.isfinite(yc[k]):
            fig.add_trace(go.Scatter(x=[lookbacks[k]], y=[yc[k]], mode="markers",
                                     marker=dict(symbol="star", size=12, color=C["magenta"],
                                                 line=dict(color=C["text"], width=1)),
                                     name=f"L={best_L}", hoverinfo="skip"))
    fig.add_hline(y=1.0, line_color="rgba(139,148,171,.4)", line_dash="dash", line_width=1)
    fig.update_xaxes(title="lookback (bars)")
    fig.update_yaxes(title="profit factor")
    return fig


def price_fig(det, style, show_overlays, title="Price"):
    fig = base_fig(410, title, top=76)
    if style == "Candles":
        d = det["daily"]
        fig.add_trace(go.Candlestick(
            x=d.index, open=d["open"], high=d["high"], low=d["low"], close=d["close"],
            increasing_line_color=C["green"], increasing_fillcolor="rgba(44,229,167,.55)",
            decreasing_line_color=C["red"], decreasing_fillcolor="rgba(255,77,94,.55)",
            name="Price", whiskerwidth=.4))
    else:
        fig.add_trace(go.Scatter(x=det["eq_x"], y=det["close_ds"], mode="lines",
                                 line=dict(color=C["btc"], width=1.8), name="Close"))
    if show_overlays:
        for name, (series, color, dash) in det["overlays"].items():
            fig.add_trace(go.Scatter(x=series.index, y=finite_only(series.to_numpy()),
                                     mode="lines", line=dict(color=color, width=1.4, dash=dash),
                                     name=name))
    fig.update_layout(xaxis_rangeslider_visible=False, hovermode="x")
    fig.update_yaxes(title="price")
    return fig


def position_fig(x, pos):
    fig = base_fig(150, "Position", top=34)
    fig.add_trace(go.Scatter(x=x, y=pos, mode="lines",
                             line=dict(color=C["cyan"], width=1.4, shape="hv"),
                             fill="tozeroy", fillcolor="rgba(34,211,238,.18)", name="Position"))
    fig.update_yaxes(range=[-1.3, 1.3], tickvals=[-1, 0, 1],
                     ticktext=["−1", "0", "+1"], title=None)
    fig.update_layout(margin=dict(l=54, r=16, t=34, b=26), showlegend=False)
    return fig


def dd_fig(x, dd):
    fig = base_fig(230, "Drawdown")
    fig.add_trace(go.Scatter(x=x, y=dd, mode="lines", fill="tozeroy",
                             line=dict(color=C["red"], width=1.2),
                             fillcolor="rgba(255,77,94,.25)", name="Drawdown"))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="Σ log return")
    return fig


def choices_fig(choices, title="Lookback per step"):
    fig = base_fig(230, title)
    fig.add_trace(go.Scatter(x=[t for t, _ in choices], y=[b for _, b in choices],
                             mode="lines", line=dict(color=C["magenta"], width=2, shape="hv"),
                             name="Chosen L"))
    fig.update_layout(showlegend=False)
    fig.update_yaxes(title="lookback (bars)")
    return fig


def acf_fig(lags, real, perm, n, title, unit="bar"):
    fig = base_fig(280, title)
    fig.add_trace(go.Bar(x=lags, y=real, marker_color=C["btc"], opacity=.85, width=.42, name="Real"))
    fig.add_trace(go.Bar(x=lags, y=perm, marker_color=C["cyan"], opacity=.6, width=.42, name="Permuted"))
    b = 1.96 / np.sqrt(max(n, 2))
    fig.add_hline(y=b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.add_hline(y=-b, line_color="rgba(139,148,171,.5)", line_dash="dash", line_width=1)
    fig.update_layout(barmode="group", bargap=.15)
    fig.update_xaxes(title=f"lag ({unit} bars)")
    return fig


def spec_fig(fx, px, fy, py):
    fig = base_fig(300, "Spectrum · |returns|")
    fig.add_trace(go.Scatter(x=fx, y=px, mode="lines", line=dict(color=C["btc"], width=2), name="Real"))
    fig.add_trace(go.Scatter(x=fy, y=py, mode="lines", line=dict(color=C["cyan"], width=1.6, dash="dot"), name="Permuted"))
    fig.update_xaxes(type="log", title="frequency (cycles / bar)")
    fig.update_yaxes(type="log", title="power")
    return fig


def corr_fig(rb, re, pb, pe, rho_r, rho_p):
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"real ρ = {rho_r:.3f}", f"permuted ρ = {rho_p:.3f}"))
    fig.add_trace(go.Scattergl(x=rb, y=re, mode="markers", marker=dict(size=3, color=C["btc"], opacity=.45), name="Real"), 1, 1)
    fig.add_trace(go.Scattergl(x=pb, y=pe, mode="markers", marker=dict(size=3, color=C["cyan"], opacity=.45), name="Permuted"), 1, 2)
    fig.update_layout(height=320, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(10,15,26,.55)",
                      font=dict(family="IBM Plex Mono", size=11, color="#aab3c8"),
                      margin=dict(l=54, r=16, t=68, b=40), showlegend=False,
                      title=dict(text="Cross-market correlation", x=0.0, xanchor="left", y=.98,
                                 yanchor="top", font=dict(size=12.5, color=C["text"], family="Space Grotesk")))
    fig.update_xaxes(gridcolor="rgba(139,148,171,.1)", title="currency 1 log return")
    fig.update_yaxes(gridcolor="rgba(139,148,171,.1)", title="currency 2 log return")
    for a in fig.layout.annotations:
        a.font = dict(family="IBM Plex Mono", size=10, color=C["muted"])
    return fig


def imp_fig(importances, spans=(6, 24, 168), unit="bar"):
    names = [f"{int(s)}-{unit} momentum" for s in spans]
    fig = base_fig(210, "Feature importances")
    fig.add_trace(go.Bar(y=names, x=importances, orientation="h",
                         marker_color=[C["btc"], C["cyan"], C["magenta"]], opacity=.85,
                         text=[f"{v:.1%}" for v in importances], textposition="outside",
                         textfont=dict(color=C["text"])))
    fig.update_layout(showlegend=False, margin=dict(l=110, r=16, t=44, b=34))
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
    if len(x) < 16:
        return np.array([1.0]), np.array([1.0])
    nbins = int(max(6, min(nbins, len(x) // 8)))
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


# ---------------------------------------------------------------- data
@st.cache_data(show_spinner="Synthesizing demo feed…")
def synth_pair(seed=REF_SEED):
    return make_synthetic_pair(seed)


@st.cache_data(show_spinner=False)
def parse_bytes(name, data):
    bio = io.BytesIO(data)
    if name.lower().endswith(".csv"):
        # round_trip keeps float64 exact, so an exported reference CSV reloads identically
        df = pd.read_csv(bio, float_precision="round_trip")
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
    if len(df) < 8:
        raise ValueError(f"{name}: needs at least 8 rows")
    return df


@st.cache_data(show_spinner=False)
def load_path(path, mtime):
    return parse_bytes(path, Path(path).read_bytes())


@st.cache_data(show_spinner=False)
def to_tf(df, tf):
    return resample_ohlc(df, tf)


@st.cache_data(show_spinner="Preparing reference CSV…")
def reference_csv(symbol):
    """The seeded reference feed as a CSV byte string — identical to the default run."""
    b, e = synth_pair()
    df = (b if symbol == "BTC" else e).copy()
    df.index.name = "timestamp"
    return df.to_csv().encode("utf-8")


@st.cache_data(show_spinner="Packaging reference pair…")
def reference_zip():
    readme = (
        "Permutation Tests · Trading Strategies — reference pair\r\n"
        "=======================================================\r\n"
        f"Provenance : seeded synthetic feed, make_synthetic_pair(seed={REF_SEED}).\r\n"
        "             GARCH(1,1) volatility, t(4) innovations, AR(1) trend regimes,\r\n"
        "             rare jumps, correlated cross-market shocks, realistic wicks.\r\n"
        "             THIS IS NOT EXCHANGE HISTORY. Do not cite it as BTC/ETH prices.\r\n"
        "Coverage   : 2016-01-01 00:00 to 2020-12-31 23:00, hourly, 43,848 bars each.\r\n"
        "Schema     : timestamp,open,high,low,close (UTC-naive, ascending, no duplicates).\r\n"
        "Usage      : sidebar > Source > Upload files > Currency 1 = BTC, Currency 2 = ETH.\r\n"
        "             Results reproduce the app's default seeded run exactly.\r\n"
        "Real data  : see Method & Guide > Data for public sources and the required schema.\r\n"
        f"Built by   : {AUTHOR}\r\n"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("BTCUSD_reference_hourly.csv", reference_csv("BTC"))
        z.writestr("ETHUSD_reference_hourly.csv", reference_csv("ETH"))
        z.writestr("README.txt", readme)
    return buf.getvalue()


def use_reference_pair():
    reset_notices()
    st.session_state["w_source"] = REF_SOURCE


def resolve_data(source, up1, up2):
    """Returns (currency1, currency2 or None, label, notice) at native resolution."""
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
    return b, e, "REFERENCE PAIR", None


# ---------------------------------------------------------------- config & callbacks
def collect_cfg():
    g = st.session_state.get
    tf = g("w_tf", "1h")
    d = tf_defaults(tf)
    depth = g("w_depth", "Unlimited")
    lbmin, lbmax = int(g("w_lbmin", d["lb_min"])), int(g("w_lbmax", d["lb_max"]))
    if lbmin >= lbmax:
        lbmin, lbmax = min(lbmin, lbmax - 1), max(lbmax, lbmin + 1)
    lbmin = max(2, lbmin)
    _btc = st.session_state.get("btc_df")
    ylo, yhi = year_bounds(_btc)
    is_def, wf_def = year_defaults(_btc)
    is_years = clamp_years(g("w_is_years", is_def), ylo, yhi, is_def)
    wf_years = clamp_years(g("w_wf_years", wf_def), ylo, yhi, wf_def)
    fast, slow = int(g("w_maf", d["ma_fast"])), int(g("w_mas", d["ma_slow"]))
    if fast >= slow:
        fast, slow = max(1, slow - 1), slow
    return dict(
        strategy=g("w_strategy", "donchian_is"),
        tf=tf,
        n_perms=int(g("w_perms", 1000)),
        seed=BASE_SEED,
        turbo=bool(g("w_turbo", True)),
        lb_min=lbmin, lb_max=lbmax, lb_step=int(g("w_lbstep", d["lb_step"])),
        is_start=int(is_years[0]), is_end=int(is_years[1]),
        wf_start=int(wf_years[0]), wf_end=int(wf_years[1]),
        wf_train_bars=int(g("w_wf_train", d["train"])),
        wf_step_bars=int(g("w_wf_step", d["step"])),
        tree_leaf=int(g("w_leaf", 5)),
        tree_depth=None if depth == "Unlimited" else int(depth),
        tree_spans=tuple(d["spans"]), tree_horizon=int(d["horizon"]),
        ma_fast=fast, ma_slow=slow,
        ma_short=bool(g("w_mashort", False)),
    )


def start_run():
    active = st.session_state.get("active")
    if active and active.get("status") == "running":
        return
    reset_notices()
    btc = st.session_state.get("btc_df")
    if btc is None or len(btc) < 4:
        notify("There are not enough bars at this bar size to run a test.",
               ["Choose a finer bar size — Hourly or Daily.",
                "Load a file covering a longer period."])
        return
    if active and "stop" in active:
        active["stop"].set()
    cfg = collect_cfg()
    if cfg["strategy"] == "donchian_wf":
        y = btc.index.year
        n_win = int(((y >= cfg["wf_start"]) & (y <= cfg["wf_end"])).sum())
        try:
            wf_plan(n_win, cfg["wf_train_bars"], cfg["wf_step_bars"])
        except ValueError as ex:
            notify(str(ex), ["Widen the walk-forward years.",
                             "Shorten the training window or the re-optimization step."],
                   technical=str(ex))
            return
    job = dict(
        id=uuid.uuid4().hex[:10], status="running", phase="Initializing…",
        config=cfg, start=time.time(), done=0, better=1, p=1.0,
        perm_pfs=[], sample_curves=[], real_pf=float("nan"),
        best_lookback=None, lookbacks=None, pf_curve=None,
        env_cnt=None, env_sum=None, env_ssq=None,
        real_curve=None, curve_x=None, importances=None, wf_choices=None,
        window_desc="", label=STRAT_LABELS[cfg["strategy"]], tf=cfg["tf"],
        stop=threading.Event(), elapsed=0.0, error=None, note=None, _finalized=False,
    )
    th = threading.Thread(target=run_mcpt, args=(job, btc.copy(), cfg), daemon=True)
    job["thread"] = th
    st.session_state["active"] = job
    th.start()


def request_stop():
    a = st.session_state.get("active")
    if a and "stop" in a:
        a["stop"].set()


def clear_active():
    a = st.session_state.get("active")
    if a and a.get("status") == "running":
        return
    reset_notices()
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
        tf = cfg.get("tf", "1h")
        d = tf_defaults(tf)
        spans = tuple(cfg.get("tree_spans", d["spans"]))
        horizon = int(cfg.get("tree_horizon", d["horizon"]))
        if strat in ("donchian_is", "tree"):
            win = btc[(btc.index.year >= cfg["is_start"]) & (btc.index.year <= cfg["is_end"])]
        elif strat == "donchian_wf":
            win = btc[(btc.index.year >= cfg["wf_start"]) & (btc.index.year <= cfg["wf_end"])]
        else:
            win = btc
        if len(win) < 4:
            return {"error": f"Only {len(win)} bar(s) in that range.",
                    "hints": ["Widen the year range.", "Pick a finer bar size."]}

        lb = clamp_grid(cfg["lb_min"], cfg["lb_max"], cfg["lb_step"], len(win))
        c = win["close"].to_numpy()
        r = np.empty(len(c)); r[:-1] = np.diff(np.log(c)); r[-1] = np.nan
        overlays, extra = {}, {"grid": lb}

        if strat == "donchian_is":
            if cfg["turbo"]:
                bL, pf, lbs, curve = optimize_donchian_fast(c, *lb)
                extra["opt"] = (lbs, curve, bL)
            else:
                bL, pf = optimize_donchian(win, *lb)
            if bL > 1:
                sig = breakout_signal_np(c, bL)
                s = win["close"]
                overlays[f"upper L={bL}"] = (s.rolling(bL - 1).max().shift(1), C["btc2"], "dot")
                overlays[f"lower L={bL}"] = (s.rolling(bL - 1).min().shift(1), C["violet"], "dot")
            else:
                sig = np.full(len(c), np.nan)
            extra["best_L"] = bL

        elif strat == "donchian_wf":
            try:
                tw, tstep, n_oos = wf_plan(len(win), cfg["wf_train_bars"], cfg["wf_step_bars"])
            except ValueError as ex:
                return {"error": str(ex),
                        "hints": ["Widen the years.", "Shorten the training window or step."]}
            lb = clamp_grid(lb[0], lb[1], lb[2], tw)
            extra["grid"] = lb
            if cfg["turbo"]:
                sig, choices = walkforward_donch_fast(c, tw, tstep, *lb, return_choices=True)
                extra["wf_choices"] = [(win.index[min(t, len(win) - 1)], int(b)) for t, b in choices]
            else:
                sig = walkforward_donch(win, tw, tstep, *lb)
            # out-of-sample segment only — no in-sample bar is scored or drawn
            sig = sig[tw:]
            win = win.iloc[tw:]
            c = win["close"].to_numpy()
            r = np.empty(len(c)); r[:-1] = np.diff(np.log(c)); r[-1] = np.nan
            extra["oos_bars"] = len(win)
            extra["wf_train"] = tw

        elif strat == "tree":
            model = train_tree(win, cfg["tree_leaf"], cfg["tree_depth"],
                               spans=spans, horizon=horizon)
            sig_s, pf = tree_strategy(win, model, spans=spans)
            sig = sig_s.to_numpy()
            extra["importances"] = model.feature_importances_.tolist()
            extra["spans"] = spans

        else:
            fast, slow = int(cfg["ma_fast"]), int(cfg["ma_slow"])
            if fast >= slow:
                fast, slow = max(1, slow - 1), slow
            sig_s, pf = ma_crossover(win, fast, slow, cfg["ma_short"])
            sig = sig_s.to_numpy()
            overlays[f"MA {fast}"] = (win["close"].rolling(fast).mean(), C["cyan"], "solid")
            overlays[f"MA {slow}"] = (win["close"].rolling(slow).mean(), C["magenta"], "solid")

        rets = sig * r
        pf = profit_factor_np(rets)
        rs = pd.Series(rets)
        eq = np.nancumsum(rets)
        step = max(1, len(eq) // 2600)
        dd = eq - np.maximum.accumulate(eq)
        valid = np.isfinite(sig)
        chg = np.flatnonzero(valid[1:] & valid[:-1] & (sig[1:] != sig[:-1])) + 1
        daily = win.resample("1D").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        if tf in ("1M", "1Y"):
            daily = win[["open", "high", "low", "close"]]
        dov = {name: (ser.resample("1D").last(), col, dash) for name, (ser, col, dash) in overlays.items()}
        if tf in ("1M", "1Y"):
            dov = overlays
        nz = rs.dropna()
        nz = nz[nz != 0]
        std = rs.std()
        sharpe = float(rs.mean() / std) if std and std > 0 else float("nan")

        return dict(
            error=None, daily=daily, overlays=dov,
            eq_x=win.index[::step], eq=eq[::step], dd=dd[::step],
            close_ds=c[::step], pos_ds=np.nan_to_num(sig[::step], nan=0.0),
            stats=dict(
                pf=pf,
                sharpe=sharpe,
                sharpe_ann=sharpe * np.sqrt(bars_per_year(tf)),
                total=float(np.nansum(rets)),
                exposure=float(np.nanmean(np.abs(sig))) if valid.any() else 0.0,
                trades=int(len(chg)),
                winrate=float((nz > 0).mean()) if len(nz) else float("nan"),
                bars=len(win),
            ),
            extra=extra,
        )
    except Exception as e:
        return {"error": str(e),
                "hints": ["Try a finer bar size or a wider date range.",
                          "For the tree, raise min_samples_leaf or cap max_depth."]}


# ---------------------------------------------------------------- anatomy
@st.cache_data(show_spinner="Shuffling the universe…")
def anatomy(btc, eth, draw, nlags, tf):
    seed = ANATOMY_SEED + int(draw)
    d = tf_defaults(tf)
    sl = btc.iloc[-min(len(btc), int(d["anat"])):]
    if len(sl) < 4:
        raise ValueError("Not enough bars at this bar size for the anatomy view.")
    nlags = int(max(2, min(int(nlags), max(2, len(sl) // 3))))
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
        span=f"{sl.index[0]:%Y-%m-%d} → {sl.index[-1]:%Y-%m-%d}", bars=len(sl),
    )
    if eth is not None:
        common = sl.index.intersection(eth.index)
        if len(common) > 30:
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


# ---------------------------------------------------------------- market notes
@st.cache_data(show_spinner="Reading the tape…")
def notes_stats(btc, eth, tf):
    """Descriptive diagnostics behind the Market Notes article — all computed, none assumed."""
    ppy = bars_per_year(tf)

    def one(df):
        c = df["close"].to_numpy(dtype=float)
        lc = np.log(c)
        r = np.diff(lc)
        n = len(r)
        if n < 8:
            return None
        s = pd.Series(r)
        a = pd.Series(np.abs(r))
        eq = np.cumsum(r)
        dd = eq - np.maximum.accumulate(eq)
        k = max(1, n // 100)
        q = int(max(2, min(24, n // 4)))
        rq = pd.Series(lc).diff(q).dropna()
        v1 = float(s.var(ddof=1))
        vr = float(rq.var(ddof=1) / (q * v1)) if n > 3 * q and v1 > 0 else float("nan")
        sd = float(s.std(ddof=1))
        z = (r - float(s.mean())) / sd if sd > 0 else r * 0.0
        lag = int(min(24, max(2, n // 10)))
        return dict(
            bars=int(len(df)), start=df.index[0], end=df.index[-1],
            first=float(c[0]), last=float(c[-1]), years=float(len(df) / ppy),
            total=float(lc[-1] - lc[0]), ann_vol=sd * np.sqrt(ppy),
            skew=float(s.skew()), kurt=float(s.kurt()),
            best=float(r.max()), worst=float(r.min()),
            mdd_log=float(dd.min()), mdd_pct=float(np.expm1(dd.min())),
            acf1=float(s.autocorr(1)), acf1_abs=float(a.autocorr(1)),
            acfL_abs=float(a.autocorr(lag)), lag=lag,
            p3=float((np.abs(z) > 3).mean()), p2=float((np.abs(z) > 2).mean()),
            top1=float(np.sort(np.abs(r))[-k:].sum() / max(np.abs(r).sum(), 1e-12)),
            vrq=q, vr=vr,
        )

    out = dict(tf=tf, btc=one(btc), eth=one(eth) if eth is not None else None,
               joint=None, perm=None, acf=None, dd=None)

    if out["btc"] is not None:
        sl = btc.iloc[-min(len(btc), int(tf_defaults(tf)["anat"])):]
        if len(sl) >= 4:
            pm = get_permutation(sl, seed=ANATOMY_SEED)
            rr = np.log(sl["close"]).diff().dropna().to_numpy()
            pr = np.log(pm["close"]).diff().dropna().to_numpy()
            nlags = int(max(2, min(48, len(rr) // 3)))
            aar, n = acf(np.abs(rr), nlags)
            aap, _ = acf(np.abs(pr), nlags)
            out["acf"] = dict(lags=np.arange(nlags + 1), real=aar, perm=aap, n=n)
            out["perm"] = dict(acf1=float(pd.Series(pr).autocorr(1)),
                               acf1_abs=float(pd.Series(np.abs(pr)).autocorr(1)))
        eq = np.cumsum(np.diff(np.log(btc["close"].to_numpy(dtype=float))))
        dd = eq - np.maximum.accumulate(eq)
        step = max(1, len(dd) // 2000)
        out["dd"] = dict(x=btc.index[1:][::step], y=dd[::step])

    if eth is not None:
        common = btc.index.intersection(eth.index)
        if len(common) > 30:
            rb = np.log(btc.loc[common, "close"]).diff()
            re = np.log(eth.loc[common, "close"]).diff()
            j = pd.concat([rb.rename("b"), re.rename("e")], axis=1).dropna()
            if len(j) > 30 and float(j["b"].var()) > 0:
                w = int(max(20, min(720, len(j) // 10)))
                roll = j["b"].rolling(w).corr(j["e"]).dropna()
                out["joint"] = dict(
                    n=int(len(j)), rho=float(j["b"].corr(j["e"])),
                    beta=float(j["b"].cov(j["e"]) / j["b"].var()),
                    rho_abs=float(j["b"].abs().corr(j["e"].abs())),
                    win=w, roll_lo=float(roll.min()) if len(roll) else float("nan"),
                    roll_hi=float(roll.max()) if len(roll) else float("nan"),
                )
    return out


def notes_table(S):
    rows = ["Bars", "Span (years)", "First close", "Last close", "Σ log return",
            "Annualised vol", "Skew", "Excess kurtosis", "Best bar", "Worst bar",
            "Max drawdown", "ACF(1) returns", "ACF(1) |returns|", "|r| > 3σ share",
            "Top 1% of bars · share of movement"]
    specs = ["{:,.0f}", "{:.2f}", "{:,.2f}", "{:,.2f}", "{:+.3f}", "{:.1%}", "{:+.3f}",
             "{:.2f}", "{:+.2%}", "{:+.2%}", "{:.1%}", "{:+.4f}", "{:+.4f}", "{:.2%}", "{:.1%}"]
    keys = ["bars", "years", "first", "last", "total", "ann_vol", "skew", "kurt", "best",
            "worst", "mdd_pct", "acf1", "acf1_abs", "p3", "top1"]
    data = {}
    for label, sym in (("BTC", "btc"), ("ETH", "eth")):
        s = S.get(sym)
        data[label] = [fmt_n(s[k], sp) if s else "—" for k, sp in zip(keys, specs)]
    return pd.DataFrame(data, index=rows)


def notes_markdown(S, tf, provenance):
    b, e, j, pm = S["btc"], S["eth"], S["joint"], S["perm"]
    L = [f"# Market notes — {TF_LABEL[tf]} bars", "",
         f"Provenance: {provenance}", "",
         "Every figure below is computed from the loaded feed; nothing is assumed.", ""]
    if b:
        L += ["## Currency 1 (BTC)",
              f"- Bars: {b['bars']:,} · {b['start']:%Y-%m-%d} → {b['end']:%Y-%m-%d}",
              f"- Σ log return {b['total']:+.4f} · annualised vol {b['ann_vol']:.1%}",
              f"- Skew {b['skew']:+.3f} · excess kurtosis {b['kurt']:.2f}",
              f"- Bars beyond 3σ: {b['p3']:.2%} (Gaussian expectation 0.27%)",
              f"- ACF(1) returns {b['acf1']:+.4f} · ACF(1) |returns| {b['acf1_abs']:+.4f}",
              f"- Max drawdown {b['mdd_pct']:.1%} · top 1% of bars carry {b['top1']:.1%} of movement",
              f"- Variance ratio q={b['vrq']}: {fmt_n(b['vr'], '{:.3f}')}", ""]
    if e:
        L += ["## Currency 2 (ETH)",
              f"- Bars: {e['bars']:,} · Σ log return {e['total']:+.4f} · vol {e['ann_vol']:.1%}",
              f"- Excess kurtosis {e['kurt']:.2f} · max drawdown {e['mdd_pct']:.1%}", ""]
    if j:
        L += ["## Co-movement",
              f"- ρ(BTC, ETH) = {j['rho']:.3f} on {j['n']:,} aligned bars",
              f"- β(ETH on BTC) = {j['beta']:.3f} · ρ of |returns| = {j['rho_abs']:.3f}",
              f"- Rolling {j['win']}-bar ρ range: {fmt_n(j['roll_lo'],'{:.3f}')} … {fmt_n(j['roll_hi'],'{:.3f}')}", ""]
    if pm:
        L += ["## After one permutation",
              f"- ACF(1) returns {pm['acf1']:+.4f} · ACF(1) |returns| {pm['acf1_abs']:+.4f}",
              "- The marginal distribution is unchanged; serial dependence is not.", ""]
    L += ["## Caveats",
          "- Gross of fees, slippage and funding. Descriptive statistics, not a backtest.",
          "- Not investment advice. Verify provenance before quoting any figure.",
          f"\nBuilt by {AUTHOR} · Permutation Tests · Trading Strategies"]
    return "\n".join(L)


# ---------------------------------------------------------------- results
def result_body(job, live=False):
    cfg = job["config"]
    tf = job.get("tf", cfg.get("tf", "1h"))
    unit = TF_UNIT[tf]
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
        meta = [f"{TF_LABEL[tf]} bars", f"elapsed {fmt_t(job.get('elapsed',0))}"]
        if job.get("grid"):
            meta.append(f"grid {job['grid'][0]}–{job['grid'][1]}")
        if job.get("best_lookback") and job["best_lookback"] > 0:
            meta.append(f"selected L={job['best_lookback']}")
        if job.get("finished_at"):
            meta.append(job["finished_at"])
        st.caption(" · ".join(meta))
        if job.get("note"):
            st.caption("⚠ " + job["note"])

    colA, colB = st.columns([1.35, 1])
    with colA:
        show_fig(hist_fig(job["perm_pfs"], real_pf, p, live), key=f"hist_{job['id']}_{live}")
    with colB:
        if job.get("real_curve") is not None:
            show_fig(equity_fig(job["curve_x"], job["real_curve"], job["sample_curves"],
                                "Equity · out-of-sample" if job.get("oos_start") else "Equity"),
                     key=f"eq_{job['id']}_{live}")
        else:
            st.info("Equity curves appear once the baseline finishes.")
    st.caption(f"Amber = real · cyan = permuted universes · {unit} bars, Σ log return, gross of costs.")

    if job.get("pf_curve") is not None and job.get("env_cnt") is not None:
        cnt = job["env_cnt"]
        mean = np.where(cnt > 0, job["env_sum"] / np.maximum(cnt, 1), np.nan)
        var = np.where(cnt > 0, job["env_ssq"] / np.maximum(cnt, 1) - mean**2, np.nan)
        std = np.sqrt(np.maximum(var, 0))
        show_fig(optcurve_fig(job["lookbacks"], job["pf_curve"], job.get("best_lookback"), mean, std),
                 key=f"opt_{job['id']}_{live}")
        st.caption("The gap above the noise envelope is the evidence — not the peak height.")

    if job.get("wf_choices"):
        show_fig(choices_fig(job["wf_choices"]), key=f"wfch_{job['id']}_{live}")

    if job.get("importances"):
        show_fig(imp_fig(job["importances"], job.get("spans", cfg["tree_spans"]), unit),
                 key=f"imp_{job['id']}_{live}")

    if not live:
        with st.expander("What this null hypothesis does — and does not — rule out"):
            st.markdown(
                "Bar permutation preserves the **marginal distribution** (mean, std, skew, kurtosis), intrabar geometry and gaps, "
                "but destroys **volatility clustering**, **long memory**, **self-similarity** and spectral structure — anything involving dependence. "
                "Bars are permuted **at the selected timeframe**, so the null matches the bars the strategy actually trades. "
                "A significant p-value means the edge needs *serial structure*, not merely fat tails. "
                "A non-significant result does not disprove every conceivable edge — only that this pipeline cannot distinguish itself from shuffled noise. "
                "All results are gross of fees, slippage and funding."
            )
        e1, e2, e3 = st.columns(3)
        payload = dict(
            id=job["id"], label=job["label"], config=cfg, timeframe=tf, real_pf=real_pf,
            p_value=p, better=better, completed=done,
            best_lookback=job.get("best_lookback"), grid=job.get("grid"),
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
  <div class="h3">Is your optimized edge real — or the best-looking lie in 157 tries?</div>
  <p>Optimization <b style="color:var(--btc2)">always</b> induces data-mining bias. This terminal rebuilds the market as
  structureless noise — same distribution, zero memory — re-runs your <i>entire</i> pipeline on each shuffled universe,
  and asks how often noise alone beats your real result. </p>
  <div class="steps">
    <div class="step"><div class="n">01</div><div class="h">Pick a bar size</div><div class="b">Hourly, daily, monthly or yearly. Every window, lookback and feature span is counted in bars of that size and auto-fits your data.</div></div>
    <div class="step"><div class="n">02</div><div class="h">Determine amount of variations</div><div class="b">Intrabar shape and gaps permuted separately. Mean/std/skew/kurt survive; volatility clustering and long memory die.</div></div>
    <div class="step"><div class="n">03</div><div class="h">Compare &amp; verdict</div><div class="b">p = (1 + #noise-PF ≥ real-PF) / (N + 1). If noise matches your edge, the edge is a mirage. Watch the histogram build live.</div></div>
  </div>
  <div class="callout" style="margin-top:.85rem"><b>→ Load Currency 1, choose a bar size and a pipeline, press RUN PERMUTATION TEST.</b> Results stream as they compute; stop anytime. Mechanics, strengths and weaknesses are in Method &amp; Guide; a downloadable reference pair is in Method &amp; Guide → Data.</div>
</div>
""", unsafe_allow_html=True)


@st.fragment(run_every=0.6)
def live_fragment():
    job = st.session_state.get("active")
    if job is None or job.get("status") != "running":
        if job is not None and job.get("status") == "done" and not job.get("_finalized"):
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
        st.info(f"{job.get('phase','Working')} — the real-data baseline runs first, then permutations stream in.")

    if job["status"] != "running":
        if job.get("status") == "done" and not job.get("_finalized"):
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
                f'<span class="m">{TF_LABEL.get(r.get("tf","1h"),"Hourly")} · n={r["done"]+1} · '
                f'PF {r["real_pf"]:.3f} · p {r["p"]:.3f}</span>'
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
    src_options.append(REF_SOURCE)
    source = st.selectbox("Source", src_options, index=0, key="w_source")

    up1 = up2 = None
    if source == "Upload files":
        up1 = st.file_uploader("Currency 1 (.pq / .parquet / .csv)",
                               type=["pq", "parquet", "csv"], key="up_1",
                               help="timestamp + open/high/low/close · max 200 MB per file")
        with st.expander("Optional · second currency"):
            up2 = st.file_uploader("Currency 2 (optional, for correlation)",
                                   type=["pq", "parquet", "csv"], key="up_2",
                                   help="Same schema and timestamps as Currency 1 · max 200 MB per file")
        st.caption("max 200 MB per file · CSV, PQ or Parquet. "
                   "No file? Download the reference pair in Method & Guide → Data.")

    notice = None
    try:
        raw1, raw2, src_label, notice = resolve_data(source, up1, up2)
    except Exception as e:
        raw1, raw2 = synth_pair()
        src_label, notice = "DEMO FEED", None
        notify("That file could not be read, so the reference feed is in use.",
               ["Check the columns: timestamp, open, high, low, close.",
                "Prices must be positive numbers with no empty rows.",
                "Keep the file under max 200 MB."],
               technical=f"{type(e).__name__}: {e}")
    if notice:
        st.caption(notice)

    st.markdown('<div class="sb-sec">02 · Bar size</div>', unsafe_allow_html=True)
    tf = st.select_slider("Timeframe", options=list(TIMEFRAMES), value="1h",
                          format_func=lambda k: TF_LABEL[k], key="w_tf")
    if st.session_state.get("tf_prev") != tf:
        st.session_state["tf_prev"] = tf
        reset_notices()
        for k in ("w_lbmin", "w_lbmax", "w_lbstep", "w_wf_train", "w_wf_step",
                  "w_maf", "w_mas", "w_lags"):
            st.session_state.pop(k, None)
    btc_df = to_tf(raw1, tf)
    eth_df = to_tf(raw2, tf) if raw2 is not None else None
    st.session_state["btc_df"] = btc_df
    st.session_state["eth_df"] = eth_df

    dflt = tf_defaults(tf)
    unit = TF_UNIT[tf]
    n_bars = len(btc_df)
    ymin, ymax = year_bounds(btc_df)          # inclusive; stops on the current year
    is_def, wf_def = year_defaults(btc_df)
    data_last = int(btc_df.index.year.max())
    st.caption(f"{src_label} · {n_bars:,} {unit} bars · data {ymin}–{data_last}" +
               (" · currency 2 loaded" if eth_df is not None else ""))
    if ymax > data_last:
        st.caption(f"Year sliders run to {ymax} (current year); this feed ends in {data_last}.")
    if n_bars < 30:
        st.caption(f"⚠ {n_bars} {unit} bars — the test still runs, but such a small sample is indicative only.")

    st.markdown('<div class="sb-sec">03 · Pipeline</div>', unsafe_allow_html=True)
    strategy = st.selectbox("Strategy under test", list(STRAT_LABELS.keys()),
                            format_func=lambda k: STRAT_LABELS[k], key="w_strategy")

    if strategy in ("donchian_is", "donchian_wf"):
        ca, cb = st.columns(2)
        with ca:
            lb_min = num_slider("Lookback min", 2, max(3, min(120, n_bars - 2)), dflt["lb_min"], "w_lbmin")
        with cb:
            lb_max = num_slider("Lookback max", 3, max(4, min(360, n_bars - 2)), dflt["lb_max"], "w_lbmax")
        lb_step = num_slider("Grid step", 1, 12, dflt["lb_step"], "w_lbstep")
    else:
        lb_min, lb_max, lb_step = dflt["lb_min"], dflt["lb_max"], dflt["lb_step"]

    if strategy in ("donchian_is", "tree"):
        year_range("In-sample years (inclusive)", ymin, ymax, is_def, "w_is_years")

    if strategy == "donchian_wf":
        y0, y1 = year_range("Walk-forward years (inclusive)", ymin, ymax, wf_def, "w_wf_years")
        n_win = int(((btc_df.index.year >= y0) & (btc_df.index.year <= y1)).sum())
        tr = num_slider(f"Training window ({unit} bars)", 2, wf_train_max(n_win), dflt["train"], "w_wf_train")
        stp = num_slider(f"Re-optimize every ({unit} bars)", 1, wf_step_max(n_win), dflt["step"], "w_wf_step")
        if n_win < 4:
            st.caption(f"⚠ {n_win} {unit} bars in that range — widen the years or pick a finer bar size.")
        else:
            tw, ts, no = wf_plan(n_win, tr, stp)
            g = clamp_grid(lb_min, lb_max, lb_step, tw)
            st.caption(f"auto-fit → train {tw:,} · step {ts:,} · OOS {no:,} · "
                       f"{max(1, no // ts)} re-optimizations · grid {g[0]}–{g[1]}")

    if strategy == "tree":
        st.slider("min_samples_leaf", 1, 100, 5, key="w_leaf")
        st.selectbox("max_depth", ["Unlimited", 4, 6, 8, 12], format_func=str, key="w_depth")
        sp = dflt["spans"]
        st.caption(f"features {sp[0]}/{sp[1]}/{sp[2]} {unit}-bar momentum · label = next {dflt['horizon']} bar(s)")

    if strategy == "ma":
        ca, cb = st.columns(2)
        with ca:
            num_slider("Fast MA", 1, max(2, min(120, n_bars - 1)), dflt["ma_fast"], "w_maf")
        with cb:
            num_slider("Slow MA", 2, max(3, min(360, n_bars - 1)), dflt["ma_slow"], "w_mas")
        st.toggle("Allow shorts (else long/flat)", key="w_mashort")
        if int(st.session_state.get("w_maf", dflt["ma_fast"])) >= int(st.session_state.get("w_mas", dflt["ma_slow"])):
            st.caption("Fast ≥ slow — the fast MA is auto-set to slow − 1.")

    st.markdown('<div class="sb-sec">04 · Permutation engine</div>', unsafe_allow_html=True)
    st.slider("Permutations", 50, 2000, 1000, step=50, key="w_perms")
    st.toggle("Turbo engine (vectorized, identical math)", value=True, key="w_turbo")
    if strategy == "donchian_wf":
        st.caption("Walk-forward re-optimizes inside every permutation — the heaviest test.")

    st.markdown('<div class="sb-sec">05 · Execute</div>', unsafe_allow_html=True)
    active = st.session_state.get("active")
    running = bool(active and active.get("status") == "running")
    st.button("🧪 RUN PERMUTATION TEST", type="primary", on_click=start_run,
              disabled=running, use_container_width=True)
    st.button("Clear results", on_click=clear_active, disabled=running, use_container_width=True)
    if running:
        st.caption(f"● running — {active['done']:,} / {active['config']['n_perms']:,}")

# ---------------------------------------------------------------- header
cfg_now = collect_cfg()
tf_now = cfg_now["tf"]
unit_now = TF_UNIT[tf_now]
st.markdown(
    f"""
<div class="hero">
  <h1>Permutation Tests<span class="grad">Trading Strategies</span></h1>
  <div class="sub">Optimization always induces <b>data-mining bias</b>. This terminal re-runs your fully-optimized pipeline on
  <b>various reference benchmarks of your choosing</b> — same distribution, zero memory — at the bar size you choose, and measures how often noise alone beats you.</div>
  <div class="chips">
    <span class="chip hot">FEED <b>{src_label}</b></span>
    <span class="chip cool">BARS <b>{TF_LABEL[tf_now]} · {len(btc_df):,}</b></span>
    <span class="chip">N <b>{cfg_now['n_perms']:,}</b></span>
    <span class="chip">BUILT BY <b>{AUTHOR}</b></span>
  </div>
  <div class="hairline"></div>
</div>
""", unsafe_allow_html=True)

tab_lab, tab_strat, tab_anat, tab_notes, tab_method = st.tabs(
    ["PERMUTATION LAB", "STRATEGY", "PERMUTATION ANATOMY", "MARKET NOTES", "METHOD & GUIDE"]
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
        notify("The test could not complete on this sample.",
               ["Pick a finer bar size, or widen the date range.",
                "Lower the lookback grid, or allow shorts for the MA rule.",
                "For the tree, raise min_samples_leaf or cap max_depth."],
               technical=active.get("error"))
        onboarding()
        history_block()
    else:
        sec("R", f"Results · {active['label']}")
        if not active.get("_finalized"):
            finalize(active)
        st.markdown(
            f'<div class="chips"><span class="chip hot">{active["label"]}</span>'
            f'<span class="chip cool">{TF_LABEL.get(active.get("tf","1h"),"Hourly")}</span>'
            f'<span class="chip">{"turbo" if active["config"]["turbo"] else "reference"}</span>'
            f'<span class="chip">{active.get("window_desc","")}</span>'
            f'<span class="chip">{active["status"].upper()}</span></div>',
            unsafe_allow_html=True)
        result_body(active, live=False)
        history_block()

# ---------------------------------------------------------------- STRATEGY
with tab_strat:
    sec("S", f"Strategy · {STRAT_LABELS[cfg_now['strategy']]}")
    c1, c2, c3 = st.columns([1, 1, 2.2])
    style = c1.selectbox("Price panel", ["Candles", "Line"], key="w_price_style")
    show_ov = c2.toggle("Indicator overlay", value=True, key="w_show_ov")
    c3.markdown(
        '<div class="callout" style="margin:.2rem 0">Separate panels: <b>price · position · equity · drawdown</b>.</div>',
        unsafe_allow_html=True)

    det = compute_details(btc_df, cfg_now)
    if det.get("error"):
        notify(det["error"], det.get("hints"), technical=det["error"])
        st.caption("Adjust the bar size or date range in the sidebar to populate this panel.")
    else:
        s = det["stats"]
        st.markdown(cards_html([
            dict(k="Profit Factor", v=f"{s['pf']:.4f}" if np.isfinite(s['pf']) else "—",
                 cls="" if s['pf'] >= 1 else "red"),
            dict(k="Sharpe (ann.)", v=f"{s['sharpe_ann']:.2f}" if np.isfinite(s['sharpe_ann']) else "—",
                 cls="cyan", d=f"per bar {s['sharpe']:.4f}" if np.isfinite(s['sharpe']) else ""),
            dict(k="Σ Log Return", v=f"{s['total']:+.3f}", cls="green" if s['total'] > 0 else "red"),
            dict(k="Exposure", v=f"{s['exposure']:.0%}", cls="cyan"),
            dict(k="Position changes", v=f"{s['trades']:,}", cls="cyan",
                 d=f"{s['bars']:,} {unit_now} bars"),
        ]), unsafe_allow_html=True)
        if det["extra"].get("oos_bars"):
            st.caption(f"Out-of-sample only · {det['extra']['oos_bars']:,} bars · "
                       f"training block {det['extra']['wf_train']:,} bars is never scored.")

        show_fig(price_fig(det, style, show_ov), key="s_price")
        show_fig(position_fig(det["eq_x"], det["pos_ds"]), key="s_pos")

        ce, cd = st.columns([1.5, 1])
        with ce:
            show_fig(equity_fig(det["eq_x"], det["eq"], [], "Equity"), key="s_eq")
        with cd:
            show_fig(dd_fig(det["eq_x"], det["dd"]), key="s_dd")

        ex = det["extra"]
        if "opt" in ex:
            lbs, curve, bL = ex["opt"]
            show_fig(optcurve_fig(lbs, curve, bL), key="s_opt")
            st.caption(f"Best lookback on real bars: {bL}. The Permutation Lab renders the verdict.")
        if "wf_choices" in ex:
            show_fig(choices_fig(ex["wf_choices"]), key="s_wf")
        if "importances" in ex:
            show_fig(imp_fig(ex["importances"], ex.get("spans", cfg_now["tree_spans"]), unit_now), key="s_imp")

# ---------------------------------------------------------------- ANATOMY
with tab_anat:
    sec("A", "What permutation preserves — and what it kills")
    ca, cb = st.columns([1, 3])
    cb.button("🔀 Draw a new permutation", on_click=new_permutation_draw)
    nlags = num_slider(f"ACF lags ({unit_now} bars)", 2, max(3, min(336, len(btc_df) // 3)),
                       tf_defaults(tf_now)["lags"], "w_lags")
    try:
        an = anatomy(btc_df, eth_df, int(st.session_state.get("anat_draw", 0)), nlags, tf_now)
    except Exception as ex:
        an = None
        notify("The anatomy view needs more bars at this bar size.",
               ["Pick Hourly or Daily.", "Load a longer date range."],
               technical=f"{type(ex).__name__}: {ex}")

    if an is None:
        st.caption("Adjust the bar size in the sidebar to populate this panel.")
    else:
        st.caption(f"{an['bars']:,} {unit_now} bars · {an['span']} · fixed seeds.")

        col1, col2 = st.columns([1.5, 1])
        with col1:
            show_fig(rebased_fig(an["x"], an["lcr"], an["lcp"]), key="an_price")
            st.caption("Same bars, same total return — different path.")
        with col2:
            st.markdown('<div class="panel"><div class="h3">Distribution preserved</div>'
                        '<p>Shuffling rearranges bars; it cannot change the histogram of returns.</p></div>',
                        unsafe_allow_html=True)
            st.table(an["moments"].style.format("{:.6f}"))

        c1, c2 = st.columns(2)
        with c1:
            show_fig(acf_fig(an["lags"], an["ar"], an["ap"], an["n"], "ACF · returns", unit_now), key="an_acf_r")
        with c2:
            show_fig(acf_fig(an["lags"], an["aar"], an["aap"], an["n"], "ACF · |returns|", unit_now), key="an_acf_a")
        st.caption("Dashed = ±1.96/√n white-noise bounds.")

        if len(an["fxr"]) > 2:
            show_fig(spec_fig(an["fxr"], an["pxr"], an["fxp"], an["pxp"]), key="an_spec")
            st.caption("Slow low-frequency decay = long memory; permutation flattens it.")

        if an["corr"]:
            cr = an["corr"]
            show_fig(corr_fig(cr["rb"], cr["re"], cr["pb"], cr["pe"], cr["rho_r"], cr["rho_p"]), key="an_corr")
            st.caption("One shared shuffle keeps contemporaneous correlation alive.")
        else:
            st.info("Load Currency 2 to see the joint-permutation correlation demo.")

# ---------------------------------------------------------------- MARKET NOTES
with tab_notes:
    sec("N", "Market notes — an empirical read of the loaded feed")
    is_ref = src_label in ("REFERENCE PAIR", "DEMO FEED")
    provenance = (
        f"Seeded synthetic reference pair (seed {REF_SEED}), BTC- and ETH-like — not exchange history"
        if is_ref else f"User-supplied feed: {src_label}"
    )
    st.markdown(
        f'<div class="callout {"cool" if is_ref else ""}"><b>Provenance.</b> {provenance}. '
        f'Every number in this article is computed from the feed currently loaded at the '
        f'<b>{TF_LABEL[tf_now]}</b> bar size; change the bar size or the data and the text updates with it. '
        'Descriptive statistics — not a backtest, not investment advice.</div>',
        unsafe_allow_html=True)

    try:
        NS = notes_stats(btc_df, eth_df, tf_now)
    except Exception as ex:
        NS = None
        notify("The market diagnostics need more bars at this bar size.",
               ["Pick Hourly or Daily.", "Load a longer date range."],
               technical=f"{type(ex).__name__}: {ex}")

    if NS is None or NS["btc"] is None:
        st.caption("Adjust the bar size in the sidebar to populate this panel.")
    else:
        b, e, jt, pm = NS["btc"], NS["eth"], NS["joint"], NS["perm"]
        st.table(notes_table(NS))
        st.caption(f"Currency 1 · {b['start']:%Y-%m-%d} → {b['end']:%Y-%m-%d} · "
                   f"{b['bars']:,} {unit_now} bars" + (" · Currency 2 aligned on the same timestamps." if e else ""))

        cn1, cn2 = st.columns([1.4, 1])
        with cn1:
            common = btc_df.index.intersection(eth_df.index) if eth_df is not None else []
            if e is not None and len(common) > 2:
                lb_ = np.log(btc_df.loc[common, "close"].to_numpy()); lb_ -= lb_[0]
                le_ = np.log(eth_df.loc[common, "close"].to_numpy()); le_ -= le_[0]
                stp = max(1, len(common) // 2000)
                show_fig(rebased_fig(common[::stp], lb_[::stp], le_[::stp],
                                     "BTC vs ETH · log price", ("BTC", "ETH")), key="n_pair")
                st.caption("Both series rebased to zero at the first shared bar.")
            elif NS["dd"] is not None:
                show_fig(dd_fig(NS["dd"]["x"], NS["dd"]["y"]), key="n_dd_solo")
        with cn2:
            if NS["acf"] is not None:
                show_fig(acf_fig(NS["acf"]["lags"], NS["acf"]["real"], NS["acf"]["perm"],
                                 NS["acf"]["n"], "ACF · |returns| · real vs one shuffle", unit_now),
                         key="n_acf")
                st.caption("The gap between amber and cyan is what the null hypothesis removes.")

        gauss = 0.0027
        ratio = b["p3"] / gauss if b["p3"] > 0 else float("nan")
        vr_txt = (f"above 1 at q={b['vrq']} ({b['vr']:.2f}), i.e. variance grows faster than linearly with horizon — trend-like"
                  if np.isfinite(b["vr"]) and b["vr"] > 1.05 else
                  (f"below 1 at q={b['vrq']} ({b['vr']:.2f}), i.e. partial mean reversion at that horizon"
                   if np.isfinite(b["vr"]) and b["vr"] < 0.95 else
                   f"close to 1 at q={b['vrq']} ({fmt_n(b['vr'],'{:.2f}')}), i.e. near-random-walk scaling"))
        perm_txt = (f"After a single permutation of the same bars, those two figures become "
                    f"<b>{fmt_n(pm['acf1'], '{:+.4f}')}</b> and <b>{fmt_n(pm['acf1_abs'], '{:+.4f}')}</b> respectively."
                    if pm else "A permutation comparison needs a few more bars at this size.")

        st.markdown(f"""
<div class="note-sec">
  <div class="h4">1 · The return distribution is decisively non-Gaussian</div>
  <p>Across <span class="fig">{b['bars']:,}</span> {unit_now} bars, currency 1 shows an annualised volatility of
  <b>{b['ann_vol']:.1%}</b>, skew of <b>{b['skew']:+.3f}</b> and excess kurtosis of <b>{b['kurt']:.2f}</b>.
  Moves beyond three standard deviations account for <b>{b['p3']:.2%}</b> of bars against a Gaussian expectation of
  0.27% — roughly <b>{fmt_n(ratio, '{:.1f}')}×</b> the normal rate. The single best bar is
  <b>{b['best']:+.2%}</b> and the worst <b>{b['worst']:+.2%}</b>.</p>
  <p>The practical consequence: any risk model, position size or stop calibrated on a normal distribution
  understates tail exposure here. It also explains why profit factor alone is a weak summary — a handful of
  extreme bars can dominate it.</p>
</div>

<div class="note-sec">
  <div class="h4">2 · Direction is close to unpredictable; risk is not</div>
  <p>First-order autocorrelation of returns is <b>{b['acf1']:+.4f}</b>, while autocorrelation of absolute returns is
  <b>{b['acf1_abs']:+.4f}</b> and remains <b>{fmt_n(b['acfL_abs'], '{:+.4f}')}</b> at lag {b['lag']}.
  {perm_txt}</p>
  <p>This asymmetry is the central empirical fact behind the test. Shuffling cannot change the histogram, so a
  strategy that only harvests the shape of the distribution keeps working on noise. A strategy that depends on
  <i>order</i> — volatility clustering, persistence, regime length — is being judged against a world where that
  order no longer exists.</p>
</div>

<div class="note-sec">
  <div class="h4">3 · Horizon changes the answer</div>
  <p>The variance ratio is {vr_txt}. Aggregation matters: the same feed read at a different bar size changes both
  the tail statistics and the dependence structure, because averaging bars pulls the distribution toward normality
  and re-mixes the clustering. This is why the terminal counts every window in bars of the selected size rather
  than in fixed hours, and why a result should always be re-checked at one neighbouring bar size before it is trusted.</p>
</div>

<div class="note-sec">
  <div class="h4">4 · Path risk dominates the average</div>
  <p>The cumulative log return is <b>{b['total']:+.3f}</b>, yet the deepest peak-to-trough drawdown on the way is
  <b>{b['mdd_pct']:.1%}</b>. The largest 1% of bars carry <b>{b['top1']:.1%}</b> of all absolute movement.</p>
  <p>Two implications for evaluation. First, an equity curve that looks acceptable in summary can contain a
  drawdown that no mandate would survive. Second, when so much movement is concentrated in so few bars, the
  effective sample size is far smaller than the bar count suggests — another reason to demand a large permutation
  count rather than a comfortable-looking profit factor.</p>
</div>
""", unsafe_allow_html=True)

        if jt is not None:
            st.markdown(f"""
<div class="note-sec">
  <div class="h4">5 · Two tickers, largely one risk factor</div>
  <p>On <span class="fig">{jt['n']:,}</span> aligned bars, the correlation between currency 1 and currency 2 returns is
  <b>{jt['rho']:.3f}</b>, with a beta of currency 2 on currency 1 of <b>{jt['beta']:.3f}</b>. The correlation of
  absolute returns is <b>{jt['rho_abs']:.3f}</b>, and a rolling {jt['win']}-bar correlation ranges from
  <b>{fmt_n(jt['roll_lo'], '{:.3f}')}</b> to <b>{fmt_n(jt['roll_hi'], '{:.3f}')}</b>.</p>
  <p>Holding both is therefore much less diversification than two independent assets would provide, and the
  correlation is not stable through time. Joint permutation preserves this contemporaneous relationship while
  destroying each market's own serial structure — which is exactly the null you want when a rule reads two markets
  at once.</p>
</div>
""", unsafe_allow_html=True)
        else:
            st.info("Load Currency 2 (the reference ETH file works) to add the co-movement section.")

        st.markdown(f"""
<div class="note-sec">
  <div class="h4">6 · What this implies for each pipeline</div>
  <ul>
    <li><b>Donchian Breakout.</b> With dependence this weak at lag 1 and tails this heavy, the lookback sweep has
    ample room to fit noise. Read the amber curve against the noise envelope, never on its own.</li>
    <li><b>Donchian Walk-Forward.</b> The most defensible of the four: parameters are re-chosen on trailing bars and
    scored only out-of-sample. Expect a lower profit factor than in-sample — that drop is the honest price of selection.</li>
    <li><b>Decision Tree.</b> With |returns| persistence at {b['acf1_abs']:+.3f} and near-zero return autocorrelation,
    a flexible classifier can appear skilful by learning volatility state rather than direction. Cap depth, raise the
    leaf size, and re-run.</li>
    <li><b>MA Crossover.</b> No optimizer, so a low p-value here is the cleanest evidence of a genuine trend pattern —
    but long/flat also embeds a bet on positive drift, which the {b['total']:+.3f} cumulative log return makes explicit.</li>
  </ul>
</div>

<div class="note-sec">
  <div class="h4">7 · Reproduction and caveats</div>
  <p>The reference pair is generated with seed {REF_SEED} and exported at full float precision, so downloading the CSVs
  and uploading them again reproduces the default run exactly. Permutation seeds are fixed at {BASE_SEED} for tests and
  {ANATOMY_SEED} for the anatomy draw, so every figure here is repeatable.</p>
  <p>All figures are gross of fees, slippage and funding; drawdowns are measured in log-equity units on close-to-close
  returns and ignore intrabar excursions. {"Because this feed is synthetic, its numbers describe the generator, not the real BTC/ETH market — load exchange history before drawing market conclusions." if is_ref else "These numbers describe the supplied file only; verify its provenance, timezone and survivorship handling before quoting them."}</p>
</div>
""", unsafe_allow_html=True)

        nd1, nd2 = st.columns(2)
        nd1.download_button("⬇ These notes (.md)", notes_markdown(NS, tf_now, provenance),
                            file_name=f"market_notes_{tf_now}.md", mime="text/markdown", key="n_md")
        nd2.download_button("⬇ Diagnostics table (.csv)", notes_table(NS).to_csv().encode(),
                            file_name=f"market_notes_{tf_now}.csv", mime="text/csv", key="n_csv")

# ---------------------------------------------------------------- METHOD & GUIDE
with tab_method:
    sec("M", "The logic of the test")
    st.markdown(
        """
<div class="panel">
<div class="h3">Why permutation testing?</div>
<p>Using the best version of a strategy <b style="color:var(--text)">requires optimization</b> — and optimization
<b style="color:var(--btc2)">always induces data-mining bias</b>: pick the best of various lookbacks and the winner's profit factor is inflated by selection alone.
The question is whether your pipeline's result is distinguishable from what the <i>same pipeline</i> achieves on data with no pattern at all.</p>
<p>MCPT builds that null by shuffling bars, then re-runs the <b style="color:var(--cyan)">entire procedure — optimization included —</b> on every permuted universe.
</p>
</div>
""", unsafe_allow_html=True)
    st.latex(r"p \;=\; \frac{1 \;+\; \#\{\,PF^{\,perm} \;\ge\; PF^{\,real}\,\}}{N_{perm} \;+\; 1}")
    st.caption("The real run counts as one observation (hence the +1). Undefined permutation PFs count as 'no edge', never as a free pass.")

    st.markdown(
        """
<div class="callout"><b>Bar size drives everything.</b> The feed is aggregated once into hourly, daily, monthly or yearly OHLC
(open = first, high = max, low = min, close = last). Lookbacks, the walk-forward training window and step, the MA pair, the tree's
momentum spans and label horizon, and the ACF lags are all counted in <b>bars of that size</b> and auto-fit the sample — there is no
minimum bar count. Permutation is applied to those same bars, so the null always matches what the strategy trades.</div>
<div class="callout cool"><b>Year ranges are automatic and inclusive.</b> The date sliders start at the first year in your data and
stop on the current calendar year — both ends are included, so selecting 2016 → 2026 keeps every bar in 2026. Nothing is hard-coded:
load a newer file and the range extends itself. Both pipelines default to the full range.</div>
<div class="pd-grid">
  <div class="pd keep"><div class="h4">✓ PRESERVED by bar permutation</div><ul>
    <li><b>Marginal distribution</b> of returns — mean, std, skew, kurtosis</li>
    <li><b>Intrabar geometry</b> — high/low/close relative to open, shuffled jointly</li>
    <li><b>Gap distribution</b> — close-to-open jumps, shuffled separately</li>
    <li><b>Cross-market correlation</b> — when currencies are permuted jointly</li>
    <li><b>Real history before start_index</b> — walk-forward permutes only the OOS region</li>
  </ul></div>
  <div class="pd kill"><div class="h4">✕ DESTROYED by bar permutation</div><ul>
    <li><b>Volatility clustering</b> — serial correlation of absolute returns</li>
    <li><b>Long memory</b> — slow decay of dependency over longer lags</li>
    <li><b>Self-similarity</b> and fractal scaling across horizons</li>
    <li><b>Frequency-domain structure</b> — the spectrum of |returns| flattens</li>
    <li><b>Anything involving dependence</b> — trends, mean-reversion, calendar effects</li>
  </ul></div>
</div>
""", unsafe_allow_html=True)

    # ---------------- data, reference downloads, quick start ----------------
    sec("D", "Data — required format and a reference pair to try")
    st.markdown(
        """
<div class="panel">
<div class="h3">File requirements</div>
<ul>
  <li><b>Format.</b> <code>.csv</code>, <code>.pq</code> or <code>.parquet</code>. Upload limit is <b>max 200 MB per file</b>
      (Streamlit's cap; raise it with <code>[server] maxUploadSize</code> in <code>.streamlit/config.toml</code> if you self-host).</li>
  <li><b>Columns.</b> A timestamp first column, then <code>open, high, low, close</code>. Names are lower-cased; extra
      columns (volume, symbol) are ignored. For CSV the first column is parsed as the index.</li>
  <li><b>Timestamps.</b> Anything <code>pandas.to_datetime</code> understands. Use one timezone throughout — UTC is safest.
      Rows are sorted and de-duplicated automatically.</li>
  <li><b>Values.</b> Strictly positive prices; rows with missing or non-positive OHLC are dropped. A file needs at least 8 usable rows.</li>
  <li><b>Second currency.</b> Optional, and only used for the joint-permutation correlation demo and the Market Notes
      co-movement section. It is aligned to Currency 1 on shared timestamps.</li>
</ul>
<div class="h4">Example — first three rows</div>
</div>
""", unsafe_allow_html=True)
    st.code("timestamp,open,high,low,close\n"
            "2016-01-01 00:00:00,430.00,431.82,428.15,430.44\n"
            "2016-01-01 01:00:00,430.44,433.10,429.90,432.02\n"
            "2016-01-01 02:00:00,432.02,432.55,427.40,428.11", language="text")

    st.markdown(
        f"""
<div class="callout cool"><b>Reference pair — download and try.</b> These two CSVs are the app's <b>default seeded run</b>
exported at full precision: <b>BTC-like</b> and <b>ETH-like</b> hourly OHLC, 2016-01-01 → 2020-12-31, 43,848 bars each
(≈4 MB per file, far below the max 200 MB cap). They are <b>synthetic</b> — GARCH(1,1) volatility, t(4) innovations,
AR(1) trend regimes, rare jumps and correlated cross-market shocks — generated with seed {REF_SEED}.
They are <b>not exchange history</b> and must not be quoted as BTC or ETH prices. Uploading them reproduces the
default run exactly, which makes them ideal for learning the workflow, checking a change, or reporting a bug.</div>
""", unsafe_allow_html=True)
    rb1, rb2, rb3, rb4 = st.columns(4)
    rb1.download_button("⬇ BTC reference (.csv)", reference_csv("BTC"),
                        file_name="BTCUSD_reference_hourly.csv", mime="text/csv", key="dl_ref_btc")
    rb2.download_button("⬇ ETH reference (.csv)", reference_csv("ETH"),
                        file_name="ETHUSD_reference_hourly.csv", mime="text/csv", key="dl_ref_eth")
    rb3.download_button("⬇ Both + README (.zip)", reference_zip(),
                        file_name="reference_pair_hourly.zip", mime="application/zip", key="dl_ref_zip")
    rb4.button("▶ Use reference pair now", on_click=use_reference_pair, key="use_ref")
    st.caption("Currency 1 = BTC file · Currency 2 = ETH file. The button switches the sidebar source without an upload.")

    st.markdown(
        """
<div class="panel">
<div class="h3">Loading real exchange history</div>
<p>For conclusions about the actual market, use real data. Public, no-cost sources that publish BTC and ETH OHLC:</p>
<ul>
  <li><b>Binance public data</b> — <code>data.binance.vision</code>, spot/futures klines aggregated into daily or monthly
      ZIPs per symbol and interval; the companion <code>binance/binance-public-data</code> repository documents the layout.</li>
  <li><b>CryptoDataDownload</b> — free hourly and daily CSVs for BTCUSD / ETHUSD and the USDT pairs across several venues.</li>
  <li><b>Kaggle</b> — the long-running “Bitcoin Historical Data” minute-level dataset, and various hourly Binance mirrors.</li>
  <li><b>Kraken OHLCVT archives</b> — free intraday history back to 2013 across pairs and intervals.</li>
</ul>
<p>Whatever the source, normalise it to the schema above before uploading: one row per bar, ascending UTC timestamps,
no duplicate bars, positive prices. Binance klines carry an <code>open_time</code> in milliseconds and extra columns —
convert and rename, then keep the four price columns.</p>
</div>
""", unsafe_allow_html=True)
    st.code(
        "import pandas as pd\n"
        "# Binance kline CSV (no header): open_time, open, high, low, close, volume, ...\n"
        "cols = ['open_time','open','high','low','close','volume','close_time',\n"
        "        'qav','trades','tbbav','tbqav','ignore']\n"
        "df = pd.read_csv('BTCUSDT-1h-2024-01.csv', header=None, names=cols)\n"
        "df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True).dt.tz_localize(None)\n"
        "df = df[['timestamp','open','high','low','close']].astype(\n"
        "        {'open':float,'high':float,'low':float,'close':float})\n"
        "df.to_csv('BTCUSD_hourly.csv', index=False)", language="python")

    sec("Q", "Quick start — a first run in five minutes")
    st.markdown(
        """
<div class="panel">
<ol>
  <li><b>Load data.</b> Sidebar → Source → <i>Reference BTC/ETH pair (seeded)</i>, or upload your own files. Currency 2 is optional.</li>
  <li><b>Choose a bar size.</b> Start with <b>Daily</b>: enough bars for stable statistics, fast enough for 1000 permutations.
      Hourly is the reference setting; monthly and yearly are for coarse regime questions.</li>
  <li><b>Choose a pipeline.</b> Begin with <b>MA Crossover</b> — no optimizer, so the result is easy to interpret.
      Then run <b>Donchian Breakout</b> to see selection bias appear, and <b>Donchian Walk-Forward</b> for the honest version.</li>
  <li><b>Set permutations.</b> 200 for a smoke test, <b>1000</b> for anything you intend to report. Keep Turbo on.</li>
  <li><b>Run and read.</b> Watch the histogram build. When it finishes, read the verdict, then the equity panel, then the
      PF-vs-lookback surface. Download the JSON — it contains the full configuration and every permuted PF.</li>
  <li><b>Re-check.</b> Change one thing — bar size, date range, or grid — and run again. A result that survives neighbouring
      settings is worth more than a single flattering number.</li>
</ol>
</div>
""", unsafe_allow_html=True)

    sec("P", "Every pipeline — how it works, strengths, weaknesses")
    st.markdown(
        """
<div class="strat-block">
  <div class="hd"><span class="nm">1 · Donchian Breakout</span><span class="tagline">TREND / BREAKOUT · 1 PARAMETER SWEPT</span></div>
  <ol>
    <li>Channel from the <b>prior L−1 closes</b>: upper = rolling max, lower = rolling min, both <b>shifted one bar</b> — causal.</li>
    <li>Close above upper → <b>+1</b>; below lower → <b>−1</b>; otherwise hold until the opposite break.</li>
    <li>Earn the next bar's log return: <code>signal × (log close[t+1] − log close[t])</code>.</li>
    <li>Sweep the lookback grid and keep the <b>highest profit factor</b> on that same data. That selection step is the bias MCPT prices.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><div class="h4">Strengths</div><ul>
      <li>Captures fat-tailed trending regimes</li>
      <li>One transparent parameter</li>
      <li>Always in the market — cannot miss a violent breakout</li>
      <li>No look-ahead in the signal</li>
    </ul></div>
    <div class="pd kill"><div class="h4">Weaknesses</div><ul>
      <li><b>Maximum selection bias</b> — the best of the grid is an upper bound</li>
      <li>Spiky PF surface; the winner is often unstable</li>
      <li>Whipsaws in ranges; no stop or sizing</li>
      <li>Gross of fees, slippage and funding</li>
    </ul></div>
  </div>
  <div class="ask">MCPT asks: does the <i>whole sweep</i> find lookbacks this good on shuffled bars?</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">2 · Donchian Walk-Forward</span><span class="tagline">SAME RULE · RE-OPTIMIZED OUT OF SAMPLE</span></div>
  <ol>
    <li>Reserve a training window measured in <b>bars of the selected timeframe</b>, auto-fitted to the sample.</li>
    <li>At each step, sweep <b>only the trailing training window</b>, then trade the next step's unseen bars.</li>
    <li>Roll forward and stitch the segments. <b>Only out-of-sample bars are scored, counted and plotted.</b></li>
    <li>Permutation starts at <code>start_index = train_bars</code>; the <b>re-optimization runs inside every permutation</b>.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><div class="h4">Strengths</div><ul>
      <li>Honest out-of-sample estimate</li>
      <li>Adapts as regimes change</li>
      <li>Tests the <b>procedure</b> you would deploy</li>
      <li>Selection bias reduced and still measured</li>
    </ul></div>
    <div class="pd kill"><div class="h4">Weaknesses</div><ul>
      <li>Bias reduced, not removed</li>
      <li>Parameter churn adds turnover</li>
      <li>One full sweep per step, per permutation</li>
      <li>Shorter traded sample → wider uncertainty</li>
    </ul></div>
  </div>
  <div class="ask">The strongest of the four tests: does the adaptive procedure beat noise re-optimized the same way?</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">3 · Decision Tree</span><span class="tagline">NON-LINEAR MOMENTUM · SUPERVISED</span></div>
  <ol>
    <li>Three momentum features whose spans <b>scale with the bar size</b> (hourly 6/24/168, daily 1/5/21, monthly 1/3/12, yearly 1/2/3).</li>
    <li>Label: sign of the next <b>horizon</b> bars, mapped to {0, 1}; incomplete rows dropped.</li>
    <li><code>DecisionTreeClassifier</code> with your leaf/depth caps; leaf size is clamped to the usable rows.</li>
    <li>Predict 1 → long, 0 → short; warm-up bars stay <b>flat (NaN)</b>, never an accidental short.</li>
    <li>MCPT <b>retrains a fresh tree on every permuted universe</b>.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><div class="h4">Strengths</div><ul>
      <li>Learns non-linear interactions</li>
      <li>Inspectable feature importances</li>
      <li>No distributional assumptions</li>
      <li>Depth/leaf caps are a direct overfitting dial</li>
    </ul></div>
    <div class="pd kill"><div class="h4">Weaknesses</div><ul>
      <li><b>Enormous overfit capacity</b></li>
      <li>Overlapping labels inflate apparent skill</li>
      <li>Label horizon ≠ scoring horizon (1 bar)</li>
      <li>Very unstable; drift imbalances classes</li>
    </ul></div>
  </div>
  <div class="ask">Can the tree find equally convincing structure in bars with no memory at all?</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">4 · MA Crossover</span><span class="tagline">TREND FILTER · LONG/FLAT (OPTIONAL SHORT)</span></div>
  <ol>
    <li>Fast and slow SMA of close, defaults scaled to the bar size.</li>
    <li>Fast above slow → <b>long</b>; otherwise <b>flat</b>, or <b>short</b> if enabled.</li>
    <li>No optimization — MCPT tests the <b>raw rule</b>. Fast ≥ slow is auto-corrected to slow − 1.</li>
    <li>Turns only at crossovers, so turnover is low.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><div class="h4">Strengths</div><ul>
      <li>Few degrees of freedom → little mining bias</li>
      <li>Low turnover, robust to costs</li>
      <li>Long/flat sidesteps borrow risk</li>
      <li>Easy to explain</li>
    </ul></div>
    <div class="pd kill"><div class="h4">Weaknesses</div><ul>
      <li>Lags every turn</li>
      <li>Whipsaws in chop; flat periods earn nothing</li>
      <li>Long/flat is partly a bet on positive drift</li>
      <li>Still sensitive to the chosen pair</li>
    </ul></div>
  </div>
  <div class="ask">With no optimizer in the loop, significance here is the cleanest evidence of a real trend pattern.</div>
</div>
""", unsafe_allow_html=True)

    sec("E", "Execution — what a permutation test does not price")
    st.markdown(
        """
<div class="callout"><b>What the test does not price.</b> A permutation test prices <b>data-mining bias</b>: how often noise beats your pipeline. It does not tell you whether the edge survives <b>execution</b>. Every figure here is gross, close-to-close, and assumes a fill at the signal bar's close. The blocks below name what stands between that and a real fill; the table after them shows how each one bears on the p-value.</div>

<div class="strat-block">
  <div class="hd"><span class="nm">1 · Transaction cost analysis (TCA)</span><span class="tagline">PRE-TRADE · INTRA-TRADE · POST-TRADE</span></div>
  <ol>
    <li><b>Implementation shortfall</b> — the paper result at the <b>decision price</b> minus the result you actually realise once the order is filled. In a bar backtest the decision price is the signal bar's close.</li>
    <li><b>Explicit costs</b> — commissions, exchange and clearing fees, taxes. <b>Implicit costs</b> — the spread, the delay between signal and order, market impact, and the <b>opportunity cost</b> of orders that never fill.</li>
    <li><b>Benchmarks</b> — the <b>arrival price</b> (the mid when the order reaches the market), <b>VWAP / TWAP</b> over the execution window, or the <b>close</b>. Pick the one that matches the decision being judged.</li>
    <li><b>Three stages</b> — <b>pre-trade</b> estimates cost before sending, <b>intra-trade</b> adapts the pace to live conditions, <b>post-trade</b> compares fills with the benchmark and feeds the estimate back.</li>
  </ol>
  <div class="ask">Every trade pays twice, in and out. Compare the cost of a round trip with the average gain per trade before trusting a profit factor.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">2 · Information leakage</span><span class="tagline">INTENT · FOOTPRINT · CROWDED LEVELS</span></div>
  <ol>
    <li><b>Leakage</b> is any way your intent becomes visible before you finish trading: repeated child-order sizes, regular timing, a resting order at an obvious level, or a venue that exposes your flow.</li>
    <li>Others who infer it trade ahead of you or step away, so price drifts against you between arrival and fill. The cost shows up as <b>shortfall</b>, not as a fee.</li>
    <li>Breakouts are especially exposed: the trendline is a public level, stops and entries cluster just beyond it, and many participants are watching the same trigger.</li>
    <li><b>Controls</b> — randomise slice size and timing, cap <b>participation</b>, use passive or midpoint orders when urgency allows, and avoid leaving an identical footprint trade after trade.</li>
  </ol>
  <div class="ask">Leakage feeds adverse selection: the more visible your intent, the more counterparties can choose when to trade against you.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">3 · Market impact</span><span class="tagline">TEMPORARY · PERMANENT</span></div>
  <ol>
    <li><b>Market impact</b> is how far your own trading moves the price against you. It grows with size, and it is absent from any backtest that fills at the close.</li>
    <li>Rule of thumb: impact grows with the <b>square root</b> of order size relative to volume and scales with volatility — <code>impact ≈ Y · σ · √(Q / ADV)</code>, where <code>Y</code> is a market-specific constant of order one. Calibrate it on your own fills; treat it as a shape, not a quote.</li>
    <li>It splits into a part that fades and a part that stays.</li>
    <li>Breakouts fire when σ is high and the book is thin, so impact is at its worst exactly when the signal appears.</li>
  </ol>
  <div class="pd-grid">
    <div class="pd keep"><div class="h4">Temporary impact — reverts</div><ul>
      <li>The price concession paid for consuming liquidity <b>now</b></li>
      <li>Rises with the <b>speed</b> of trading (participation rate)</li>
      <li>Fades once you stop, as the book refills</li>
      <li>Lowered by trading slower, at the price of more delay and leakage</li>
    </ul></div>
    <div class="pd kill"><div class="h4">Permanent impact — stays</div><ul>
      <li>A new price level that reflects the <b>information</b> your order revealed</li>
      <li>Rises with the <b>size</b> traded; largely independent of how you slice it</li>
      <li>Does not revert; it becomes part of the next fair price</li>
      <li>Avoided only by not revealing the information in the first place</li>
    </ul></div>
  </div>
  <div class="ask">Temporary impact is a cost of urgency; permanent impact is a cost of size and information.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">4 · ADV — average daily volume</span><span class="tagline">PARTICIPATION = ORDER SIZE ÷ ADV</span></div>
  <ol>
    <li><b>ADV</b> is the mean traded volume per day over a trailing window, commonly 20 to 30 sessions, in shares, contracts or coins.</li>
    <li><b>Participation rate</b> is order size ÷ ADV for a whole-day view, or order size ÷ interval volume for intraday. It is the input to every impact estimate above.</li>
    <li><b>Uses</b> — a <b>capacity cap</b> (the largest size that still clears costs), a <b>liquidity screen</b> for which markets to trade, and the denominator in the √(Q / ADV) rule.</li>
    <li>None of the four pipelines reads volume, and the loader ignores that column, so this app cannot size participation for you. Take ADV from your own market data and apply it to the strategy's real order size.</li>
  </ol>
  <div class="ask">ADV is a trailing average. On a liquidity event it overstates what the market can absorb that day.</div>
</div>
""", unsafe_allow_html=True)

    sec("X", "Market structure — order book, auctions, sessions, liquidity events")
    st.markdown(
        """
<div class="strat-block">
  <div class="hd"><span class="nm">5 · Order book</span><span class="tagline">BID · ASK · DEPTH · QUEUE</span></div>
  <ol>
    <li>A <b>limit order book</b> lists resting buy orders (bids) and sell orders (asks) by price; orders at the same price fill by arrival time. <b>Spread</b> = best ask − best bid. <b>Depth</b> = the size resting at each level.</li>
    <li>A <b>market order</b> walks the book: it fills level by level, so the average price worsens with size. That worsening against the arrival mid is <b>slippage</b>.</li>
    <li>A buy <b>stop</b> placed above a trendline becomes a market order the moment price trades through it, so it demands liquidity from the side of the book that is thinning.</li>
    <li>A passive <b>limit</b> order earns the spread but is most likely to fill when the breakout fails and price trades back through it — <b>adverse selection</b>.</li>
  </ol>
  <div class="ask">OHLCV bars carry no book data. Spread and depth are invisible here; measure them with level-1 or level-2 data.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">6 · Open and close auctions</span><span class="tagline">SINGLE-PRICE CALLS · BENCHMARK PRICES</span></div>
  <ol>
    <li>In an <b>auction</b>, orders accumulate during a call period and cross at one price that maximises matched volume. Most exchanges run one at the <b>open</b> and one at the <b>close</b>.</li>
    <li>The opening auction does the price discovery for the overnight gap. The closing auction sets the official close used by index funds, fund valuation and market-on-close orders, and often draws a large share of the day's volume.</li>
    <li>Every pipeline here sets its position from the <b>close</b> and earns the <b>next bar's</b> close-to-close return, which assumes a fill at that close. In real trading you only know the close after the closing cross, so the realistic fill is the next open or the next continuous session. That gap is delay cost, and for daily bars it can dominate the other cost terms.</li>
    <li>24/7 markets such as crypto have no auctions, so a daily close is a timestamp convention. Liquidity still follows the clock, as the next block shows.</li>
  </ol>
  <div class="ask">If a rule decides at the close, also test it with fills at the next open. The drop is the delay cost.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">7 · Continuous trading sessions</span><span class="tagline">BETWEEN THE AUCTIONS</span></div>
  <ol>
    <li>Between the auctions, <b>continuous trading</b> matches orders as they arrive, by price then time, with a live spread and live depth.</li>
    <li>Volume and spreads follow an intraday pattern: typically heavy and wide around the open, thinner midday, and heavy again into the close. Pre-market and post-market sessions are thinner still.</li>
    <li>Hourly bars therefore straddle different regimes. A breakout flagged in a midday bar fills into a thinner book than one at the open: same signal, different cost.</li>
    <li><b>Control</b> — schedule execution to the volume profile (VWAP-style) instead of a clock time, and prefer liquid windows for size.</li>
  </ol>
  <div class="ask">The same breakout signal can carry two very different costs depending on where in the session it fires.</div>
</div>

<div class="strat-block">
  <div class="hd"><span class="nm">8 · Liquidity events</span><span class="tagline">WHEN THE BOOK THINS</span></div>
  <ol>
    <li><b>Scheduled</b> — earnings and macro releases, index rebalances, futures and options expiries, auction imbalances. <b>Unscheduled</b> — news shocks, trading halts and circuit breakers, flash moves, venue outages, liquidation cascades in crypto.</li>
    <li>In these windows liquidity providers widen spreads or pull quotes: depth falls, the spread rises, and temporary impact jumps for the same order size.</li>
    <li>Breakouts often coincide with liquidity events, since a break of a level is frequently the news itself. The cost on a breakout is therefore likely higher than the all-weather average.</li>
    <li><b>Controls</b> — size down or stand aside around scheduled events, and cost the real run's worst stretches (its deepest drawdown) at stressed spreads rather than at the average.</li>
  </ol>
  <div class="ask">Liquidity is cheapest when you do not need it and dearest when you do.</div>
</div>
""", unsafe_allow_html=True)

    sec("W", "How each concept bears on a permutation test")
    st.table(pd.DataFrame({
        "Effect on the test": [
            "The test is gross. A small p-value says the edge is unlikely to be noise, not that it is larger than the cost of capturing it.",
            "Permutation removes serial structure, including any footprint other traders' flow leaves. Your own orders never feed back into prices, in the real run or in the null.",
            "The real run and every permutation assume you are a price-taker, so the p-value carries no information about capacity.",
            "Bar permutation shuffles intrabar shapes and gaps separately and across time, so hourly bars lose their time-of-day pattern. Gaps, where the opening auction does its price discovery, keep their distribution but are detached from the bars they belong to.",
            "Permutation destroys volatility clustering, so the null contains no stress regimes. Real liquidity events cluster, and execution costs peak inside them.",
        ],
        "What to do": [
            "Charge a per-trade cost to the real run and to every permutation, re-run, and report net PF beside gross.",
            "Treat the result as a signal-quality test. Slice size, timing and venue are execution choices it cannot see.",
            "Estimate impact with the square-root rule (impact ≈ Y · σ · √(Q / ADV)) and compare it with the per-trade edge at the size you intend to run.",
            "A rule that leans on session effects is tested against a null without them. For rules that decide at the close, also run a daily-bar check.",
            "Look at the real run's deepest drawdown stretch and ask what it would have cost to trade through it.",
        ],
    }, index=["Transaction costs", "Information leakage", "Market impact and ADV",
               "Auctions and sessions", "Order book and liquidity events"]))

    sec("S", "Recommended settings by bar size")
    st.table(pd.DataFrame(
        {
            "Lookback grid": ["12–168", "5–60", "3–24", "2–5"],
            "Training window": ["35,040 bars (≈4 y)", "365 bars (≈1 y)", "36 bars (3 y)", "5 bars"],
            "Re-optimize every": ["720 bars (≈30 d)", "30 bars", "3 bars", "1 bar"],
            "MA fast / slow": ["10 / 30", "5 / 20", "3 / 12", "2 / 3"],
            "Tree spans / horizon": ["6·24·168 / 24", "1·5·21 / 5", "1·3·12 / 3", "1·2·3 / 1"],
            "Typical use": ["reference setting", "default recommendation", "regime questions", "illustration only"],
        },
        index=["Hourly", "Daily", "Monthly", "Yearly"]))
    st.caption("Defaults load automatically when you move the Timeframe slider; every value is clamped to the bars you actually have.")

    sec("G", "Reading the lab")
    st.markdown(
        """
<div class="panel">
<div class="h3">Workflow</div>
<p><b style="color:var(--text)">1.</b> Load Currency 1. <b style="color:var(--text)">2.</b> Pick a bar size.
<b style="color:var(--text)">3.</b> Pick a pipeline; windows auto-fit. <b style="color:var(--text)">4.</b> Set permutations and run.</p>
<p><b style="color:var(--btc2)">PF histogram</b> — the amber line is your real result, the red tail everything that beat it.
<b style="color:var(--btc2)">Equity</b> — thick amber real curve on top of the cyan noise bundle, its median and 10–90% band; inside the band means indistinguishable from noise.
<b style="color:var(--btc2)">PF vs lookback</b> — the vertical gap above the noise envelope is the evidence.</p>
<p><b style="color:var(--green)">p ≤ 0.05</b> edge · <b style="color:var(--btc2)">0.05 &lt; p ≤ 0.20</b> inconclusive · <b style="color:var(--red)">p &gt; 0.20</b> noise.
The floor is 1/(N+1) — ≈0.001 at 1000 permutations.</p>
</div>
<div class="callout"><b>Limitations.</b> Gross of fees, slippage and funding. Profit factor ignores drawdown and path risk.
Very coarse bar sizes produce very few bars — the test still runs, but treat such results as indicative.
MCPT prices the procedure you actually ran; if you tried several pipelines and only tested the winner, the honest N is larger.</div>
""", unsafe_allow_html=True)

    sec("T", "Troubleshooting")
    st.table(pd.DataFrame(
        {
            "Likely cause": [
                "Coarse bar size, or a long MA/lookback on a short sample",
                "Feature spans plus label horizon exceed the rows available",
                "Date range or bar size leaves almost nothing to trade",
                "Selected years sit beyond the end of the loaded data",
                "Missing / misnamed OHLC columns, or a non-datetime first column",
                "Fewer than 8 usable rows after cleaning",
                "File larger than the upload cap",
                "Walk-forward with a wide grid on hourly bars",
                "Two currencies with different timestamps or timezones",
            ],
            "What to do": [
                "Pick a finer bar size, widen the range, lower the grid, or allow shorts",
                "Use daily or hourly bars, or widen the date range",
                "Widen the years; the sidebar shows the auto-fitted train / step / OOS counts",
                "The sidebar states the feed's last year — drag the end back to it, or load newer data",
                "Match the schema in section D; extra columns are fine, names are lower-cased",
                "Check for non-positive prices, NaNs and duplicate timestamps",
                "Resample to daily before upload, or split by year (max 200 MB per file)",
                "Keep Turbo on, raise the grid step, or reduce permutations for a first pass",
                "Convert both to UTC and align; only shared timestamps are used",
            ],
        },
        index=[
            "Strategy never traded",
            "Too few usable rows for the tree",
            "Too few bars at this timeframe",
            "No bars in the selected years",
            "Missing OHLC columns",
            "File too short after cleaning",
            "Upload rejected",
            "Run feels slow",
            "Correlation panel is empty",
        ]))
    st.caption("Any of these conditions raises a single dismissible popup — the page itself stays clean.")

    sec("F", "Frequently asked questions")
    st.markdown(
        """
<div class="panel">
<div class="h4">Does a low p-value prove the strategy is profitable?</div>
<p>No. It says the pipeline's result is unlikely to be reproduced by the same pipeline on bars with no serial
structure. Costs, capacity, slippage, funding and regime change are all outside the test.</p>
<div class="h4">Does a high p-value prove there is no edge?</div>
<p>No. It says this procedure, on this sample, at this bar size, cannot distinguish itself from shuffled noise.
A different horizon or a cost-aware variant may behave differently.</p>
<div class="h4">Why 1000 permutations?</div>
<p>Resolution. The smallest attainable p-value is 1/(N+1): 0.02 at 50, 0.005 at 200, ≈0.001 at 1000. Monte Carlo error
on p shrinks roughly with 1/√N, so 1000 is the point where a 0.05 verdict becomes stable.</p>
<div class="h4">What do the year sliders include?</div>
<p>Both ends. They run from the first year in your data to the current calendar year, and the default selection covers
that whole span. Years past the end of your feed simply contain no bars — the sidebar states the feed's last year.</p>
<div class="h4">Why is walk-forward so much slower?</div>
<p>It re-runs the full lookback sweep at every step, inside every permutation. Cost scales with
steps × grid size × permutations. Turbo, a coarser grid step, or daily bars are the three effective levers.</p>
<div class="h4">Are the seeds really fixed?</div>
<p>Yes — tests use seed 69, the anatomy draw uses 4242, and the reference pair uses 7. Identical inputs give identical
outputs, which is what makes a reported result checkable.</p>
<div class="h4">Can I trust the reference CSVs as market data?</div>
<p>No. They are a seeded synthetic pair designed to behave like crypto, provided so the workflow can be exercised and
reproduced offline. Use exchange history for any claim about the real market.</p>
<div class="h4">What exactly does the Turbo toggle change?</div>
<p>Nothing mathematically. It replaces the pandas loops with suffix-extreme matrices and segment cumulative sums.
Turn it off to run the literal reference implementation and confirm the numbers agree.</p>
<div class="h4">Why is my equity curve flat at the start?</div>
<p>It should not be for walk-forward: only out-of-sample bars are drawn. For the other pipelines the warm-up period is
flat by design — the channel, the MAs and the tree features need history before the first position.</p>
</div>
""", unsafe_allow_html=True)

    sec("V", "Glossary")
    st.table(pd.DataFrame(
        {"Meaning": [
            "Gross profit divided by gross loss; scale-free, ignores path and drawdown.",
            "(1 + #permuted PF ≥ real PF) / (N + 1); the real run is one observation.",
            "One shuffled universe: same bars, same distribution, no serial dependence.",
            "The distribution of the best PF the pipeline achieves on noise.",
            "Bars used to choose parameters; never scored in walk-forward.",
            "Bars traded with parameters chosen before them; the only bars walk-forward scores.",
            "Mean / standard deviation of per-bar returns; annualised by √(bars per year).",
            "Peak-to-trough fall in cumulative log return.",
            "Correlation of a series with its own lag; |returns| reveals volatility clustering.",
            "Var(q-bar returns) / (q · Var(1-bar returns)); >1 trend-like, <1 mean-reverting.",
            "Transaction cost analysis: estimating and measuring execution cost before, during and after a trade.",
            "Realised result versus the paper result at the decision price: delay, spread, impact, fees and missed fills.",
            "Mid price when the order reaches the market; the usual benchmark for slippage.",
            "Average daily volume: mean traded volume per day over a trailing window.",
            "Order size divided by the volume traded in the same window (ADV for a daily view).",
            "Price concession for consuming liquidity now; fades once trading stops.",
            "Lasting price shift from the information your order reveals; does not revert.",
            "Your intent becoming visible before you finish trading, so others trade ahead of it.",
            "Best ask minus best bid / the size resting at each price level.",
            "Orders accumulate and cross at one price; run at the open and at the close.",
        ]},
        index=["Profit factor", "MCPT p-value", "Permutation", "Null distribution",
               "Training window", "Out-of-sample", "Sharpe", "Max drawdown",
               "ACF", "Variance ratio",
               "TCA", "Implementation shortfall", "Arrival price", "ADV", "Participation rate", "Temporary impact", "Permanent impact", "Information leakage", "Spread / depth", "Auction (call)"]))

    sec("K", "Before you report a result")
    st.markdown(
        """
<div class="panel">
<ul>
  <li><b>State the feed.</b> File name or reference pair, date range, bar size, bar count.</li>
  <li><b>State the procedure.</b> Pipeline, grid, training window and step, tree caps or MA pair.</li>
  <li><b>State N and p.</b> Permutations completed, real PF, noise median, p-value.</li>
  <li><b>Show the null.</b> Include the histogram and the equity panel, not just the headline number.</li>
  <li><b>Report the neighbours.</b> One adjacent bar size and one adjacent grid setting.</li>
  <li><b>Declare costs and capacity.</b> Every figure here is gross and assumes a price-taker; say so, and state the per-trade cost and order size ÷ ADV you would assume.</li>
</ul>
<p>The results JSON exported from the Lab already carries the full configuration and every permuted profit factor —
attach it, and the run can be reproduced exactly.</p>
</div>
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
        st.caption("That file is not present in this deployment.")
    st.markdown(
        '<div class="callout"><b>Run it:</b> <code>pip install -r requirements.txt</code> → <code>streamlit run app.py</code>. '
        'The standalone scripts remain the classic hourly tqdm + matplotlib workflow.</div>',
        unsafe_allow_html=True)

st.markdown(
    f'<div class="foot">PERMUTATION TESTS · TRADING STRATEGIES · BUILT BY <b>{AUTHOR.upper()}</b> · '
    'DISTRIBUTION SURVIVES THE SHUFFLE — DEPENDENCE DOES NOT</div>',
    unsafe_allow_html=True,
)

flush_popup()
