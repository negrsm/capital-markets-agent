"""
dashboard.py — Professional Streamlit dashboard for the AI Capital Markets Decision Agent.

Run from the project root:
    streamlit run app/dashboard.py
"""

import math
import os
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from matplotlib.patches import Patch

# ── make src/ importable regardless of launch directory ──────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# ── resolve OpenAI API key before importing agents (which creates the LLM) ───
# Priority: st.secrets (Streamlit Cloud) → os.getenv / .env (local dev)
try:
    _openai_key = st.secrets["OPENAI_API_KEY"]
except Exception:
    from dotenv import load_dotenv
    load_dotenv()
    _openai_key = os.getenv("OPENAI_API_KEY", "")
if _openai_key:
    os.environ["OPENAI_API_KEY"] = _openai_key

from agents import build_graph, MarketState          # streaming execution
from data_loader import DEFAULT_TICKERS, download_data, get_last_updated
from graph import run_pipeline                        # fallback / import as requested

# ── design tokens ─────────────────────────────────────────────────────────────
_NAVY       = "#003F8A"
_BLUE       = "#0056A2"
_LIGHT_BLUE = "#E8F0F9"

_RECO_STYLE = {
    "Overweight":  ("background-color:#d4edda", "color:#155724"),
    "Neutral":     ("background-color:#fff3cd", "color:#856404"),
    "Underweight": ("background-color:#ffe5cc", "color:#7d4f00"),
    "Avoid":       ("background-color:#f8d7da", "color:#721c24"),
}

# Maps LangGraph node name → (progress %, status message)
_NODE_STAGES = {
    "market_data_agent": (15, "📥  Loading market data..."),
    "metrics_agent":     (30, "📊  Computing financial metrics..."),
    "forecasting_agent": (52, "🤖  Training ML forecasting models..."),
    "sentiment_agent":   (68, "🧠  Running AI sentiment analysis..."),
    "decision_agent":    (82, "📋  Scoring portfolio decisions..."),
    "risk_agent":        (90, "⚠️  Evaluating risk flags..."),
    "report_agent":      (97, "📝  Generating executive report..."),
}

_PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# ── page config — must be the first Streamlit call ───────────────────────────
st.set_page_config(
    page_title="AI Capital Markets Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
    <style>
    /* ── sidebar background + text ── */
    [data-testid="stSidebar"] > div:first-child {{
        background-color: {_NAVY};
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] .stButton > button {{
        background-color: #ffffff !important;
        color: {_NAVY} !important;
        font-weight: 700;
        width: 100%;
        border-radius: 6px;
        border: none;
        padding: 0.55rem 1rem;
        margin-top: 0.4rem;
        font-size: 0.95rem;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {_LIGHT_BLUE} !important;
    }}
    /* ── metric cards ── */
    .metric-card {{
        background: {_LIGHT_BLUE};
        border-left: 5px solid {_NAVY};
        border-radius: 6px;
        padding: 1rem 1.3rem;
        min-height: 88px;
    }}
    .metric-label {{
        color: #555;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        margin-bottom: 0.35rem;
    }}
    .metric-value {{
        color: {_NAVY};
        font-size: 2.1rem;
        font-weight: 700;
        line-height: 1.1;
    }}
    /* ── executive summary box ── */
    .report-box {{
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-left: 5px solid {_BLUE};
        border-radius: 6px;
        padding: 1.5rem 1.75rem;
        font-size: 0.97rem;
        line-height: 1.9;
        color: #222;
    }}
    /* ── risk warning box ── */
    .risk-box {{
        background: #fff8e1;
        border-left: 5px solid #ffa000;
        border-radius: 6px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.55rem;
        color: #5d4037;
        font-size: 0.93rem;
    }}
    /* ── section headers inside tabs ── */
    .section-head {{
        color: {_NAVY};
        font-size: 1.05rem;
        font-weight: 700;
        border-bottom: 2px solid {_BLUE};
        padding-bottom: 0.28rem;
        margin-top: 1.2rem;
        margin-bottom: 0.75rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── session state defaults ────────────────────────────────────────────────────
for _k, _v in [("result", None), ("last_run", None), ("run_error", None)]:
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ════════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── branding ──
    st.markdown(
        "<h2 style='color:#fff; margin-bottom:0.15rem; line-height:1.25;'>"
        "📊 AI Capital<br>Markets Agent"
        "</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='font-size:0.82rem; opacity:0.78; margin-top:0.25rem;'>"
        "Multi-agent portfolio analysis powered by LangGraph, "
        "Random Forest ML, and OpenAI GPT-4o-mini."
        "</p>",
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ── data settings ──
    st.markdown(
        "<p style='font-size:0.88rem; font-weight:700; letter-spacing:0.5px; margin-bottom:0.4rem;'>"
        "⚙️  DATA SETTINGS"
        "</p>",
        unsafe_allow_html=True,
    )

    fresh_data = st.checkbox(
        "Download Fresh Data",
        value=False,
        help="Re-download all tickers from Yahoo Finance before running the pipeline.",
    )

    # Show last-updated dates from local CSVs
    try:
        last_updated = get_last_updated()
        non_null     = {k: v for k, v in last_updated.items() if v is not None}
        if non_null:
            oldest = min(non_null.values())
            newest = max(non_null.values())
            st.markdown(
                f"<p style='font-size:0.78rem; opacity:0.72; margin:0.3rem 0 0;'>"
                f"Local data: {oldest} → {newest}"
                f"</p>",
                unsafe_allow_html=True,
            )
            missing = [t for t in DEFAULT_TICKERS if t not in non_null]
            if missing:
                st.markdown(
                    f"<p style='font-size:0.78rem; color:#FFD54F; margin-top:0.2rem;'>"
                    f"⚠  Missing: {', '.join(missing)}"
                    f"</p>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                "<p style='font-size:0.78rem; color:#FFD54F; margin-top:0.3rem;'>"
                "⚠  No local data found. Enable Fresh Data above."
                "</p>",
                unsafe_allow_html=True,
            )
    except Exception:
        pass  # suppress any import-time errors before data exists

    st.markdown("---")

    run_btn = st.button("▶  Run Full Analysis", use_container_width=True)

    # ── post-run summary ──
    if st.session_state.last_run:
        st.markdown("---")
        st.markdown(
            f"<p style='font-size:0.82rem; font-weight:700; margin-bottom:0.1rem;'>"
            f"✅  Last run</p>"
            f"<p style='font-size:0.82rem; opacity:0.82; margin:0;'>"
            f"{st.session_state.last_run.strftime('%Y-%m-%d  %H:%M:%S')}"
            f"</p>",
            unsafe_allow_html=True,
        )
        if st.session_state.result:
            dec = st.session_state.result.get("decisions", {})
            _icons = {
                "Overweight": "▲", "Neutral": "─",
                "Underweight": "▼", "Avoid": "✕",
            }
            st.markdown(
                "<p style='font-size:0.82rem; font-weight:700; margin:0.5rem 0 0.2rem;'>"
                "Tickers analysed</p>",
                unsafe_allow_html=True,
            )
            for t in sorted(dec):
                action = dec[t].get("action", "")
                st.markdown(
                    f"<p style='font-size:0.80rem; margin:0.04rem 0;'>"
                    f"&nbsp;&nbsp;{_icons.get(action, '·')} {t}  —  {action}"
                    f"</p>",
                    unsafe_allow_html=True,
                )

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.73rem; opacity:0.5; line-height:1.7;'>"
        "LangGraph · scikit-learn<br>"
        "OpenAI GPT-4o-mini · yfinance"
        "</p>",
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════════════════════
# PAGE HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(
    f"<h1 style='color:{_NAVY}; margin-bottom:0; font-size:2rem; font-weight:800;'>"
    "📊  AI Capital Markets Decision Agent"
    "</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:#666; font-size:0.9rem; margin-top:0.2rem; margin-bottom:1.1rem;'>"
    "LangGraph multi-agent pipeline  ·  "
    "Random Forest forecasting  ·  "
    "OpenAI GPT-4o-mini sentiment analysis"
    "</p>",
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════════════════════
# PIPELINE EXECUTION
# ════════════════════════════════════════════════════════════════════════════════
if run_btn:
    prog_slot   = st.empty()
    status_slot = st.empty()

    try:
        prog_slot.progress(0)
        status_slot.info("🚀  Initializing pipeline...")

        # Optional fresh download before pipeline runs
        if fresh_data:
            status_slot.info("📥  Downloading fresh market data from Yahoo Finance...")
            prog_slot.progress(5)
            download_data(tickers=list(DEFAULT_TICKERS), refresh=True)

        # Build graph and run with per-node streaming so progress updates are real
        graph = build_graph()
        initial_state: MarketState = {
            "tickers":    list(DEFAULT_TICKERS),
            "price_data": {},
            "metrics":    {},
            "forecasts":  {},
            "sentiment":  {},
            "decisions":  {},
            "risk_flags": {},
            "report":     "",
            "timestamp":  "",
        }

        final_state = dict(initial_state)

        try:
            # stream_mode="updates" yields {node_name: partial_state} per node,
            # allowing the progress bar to update between each agent's execution.
            for event in graph.stream(initial_state, stream_mode="updates"):
                for node_name, partial in event.items():
                    pct, msg = _NODE_STAGES.get(node_name, (95, "⏳  Processing..."))
                    prog_slot.progress(pct)
                    status_slot.info(msg)
                    final_state.update(partial)

        except TypeError:
            # Older LangGraph without stream_mode kwarg — fall back to invoke
            status_slot.info("⏳  Running full pipeline (this may take ~30 s)...")
            prog_slot.progress(20)
            final_state = run_pipeline(list(DEFAULT_TICKERS))

        prog_slot.progress(100)
        status_slot.success("✅  Analysis complete!")

        st.session_state.result    = final_state
        st.session_state.last_run  = datetime.now()
        st.session_state.run_error = None

        time.sleep(0.8)
        prog_slot.empty()
        status_slot.empty()
        st.rerun()

    except Exception as exc:
        st.session_state.run_error = str(exc)
        prog_slot.empty()
        status_slot.error(f"❌  Pipeline failed: {exc}")

# Persist any error from the previous run
if st.session_state.run_error and not run_btn:
    st.error(f"❌  Last run failed: {st.session_state.run_error}")

# ════════════════════════════════════════════════════════════════════════════════
# GATE: show friendly placeholder if pipeline hasn't run yet
# ════════════════════════════════════════════════════════════════════════════════
if st.session_state.result is None:
    st.info(
        "👈  **Click 'Run Full Analysis'** in the sidebar to start.\n\n"
        "The pipeline will:\n"
        "1. Load market data for 8 sector ETFs (SPY, QQQ, XLF, XLE, XLV, TLT, GLD, UUP)\n"
        "2. Compute financial metrics — Sharpe ratio, max drawdown, beta, and more\n"
        "3. Train per-ticker Random Forest models for next-week return forecasting\n"
        "4. Classify market sentiment with OpenAI GPT-4o-mini\n"
        "5. Score each ticker against a 7-factor portfolio rubric\n"
        "6. Identify concentration and volatility risk flags\n"
        "7. Generate an AI-written executive summary (~200 words)"
    )
    st.stop()

# ════════════════════════════════════════════════════════════════════════════════
# UNPACK STATE
# ════════════════════════════════════════════════════════════════════════════════
_r          = st.session_state.result
price_data: dict = _r.get("price_data", {})
metrics:    dict = _r.get("metrics",    {})
forecasts:  dict = _r.get("forecasts",  {})
sentiment:  dict = _r.get("sentiment",  {})
decisions:  dict = _r.get("decisions",  {})
risk_flags: dict = _r.get("risk_flags", {})
report:     str  = _r.get("report",     "")
timestamp:  str  = _r.get("timestamp",  "")

# ── build recommendations DataFrame ──────────────────────────────────────────
def _reco_df() -> pd.DataFrame:
    rows = []
    for ticker in sorted(decisions):
        d    = decisions[ticker]
        m    = metrics.get(ticker, {})
        prob = forecasts.get(ticker, float("nan"))
        rows.append({
            "Ticker":          ticker,
            "Recommendation":  d.get("action", "N/A"),
            "Score":           d.get("score", 0),
            "Sentiment":       sentiment.get(ticker, "N/A"),
            "Forecast Prob":   prob,
            "Sharpe Ratio":    m.get("sharpe_ratio",   float("nan")),
            "Ann. Return":     m.get("ann_return",     float("nan")),
            "Max Drawdown":    m.get("max_drawdown",   float("nan")),
        })
    return pd.DataFrame(rows).set_index("Ticker")


def _style_reco(df: pd.DataFrame):
    def _row(row):
        bg, fg = _RECO_STYLE.get(row["Recommendation"], ("", ""))
        return [f"{bg}; {fg}"] * len(row)
    return (
        df.style
        .apply(_row, axis=1)
        .format({
            "Score":          "{:.0f}/7",
            "Forecast Prob":  "{:.1%}",
            "Sharpe Ratio":   "{:.2f}",
            "Ann. Return":    "{:.1%}",
            "Max Drawdown":   "{:.1%}",
        }, na_rep="N/A")
    )


reco_df = _reco_df()

# ── daily return series (used by Charts tab) ──────────────────────────────────
_close_series = {
    t: df["Close"].pct_change().dropna()
    for t, df in price_data.items()
    if df is not None and "Close" in df.columns
}
returns_df = pd.DataFrame(_close_series).dropna() if _close_series else pd.DataFrame()

# ════════════════════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════════════════════
tab_ov, tab_ch, tab_rpt, tab_raw = st.tabs([
    "📋  Overview",
    "📊  Charts",
    "📝  AI Report",
    "🗂  Raw Data",
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════════════════════════════════════════
with tab_ov:
    # ── summary metric cards ──────────────────────────────────────────────────
    n_tickers    = len(decisions)
    n_overweight = sum(1 for d in decisions.values() if d.get("action") == "Overweight")
    n_avoid      = sum(1 for d in decisions.values() if d.get("action") == "Avoid")
    n_flags      = len(risk_flags)

    def _card(col, label: str, value, color: str = _NAVY) -> None:
        col.markdown(
            f'<div class="metric-card" style="border-left-color:{color}">'
            f'<div class="metric-label">{label}</div>'
            f'<div class="metric-value" style="color:{color}">{value}</div>'
            f"</div>",
            unsafe_allow_html=True,
        )

    c1, c2, c3, c4 = st.columns(4)
    _card(c1, "Tickers Analysed",  n_tickers)
    _card(c2, "Overweight",        n_overweight, "#155724")
    _card(c3, "Avoid",             n_avoid,      "#721c24")
    _card(c4, "Risk Flags",        n_flags,      "#856404" if n_flags else _NAVY)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── recommendations table ─────────────────────────────────────────────────
    st.markdown(
        '<p class="section-head">📋  Portfolio Recommendations</p>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        _style_reco(reco_df),
        use_container_width=True,
        height=min(112 + len(reco_df) * 38, 510),
    )

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHARTS
# ════════════════════════════════════════════════════════════════════════════════
with tab_ch:

    # ── Chart 1: Forecast probability bars ────────────────────────────────────
    st.markdown(
        '<p class="section-head">ML Forecast Probabilities — Next-Week Positive Return</p>',
        unsafe_allow_html=True,
    )

    if forecasts:
        fc_sorted  = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
        tfc        = [t for t, _ in fc_sorted]
        pfc        = [p for _, p in fc_sorted]
        bar_colors = [
            "#2E7D32" if p > 0.55 else ("#C62828" if p < 0.45 else "#F9A825")
            for p in pfc
        ]

        fig, ax = plt.subplots(figsize=(10, 4.2))
        x    = range(len(tfc))
        bars = ax.bar(x, pfc, color=bar_colors, edgecolor="white", linewidth=0.8, width=0.58)

        # 55% threshold line
        ax.axhline(0.55, color=_NAVY, linewidth=1.6, linestyle="--", zorder=5)
        ax.text(
            len(tfc) - 0.52, 0.562,
            "55% buy threshold",
            color=_NAVY, fontsize=8.5, ha="right", va="bottom", style="italic",
        )

        # Value labels on top of bars
        for bar, prob in zip(bars, pfc):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.013,
                f"{prob:.1%}",
                ha="center", va="bottom", fontsize=9, fontweight="600",
            )

        # x-tick labels: ticker + recommendation action
        ax.set_xticks(list(x))
        ax.set_xticklabels(
            [f"{t}\n({decisions.get(t, {}).get('action', '—')})" for t in tfc],
            fontsize=9,
        )
        ax.set_ylim(0, 1.1)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
        ax.set_ylabel("P(positive next-week return)", fontsize=9.5)
        ax.set_title(
            "ML Forecast Probabilities by Ticker",
            fontsize=12, color=_NAVY, fontweight="bold", pad=10,
        )
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
        ax.legend(
            handles=[
                Patch(facecolor="#2E7D32", label="Bullish  (> 55%)"),
                Patch(facecolor="#F9A825", label="Neutral  (45–55%)"),
                Patch(facecolor="#C62828", label="Bearish  (< 45%)"),
            ],
            fontsize=9, loc="lower left",
        )
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No forecast data available.")

    st.markdown("---")

    # ── Chart 2: Cumulative returns ────────────────────────────────────────────
    st.markdown(
        '<p class="section-head">Cumulative Returns — Full History</p>',
        unsafe_allow_html=True,
    )

    if not returns_df.empty:
        cum     = (1 + returns_df).cumprod() - 1
        palette = plt.cm.tab10.colors

        fig, ax = plt.subplots(figsize=(12, 5))
        for i, ticker in enumerate(sorted(cum.columns)):
            action = decisions.get(ticker, {}).get("action", "")
            ax.plot(
                cum.index, cum[ticker] * 100,
                label=f"{ticker}  ({action})",
                color=palette[i % len(palette)],
                linewidth=2.4 if action == "Overweight" else 1.5,
                linestyle="--" if action == "Avoid" else "-",
            )

        ax.axhline(0, color="#bbb", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_ylabel("Cumulative Return (%)", fontsize=10)
        ax.set_title(
            "Cumulative Returns — All Tickers",
            fontsize=12, color=_NAVY, fontweight="bold", pad=10,
        )
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.legend(fontsize=9, loc="upper left", ncol=2, framealpha=0.85)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No price data available.")

    st.markdown("---")

    # ── Chart 3: Correlation heatmap ───────────────────────────────────────────
    st.markdown(
        '<p class="section-head">Daily Return Correlation Heatmap</p>',
        unsafe_allow_html=True,
    )

    col_heat, _gap = st.columns([1.0, 0.22])
    with col_heat:
        if not returns_df.empty and returns_df.shape[1] > 1:
            corr = returns_df.corr()
            n    = len(corr)
            sz   = max(5.5, n * 0.72)
            fig, ax = plt.subplots(figsize=(sz, sz * 0.88))
            sns.heatmap(
                corr, ax=ax,
                annot=True, fmt=".2f",
                cmap=sns.light_palette(_BLUE, as_cmap=True),
                vmin=-1, vmax=1,
                linewidths=0.5, linecolor="#dee2e6",
                annot_kws={"size": max(7, 10 - n)},
                cbar_kws={"shrink": 0.72, "label": "Pearson ρ"},
            )
            ax.set_title(
                "Return Correlation Matrix",
                fontsize=12, color=_NAVY, fontweight="bold", pad=10,
            )
            ax.tick_params(axis="both", labelsize=9)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("Need at least 2 tickers with price data to render the heatmap.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3 — AI REPORT
# ════════════════════════════════════════════════════════════════════════════════
with tab_rpt:
    st.markdown(
        '<p class="section-head">📝  Executive Summary</p>',
        unsafe_allow_html=True,
    )
    st.caption(f"Generated at {timestamp}  ·  OpenAI GPT-4o-mini")

    if report:
        safe_report = (
            report
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n\n", "</p><p>")
            .replace("\n", "<br>")
        )
        st.markdown(
            f'<div class="report-box"><p>{safe_report}</p></div>',
            unsafe_allow_html=True,
        )
    else:
        st.info("No executive summary was generated.")

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<p class="section-head">⚠️  Risk Warnings</p>',
        unsafe_allow_html=True,
    )
    if risk_flags:
        for flag_name, message in risk_flags.items():
            label = flag_name.replace("_", " ").title()
            st.markdown(
                f'<div class="risk-box">'
                f"⚠️ <strong>{label}</strong><br>{message}"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("✅  No risk warnings identified for this portfolio configuration.")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4 — RAW DATA
# ════════════════════════════════════════════════════════════════════════════════
with tab_raw:
    # ── financial metrics table ───────────────────────────────────────────────
    st.markdown(
        '<p class="section-head">Financial Metrics</p>',
        unsafe_allow_html=True,
    )

    if metrics:
        met_df = pd.DataFrame(metrics).T.sort_index()

        _pct_cols   = [c for c in ["ann_return", "ann_volatility", "max_drawdown"] if c in met_df.columns]
        _float_cols = [c for c in ["sharpe_ratio", "beta"] if c in met_df.columns]
        _price_cols = [c for c in ["ma_50", "ma_200", "last_close"] if c in met_df.columns]

        _fmt = {}
        _fmt.update({c: "{:.2%}" for c in _pct_cols})
        _fmt.update({c: "{:.3f}" for c in _float_cols})
        _fmt.update({c: "{:.2f}" for c in _price_cols})

        st.dataframe(
            met_df.style.format(_fmt, na_rep="N/A"),
            use_container_width=True,
        )
    else:
        st.info("No metrics data available.")

    st.markdown("---")

    # ── ML forecasting results ────────────────────────────────────────────────
    st.markdown(
        '<p class="section-head">ML Forecasting Results</p>',
        unsafe_allow_html=True,
    )

    proba_path = os.path.join(_PROCESSED_DIR, "forecasting_probabilities.csv")
    if os.path.exists(proba_path):
        proba_df = pd.read_csv(proba_path)
        st.dataframe(
            proba_df.style.format({"probability": "{:.4f}"}, na_rep="N/A"),
            use_container_width=True,
            height=min(60 + len(proba_df) * 38, 380),
        )
    else:
        st.info("forecasting_probabilities.csv not found — run the pipeline first.")

    report_path = os.path.join(_PROCESSED_DIR, "forecasting_model_report.txt")
    if os.path.exists(report_path):
        with st.expander("📄  Full Model Report (accuracy, precision, recall per ticker)", expanded=False):
            with open(report_path, "r", encoding="utf-8") as fh:
                st.text(fh.read())

# ════════════════════════════════════════════════════════════════════════════════
# FOOTER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#aaa; font-size:0.8rem;'>"
    f"Analysis timestamp: {timestamp if timestamp else 'not yet run'}"
    "  &nbsp;·&nbsp;  "
    "Powered by LangGraph  ·  Random Forest  ·  OpenAI GPT-4o-mini  ·  yfinance"
    "</p>",
    unsafe_allow_html=True,
)
