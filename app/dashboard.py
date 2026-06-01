"""
dashboard.py — Professional Streamlit dashboard for the AI Capital Markets Decision Agent.

Run from the project root:
    streamlit run app/dashboard.py
"""

import os
import sys
from datetime import datetime

# Make src/ importable regardless of the working directory Streamlit is launched from
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Capital Markets Decision Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# BMO-style design tokens + global CSS
# ---------------------------------------------------------------------------

_NAVY = "#003F8A"
_BLUE = "#0056A2"
_LIGHT_BLUE = "#E8F0F9"

st.markdown(
    f"""
    <style>
        /* ---- typography ---- */
        .bmo-title {{
            color: {_NAVY};
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: -0.5px;
            margin-bottom: 0;
        }}
        .bmo-subtitle {{
            color: #555;
            font-size: 0.97rem;
            margin-top: 0.25rem;
            margin-bottom: 1.6rem;
        }}
        .section-header {{
            color: {_NAVY};
            font-size: 1.1rem;
            font-weight: 700;
            border-bottom: 2px solid {_BLUE};
            padding-bottom: 0.25rem;
            margin-top: 1.8rem;
            margin-bottom: 0.75rem;
        }}

        /* ---- metric cards ---- */
        .metric-card {{
            background: {_LIGHT_BLUE};
            border-left: 5px solid {_NAVY};
            border-radius: 6px;
            padding: 1rem 1.25rem;
        }}
        .metric-label {{
            color: #555;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 0.3rem;
        }}
        .metric-value {{
            color: {_NAVY};
            font-size: 2.1rem;
            font-weight: 700;
            line-height: 1;
        }}

        /* ---- report box ---- */
        .report-box {{
            background: #F8F9FA;
            border: 1px solid #DEE2E6;
            border-left: 5px solid {_BLUE};
            border-radius: 6px;
            padding: 1.4rem 1.6rem;
            font-size: 0.97rem;
            line-height: 1.75;
            color: #222;
        }}

        /* ---- risk warning box ---- */
        .risk-box {{
            background: #FFF8E1;
            border-left: 5px solid #FFA000;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.5rem;
            color: #5D4037;
            font-size: 0.93rem;
        }}

        /* ---- sidebar ---- */
        [data-testid="stSidebar"] > div:first-child {{
            background-color: {_NAVY};
        }}
        [data-testid="stSidebar"] * {{
            color: white !important;
        }}
        [data-testid="stSidebar"] .stButton > button {{
            background-color: white !important;
            color: {_NAVY} !important;
            font-weight: 700;
            width: 100%;
            border-radius: 6px;
            border: none;
        }}
        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: {_LIGHT_BLUE} !important;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Import pipeline — surface import errors clearly before anything renders
# ---------------------------------------------------------------------------

_import_ok = True
_import_error: str = ""

try:
    from graph import run_pipeline
except Exception as exc:
    _import_ok = False
    _import_error = str(exc)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

for _key, _default in [
    ("pipeline_state", None),
    ("last_run", None),
    ("pipeline_error", None),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    run_clicked = st.button("▶  Run Full Pipeline", use_container_width=True)

    st.markdown("### Last run")
    if st.session_state.last_run:
        st.markdown(st.session_state.last_run.strftime("%Y-%m-%d  %H:%M:%S"))
    else:
        st.markdown("*Not yet run*")

    st.markdown("---")
    st.markdown("### Tickers")
    if st.session_state.pipeline_state:
        for t in sorted(st.session_state.pipeline_state.get("price_data", {}).keys()):
            st.markdown(f"&nbsp;&nbsp;• {t}")
    else:
        st.markdown("*Run pipeline to see tickers*")

    st.markdown("---")
    st.markdown(
        "<span style='opacity:0.65; font-size:0.8rem;'>"
        "AI Capital Markets Agent<br>"
        "LangGraph · scikit-learn · OpenAI"
        "</span>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="bmo-title">📊 AI Capital Markets Decision Agent</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="bmo-subtitle">'
    "Multi-agent portfolio analysis powered by LangGraph · "
    "Random Forest forecasting · OpenAI sentiment classification"
    "</p>",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Pipeline execution (triggered by sidebar button)
# ---------------------------------------------------------------------------

if run_clicked:
    if not _import_ok:
        st.error(f"❌ Cannot import pipeline: {_import_error}")
    else:
        with st.spinner("🔄 Running full pipeline — this may take a minute..."):
            try:
                result = run_pipeline()
                st.session_state.pipeline_state = result
                st.session_state.last_run = datetime.now()
                st.session_state.pipeline_error = None
                st.success("✅ Pipeline complete!")
            except Exception as exc:
                st.session_state.pipeline_error = str(exc)
                st.error(f"❌ Pipeline failed: {exc}")

if st.session_state.pipeline_error and not run_clicked:
    st.error(f"❌ Last pipeline run failed: {st.session_state.pipeline_error}")

# ---------------------------------------------------------------------------
# Gate: show placeholder until the pipeline has been run at least once
# ---------------------------------------------------------------------------

if st.session_state.pipeline_state is None:
    st.info(
        "👈  Click **Run Full Pipeline** in the sidebar to start the analysis.\n\n"
        "The pipeline will load market data, compute financial metrics, train "
        "per-ticker ML models, classify sentiment with OpenAI, score each ticker "
        "against a 7-factor rubric, check for portfolio risks, and produce an "
        "AI-generated executive summary."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Unpack state
# ---------------------------------------------------------------------------

state = st.session_state.pipeline_state
price_data: dict = state.get("price_data", {})
metrics: dict     = state.get("metrics", {})
forecasts: dict   = state.get("forecasts", {})
sentiment: dict   = state.get("sentiment", {})
decisions: dict   = state.get("decisions", {})
risk_flags: dict  = state.get("risk_flags", {})
report: str       = state.get("report", "")
timestamp: str    = state.get("timestamp", "")

# ---------------------------------------------------------------------------
# Derived data helpers (cached to avoid recomputing on every widget interaction)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _build_close_returns(close_dict: dict[str, pd.Series]) -> pd.DataFrame:
    """Daily returns DataFrame from a {ticker: Close series} dict."""
    frames = {t: s.pct_change().dropna() for t, s in close_dict.items()}
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).dropna()


@st.cache_data(show_spinner=False)
def _build_reco_df(
    decisions: dict, metrics: dict, forecasts: dict, sentiment: dict
) -> pd.DataFrame:
    """Build the styled recommendations table DataFrame."""
    rows = []
    for ticker in sorted(decisions):
        d = decisions[ticker]
        m = metrics.get(ticker, {})
        rows.append(
            {
                "Ticker": ticker,
                "Recommendation": d.get("action", "N/A"),
                "Score": d.get("score", 0),
                "Sentiment": sentiment.get(ticker, "N/A"),
                "Forecast Prob": forecasts.get(ticker, float("nan")),
                "Sharpe Ratio": m.get("sharpe_ratio", float("nan")),
                "Ann. Return": m.get("ann_return", float("nan")),
            }
        )
    return pd.DataFrame(rows).set_index("Ticker")


# Extract Close price series from price_data for caching
_close_dict = {
    ticker: df["Close"]
    for ticker, df in price_data.items()
    if df is not None and "Close" in df.columns
}
returns_df = _build_close_returns(_close_dict)
reco_df    = _build_reco_df(decisions, metrics, forecasts, sentiment)

# ---------------------------------------------------------------------------
# Section 1 — Summary metric cards
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">📈 Summary</p>', unsafe_allow_html=True)

n_tickers    = len(decisions)
n_overweight = sum(1 for d in decisions.values() if d.get("action") == "Overweight")
n_avoid      = sum(1 for d in decisions.values() if d.get("action") == "Avoid")
n_flags      = len(risk_flags)


def _metric_card(col, label: str, value, accent_color: str = _NAVY) -> None:
    col.markdown(
        f'<div class="metric-card" style="border-left-color:{accent_color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value" style="color:{accent_color}">{value}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


c1, c2, c3, c4 = st.columns(4)
_metric_card(c1, "Tickers Analysed", n_tickers)
_metric_card(c2, "Overweight", n_overweight, "#155724")
_metric_card(c3, "Avoid", n_avoid, "#721c24")
_metric_card(c4, "Risk Flags", n_flags, "#856404" if n_flags else _NAVY)

# ---------------------------------------------------------------------------
# Section 2 — Recommendations table (colour-coded by action)
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="section-header">📋 Portfolio Recommendations</p>',
    unsafe_allow_html=True,
)

_ACTION_STYLE: dict[str, tuple[str, str]] = {
    "Overweight":  ("background-color: #d4edda", "color: #155724"),
    "Neutral":     ("background-color: #fff3cd", "color: #856404"),
    "Underweight": ("background-color: #ffe5cc", "color: #7d4f00"),
    "Avoid":       ("background-color: #f8d7da", "color: #721c24"),
}


def _colour_rows(row: pd.Series) -> list[str]:
    bg, fg = _ACTION_STYLE.get(row["Recommendation"], ("", ""))
    return [f"{bg}; {fg}"] * len(row)


styled_reco = (
    reco_df.style
    .apply(_colour_rows, axis=1)
    .format(
        {
            "Score":        "{:.0f}/7",
            "Forecast Prob": "{:.1%}",
            "Sharpe Ratio": "{:.2f}",
            "Ann. Return":  "{:.1%}",
        },
        na_rep="N/A",
    )
)

st.dataframe(
    styled_reco,
    use_container_width=True,
    height=min(80 + len(reco_df) * 38, 440),
)

# ---------------------------------------------------------------------------
# Section 3a + 3b — Forecast bar chart  |  Correlation heatmap
# ---------------------------------------------------------------------------

st.markdown('<p class="section-header">📊 Charts</p>', unsafe_allow_html=True)

col_bar, col_heat = st.columns(2)

# — Forecast probability bar chart —
with col_bar:
    st.markdown("**Next-Week Positive Return Probability**")

    if forecasts:
        fc_sorted = sorted(forecasts.items(), key=lambda x: x[1], reverse=True)
        tickers_sorted = [t for t, _ in fc_sorted]
        probs_sorted   = [p for _, p in fc_sorted]

        bar_colors = [
            "#2e7d32" if p > 0.55 else ("#c62828" if p < 0.45 else "#f9a825")
            for p in probs_sorted
        ]

        fig, ax = plt.subplots(figsize=(6, 4))
        x_pos = range(len(tickers_sorted))

        ax.bar(x_pos, probs_sorted, color=bar_colors, edgecolor="white", linewidth=0.5)
        ax.axhline(0.55, color=_NAVY, linewidth=1.4, linestyle="--")
        ax.text(
            len(tickers_sorted) - 0.5, 0.57,
            "55% threshold",
            color=_NAVY, fontsize=8, ha="right", va="bottom",
        )

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(tickers_sorted, fontsize=9)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.set_ylabel("Probability", fontsize=9)
        ax.set_title("Next-Week Positive Return Probability", fontsize=10,
                     color=_NAVY, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("No forecast data available.")

# — Correlation heatmap —
with col_heat:
    st.markdown("**Daily Return Correlation Heatmap**")

    if not returns_df.empty and returns_df.shape[1] > 1:
        corr = returns_df.corr()
        n_cols = len(corr)
        fig_size = max(4.5, n_cols * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))

        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=sns.light_palette(_BLUE, as_cmap=True),
            vmin=-1,
            vmax=1,
            linewidths=0.4,
            linecolor="#dee2e6",
            annot_kws={"size": max(6, 9 - n_cols)},
            cbar_kws={"shrink": 0.75, "label": "Correlation"},
        )
        ax.set_title("Return Correlations", fontsize=10, color=_NAVY, pad=8, fontweight="bold")
        ax.tick_params(axis="both", labelsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Insufficient data for heatmap (need ≥ 2 tickers with Close prices).")

# ---------------------------------------------------------------------------
# Section 3c — Cumulative returns line chart (full width)
# ---------------------------------------------------------------------------

st.markdown("**Cumulative Returns (full history)**")

if not returns_df.empty:
    cum_ret = (1 + returns_df).cumprod() - 1
    cum_ret.index.name = "Date"

    cum_long = (
        cum_ret
        .reset_index()
        .melt(id_vars="Date", var_name="Ticker", value_name="Cumulative Return")
    )

    zero_rule = (
        alt.Chart(pd.DataFrame({"y": [0]}))
        .mark_rule(color="#aaa", strokeDash=[4, 4], size=1)
        .encode(y=alt.Y("y:Q"))
    )

    lines = (
        alt.Chart(cum_long)
        .mark_line(strokeWidth=1.8)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y(
                "Cumulative Return:Q",
                axis=alt.Axis(format=".0%", title="Cumulative Return"),
            ),
            color=alt.Color(
                "Ticker:N",
                legend=alt.Legend(title="Ticker", orient="right"),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date"),
                alt.Tooltip("Ticker:N"),
                alt.Tooltip("Cumulative Return:Q", format=".2%", title="Cum. Return"),
            ],
        )
        .properties(height=340)
        .interactive()
    )

    st.altair_chart(zero_rule + lines, use_container_width=True)
else:
    st.info("No return data available for cumulative chart.")

# ---------------------------------------------------------------------------
# Section 4 — Executive summary
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="section-header">📝 Executive Summary</p>',
    unsafe_allow_html=True,
)

if report:
    # Preserve paragraph breaks while escaping HTML
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

# ---------------------------------------------------------------------------
# Section 5 — Risk warnings
# ---------------------------------------------------------------------------

st.markdown(
    '<p class="section-header">⚠️ Risk Warnings</p>',
    unsafe_allow_html=True,
)

if risk_flags:
    for flag_name, message in risk_flags.items():
        label = flag_name.replace("_", " ").title()
        st.markdown(
            f'<div class="risk-box">⚠️ <strong>{label}</strong><br>{message}</div>',
            unsafe_allow_html=True,
        )
else:
    st.success("✅ No risk warnings identified for this portfolio configuration.")

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    f"<small style='color:#999;'>"
    f"Analysis timestamp: {timestamp} &nbsp;·&nbsp; "
    f"AI Capital Markets Decision Agent"
    f"</small>",
    unsafe_allow_html=True,
)
