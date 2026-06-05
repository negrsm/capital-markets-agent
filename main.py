"""
main.py — Command-line entry point for the AI Capital Markets Decision Agent.

Usage:
    python main.py                          # run on all 8 default tickers
    python main.py --fresh                  # re-download data first
    python main.py --tickers SPY,QQQ,GLD   # custom ticker list
    python main.py --fresh --dashboard      # refresh data then open dashboard
"""

import argparse
import json
import math
import os
import subprocess
import sys
import textwrap
from datetime import date

# Make src/ importable from the project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import DEFAULT_TICKERS, download_data  # noqa: E402
from graph import run_pipeline                           # noqa: E402

import pandas as pd  # noqa: E402

PROCESSED_DIR  = os.path.join(os.path.dirname(__file__), "data", "processed")
DASHBOARD_PATH = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")

WIDTH = 62


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="AI Capital Markets Decision Agent — CLI entry point.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py
              python main.py --fresh
              python main.py --tickers SPY,QQQ,GLD
              python main.py --fresh --dashboard
        """),
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Download fresh market data from Yahoo Finance before running.",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the Streamlit dashboard after the pipeline finishes.",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        metavar="TICKER,...",
        help=(
            "Comma-separated list of tickers to analyse "
            "(default: all 8 — SPY QQQ XLF XLE XLV TLT GLD UUP)."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _rule(char: str = "─") -> None:
    print(char * WIDTH)


def _section(title: str) -> None:
    print()
    _rule("─")
    print(f"  {title}")
    _rule("─")


def print_header(tickers: list[str]) -> None:
    today = date.today().isoformat()
    _rule("=")
    print("  AI CAPITAL MARKETS DECISION AGENT")
    print(f"  Date    : {today}")
    print(f"  Tickers : {', '.join(sorted(tickers))}  ({len(tickers)} total)")
    _rule("=")


def print_report(report: str) -> None:
    _section("EXECUTIVE SUMMARY")
    if not report:
        print("  (no report generated)")
        return
    for para in report.strip().split("\n\n"):
        for line in textwrap.wrap(para.strip(), width=WIDTH - 2):
            print(f"  {line}")
        print()


def print_decisions(
    decisions: dict,
    sentiment: dict,
    forecasts: dict,
) -> None:
    _section("PORTFOLIO DECISIONS")

    _ACTION_ICON = {
        "Overweight":  "▲",
        "Neutral":     "─",
        "Underweight": "▼",
        "Avoid":       "✕",
    }

    header = (
        f"  {'Ticker':<7}  {'Action':<13}  {'Score':>5}  "
        f"{'Sentiment':<10}  {'Forecast':>8}"
    )
    print(header)
    print(
        f"  {'─'*7}  {'─'*13}  {'─'*5}  {'─'*10}  {'─'*8}"
    )

    for ticker in sorted(decisions):
        d     = decisions[ticker]
        icon  = _ACTION_ICON.get(d["action"], " ")
        sent  = sentiment.get(ticker, "N/A")
        prob  = forecasts.get(ticker, float("nan"))
        prob_str = f"{prob:.1%}" if not math.isnan(prob) else " N/A"
        print(
            f"  {ticker:<7}  {icon} {d['action']:<12}  {d['score']:>3}/7  "
            f"{sent:<10}  {prob_str:>8}"
        )


def print_risk_warnings(risk_flags: dict) -> None:
    _section("RISK WARNINGS")
    if not risk_flags:
        print("  ✓  No risk warnings identified.")
        return
    for flag_name, message in risk_flags.items():
        label = flag_name.replace("_", " ").upper()
        print(f"  ⚠  [{label}]")
        for line in textwrap.wrap(message, width=WIDTH - 5):
            print(f"      {line}")
        print()


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(state: dict, tickers: list[str]) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    decisions = state.get("decisions", {})
    sentiment = state.get("sentiment", {})
    forecasts = state.get("forecasts", {})
    metrics   = state.get("metrics", {})

    # ── decisions.csv ───────────────────────────────────────────────────────
    rows = []
    for ticker in sorted(decisions):
        d = decisions[ticker]
        m = metrics.get(ticker, {})
        prob = forecasts.get(ticker, float("nan"))
        rows.append({
            "ticker":       ticker,
            "action":       d.get("action"),
            "score":        d.get("score"),
            "sentiment":    sentiment.get(ticker, "N/A"),
            "forecast_prob": None if math.isnan(prob) else round(prob, 4),
            "ann_return":   m.get("ann_return"),
            "sharpe_ratio": m.get("sharpe_ratio"),
            "max_drawdown": m.get("max_drawdown"),
            "ann_volatility": m.get("ann_volatility"),
        })

    if rows:
        decisions_path = os.path.join(PROCESSED_DIR, "decisions.csv")
        pd.DataFrame(rows).to_csv(decisions_path, index=False)
        print(f"  Saved decisions     -> {decisions_path}")

    # ── pipeline_summary.json ───────────────────────────────────────────────
    def _safe_float(v):
        if v is None:
            return None
        try:
            return None if math.isnan(v) else round(float(v), 6)
        except (TypeError, ValueError):
            return None

    summary = {
        "timestamp":  state.get("timestamp"),
        "tickers":    tickers,
        "decisions":  decisions,
        "sentiment":  sentiment,
        "forecasts":  {k: _safe_float(v) for k, v in forecasts.items()},
        "risk_flags": state.get("risk_flags", {}),
        "report":     state.get("report", ""),
    }
    summary_path = os.path.join(PROCESSED_DIR, "pipeline_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(f"  Saved pipeline JSON -> {summary_path}")


# ---------------------------------------------------------------------------
# Dashboard launcher
# ---------------------------------------------------------------------------

def launch_dashboard() -> None:
    print()
    _rule("=")
    print("  Launching Streamlit dashboard ...")
    print("  Press Ctrl+C to stop.")
    _rule("=")
    print()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", DASHBOARD_PATH],
        check=False,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # Resolve ticker list
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        if not tickers:
            print("ERROR: --tickers produced an empty list. Check the format (e.g. SPY,QQQ).")
            sys.exit(1)
    else:
        tickers = list(DEFAULT_TICKERS)

    # ── Header ──────────────────────────────────────────────────────────────
    print_header(tickers)

    # ── Step 1: Data download ────────────────────────────────────────────────
    print(f"\n[1/4] Data")
    if args.fresh:
        print("  Downloading fresh data from Yahoo Finance ...")
        download_data(tickers=tickers, refresh=True)
        print("  Download complete.")
    else:
        print("  Skipping download (pass --fresh to re-download from Yahoo Finance).")

    # ── Step 2: Pipeline ────────────────────────────────────────────────────
    print(f"\n[2/4] Running LangGraph pipeline ...")
    state = run_pipeline(tickers)
    print(f"  Pipeline complete.  Timestamp: {state['timestamp']}")

    # ── Step 3: Print results ────────────────────────────────────────────────
    print(f"\n[3/4] Results")
    print_report(state.get("report", ""))
    print_decisions(
        state.get("decisions", {}),
        state.get("sentiment", {}),
        state.get("forecasts", {}),
    )
    print_risk_warnings(state.get("risk_flags", {}))

    # ── Step 4: Save ─────────────────────────────────────────────────────────
    print(f"\n[4/4] Saving to data/processed/ ...")
    save_results(state, tickers)

    print()
    _rule("=")
    print("  Done.")
    _rule("=")

    # ── Optional: dashboard ──────────────────────────────────────────────────
    if args.dashboard:
        launch_dashboard()


if __name__ == "__main__":
    main()
