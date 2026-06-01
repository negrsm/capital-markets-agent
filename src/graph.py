"""
graph.py — Pipeline entry point: assembles the LangGraph and exposes run_pipeline().

This module is the single import surface for the dashboard and any other caller
that wants to run the full capital markets agent pipeline.

Usage:
    from graph import run_pipeline
    state = run_pipeline()          # uses DEFAULT_TICKERS
    state = run_pipeline(["SPY", "QQQ"])
"""

from agents import MarketState, build_graph
from data_loader import DEFAULT_TICKERS


def run_pipeline(tickers: list[str] | None = None) -> MarketState:
    """Build and execute the full capital markets agent pipeline.

    Constructs a fresh LangGraph, sets up the initial state, and invokes
    the full node chain:
        market_data → metrics → forecasting → sentiment → decision → risk → report

    Args:
        tickers: List of ticker symbols to analyse.  Defaults to
                 DEFAULT_TICKERS from data_loader.py if not provided.

    Returns:
        Final MarketState dict populated by every agent node.  Key fields:

        - ``price_data``  — raw OHLCV DataFrames (dict[str, pd.DataFrame])
        - ``metrics``     — financial metrics (dict[str, dict])
        - ``forecasts``   — ML probabilities (dict[str, float])
        - ``sentiment``   — LLM labels (dict[str, str])
        - ``decisions``   — portfolio actions (dict[str, dict])
        - ``risk_flags``  — named warnings (dict[str, str])
        - ``report``      — 200-word executive summary (str)
        - ``timestamp``   — ISO-8601 UTC run time (str)
    """
    if tickers is None:
        tickers = list(DEFAULT_TICKERS)

    graph = build_graph()

    initial_state: MarketState = {
        "tickers": tickers,
        "price_data": {},
        "metrics": {},
        "forecasts": {},
        "sentiment": {},
        "decisions": {},
        "risk_flags": {},
        "report": "",
        "timestamp": "",
    }

    return graph.invoke(initial_state)


if __name__ == "__main__":
    state = run_pipeline()
    print(f"\nPipeline complete — timestamp: {state['timestamp']}")
    print(f"Tickers processed: {sorted(state['decisions'].keys())}")
    print(f"\nDecisions:")
    for ticker, d in sorted(state["decisions"].items()):
        print(f"  {ticker:<6} {d['action']:<12}  score={d['score']}/7")
    print(f"\nReport:\n{state['report']}")
