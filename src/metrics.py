"""
metrics.py — Compute financial performance metrics for a set of OHLCV DataFrames.

Each public function operates on a single ticker's Close price series.
The top-level `compute_metrics` function aggregates results into a summary DataFrame
with one row per ticker.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252          # conventional annualisation factor
RISK_FREE_RATE = 0.05       # annual risk-free rate used for Sharpe ratio
MA_SHORT = 50               # short moving-average window (days)
MA_LONG = 200               # long moving-average window (days)


# ---------------------------------------------------------------------------
# Per-metric helpers
# ---------------------------------------------------------------------------

def _daily_returns(close: pd.Series) -> pd.Series:
    """Return the daily percentage change of a Close price series.

    pct_change() gives (P_t - P_{t-1}) / P_{t-1}.  The first row is NaN
    and is dropped so downstream calculations are not affected.
    """
    return close.pct_change().dropna()


def _annualized_return(daily_returns: pd.Series) -> float:
    """Compound annualised return from a series of daily returns.

    Formula: (product of (1 + r_t))^(252/N) - 1
    Geometric compounding is used rather than a simple arithmetic mean
    because it correctly accounts for the order of gains and losses.
    """
    n = len(daily_returns)
    if n == 0:
        return float("nan")
    cumulative = (1 + daily_returns).prod()
    return float(cumulative ** (TRADING_DAYS / n) - 1)


def _annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualised standard deviation of daily returns (historical volatility).

    Daily std is scaled by sqrt(252) under the assumption that daily returns
    are i.i.d. — the standard square-root-of-time rule.
    """
    if len(daily_returns) < 2:
        return float("nan")
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS))


def _sharpe_ratio(ann_return: float, ann_vol: float) -> float:
    """Sharpe ratio: excess return per unit of total risk.

    Sharpe = (annualised_return - risk_free_rate) / annualised_volatility

    A ratio > 1 is generally considered acceptable; > 2 is strong.
    Returns NaN when volatility is zero to avoid division by zero.
    """
    if ann_vol == 0 or np.isnan(ann_vol):
        return float("nan")
    return float((ann_return - RISK_FREE_RATE) / ann_vol)


def _max_drawdown(close: pd.Series) -> float:
    """Maximum drawdown: the largest peak-to-trough decline in price.

    Drawdown at time t = (price_t - running_peak_t) / running_peak_t
    Max drawdown is the minimum (most negative) value across all t.
    Result is expressed as a negative decimal (e.g. -0.34 means -34 %).
    """
    if close.empty:
        return float("nan")
    running_peak = close.cummax()
    drawdown = (close - running_peak) / running_peak
    return float(drawdown.min())


def _moving_averages(close: pd.Series) -> tuple[float, float]:
    """Return the most recent 50-day and 200-day simple moving averages.

    Uses all available history; if fewer bars exist than the window the
    result for that window is NaN (e.g. a ticker with < 200 days of data).
    """
    ma50 = float(close.rolling(MA_SHORT).mean().iloc[-1])
    ma200 = float(close.rolling(MA_LONG).mean().iloc[-1])
    return ma50, ma200


def _beta(ticker_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Beta of the ticker relative to the benchmark (SPY).

    Beta = Cov(r_ticker, r_benchmark) / Var(r_benchmark)

    Beta > 1 means the asset amplifies benchmark moves (more volatile);
    Beta < 1 means it is less sensitive to market swings.
    Aligns the two series on their shared dates before computing.
    """
    # Align on common dates
    aligned = pd.concat([ticker_returns, benchmark_returns], axis=1).dropna()
    if aligned.shape[0] < 2:
        return float("nan")

    t_ret = aligned.iloc[:, 0]
    b_ret = aligned.iloc[:, 1]

    cov_matrix = np.cov(t_ret, b_ret, ddof=1)   # 2x2 covariance matrix
    cov_tb = cov_matrix[0, 1]
    var_b = cov_matrix[1, 1]

    if var_b == 0:
        return float("nan")
    return float(cov_tb / var_b)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_metrics(
    data: dict[str, pd.DataFrame],
    benchmark: str = "SPY",
    close_col: str = "Close",
) -> pd.DataFrame:
    """Compute financial metrics for every ticker in *data*.

    Args:
        data:       Dictionary mapping ticker symbol -> OHLCV DataFrame.
                    Each DataFrame must contain a column named *close_col*.
        benchmark:  Ticker key in *data* to use as the market benchmark for
                    beta calculations.  Defaults to ``"SPY"``.
        close_col:  Name of the column holding closing prices.

    Returns:
        A DataFrame with one row per ticker and the following columns:

        ================  ============================================================
        ann_return        Annualised geometric return (decimal, e.g. 0.12 = 12 %)
        ann_volatility    Annualised volatility / standard deviation (decimal)
        sharpe_ratio      (ann_return - 0.05) / ann_volatility
        max_drawdown      Largest peak-to-trough decline (negative decimal)
        ma_50             Most recent 50-day simple moving average of Close
        ma_200            Most recent 200-day simple moving average of Close
        beta              Covariance with SPY / variance of SPY
        last_close        Most recent closing price
        ================  ============================================================
    """
    print(f"[metrics] Computing metrics for {len(data)} tickers "
          f"(benchmark: {benchmark}, risk-free rate: {RISK_FREE_RATE:.0%}) ...")

    # Pre-compute benchmark returns once; reused for every beta calculation
    if benchmark in data:
        bench_close = data[benchmark][close_col].dropna()
        bench_returns = _daily_returns(bench_close)
    else:
        print(f"  WARNING: benchmark '{benchmark}' not in data — beta will be NaN for all tickers.")
        bench_returns = pd.Series(dtype=float)

    rows: list[dict] = []

    for ticker, df in data.items():
        print(f"  -> {ticker} ...", end=" ", flush=True)

        if close_col not in df.columns:
            print(f"WARNING: column '{close_col}' not found, skipping.")
            continue

        close = df[close_col].dropna()

        if close.empty:
            print("WARNING: no valid close prices, skipping.")
            continue

        # --- core return statistics ---
        daily_ret = _daily_returns(close)
        ann_ret = _annualized_return(daily_ret)
        ann_vol = _annualized_volatility(daily_ret)
        sharpe = _sharpe_ratio(ann_ret, ann_vol)

        # --- drawdown ---
        max_dd = _max_drawdown(close)

        # --- moving averages ---
        ma50, ma200 = _moving_averages(close)

        # --- beta vs benchmark (NaN for the benchmark itself is fine) ---
        beta = _beta(daily_ret, bench_returns)

        last_close = float(close.iloc[-1])

        rows.append({
            "ticker":         ticker,
            "ann_return":     ann_ret,
            "ann_volatility": ann_vol,
            "sharpe_ratio":   sharpe,
            "max_drawdown":   max_dd,
            "ma_50":          ma50,
            "ma_200":         ma200,
            "beta":           beta,
            "last_close":     last_close,
        })

        print(
            f"ann_ret={ann_ret:+.1%}  vol={ann_vol:.1%}  "
            f"sharpe={sharpe:.2f}  maxDD={max_dd:.1%}  beta={beta:.2f}"
        )

    summary = (
        pd.DataFrame(rows)
        .set_index("ticker")
        .sort_index()
    )

    print(f"[metrics] Done. Summary shape: {summary.shape}")
    return summary


if __name__ == "__main__":
    from data_loader import load_data
    data = load_data()
    summary = compute_metrics(data)
    print("\n--- Metrics Summary ---")
    print(summary.to_string())
