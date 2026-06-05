"""
data_loader.py — Download and load daily OHLCV data for a set of tickers.
"""

import os
from datetime import date

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["SPY", "QQQ", "XLF", "XLE", "XLV", "TLT", "GLD", "UUP"]
DEFAULT_START = "2022-01-01"
DEFAULT_END = date.today().isoformat()
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_data(
    tickers: list[str] = DEFAULT_TICKERS,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for each ticker and save to data/raw/<TICKER>.csv.

    Args:
        tickers: List of ticker symbols to download.
        start:   Start date string in YYYY-MM-DD format.
        end:     End date string in YYYY-MM-DD format (defaults to today).
        refresh: If True, always download fresh data even if the CSV already exists.
                 If False, skip tickers whose CSV files already exist.

    Returns:
        Dictionary mapping each ticker to its OHLCV DataFrame.
    """
    os.makedirs(RAW_DIR, exist_ok=True)

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")

        if not refresh and os.path.exists(csv_path):
            print(f"  -> {ticker}: CSV exists, skipping (use refresh=True to re-download).")
            results[ticker] = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            continue

        print(f"  -> Fetching {ticker} ...", end=" ", flush=True)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty:
            print("WARNING: no data returned, skipping.")
            continue

        # Flatten MultiIndex columns that yfinance may return for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        df.to_csv(csv_path)
        print(f"saved {len(df)} rows -> {csv_path}")
        results[ticker] = df

    downloaded = sum(1 for t in tickers if not (not refresh and os.path.exists(os.path.join(RAW_DIR, f"{t}.csv"))))
    print(f"[data_loader] Download complete. {len(results)}/{len(tickers)} tickers available.")
    return results


def load_data(
    tickers: list[str] = DEFAULT_TICKERS,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for each ticker from data/raw/ CSVs.

    Automatically calls download_data() if any ticker CSV is missing or
    if refresh=True.

    Args:
        tickers: List of ticker symbols to load.
        refresh: If True, re-download all tickers from Yahoo Finance before loading.
                 If False, only download tickers whose CSV files are missing.

    Returns:
        Dictionary mapping each ticker to its OHLCV DataFrame.
    """
    missing = [t for t in tickers if not os.path.exists(os.path.join(RAW_DIR, f"{t}.csv"))]

    if refresh or missing:
        if refresh:
            print(f"[data_loader] refresh=True — re-downloading all {len(tickers)} tickers...")
        else:
            print(f"[data_loader] Missing CSVs for {missing} — downloading now...")
        download_data(tickers=tickers, refresh=refresh)

    print(f"[data_loader] Loading {len(tickers)} tickers from {RAW_DIR} ...")
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — skipping.")
            continue

        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        print(f"  -> Loaded {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        results[ticker] = df

    print(f"[data_loader] Load complete. {len(results)}/{len(tickers)} tickers available.")
    return results


def get_last_updated(tickers: list[str] = DEFAULT_TICKERS) -> dict[str, str]:
    """Return the most recent date in each ticker's CSV file.

    Args:
        tickers: List of ticker symbols to check.

    Returns:
        Dictionary mapping each ticker to its last date string (YYYY-MM-DD),
        or None if the CSV does not exist.
    """
    result: dict[str, str | None] = {}
    for ticker in tickers:
        csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            result[ticker] = None
            continue
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        result[ticker] = df.index[-1].date().isoformat() if not df.empty else None
    return result


if __name__ == "__main__":
    download_data()
    load_data()
