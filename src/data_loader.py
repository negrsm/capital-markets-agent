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
) -> dict[str, pd.DataFrame]:
    """Download daily OHLCV data for each ticker and save to data/raw/<TICKER>.csv.

    Args:
        tickers: List of ticker symbols to download.
        start:   Start date string in YYYY-MM-DD format.
        end:     End date string in YYYY-MM-DD format (defaults to today).

    Returns:
        Dictionary mapping each ticker to its OHLCV DataFrame.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    print(f"[data_loader] Downloading {len(tickers)} tickers from {start} to {end}...")

    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        print(f"  -> Fetching {ticker} ...", end=" ", flush=True)
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if df.empty:
            print("WARNING: no data returned, skipping.")
            continue

        # Flatten MultiIndex columns that yfinance may return for a single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.index.name = "Date"
        csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        df.to_csv(csv_path)
        print(f"saved {len(df)} rows -> {csv_path}")
        results[ticker] = df

    print(f"[data_loader] Download complete. {len(results)}/{len(tickers)} tickers saved.")
    return results


def load_data(tickers: list[str] = DEFAULT_TICKERS) -> dict[str, pd.DataFrame]:
    """Load previously downloaded CSVs from data/raw/ into DataFrames.

    Args:
        tickers: List of ticker symbols to load.

    Returns:
        Dictionary mapping each ticker to its OHLCV DataFrame.
        Tickers whose CSV files are not found are skipped with a warning.
    """
    print(f"[data_loader] Loading {len(tickers)} tickers from {RAW_DIR} ...")
    results: dict[str, pd.DataFrame] = {}

    for ticker in tickers:
        csv_path = os.path.join(RAW_DIR, f"{ticker}.csv")
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found — run download_data() first.")
            continue

        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
        print(f"  -> Loaded {ticker}: {len(df)} rows ({df.index[0].date()} to {df.index[-1].date()})")
        results[ticker] = df

    print(f"[data_loader] Load complete. {len(results)}/{len(tickers)} tickers available.")
    return results


if __name__ == "__main__":
    download_data()
    load_data()
