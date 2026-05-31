"""
forecasting.py — ML module for predicting next-week (5-day) return direction.

Workflow:
    data_dict = load_data()
    probabilities = run_forecasting(data_dict)
    # probabilities = {"SPY": 0.62, "QQQ": 0.48, ...}
"""

import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Resolve paths relative to the project root (one level above src/)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct ML features and a binary target from a single ticker's OHLCV DataFrame.

    All features are computed from data available at time t to avoid lookahead bias.
    The target uses the 5-day forward return, so the last 5 rows will have NaN targets
    and are dropped.

    Args:
        df: OHLCV DataFrame with a DatetimeIndex and at minimum Close and Volume columns.

    Returns:
        DataFrame with feature columns and a ``target`` column (1 = positive next-week
        return, 0 = flat or negative).  Rows with any NaN are dropped.

    Feature glossary:
        lag_ret_1d   — 1-day percentage return (today vs yesterday)
        lag_ret_5d   — 5-day percentage return (today vs 5 days ago)
        lag_ret_20d  — 20-day percentage return (today vs 20 days ago)
        vol_20d      — Rolling 20-day standard deviation of daily returns (realised vol)
        momentum_5d  — Rolling 5-day mean of daily returns
        momentum_20d — Rolling 20-day mean of daily returns
        above_ma50   — 1 if Close > 50-day SMA, else 0
        above_ma200  — 1 if Close > 200-day SMA, else 0
        vol_change   — 1-day percentage change in Volume
        target       — 1 if the next-5-day return is positive, else 0
    """
    close = df["Close"]
    volume = df["Volume"]

    daily_ret = close.pct_change(1)

    feat = pd.DataFrame(index=df.index)

    # Lag returns — how far price has moved over the last N days
    feat["lag_ret_1d"] = daily_ret
    feat["lag_ret_5d"] = close.pct_change(5)
    feat["lag_ret_20d"] = close.pct_change(20)

    # Realised volatility — rolling std of daily returns over 20 days
    feat["vol_20d"] = daily_ret.rolling(20).std()

    # Momentum — rolling mean of daily returns (different from a single-period return)
    feat["momentum_5d"] = daily_ret.rolling(5).mean()
    feat["momentum_20d"] = daily_ret.rolling(20).mean()

    # Moving-average signals — binary: 1 = price is above the MA, 0 = below
    feat["above_ma50"] = (close > close.rolling(50).mean()).astype(int)
    feat["above_ma200"] = (close > close.rolling(200).mean()).astype(int)

    # Volume change — 1-day percentage change in traded volume
    feat["vol_change"] = volume.pct_change(1)

    # Target — 1 if the close 5 trading days from now is higher than today's close
    future_return = close.shift(-5) / close - 1
    feat["target"] = (future_return > 0).astype(int)

    # Drop rows where any feature or the target is NaN (start of series + last 5 rows)
    feat = feat.dropna()

    return feat


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "lag_ret_1d",
    "lag_ret_5d",
    "lag_ret_20d",
    "vol_20d",
    "momentum_5d",
    "momentum_20d",
    "above_ma50",
    "above_ma200",
    "vol_change",
]


def train_model(
    ticker: str,
    data_dict: dict[str, pd.DataFrame],
    test_size: float = 0.20,
    n_estimators: int = 200,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, pd.DataFrame, dict]:
    """Build features, split chronologically, train a Random Forest, and print metrics.

    Uses a time-based train/test split (no shuffling) so the test set always
    represents an out-of-sample future period — the only valid evaluation for
    financial time-series models.

    Args:
        ticker:       Ticker symbol key in *data_dict*.
        data_dict:    Dictionary mapping ticker symbols to OHLCV DataFrames.
        test_size:    Fraction of data reserved for the test set (default 0.20 = 20 %).
        n_estimators: Number of trees in the Random Forest.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of:
          - trained RandomForestClassifier
          - test-set feature DataFrame (includes ``target`` column)
          - metrics dict with keys: ``accuracy``, ``precision``, ``recall``, ``report``

    Raises:
        KeyError: if *ticker* is not found in *data_dict*.
        ValueError: if the feature DataFrame has fewer than 50 usable rows.
    """
    if ticker not in data_dict:
        raise KeyError(f"Ticker '{ticker}' not found in data_dict.")

    feat = build_features(data_dict[ticker])

    if len(feat) < 50:
        raise ValueError(
            f"[{ticker}] Only {len(feat)} usable rows after feature construction "
            "(need ≥ 50).  Download more history."
        )

    # Chronological split — no shuffle, no look-ahead
    split_idx = int(len(feat) * (1 - test_size))
    train_df = feat.iloc[:split_idx]
    test_df = feat.iloc[split_idx:]

    X_train, y_train = train_df[FEATURE_COLS], train_df["target"]
    X_test, y_test = test_df[FEATURE_COLS], test_df["target"]

    print(f"\n[forecasting] {ticker} — training on {len(train_df)} rows, "
          f"testing on {len(test_df)} rows")
    print(f"  Train period: {train_df.index[0].date()} → {train_df.index[-1].date()}")
    print(f"  Test  period: {test_df.index[0].date()}  → {test_df.index[-1].date()}")

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=6,          # shallow trees reduce overfitting on financial data
        min_samples_leaf=10,  # avoid memorising individual trading days
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    report_str = classification_report(y_test, y_pred, zero_division=0)

    print(f"  Accuracy : {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall   : {recall:.3f}")
    print(f"\n  Classification Report:\n{report_str}")

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "report": report_str,
    }
    return model, test_df, metrics


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_proba(
    model: RandomForestClassifier,
    latest_features: pd.DataFrame,
) -> float:
    """Return the probability of a positive next-week return.

    Args:
        model:           A trained RandomForestClassifier returned by ``train_model``.
        latest_features: Single-row DataFrame with the same feature columns used
                         during training (see FEATURE_COLS).  Typically the last
                         row of the feature DataFrame produced by ``build_features``.

    Returns:
        Float in [0, 1]: estimated probability that the next 5-day return is positive.
    """
    # predict_proba returns shape (n_samples, n_classes); class order is [0, 1]
    proba = model.predict_proba(latest_features[FEATURE_COLS])[0, 1]
    return float(proba)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

_SIGNAL_BULLISH_THRESHOLD = 0.55
_SIGNAL_BEARISH_THRESHOLD = 0.45


def save_results(
    probabilities: dict[str, float],
    model_reports: dict[str, dict],
    output_dir: str = "data/processed",
) -> None:
    """Persist forecasting outputs to disk.

    Writes two files into *output_dir*:

    * ``forecasting_probabilities.csv`` — one row per ticker with columns:
      ``ticker``, ``probability``, ``signal``, ``timestamp``.
      Signal thresholds: Bullish > 0.55, Bearish < 0.45, Neutral otherwise.

    * ``forecasting_model_report.txt`` — plain-text file with per-ticker
      accuracy, precision, recall, and the full sklearn classification report.

    Args:
        probabilities: Mapping of ticker → probability as returned by
                       ``run_forecasting()``.
        model_reports: Mapping of ticker → metrics dict as collected during
                       training (keys: ``accuracy``, ``precision``, ``recall``,
                       ``report``).
        output_dir:    Destination directory.  Relative paths are resolved from
                       the project root; absolute paths are used as-is.
    """
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(_PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # --- forecasting_probabilities.csv ---
    def _signal(p: float) -> str:
        if p > _SIGNAL_BULLISH_THRESHOLD:
            return "Bullish"
        if p < _SIGNAL_BEARISH_THRESHOLD:
            return "Bearish"
        return "Neutral"

    rows = [
        {
            "ticker": ticker,
            "probability": round(prob, 6),
            "signal": _signal(prob),
            "timestamp": timestamp,
        }
        for ticker, prob in sorted(probabilities.items())
    ]
    csv_path = os.path.join(output_dir, "forecasting_probabilities.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"[forecasting] Saved probabilities → {csv_path}")

    # --- forecasting_model_report.txt ---
    txt_path = os.path.join(output_dir, "forecasting_model_report.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Forecasting Model Report\nGenerated: {timestamp}\n")
        fh.write("=" * 60 + "\n\n")
        for ticker in sorted(model_reports):
            m = model_reports[ticker]
            fh.write(f"Ticker: {ticker}\n")
            fh.write("-" * 40 + "\n")
            fh.write(f"Accuracy : {m['accuracy']:.4f}\n")
            fh.write(f"Precision: {m['precision']:.4f}\n")
            fh.write(f"Recall   : {m['recall']:.4f}\n")
            fh.write("\nClassification Report:\n")
            fh.write(m["report"])
            fh.write("\n" + "=" * 60 + "\n\n")
    print(f"[forecasting] Saved model report → {txt_path}")


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def run_forecasting(data_dict: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Train a model for each ticker, save results to disk, and return probabilities.

    Iterates over all tickers in *data_dict*, trains an independent Random Forest for
    each, and predicts the probability that the next 5-day return will be positive
    using the most recent available feature row.  Model metrics are collected across
    all tickers and persisted alongside the probability table via ``save_results()``.

    Args:
        data_dict: Dictionary mapping ticker symbols to OHLCV DataFrames, as returned
                   by ``data_loader.load_data()``.

    Returns:
        Dictionary mapping each ticker to its predicted probability in [0, 1].
        Tickers that fail feature construction or training are skipped with a warning.

    Side effects:
        Writes ``data/processed/forecasting_probabilities.csv`` and
        ``data/processed/forecasting_model_report.txt`` via ``save_results()``.

    Example:
        >>> probabilities = run_forecasting(data_dict)
        >>> for ticker, p in sorted(probabilities.items()):
        ...     print(f"{ticker}: {p:.1%} chance of positive next-week return")
    """
    results: dict[str, float] = {}
    model_reports: dict[str, dict] = {}

    for ticker in data_dict:
        try:
            model, _, metrics = train_model(ticker, data_dict)

            # Use the most recent row (excluding the NaN-target tail) for inference.
            # build_features already drops the last 5 rows (no target), so the last
            # row of feat is the most recent day with a known, usable feature vector.
            feat = build_features(data_dict[ticker])
            latest = feat.iloc[[-1]]  # keep as DataFrame for predict_proba

            prob = predict_proba(model, latest)
            results[ticker] = prob
            model_reports[ticker] = metrics

            print(f"  [{ticker}] P(positive next-week return) = {prob:.1%}")

        except (KeyError, ValueError) as exc:
            print(f"  WARNING: skipping {ticker} — {exc}")

    print(f"\n[forecasting] Done. {len(results)}/{len(data_dict)} tickers processed.")

    if results:
        save_results(results, model_reports)

    return results


# ---------------------------------------------------------------------------
# Module smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from data_loader import load_data

    data = load_data()
    probabilities = run_forecasting(data)

    print("\n--- Next-Week Return Probabilities ---")
    for ticker, prob in sorted(probabilities.items()):
        bar = "#" * int(prob * 20)
        print(f"  {ticker:<6} {prob:.1%}  [{bar:<20}]")
