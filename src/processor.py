"""Data processing utilities for preparing and formatting OHLCV candle data."""

from datetime import date, timedelta

import pandas as pd


def preprocess_data(
    all_stock_data: dict[str, pd.DataFrame], as_date_index: bool = True
) -> dict[str, pd.DataFrame]:
    """
    Normalize candle data by enforcing date or datetime indexes, sorting, and dropping missing rows.

    Args:
        all_stock_data: Dictionary mapping ticker to DataFrame with date/datetime index
        as_date_index: If True, convert index to dates (daily bars). If False, keep datetime.

    Returns:
        Dictionary of DataFrames with cleaned indexes
    """
    if not all_stock_data:
        return {}

    normalised_all_stock_data: dict[str, pd.DataFrame] = {}
    for ticker, df in all_stock_data.items():
        df_copy = df.copy()
        if as_date_index:
            df_copy.index = pd.to_datetime(df_copy.index).date
        else:
            df_copy.index = pd.to_datetime(df_copy.index)
        df_copy = df_copy.sort_index()
        df_copy = df_copy.dropna()
        normalised_all_stock_data[ticker] = df_copy

    return normalised_all_stock_data


def candles_to_records(df: pd.DataFrame) -> list[dict[str, float | str]]:
    """Convert a candle DataFrame into JSON-serializable records."""
    if df.empty:
        return []

    records: list[dict[str, float | str]] = []
    for idx, row in df.iterrows():
        if isinstance(idx, pd.Timestamp):
            date_str = idx.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(idx, date):
            date_str = idx.strftime("%Y-%m-%d")
        else:
            date_str = str(idx)

        record = {
            "date": date_str,
            "open": float(row["Open"]),
            "high": float(row["High"]),
            "low": float(row["Low"]),
            "close": float(row["Close"]),
            "volume": float(row["Volume"]),
        }
        records.append(record)
    return records


def collect_recent_candles(
    portfolio_data: dict[str, pd.DataFrame],
    window: pd.Timedelta | timedelta = timedelta(days=30),
    days: int | None = None,
) -> dict[str, list[dict[str, float | str]]]:
    """
    Collect the most recent candles for each ticker over the given trailing window.

    Args:
        portfolio_data: Dictionary of historical DataFrames per ticker.
        window: Time window to include (timedelta). Defaults to 30 days.

    Returns:
        Dictionary mapping ticker to a list of candle dicts ordered by date.
    """
    recent_candles: dict[str, list[dict[str, float | str]]] = {}

    if days is not None:
        window = timedelta(days=days)

    for ticker, df in portfolio_data.items():
        if df.empty:
            recent_candles[ticker] = []
            continue

        last_timestamp = df.index[-1]
        cutoff = last_timestamp - window
        recent_df = df.loc[df.index >= cutoff]
        recent_candles[ticker] = candles_to_records(recent_df)

    return recent_candles
