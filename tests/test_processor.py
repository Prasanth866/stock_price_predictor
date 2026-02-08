"""Tests for processor module."""

from datetime import date

import numpy as np
import pandas as pd

from src.processor import candles_to_records, collect_recent_candles, preprocess_data


class TestProcessor:
    """Test data processing utilities."""

    def test_preprocess_data(self) -> None:
        """Test preprocessing data - ensures date index and sorting."""
        dates1 = pd.date_range("2024-01-01", periods=10, freq="D")
        dates2 = pd.date_range("2024-01-03", periods=8, freq="D")

        data1 = pd.DataFrame(
            {
                "Open": np.random.randn(10) * 10 + 100,
                "High": np.random.randn(10) * 10 + 101,
                "Low": np.random.randn(10) * 10 + 99,
                "Close": np.random.randn(10) * 10 + 100,
                "Volume": np.random.randint(1000, 2000, size=10),
            },
            index=dates1,
        )
        data2 = pd.DataFrame(
            {
                "Open": np.random.randn(8) * 10 + 50,
                "High": np.random.randn(8) * 10 + 51,
                "Low": np.random.randn(8) * 10 + 49,
                "Close": np.random.randn(8) * 10 + 50,
                "Volume": np.random.randint(1000, 2000, size=8),
            },
            index=dates2,
        )

        data_dict = {"TICKER1": data1, "TICKER2": data2}
        cleaned = preprocess_data(data_dict)

        assert isinstance(cleaned, dict)
        assert len(cleaned) == 2
        assert all(isinstance(d, date) for d in cleaned["TICKER1"].index)
        assert all(isinstance(d, date) for d in cleaned["TICKER2"].index)
        assert cleaned["TICKER1"].index.is_monotonic_increasing

    def test_candles_to_records(self) -> None:
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000, 1100, 1200],
            },
            index=[d.date() for d in dates],
        )

        records = candles_to_records(df)
        assert len(records) == 3
        assert records[0]["open"] == 100.0
        assert records[0]["high"] == 101.0
        assert records[0]["low"] == 99.0
        assert records[0]["close"] == 100.5
        assert records[0]["volume"] == 1000.0

    def test_collect_recent_candles(self) -> None:
        dates = pd.date_range("2024-01-01", periods=40, freq="D")
        df = pd.DataFrame(
            {
                "Open": np.linspace(100, 140, num=40),
                "High": np.linspace(101, 141, num=40),
                "Low": np.linspace(99, 139, num=40),
                "Close": np.linspace(100, 140, num=40),
                "Volume": np.random.randint(1000, 2000, size=40),
            },
            index=[d.date() for d in dates],
        )
        portfolio_data = {"TICKER1": df}

        recent_candles = collect_recent_candles(portfolio_data, days=30)
        assert "TICKER1" in recent_candles
        assert isinstance(recent_candles["TICKER1"], list)
        assert len(recent_candles["TICKER1"]) >= 30
