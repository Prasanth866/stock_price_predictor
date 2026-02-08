"""Tests for extractor module."""

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd

from src.extractor import extract_data


class TestExtractor:
    """Test data extraction."""

    @patch("src.extractor.yf.Ticker")
    def test_extract_data(self, mock_ticker: MagicMock) -> None:
        """Test extracting historical data with mocked yfinance."""
        tickers = ["MSFT", "AAPL"]

        # Mock yfinance history output
        mock_history = pd.DataFrame(
            {
                "Open": [1.0, 2.0],
                "High": [1.2, 2.2],
                "Low": [0.8, 1.8],
                "Close": [1.1, 2.1],
                "Volume": [100, 200],
            },
            index=pd.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        mock_ticker.return_value.history.return_value = mock_history

        data = extract_data(tickers, start_date="2024-01-01")

        assert isinstance(data, dict)
        assert len(data) > 0
        for ticker in tickers:
            if ticker in data:
                assert isinstance(data[ticker], pd.DataFrame)
                assert "Open" in data[ticker].columns
                assert "High" in data[ticker].columns
                assert "Low" in data[ticker].columns
                assert "Close" in data[ticker].columns
                assert "Volume" in data[ticker].columns
                assert data[ticker].index.name == "Date"
                # Check that index is date type
                assert all(isinstance(d, date) for d in data[ticker].index)

    @patch("src.extractor.yf.Ticker")
    def test_extract_data_with_end_date(self, mock_ticker: MagicMock) -> None:
        """Test extracting data with end_date filter using mocked yfinance."""
        tickers = ["KO"]
        end_date = "2024-06-01"

        mock_history = pd.DataFrame(
            {
                "Open": [10.0, 11.0],
                "High": [10.5, 11.5],
                "Low": [9.5, 10.5],
                "Close": [10.2, 11.2],
                "Volume": [1000, 1100],
            },
            index=pd.to_datetime(["2024-01-02", "2024-05-30"]),
        )
        mock_ticker.return_value.history.return_value = mock_history

        data = extract_data(tickers, start_date="2024-01-01", end_date=end_date)

        assert isinstance(data, dict)
        if tickers[0] in data:
            assert isinstance(data[tickers[0]], pd.DataFrame)
            # Check that all dates are <= end_date
            if len(data[tickers[0]]) > 0:
                assert all(
                    pd.Timestamp(d) <= pd.Timestamp(end_date) for d in data[tickers[0]].index
                )
