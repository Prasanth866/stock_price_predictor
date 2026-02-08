"""Tests for database module."""

import json
import uuid
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.database import get_supabase_client, save_results_to_supabase
from src.settings import SUPABASE_TABLE_NAME


class TestGetSupabaseClient:
    """Test Supabase client creation."""

    @patch.dict(
        "os.environ", {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
    )
    @patch("src.database.create_client")
    def test_get_supabase_client_with_credentials(self, mock_create_client: MagicMock) -> None:
        """Test get_supabase_client returns client when credentials are available."""
        mock_client = MagicMock()
        mock_create_client.return_value = mock_client

        result = get_supabase_client()

        assert result is not None
        assert result == mock_client
        mock_create_client.assert_called_once_with("https://test.supabase.co", "test-key")

    @patch.dict("os.environ", {}, clear=True)
    def test_get_supabase_client_without_credentials(self) -> None:
        """Test get_supabase_client returns None when both credentials are missing."""
        result = get_supabase_client()
        assert result is None


class TestSaveResultsToSupabase:
    """Test saving results to Supabase."""

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_success(self, mock_get_client: MagicMock) -> None:
        """Test successfully saving results to Supabase."""
        # Setup mocks
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Test data
        result = {
            "date": date(2024, 1, 31),
            "predictions": {
                "AAPL": [
                    {
                        "date": "2024-02-01",
                        "open": 150.0,
                        "high": 151.0,
                        "low": 149.0,
                        "close": 150.25,
                        "volume": 1000,
                    }
                ],
                "MSFT": [
                    {
                        "date": "2024-02-01",
                        "open": 380.0,
                        "high": 382.0,
                        "low": 379.0,
                        "close": 380.5,
                        "volume": 2000,
                    }
                ],
            },
            "recent_candles": {"AAPL": [], "MSFT": []},
            "prediction_horizon": 1,
            "model_paths": {"AAPL": "models/AAPL.pt", "MSFT": "models/MSFT.pt"},
        }

        # Call function
        save_results_to_supabase(result)

        # Verify calls
        mock_get_client.assert_called_once()
        mock_client.table.assert_called_once_with(SUPABASE_TABLE_NAME)
        mock_table.insert.assert_called_once()

        # Check that insert was called with correct structure
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 2  # Two stocks

        # Check first row structure
        first_row = insert_call_args[0]
        assert "id" in first_row
        assert "created_at" in first_row
        assert "as_of_date" in first_row
        assert "ticker" in first_row
        assert "prediction_start_date" in first_row
        assert "prediction_horizon" in first_row
        assert "predicted_candles" in first_row
        assert "recent_candles" in first_row
        assert "model_path" in first_row

        # Check ID is a valid UUID string
        uuid.UUID(first_row["id"])  # Will raise if invalid UUID

        # Check values
        assert first_row["ticker"] in ("AAPL", "MSFT")
        assert first_row["as_of_date"] == "2024-01-31"
        if first_row["ticker"] == "AAPL":
            predicted = json.loads(first_row["predicted_candles"])
            assert predicted[0]["close"] == 150.25
            assert first_row["prediction_start_date"] == "2024-02-01"
            assert first_row["prediction_horizon"] == 1
            assert first_row["model_path"] == "models/AAPL.pt"
        else:
            predicted = json.loads(first_row["predicted_candles"])
            assert predicted[0]["close"] == 380.5
            assert first_row["prediction_start_date"] == "2024-02-01"
            assert first_row["prediction_horizon"] == 1
            assert first_row["model_path"] == "models/MSFT.pt"

        mock_insert.execute.assert_called_once()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_no_predictions(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase returns early when no predictions."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        result = {
            "date": date(2024, 1, 31),
            "predictions": {},
            "recent_candles": {},
            "prediction_horizon": 1,
        }

        save_results_to_supabase(result)

        # Should not attempt to insert
        mock_client.table.assert_not_called()

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_missing_keys(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles missing keys gracefully."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()
        mock_execute = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.return_value = mock_execute
        mock_get_client.return_value = mock_client

        # Missing optional fields like model_paths
        result = {
            "date": date(2024, 1, 31),
            "predictions": {
                "AAPL": [
                    {
                        "date": "2024-02-01",
                        "open": 150.0,
                        "high": 151.0,
                        "low": 149.0,
                        "close": 150.25,
                        "volume": 1000,
                    }
                ]
            },
            "recent_candles": {"AAPL": []},
            "prediction_horizon": 1,
        }

        save_results_to_supabase(result)

        # Should still work, using defaults
        insert_call_args = mock_table.insert.call_args[0][0]
        assert len(insert_call_args) == 1
        assert insert_call_args[0]["ticker"] == "AAPL"
        assert json.loads(insert_call_args[0]["predicted_candles"])[0]["close"] == 150.25
        assert insert_call_args[0]["prediction_horizon"] == 1

    @patch("src.database.get_supabase_client")
    def test_save_results_to_supabase_insert_failure(self, mock_get_client: MagicMock) -> None:
        """Test save_results_to_supabase handles insertion failure."""
        mock_client = MagicMock()
        mock_table = MagicMock()
        mock_insert = MagicMock()

        mock_client.table.return_value = mock_table
        mock_table.insert.return_value = mock_insert
        mock_insert.execute.side_effect = Exception("Database connection error")
        mock_get_client.return_value = mock_client

        result = {
            "date": date(2024, 1, 31),
            "predictions": {
                "AAPL": [
                    {
                        "date": "2024-02-01",
                        "open": 150.0,
                        "high": 151.0,
                        "low": 149.0,
                        "close": 150.25,
                        "volume": 1000,
                    }
                ]
            },
            "recent_candles": {"AAPL": []},
            "prediction_horizon": 1,
        }

        # Should propagate the exception
        with pytest.raises(Exception, match="Database connection error"):
            save_results_to_supabase(result)
