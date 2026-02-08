"""Database operations for saving prediction results to Supabase."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any

from supabase import Client, create_client

from src.settings import SUPABASE_TABLE_NAME

logger = logging.getLogger(__name__)


def get_supabase_client() -> Client | None:
    """
    Create and return Supabase client from environment variables.

    Returns:
        Supabase client if credentials are available, None otherwise
    """
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.warning("Supabase credentials not found in environment variables")
        return None

    return create_client(url, key)


def save_results_to_supabase(result: dict[str, Any]) -> bool:
    """
    Save prediction results to Supabase database.

    Args:
        result: Dictionary containing prediction results from run_prediction()
            Expected keys: predictions, recent_candles, prediction_horizon

    Returns:
        True if rows were inserted, False if skipped due to missing credentials or data

    Raises:
        Exception: If insertion fails after a client is available
    """
    supabase = get_supabase_client()
    if supabase is None:
        logger.info("Skipping Supabase save because credentials are missing.")
        return False

    as_of_date = result.get("date")
    predictions = result.get("predictions", {})
    recent_candles = result.get("recent_candles", {})
    prediction_horizon = result.get("prediction_horizon")
    model_paths = result.get("model_paths", {})

    if not predictions:
        logger.warning("No predictions to save")
        return False

    # Prepare rows for insertion - one row per stock
    rows = []
    for ticker in predictions.keys():
        ticker_predictions = predictions.get(ticker, [])
        prediction_start_date = None
        if ticker_predictions:
            prediction_start_date = ticker_predictions[0].get("date")
        row = {
            "id": str(uuid.uuid4()),
            "created_at": datetime.now().isoformat(),
            "as_of_date": as_of_date.isoformat() if as_of_date else None,
            "ticker": ticker,
            "prediction_start_date": prediction_start_date,
            "prediction_horizon": int(prediction_horizon) if prediction_horizon is not None else None,
            "predicted_candles": json.dumps(ticker_predictions),
            "recent_candles": json.dumps(recent_candles.get(ticker, [])),
            "model_path": model_paths.get(ticker),
        }
        rows.append(row)

    logger.info(f"Inserting {len(rows)} rows into Supabase...")
    (supabase.table(SUPABASE_TABLE_NAME).insert(rows).execute())

    logger.info("Successfully saved %s predictions to Supabase", len(rows))
    return True
