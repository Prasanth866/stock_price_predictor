"""Main entry point for transformer-based stock price prediction."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any

import pandas as pd

from src.database import save_results_to_supabase
from src.extractor import extract_data
from src.model import TransformerPricePredictor, build_model_path
from src.processor import candles_to_records, collect_recent_candles, preprocess_data
from src.settings import (
    BATCH_SIZE,
    END_DATE,
    FEATURE_COLUMNS,
    INTRADAY_INTERVAL,
    INTRADAY_LOOKBACK_DAYS,
    INTRADAY_PREDICTION_STEPS,
    LEARNING_RATE,
    MODEL_DIR,
    PREDICTION_HORIZON,
    SEQUENCE_LENGTH,
    START_DATE,
    TOP_10_INDIAN_TICKERS,
    TRAIN_EPOCHS,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_prediction(
    tickers: list[str],
    start_date: str | None = START_DATE,
    end_date: str | None = END_DATE,
    retrain: bool = False,
    interval: str = "1d",
    prediction_horizon: int = PREDICTION_HORIZON,
    lookback_days: int | None = None,
) -> dict[str, Any]:
    """
    Run stock price prediction: pull data, train/load models, and forecast future candles.

    Args:
        tickers: List of stock ticker symbols
        start_date: Start date for historical data (YYYY-MM-DD format).
        end_date: End date for historical data (YYYY-MM-DD format).
        retrain: If True, forces retraining even if a saved model exists.

    Returns:
        Dictionary containing prediction results with keys:
        - date: date object representing date prediction was run
        - predictions: dict[str, list[dict]] of predicted candles per ticker
        - recent_candles: dict[str, list[dict]] of recent actual candles
        - prediction_horizon: int number of future candles predicted
        - model_paths: dict[str, str] of saved model paths per ticker

    Returns empty dict if data extraction fails.
    """

    as_of_date = pd.to_datetime(end_date or pd.Timestamp.utcnow().normalize()).date()
    logger.info(
        "Starting price prediction for tickers: %s as of %s (interval=%s, horizon=%s)",
        tickers,
        as_of_date,
        interval,
        prediction_horizon,
    )

    # 1. Extract historical data
    logger.info("Extracting historical data...")
    all_stock_data = extract_data(
        tickers,
        start_date=start_date,
        end_date=end_date,
        interval=interval,
        lookback_days=lookback_days,
    )
    if not all_stock_data:
        logger.warning("No data extracted. Exiting prediction.")
        return {}

    # 2. Preprocess historical data
    logger.info("Preprocessing data...")
    portfolio_data = preprocess_data(all_stock_data, as_date_index=(interval == "1d"))

    # 3. Train/load transformer models and generate predictions
    logger.info("Training/loading transformer models...")
    predictions: dict[str, list[dict]] = {}
    model_paths: dict[str, str] = {}

    for ticker, df in portfolio_data.items():
        try:
            predictor = TransformerPricePredictor(
                seq_len=SEQUENCE_LENGTH,
                pred_len=prediction_horizon,
                feature_cols=FEATURE_COLUMNS,
                batch_size=BATCH_SIZE,
                epochs=TRAIN_EPOCHS,
                learning_rate=LEARNING_RATE,
            )

            model_path = build_model_path(MODEL_DIR, ticker, interval=interval)
            if model_path.exists() and not retrain:
                logger.info("Loading saved model for %s", ticker)
                predictor.load(model_path)
            else:
                logger.info("Training model for %s", ticker)
                predictor.fit(df)
                predictor.save(model_path, trained_until=str(df.index[-1]))

            future_df = predictor.predict_future(df)

            # Assign future timestamps to predictions
            last_ts = pd.to_datetime(df.index[-1])
            if interval == "1d":
                future_index = pd.bdate_range(
                    start=last_ts + pd.Timedelta(days=1), periods=prediction_horizon
                )
                future_df.index = [d.date() for d in future_index]
            else:
                step = pd.Timedelta(interval)
                future_index = pd.date_range(
                    start=last_ts + step, periods=prediction_horizon, freq=step
                )
                future_df.index = future_index
            future_df.index.name = "Date"

            predictions[ticker] = candles_to_records(future_df)
            model_paths[ticker] = str(model_path)
        except Exception as err:
            logger.error("Failed to process %s: %s", ticker, err)
            continue

    # 4. Collect recent candle history for the past month
    window = pd.Timedelta(days=30) if interval == "1d" else pd.Timedelta(days=lookback_days or 7)
    recent_candles = collect_recent_candles(portfolio_data, window=window)

    # 5. Log results
    logger.info("Prediction Results")
    logger.info("Date: %s", as_of_date)
    for ticker, candle_list in predictions.items():
        if not candle_list:
            logger.info("  %s: No predictions", ticker)
        else:
            first = candle_list[0]
            logger.info(
                "  %s: Next close %.2f (horizon=%s)",
                ticker,
                first.get("close", 0.0),
                prediction_horizon,
            )

    return {
        "date": as_of_date,
        "predictions": predictions,
        "recent_candles": recent_candles,
        "prediction_horizon": prediction_horizon,
        "model_paths": model_paths,
    }


def main() -> None:
    """Main CLI entry point - saves results to Supabase."""
    parser = argparse.ArgumentParser(description="Stock price prediction")
    parser.add_argument("--intraday", action="store_true", help="Use intraday (hourly) mode")
    parser.add_argument("--interval", default=None, help="yfinance interval (e.g., 60m, 30m)")
    parser.add_argument("--horizon", type=int, default=None, help="Prediction horizon in steps")
    parser.add_argument(
        "--lookback-days", type=int, default=None, help="Lookback window (days) for intraday mode"
    )
    args = parser.parse_args()

    interval = args.interval or ("1d" if not args.intraday else INTRADAY_INTERVAL)
    prediction_horizon = args.horizon or (
        PREDICTION_HORIZON if interval == "1d" else INTRADAY_PREDICTION_STEPS
    )
    lookback_days = args.lookback_days or (None if interval == "1d" else INTRADAY_LOOKBACK_DAYS)

    try:
        result = run_prediction(
            tickers=TOP_10_INDIAN_TICKERS,
            start_date=START_DATE if interval == "1d" else None,
            end_date=END_DATE if interval == "1d" else None,
            interval=interval,
            prediction_horizon=prediction_horizon,
            lookback_days=lookback_days,
        )

        if not result:
            logger.error("Prediction returned empty result")
            sys.exit(1)

        try:
            saved = save_results_to_supabase(result)
            if saved:
                print("\nResults successfully saved to Supabase database")
            else:
                print("\nSupabase not configured; skipping database save")
        except Exception as db_error:
            logger.error(f"Failed to save to Supabase: {db_error}")
            print(f"\nWarning: Failed to save to Supabase: {db_error}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        print(f"Error during prediction: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
