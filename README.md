# Transformer Stock Price Predictor

## Project Overview
This project forecasts **future OHLCV candles** for Indian stocks using a transformer-based time series model (iTransformer). The pipeline pulls historical data, trains (or loads) a model per ticker, and predicts the next few trading-day candles.

**Note:** This is for educational purposes and not financial advice.

## What It Does
1. Pulls historical OHLCV data from Yahoo Finance via `yfinance`.
2. Trains an iTransformer per ticker (or loads a saved model from `models/`).
3. Predicts the next `PREDICTION_HORIZON` candles.
4. Stores predictions in Supabase (optional).

## Default Tickers (Top 10 India)
The default list is defined in `src/settings.py` as `TOP_10_INDIAN_TICKERS` and uses NSE symbols (e.g., `RELIANCE.NS`, `TCS.NS`).

## Installation
```bash
make install-dev
```

## Running Predictions
```bash
poetry run python -m src.main
# or
./scripts/run_app.sh
```

### Intraday (next few hours)
Predict the next few hourly candles using the intraday mode:
```bash
# defaults: interval=60m, horizon=6 steps (next 6 hours), lookback=10 days
./scripts/run_app.sh --intraday

# customise interval / horizon / lookback
./scripts/run_app.sh --intraday --interval 30m --horizon 8 --lookback-days 7
```

## Configuration
Edit `src/settings.py` to adjust:
- `TOP_10_INDIAN_TICKERS`
- `SEQUENCE_LENGTH`
- `PREDICTION_HORIZON`
- `TRAIN_EPOCHS`, `BATCH_SIZE`, `LEARNING_RATE`
- `MODEL_DIR`
- `START_DATE`, `END_DATE`

## Streamlit Dashboard
```bash
poetry run streamlit run src/streamlit_app.py
```

## Supabase
If you want to store prediction runs, set:
```
SUPABASE_URL=...
SUPABASE_KEY=...
```

The app writes to the table specified by `SUPABASE_TABLE_NAME` in `src/settings.py`.
