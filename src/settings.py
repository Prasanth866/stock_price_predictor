"""Settings and constants for transformer-based stock forecasting."""
from datetime import datetime

# Date defaults
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Top 10 Indian stocks (NSE tickers) for default forecasting
TOP_10_INDIAN_TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "ICICIBANK.NS",
    "INFY.NS",
    "HINDUNILVR.NS",
    "ITC.NS",
    "SBIN.NS",
    "BHARTIARTL.NS",
    "LT.NS",
]

# Feature set for candle prediction
FEATURE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

# Transformer model configuration
SEQUENCE_LENGTH = 40
PREDICTION_HORIZON = 3
TRAIN_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MODEL_DIR = "models"

# Intraday / real-time defaults
INTRADAY_INTERVAL = "60m"  # one-hour bars
INTRADAY_LOOKBACK_DAYS = 10
INTRADAY_PREDICTION_STEPS = 6  # predict next 6 hours

# Database
SUPABASE_TABLE_NAME = "stock_price_predictions"
