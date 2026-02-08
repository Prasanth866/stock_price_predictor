"""Streamlit dashboard for transformer-based candle forecasts."""

from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.database import get_supabase_client
from src.settings import SUPABASE_TABLE_NAME

st.set_page_config(page_title="Stock Price Predictor", layout="wide")


@st.cache_data(ttl=300)
def load_supabase_predictions() -> pd.DataFrame:
    """Return latest Supabase rows (one per ticker per date)."""
    client = get_supabase_client()
    if client is None:
        return pd.DataFrame()

    response = (
        client.table(SUPABASE_TABLE_NAME)
        .select("*")
        .order("as_of_date", desc=True)
        .order("created_at", desc=True)
        .execute()
    )
    data = getattr(response, "data", None)
    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    if "as_of_date" in df.columns:
        df["as_of_date"] = pd.to_datetime(df["as_of_date"]).dt.date
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"])

    df = df.sort_values(["as_of_date", "created_at"], ascending=[True, False])
    df = df.drop_duplicates(subset=["as_of_date", "ticker"], keep="first")

    if "predicted_candles" in df.columns:
        df["predicted_candles"] = df["predicted_candles"].apply(_parse_candles)
    if "recent_candles" in df.columns:
        df["recent_candles"] = df["recent_candles"].apply(_parse_candles)

    return df


def _parse_candles(raw: object) -> list[dict[str, float | str]]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if isinstance(decoded, list):
            return decoded
    return []


def _candles_to_df(records: list[dict[str, float | str]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def _latest_close(records: list[dict[str, float | str]]) -> float | None:
    if not records:
        return None
    return float(records[-1].get("close", 0.0))


def run_dashboard() -> None:
    st.title("Stock Price Predictor Dashboard")
    st.caption("Transformer-based forecasts for future OHLCV candles.")

    df = load_supabase_predictions()
    if df.empty:
        st.info("No prediction data available. Run the prediction pipeline to populate Supabase.")
        return

    available_dates = sorted(df["as_of_date"].unique(), reverse=True)
    selected_date = st.selectbox(
        "Select as-of date", options=available_dates, format_func=lambda d: d.strftime("%Y-%m-%d")
    )

    date_df = df[df["as_of_date"] == selected_date].copy().sort_values("ticker")
    if "prediction_horizon" not in date_df.columns:
        date_df["prediction_horizon"] = None
    date_df["next_close"] = date_df["predicted_candles"].apply(
        lambda x: float(x[0]["close"]) if x else None
    )

    st.subheader("Next Candle Preview")
    summary_table = date_df[["ticker", "next_close", "prediction_horizon"]].copy()
    summary_table = summary_table.rename(
        columns={
            "ticker": "Ticker",
            "next_close": "Next Close",
            "prediction_horizon": "Horizon",
        }
    )
    st.dataframe(
        summary_table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Next Close": st.column_config.NumberColumn(format="%.2f"),
        },
    )

    tickers = date_df["ticker"].tolist()
    selected_ticker = st.selectbox("Select ticker for detail view", options=tickers, index=0)

    ticker_row = date_df.set_index("ticker").loc[selected_ticker]
    actual_records = ticker_row.get("recent_candles", [])
    predicted_records = ticker_row.get("predicted_candles", [])

    col1, col2, col3 = st.columns(3)
    with col1:
        latest_actual = _latest_close(actual_records)
        st.metric("Latest Actual Close", f"{latest_actual:.2f}" if latest_actual else "—")
    with col2:
        next_close = _latest_close(predicted_records[:1])
        st.metric("Next Predicted Close", f"{next_close:.2f}" if next_close else "—")
    with col3:
        horizon = ticker_row.get("prediction_horizon", "—")
        st.metric("Prediction Horizon", str(horizon))

    actual_df = _candles_to_df(actual_records)
    predicted_df = _candles_to_df(predicted_records)

    st.subheader(f"Candles · {selected_ticker}")
    if actual_df.empty and predicted_df.empty:
        st.info("No candle data available for this ticker yet.")
    else:
        fig = go.Figure()
        if not actual_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=actual_df["date"],
                    open=actual_df["open"],
                    high=actual_df["high"],
                    low=actual_df["low"],
                    close=actual_df["close"],
                    name="Actual",
                )
            )
        if not predicted_df.empty:
            fig.add_trace(
                go.Candlestick(
                    x=predicted_df["date"],
                    open=predicted_df["open"],
                    high=predicted_df["high"],
                    low=predicted_df["low"],
                    close=predicted_df["close"],
                    name="Predicted",
                    increasing_line_color="#ff7f0e",
                    decreasing_line_color="#ff7f0e",
                    opacity=0.7,
                )
            )
        fig.update_layout(height=480, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Predicted Candles")
    if predicted_df.empty:
        st.info("No predicted candles available for this ticker.")
    else:
        st.dataframe(
            predicted_df.rename(
                columns={
                    "date": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )


def main() -> None:
    run_dashboard()


if __name__ == "__main__":
    main()
