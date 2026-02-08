"""Tests for transformer model module."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.model import TransformerPricePredictor, build_model_path


class TestTransformerModel:
    """Test transformer-based forecasting."""

    def _build_sample_df(self, rows: int = 80) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        base = np.linspace(100, 120, num=rows)
        data = {
            "Open": base + np.random.randn(rows),
            "High": base + np.random.rand(rows) * 2,
            "Low": base - np.random.rand(rows) * 2,
            "Close": base + np.random.randn(rows) * 0.5,
            "Volume": np.random.randint(1_000_000, 2_000_000, size=rows),
        }
        return pd.DataFrame(data, index=[d.date() for d in dates])

    def test_fit_and_predict(self) -> None:
        df = self._build_sample_df()
        predictor = TransformerPricePredictor(
            seq_len=10,
            pred_len=3,
            feature_cols=["Open", "High", "Low", "Close", "Volume"],
            epochs=1,
            batch_size=4,
        )

        predictor.fit(df)
        preds = predictor.predict_future(df)

        assert isinstance(preds, pd.DataFrame)
        assert preds.shape == (3, 5)

    def test_save_and_load(self, tmp_path: Path) -> None:
        df = self._build_sample_df()
        predictor = TransformerPricePredictor(
            seq_len=10,
            pred_len=2,
            feature_cols=["Open", "High", "Low", "Close", "Volume"],
            epochs=1,
            batch_size=4,
        )
        predictor.fit(df)

        model_path = tmp_path / "TEST.pt"
        predictor.save(model_path, trained_until=str(df.index[-1]))

        reloaded = TransformerPricePredictor(
            seq_len=10,
            pred_len=2,
            feature_cols=["Open", "High", "Low", "Close", "Volume"],
        )
        reloaded.load(model_path)
        preds = reloaded.predict_future(df)
        assert preds.shape == (2, 5)

    def test_build_model_path(self) -> None:
        path = build_model_path("models", "TCS.NS")
        assert path.as_posix().endswith("models/TCS.NS.pt")

    def test_build_model_path_intraday(self) -> None:
        path = build_model_path("models", "TCS.NS", interval="60m")
        assert path.as_posix().endswith("models/TCS.NS_60m.pt")
