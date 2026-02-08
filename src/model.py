"""Transformer-based model utilities for forecasting future candles."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

from src.transformer_model import iTransformer

logger = logging.getLogger(__name__)


class TransformerPricePredictor:
    """Train and use an iTransformer model to forecast OHLCV candles."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        feature_cols: Iterable[str],
        d_model: int = 128,
        n_heads: int = 4,
        num_encoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        batch_size: int = 32,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        grad_clip: float | None = 1.0,
        device: str | None = None,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_cols = list(feature_cols)
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.model = iTransformer(
            pred_len=pred_len,
            seq_len=seq_len,
            num_variates=len(self.feature_cols),
            d_model=d_model,
            n_heads=n_heads,
            num_encoder_layers=num_encoder_layers,
            d_ff=d_ff,
            dropout=dropout,
            activation=activation,
        ).to(self.device)
        self.scaler: MinMaxScaler | None = None

    def _build_sequences(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
        values = df[self.feature_cols].astype("float32").values
        if len(values) < self.seq_len + self.pred_len:
            raise ValueError(
                "Not enough data to build training sequences "
                f"(need >= {self.seq_len + self.pred_len}, got {len(values)})."
            )

        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(values)

        x_chunks = []
        y_chunks = []
        max_start = len(scaled) - self.seq_len - self.pred_len + 1
        for idx in range(max_start):
            x_chunks.append(scaled[idx : idx + self.seq_len])
            y_chunks.append(scaled[idx + self.seq_len : idx + self.seq_len + self.pred_len])

        x_array = np.array(x_chunks, dtype=np.float32)
        y_array = np.array(y_chunks, dtype=np.float32)
        return x_array, y_array, scaler

    def fit(self, df: pd.DataFrame) -> None:
        """Train the transformer model on a single ticker's OHLCV data."""
        x_train, y_train, scaler = self._build_sequences(df)
        self.scaler = scaler

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(x_train), torch.tensor(y_train)
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, drop_last=False
        )

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        self.model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                output = self.model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                if self.grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                optimizer.step()
                epoch_loss += float(loss.item())

            avg_loss = epoch_loss / max(len(loader), 1)
            scheduler.step()
            logger.info("Epoch %s/%s - loss: %.6f", epoch, self.epochs, avg_loss)

    def predict_future(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict the next pred_len candles."""
        if self.scaler is None:
            raise RuntimeError("Model must be fitted or loaded before predicting.")
        if len(df) < self.seq_len:
            raise ValueError(
                f"Not enough data to predict (need >= {self.seq_len}, got {len(df)})."
            )

        recent = df[self.feature_cols].astype("float32").values[-self.seq_len :]
        scaled_recent = self.scaler.transform(recent)

        input_tensor = torch.tensor(scaled_recent, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            pred_scaled = self.model(input_tensor).cpu().numpy()

        pred_scaled = pred_scaled.reshape(self.pred_len, len(self.feature_cols))
        pred_values = self.scaler.inverse_transform(pred_scaled)
        return pd.DataFrame(pred_values, columns=self.feature_cols)

    def save(self, path: Path, trained_until: str | None = None) -> None:
        if self.scaler is None:
            raise RuntimeError("Cannot save model without a fitted scaler.")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "seq_len": self.seq_len,
                "pred_len": self.pred_len,
                "trained_until": trained_until,
            },
            path,
        )

    def load(self, path: Path) -> None:
        # Checkpoints include a fitted scaler, so we opt out of weights_only loading.
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint.get("seq_len") != self.seq_len or checkpoint.get("pred_len") != self.pred_len:
            raise ValueError("Checkpoint config does not match current model settings.")
        self.feature_cols = list(checkpoint.get("feature_cols", self.feature_cols))
        load_result = self.model.load_state_dict(checkpoint["state_dict"], strict=False)
        if load_result.missing_keys:
            logger.warning("Missing keys while loading checkpoint (initialized randomly): %s", load_result.missing_keys)
        if load_result.unexpected_keys:
            logger.warning("Unexpected keys in checkpoint (ignored): %s", load_result.unexpected_keys)
        self.scaler = checkpoint["scaler"]


def build_model_path(model_dir: str | Path, ticker: str, interval: str = "1d") -> Path:
    safe_ticker = ticker.replace("/", "-")
    suffix = f"_{interval}" if interval != "1d" else ""
    return Path(model_dir) / f"{safe_ticker}{suffix}.pt"
