import torch
import torch.nn as nn


class iTransformer(nn.Module):
    """
    PyTorch implementation of the iTransformer model for time series forecasting.

    The iTransformer inverts the usual time/feature roles: each variate (feature) is treated
    as a token, and the transformer attends across variates. This improves how feature
    interactions are modeled for multivariate series.
    """

    def __init__(
        self,
        pred_len: int,
        seq_len: int,
        num_variates: int,
        d_model: int = 512,
        n_heads: int = 8,
        num_encoder_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
    ):
        super().__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.num_variates = num_variates

        # Embeds each univariate history of length seq_len into d_model.
        self.embedding = nn.Linear(seq_len, d_model)
        self.embed_norm = nn.LayerNorm(d_model)
        self.embed_dropout = nn.Dropout(dropout)
        # Learnable variate (token) embeddings to encode identity of each feature.
        self.variate_emb = nn.Parameter(torch.randn(num_variates, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        # Projects back to pred_len horizon.
        self.projection = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shaped (batch_size, seq_len, num_variates).
        Returns:
            Tensor shaped (batch_size, pred_len, num_variates).
        """
        # Treat each variate as a token -> (batch, num_variates, seq_len)
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.embed_norm(x)
        x = self.embed_dropout(x)
        x = x + self.variate_emb.unsqueeze(0)

        x = self.transformer_encoder(x)
        x = self.projection(x)

        # Restore to (batch, pred_len, num_variates)
        return x.permute(0, 2, 1)


if __name__ == "__main__":
    # Lightweight sanity check
    batch_size = 8
    seq_len = 96
    pred_len = 24
    num_variates = 7

    model = iTransformer(pred_len, seq_len, num_variates)
    dummy = torch.randn(batch_size, seq_len, num_variates)
    out = model(dummy)
    assert out.shape == (batch_size, pred_len, num_variates)
    print("Output shape:", out.shape)
