"""Transformer blocks and positional encodings."""
from .._imports import *

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]

class TransformerEncoder(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=4, dim_feedforward=256, dropout=0.1):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, key_padding_mask=None, attn_mask=None) -> torch.Tensor:
        """
        Args:
            x: [B, L, d_model] input sequence.
            key_padding_mask: optional [B, L] mask (True for padded positions).
            attn_mask: optional [L, L] attention mask (e.g., causal mask).
        """
        return self.encoder(
            x,
            mask=attn_mask,
            src_key_padding_mask=key_padding_mask,
        )
