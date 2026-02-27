"""
Temporal Transformer
====================
Self-attention module that models long-range temporal dependencies
across video frame features. Uses sinusoidal positional encoding so
the model knows the order of frames.

Architecture:
    Input (B, T, D) → PositionalEncoding → TransformerEncoder → LayerNorm → MeanPool → Output (B, D)
"""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (from "Attention Is All You Need").

    Adds a unique signal to each position in the sequence so the
    Transformer can distinguish frame 1 from frame 16, etc.
    """

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            (batch_size, seq_len, d_model) with positional encoding added
        """
        x = x + self.pe[:, : x.size(1), :]
        return x


class TemporalTransformer(nn.Module):
    """
    Wraps a standard Transformer encoder with positional encoding
    and global average pooling.

    Args:
        input_dim        : feature dimension of each frame (e.g. 512)
        num_heads        : attention heads
        num_layers       : number of encoder layers
        dim_feedforward  : hidden size of the feed-forward sub-layer
        dropout          : dropout rate
    """

    def __init__(
        self,
        input_dim,
        num_heads=8,
        num_layers=1,
        dim_feedforward=512,
        dropout=0.1,
    ):
        super(TemporalTransformer, self).__init__()

        self.input_dim = input_dim
        self.pos_encoder = PositionalEncoding(input_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
        Returns:
            (batch_size, feature_dim) — globally averaged features
        """
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.norm(x)
        x = torch.mean(x, dim=1)  # Global Average Pooling over time
        return x
