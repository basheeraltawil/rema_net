"""
REMA-Net Models
===============
Contains all neural network components:
- TemporalTransformer (positional encoding + self-attention over time)
- SpatialStream  (RGB frames → ResNet-18 → Transformer → class scores)
- TemporalStream (Optical flow → ResNet-18 → Transformer → class scores)
- CaptionStream  (PIL image → BLIP caption → BERT-tiny embedding)
- CaptionGating  (learns to blend RGB and Caption features)
- MATResNet      (full multi-stream model that combines everything)
"""

from .temporal_transformer import TemporalTransformer, PositionalEncoding
from .streams import (
    MultiHeadAttention,
    SpatialStream,
    TemporalStream,
    CaptionStream,
    CaptionGating,
)
from .rema_net import REMANet

__all__ = [
    "TemporalTransformer",
    "PositionalEncoding",
    "MultiHeadAttention",
    "SpatialStream",
    "TemporalStream",
    "CaptionStream",
    "CaptionGating",
    "REMANet",
]
