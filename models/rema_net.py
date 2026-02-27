"""
MATResNet — Main Multi-Stream Model
====================================
Combines all four streams into a single model:

    1. Spatial  (RGB)      → class scores + features
    2. Temporal (Flow)     → class scores + features
    3. Caption  (BLIP)     → class scores + features  (via attention)
    4. Gated Fusion        → class scores             (RGB ⊕ Caption)

Final prediction = average of all four softmax outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .streams import (
    MultiHeadAttention,
    SpatialStream,
    TemporalStream,
    CaptionGating,
)
from config.default import SPATIAL_DIM, CAPTION_DIM


class REMANet(nn.Module):
    """
    MAT-ResNet: Multi-stream Action Transformer with ResNet-18 backbone.

    Args:
        num_classes  : number of action categories (51 for HMD51)
        num_segments : number of frames sampled per video clip
        pretrained   : use ImageNet-pretrained ResNet-18
        num_heads    : attention heads for Transformer and MHA
    """

    def __init__(self, num_classes, num_segments=16, pretrained=True, num_heads=8):
        super(REMANet, self).__init__()

        # Stream 1: RGB
        self.spatial_stream = SpatialStream(num_classes, pretrained, num_heads)

        # Stream 2: Optical Flow
        self.temporal_stream = TemporalStream(num_classes, pretrained, num_heads)

        # Stream 3: Caption — attention over per-frame embeddings
        self.caption_attention  = MultiHeadAttention(CAPTION_DIM, num_heads=num_heads)
        self.caption_classifier = nn.Linear(CAPTION_DIM, num_classes)

        # Stream 4: Gated fusion of RGB + Caption
        self.caption_gating   = CaptionGating(SPATIAL_DIM, CAPTION_DIM)
        self.fused_classifier = nn.Linear(CAPTION_DIM, num_classes)

    def forward(self, rgb, flow, embeddings):
        """
        Args:
            rgb        : (B, T, 3, 224, 224)  — colour frames
            flow       : (B, T, 2, 224, 224)  — optical flow
            embeddings : (B, T, 128)           — caption embeddings

        Returns:
            spatial_out  : RGB stream logits
            temporal_out : Flow stream logits
            caption_out  : Caption stream logits
            fused_out    : Gated fusion logits
            gate_weight  : learned gate value (for monitoring)
        """
        # --- RGB stream ---
        spatial_out, spatial_feat = self.spatial_stream(rgb)

        # --- Flow stream ---
        temporal_out, _ = self.temporal_stream(flow)

        # --- Caption stream ---
        if embeddings.dim() == 4:
            embeddings = embeddings.squeeze(2)

        caption_feat = self.caption_attention(embeddings).mean(dim=1)
        caption_out  = self.caption_classifier(caption_feat)

        # --- Gated fusion ---
        fused_feat, gate_weight = self.caption_gating(spatial_feat, caption_feat)
        fused_out = self.fused_classifier(fused_feat)

        return spatial_out, temporal_out, caption_out, fused_out, gate_weight

    def predict(self, rgb, flow, embeddings):
        """
        Inference helper — returns predicted class index and averaged probs.
        """
        spatial_out, temporal_out, caption_out, fused_out, _ = self.forward(
            rgb, flow, embeddings
        )

        avg_probs = (
            F.softmax(spatial_out, dim=1)
            + F.softmax(temporal_out, dim=1)
            + F.softmax(caption_out, dim=1)
            + F.softmax(fused_out, dim=1)
        ) / 4.0

        predicted_class = avg_probs.argmax(dim=1)
        return predicted_class, avg_probs
