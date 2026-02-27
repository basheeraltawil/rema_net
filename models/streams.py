"""
Stream Modules
==============
Individual stream components of REMA-Net:

- MultiHeadAttention : self-attention over a sequence of features
- SpatialStream      : RGB frames  → ResNet-18 backbone → TemporalTransformer
- TemporalStream     : Optical flow → ResNet-18 backbone → TemporalTransformer
- CaptionStream      : PIL image   → BLIP caption → BERT-tiny embedding
- CaptionGating      : learns a gate to blend RGB + Caption features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    AutoTokenizer,
    AutoModel,
)

from .temporal_transformer import TemporalTransformer
from config.default import (
    DEVICE,
    SPATIAL_DIM,
    CAPTION_DIM,
    BLIP_MODEL_NAME,
    TEXT_MODEL_NAME,
)


# ==============================================================================
# Multi-Head Self-Attention
# ==============================================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention layer.

    Lets the model focus on the most important parts of a sequence.
    For example, given 16 caption embeddings it learns which frames
    contribute most to the action prediction.

    Args:
        in_features : size of each input feature vector
        num_heads   : number of parallel attention heads
        dropout     : dropout rate
    """

    def __init__(self, in_features, num_heads=8, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert in_features % num_heads == 0, \
            "in_features must be divisible by num_heads"

        self.num_heads = num_heads
        self.head_dim  = in_features // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(in_features, 3 * in_features)
        self.out_proj = nn.Linear(in_features, in_features)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):
        batch_size, num_frames, feature_size = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, num_frames, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn_scores  = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = (attn_weights @ v)
        out = out.transpose(1, 2).reshape(batch_size, num_frames, feature_size)

        return self.out_proj(out)


# ==============================================================================
# RGB (Spatial) Stream
# ==============================================================================

class SpatialStream(nn.Module):
    """
    Processes colour video frames using ResNet-18 + TemporalTransformer.

    Input : (batch, num_frames, 3, 224, 224)
    Output: (class_scores, feature_vector)
    """

    def __init__(self, num_classes, pretrained=True, num_heads=8):
        super(SpatialStream, self).__init__()

        weights  = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.feature_dim = backbone.fc.in_features  # 512

        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_transformer = TemporalTransformer(
            self.feature_dim, num_heads=num_heads
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape

        x = x.view(batch_size * num_frames, C, H, W)
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(batch_size, num_frames, self.feature_dim)

        pooled_features = self.temporal_transformer(features)

        output = self.fc(pooled_features)
        return output, pooled_features


# ==============================================================================
# Optical-Flow (Temporal) Stream
# ==============================================================================

class TemporalStream(nn.Module):
    """
    Processes 2-channel optical flow (dx, dy) through ResNet-18 + Transformer.

    Input : (batch, num_frames, 2, 224, 224)
    Output: (class_scores, feature_vector)
    """

    def __init__(self, num_classes, pretrained=True, num_heads=8):
        super(TemporalStream, self).__init__()

        weights  = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        self.feature_dim = backbone.fc.in_features

        # Adapt first conv from 3-channel RGB to 2-channel flow
        backbone.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.backbone    = nn.Sequential(*list(backbone.children())[:-2])
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.temporal_transformer = TemporalTransformer(
            self.feature_dim, num_heads=num_heads
        )
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.shape

        x = x.view(batch_size * num_frames, C, H, W)
        features = self.backbone(x)
        features = self.global_pool(features)
        features = features.view(batch_size, num_frames, self.feature_dim)

        pooled_features = self.temporal_transformer(features)

        output = self.fc(pooled_features)
        return output, pooled_features


# ==============================================================================
# Caption Confidence Gating
# ==============================================================================

class CaptionGating(nn.Module):
    """
    Learns a gate (0-1) that blends caption features and RGB features.

        gate ≈ 1.0  →  trust the caption (e.g. "a person is waving")
        gate ≈ 0.0  →  trust the visual appearance

    Args:
        rgb_dim     : size of RGB feature vector  (default 512)
        caption_dim : size of caption feature vector (default 128)
    """

    def __init__(self, rgb_dim=SPATIAL_DIM, caption_dim=CAPTION_DIM):
        super(CaptionGating, self).__init__()

        self.rgb_proj = nn.Linear(rgb_dim, caption_dim)

        self.gate_net = nn.Sequential(
            nn.Linear(rgb_dim + caption_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, rgb_features, caption_features):
        combined    = torch.cat([rgb_features, caption_features], dim=1)
        gate_weight = self.gate_net(combined)

        rgb_projected = self.rgb_proj(rgb_features)

        fused = gate_weight * caption_features + (1 - gate_weight) * rgb_projected
        return fused, gate_weight


# ==============================================================================
# Caption Stream  (BLIP + BERT-tiny — frozen)
# ==============================================================================

class CaptionStream(nn.Module):
    """
    Generates a text caption from a video frame using BLIP, then
    encodes it into a 128-d feature vector using BERT-tiny.

    Both sub-models are frozen (not trained).

    Example flow:
        PIL Image → BLIP → "a man is running" → BERT-tiny → [0.12, -0.34, ...] (128-d)
    """

    def __init__(
        self,
        blip_model_name=BLIP_MODEL_NAME,
        text_model_name=TEXT_MODEL_NAME,
        device=None,
    ):
        super(CaptionStream, self).__init__()

        self.device = device if device is not None else DEVICE

        # BLIP: image → text
        self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
        self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name)
        self.blip_model = self.blip_model.to(self.device)
        self.blip_model.eval()
        self.blip_model.requires_grad_(False)

        # BERT-tiny: text → embedding
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            text_model_name, use_fast=False
        )
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_encoder = self.text_encoder.to(self.device)
        self.text_encoder.eval()
        self.text_encoder.requires_grad_(False)

    @torch.no_grad()
    def forward(self, frame):
        """
        Args:
            frame: a PIL Image
        Returns:
            cls_embedding : (1, 128)
            caption       : generated text string
        """
        inputs  = self.blip_processor(images=frame, return_tensors="pt").to(self.device)
        ids     = self.blip_model.generate(**inputs, max_length=20)
        caption = self.blip_processor.decode(ids[0], skip_special_tokens=True)

        tokenized     = self.text_tokenizer(
            caption, return_tensors="pt", truncation=True, max_length=64
        ).to(self.device)
        outputs       = self.text_encoder(**tokenized)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

        return cls_embedding, caption
