import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """Fuses vision and text features using cross-attention."""
    def __init__(self, hidden_size=768):
        super().__init__()
        # Dummy linear layer for mock
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, batch_first=True)

    def forward(self, vision_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        # vision_features: (B, N, D), text_features: (B, L, D)
        # For mock, we simply return vision features passed through a linear layer
        if text_features is None: # e.g. phase 1 might not have text text_features yet
            return vision_features
        fused, _ = self.attention(text_features, vision_features, vision_features)
        return fused
