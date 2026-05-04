"""
models/cnn_models/densenet_adapter.py
======================================
DenseNet-121 adapter implementing BaseMedicalClassifier.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from models.base_classifier import BaseMedicalClassifier


class DenseNetAdapter(BaseMedicalClassifier):
    """DenseNet-121 pre-trained on ImageNet, fine-tuned for multi-label chest X-ray.

    Parameters
    ----------
    num_classes : number of output pathology classes (default 14 for NIH CheXpert)
    weights     : torchvision weight identifier; use ``None`` for random init
    drop_rate   : dropout probability before the final classifier
    """

    def __init__(
        self,
        num_classes: int = 14,
        weights: str = "IMAGENET1K_V1",
        drop_rate: float = 0.3,
    ) -> None:
        super().__init__(num_classes)
        self.model = models.densenet121(weights=weights)
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_ftrs, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, num_classes)."""
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps for Grad-CAM (before global pool).

        Shape: (B, 1024, 7, 7) for a 224×224 input.
        """
        features = self.model.features(x)
        return F.relu(features, inplace=False)


if __name__ == "__main__":
    model = DenseNetAdapter(num_classes=14)
    dummy = torch.randn(2, 3, 224, 224)
    print("Logits:", model(dummy).shape)          # (2, 14)
    print("Features:", model.extract_features(dummy).shape)  # (2, 1024, 7, 7)
