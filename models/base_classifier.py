"""
models/base_classifier.py
==========================
Abstract base class for all medical image classifiers in this framework.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseMedicalClassifier(nn.Module, ABC):
    """Abstract base for classification models (CNNs, ViTs, etc.).

    All concrete adapters must implement ``forward`` and ``extract_features``.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape (B, num_classes)."""

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return spatial feature maps required for explainability (Grad-CAM)."""

    def predict(
        self, x: torch.Tensor, activation: str = "sigmoid"
    ) -> torch.Tensor:
        """Forward pass with optional probability activation.

        Parameters
        ----------
        x          : (B, C, H, W) input tensor
        activation : ``'sigmoid'`` for multi-label, ``'softmax'`` for single-label

        Returns
        -------
        (B, num_classes) probability tensor
        """
        logits = self.forward(x)
        if activation == "sigmoid":
            return torch.sigmoid(logits)
        if activation == "softmax":
            return torch.softmax(logits, dim=1)
        return logits

    def save_checkpoint(self, path: str) -> None:
        """Save state_dict to ``path``."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, device: torch.device | None = None) -> None:
        """Load state_dict from ``path``, mapping to ``device`` if provided."""
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
