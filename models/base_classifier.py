import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseMedicalClassifier(nn.Module, ABC):
    """
    Abstract base class for all medical image classification models defined in this framework.
    All adapters (CNNs, ViTs) should inherit from this to ensure a standard API.
    """
    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass returning raw logits of shape (batch, num_classes).
        """
        pass

    @abstractmethod
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps specifically required for explainability (like Grad-CAM).
        Should return the spatial feature maps prior to global average pooling.
        """
        pass

    def predict(self, x: torch.Tensor, activation: str = 'sigmoid') -> torch.Tensor:
        """
        Forward pass with optional activation (useful for inference).
        
        Args:
            x: Input image tensor (B, C, H, W)
            activation: Activation function to apply to logits ('sigmoid' or 'softmax')
                        Medical multi-label typically requires 'sigmoid'.
        Returns:
            Probabilities (B, num_classes)
        """
        logits = self.forward(x)
        if activation == 'sigmoid':
            return torch.sigmoid(logits)
        elif activation == 'softmax':
            return torch.softmax(logits, dim=1)
        return logits

    def save_checkpoint(self, path: str):
        """Standard utility to save state_dict."""
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path: str, device: torch.device):
        """Standard utility to load state_dict."""
        self.load_state_dict(torch.load(path, map_location=device))
