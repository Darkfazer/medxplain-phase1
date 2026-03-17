import torch
import torch.nn as nn
import timm
from models.base_classifier import BaseMedicalClassifier

class SwinAdapter(BaseMedicalClassifier):
    """
    Adapter for Swin Transformer models.
    Uses timm (default: swin_base_patch4_window7_224).
    """
    def __init__(self, num_classes: int, model_name: str = "swin_base_patch4_window7_224", pretrained: bool = True):
        super().__init__(num_classes)
        
        # Load pre-trained Swin via timm
        # For Swin, timm handles spatial features gracefully.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for explainability.
        Timm Swins usually output spatial maps directly from `forward_features`
        or `forward_head(..., pre_logits=True)`.
        Output shape is typically (B, H, W, C) or (B, C, H, W) before pooling.
        """
        features = self.model.forward_features(x)
        # Typically for Swin (B, H, W, C) -> Need to map to (B, C, H, W) for standard Grad-CAM if needed.
        # But we'll return raw for now.
        return features

if __name__ == "__main__":
    # Test script
    model = SwinAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: Varies by Swin version, often (B, 49, 1024) or (B, 7, 7, 1024)
