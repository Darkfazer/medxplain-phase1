import torch
import torch.nn as nn
import timm
from models.base_classifier import BaseMedicalClassifier

class ViTAdapter(BaseMedicalClassifier):
    """
    Adapter for Vision Transformer (ViT) models.
    Uses timm to load pre-trained ViT models (default: vit_base_patch16_224).
    """
    def __init__(self, num_classes: int, model_name: str = "vit_base_patch16_224", pretrained: bool = True):
        super().__init__(num_classes)
        
        # Load pre-trained ViT via timm. 
        # timm automatically replaces the head if num_classes is specified.
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract feature maps for explainability (like Attention Rollout or Grad-CAM for ViTs).
        For timm's ViT, `forward_features` returns the sequence of tokens including the CLS token.
        Shape: (B, num_tokens, embed_dim) -> e.g., (B, 197, 768)
        """
        # Expose the patch embeddings (before the final classification head)
        features = self.model.forward_features(x)
        return features

if __name__ == "__main__":
    # Test script
    model = ViTAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: [2, 197, 768] (196 patches + 1 CLS token)
