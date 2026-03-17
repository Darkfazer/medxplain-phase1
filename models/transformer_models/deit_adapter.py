import torch
import torch.nn as nn
import timm
from models.base_classifier import BaseMedicalClassifier

class DeiTAdapter(BaseMedicalClassifier):
    """
    Adapter for Data-efficient Image Transformers (DeiT).
    Uses timm (default: deit_base_patch16_224).
    """
    def __init__(self, num_classes: int, model_name: str = "deit_base_patch16_224", pretrained: bool = True):
        super().__init__(num_classes)
        
        # Load pre-trained DeiT via timm
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract sequence of tokens for explainability.
        Similar to ViT, DeiT has distillation tokens alongside CLS tokens.
        Shape: (B, num_tokens, embed_dim)
        """
        features = self.model.forward_features(x)
        return features

if __name__ == "__main__":
    # Test script
    model = DeiTAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: typically (2, 198, 768) for DeiT (196 patches + CLS + DISTILL)
