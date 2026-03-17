import torch
import torch.nn as nn
import torchvision.models as models
from models.base_classifier import BaseMedicalClassifier

class EfficientNetAdapter(BaseMedicalClassifier):
    """
    Adapter for EfficientNet family models (default EfficientNet-B4).
    """
    def __init__(self, num_classes: int, weights: str = "IMAGENET1K_V1", drop_rate: float = 0.4):
        super().__init__(num_classes)
        
        # Load pre-trained EfficientNet-B4
        self.model = models.efficientnet_b4(weights=weights)
        
        # Replace the classifier block
        # classifier usually is a Sequential(Dropout, Linear)
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for Grad-CAM.
        For EfficientNet, it is the output of the `features` sequential block.
        """
        return self.model.features(x)

if __name__ == "__main__":
    # Test script
    model = EfficientNetAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: [2, 1792, 7, 7]
