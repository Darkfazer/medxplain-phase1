import torch
import torch.nn as nn
import torchvision.models as models
from models.base_classifier import BaseMedicalClassifier

class DenseNetAdapter(BaseMedicalClassifier):
    """
    Adapter for DenseNet family models (default DenseNet-121).
    DenseNet-121 is highly popular for chest X-rays (e.g., CheXNet).
    """
    def __init__(self, num_classes: int, weights: str = "IMAGENET1K_V1", drop_rate: float = 0.3):
        super().__init__(num_classes)
        
        # Load pre-trained DenseNet-121
        self.model = models.densenet121(weights=weights)
        
        # Replace the final classifier layer
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for Grad-CAM.
        For DenseNet, this is the output of `features` directly before the ReLU/Pooling.
        """
        features = self.model.features(x)
        # Apply ReLU as done in the forward pass before pooling
        import torch.nn.functional as F
        out = F.relu(features, inplace=True)
        return out

if __name__ == "__main__":
    # Test script
    model = DenseNetAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: [2, 1024, 7, 7]
