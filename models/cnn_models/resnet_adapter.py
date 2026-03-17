import torch
import torch.nn as nn
import torchvision.models as models
from models.base_classifier import BaseMedicalClassifier

class ResNetAdapter(BaseMedicalClassifier):
    """
    Adapter for ResNet family models (default ResNet-50).
    Uses torchvision pre-trained models and modifies the final FC layer.
    """
    def __init__(self, num_classes: int, weights: str = "IMAGENET1K_V2", drop_rate: float = 0.3):
        super().__init__(num_classes)
        
        # Load pre-trained ResNet-50
        self.model = models.resnet50(weights=weights)
        
        # Replace the final fully connected layer
        num_ftrs = self.model.fc.in_features
        
        # Adding a dropout layer before the final classification head can help prevent overfitting
        # especially important in medical datasets.
        self.model.fc = nn.Sequential(
            nn.Dropout(p=drop_rate),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for Grad-CAM.
        For ResNet, this is typically the output of `layer4` before the global average pooling.
        """
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x) # Shape: (B, C, H, W) e.g. (B, 2048, 7, 7)
        return x

if __name__ == "__main__":
    # Test script
    model = ResNetAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: [2, 2048, 7, 7]
