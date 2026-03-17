import torch
import torch.nn as nn
import torchxrayvision as xrv
from models.base_classifier import BaseMedicalClassifier

class TorchXRayAdapter(BaseMedicalClassifier):
    """
    Adapter for TorchXRayVision models.
    Default uses DenseNet121 from the 'all' dataset weights.
    Native TorchXRayVision handles specific grayscale mappings and 14 classes (or more).
    """
    def __init__(self, num_classes: int, weights: str = "densenet121-res224-all"):
        super().__init__(num_classes)
        
        # Load the pre-trained chest x-ray model
        self.model = xrv.models.DenseNet(weights=weights)
        
        # The number of classes in xrv might be 18 (for the 'all' weights).
        # We need to map or replace the final linear layer if we are specifically
        # enforcing exactly `num_classes` (like 14 for NIH only).
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Turn off op_threshs as it expects the original 18 class outputs
        if hasattr(self.model, 'op_threshs'):
            self.model.op_threshs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        TorchXRayVision models expect images with shapes [B, 1, 224, 224] (grayscale) typically.
        If inputs are RGB, we convert them.
        """
        if x.shape[1] == 3:
            # Simple conversion to grayscale by averaging channels
            x = x.mean(dim=1, keepdim=True)
            
        return self.model(x)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps for explainability.
        """
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
            
        features = self.model.features(x)
        import torch.nn.functional as F
        out = F.relu(features, inplace=True)
        return out

if __name__ == "__main__":
    # Test script
    model = TorchXRayAdapter(num_classes=14)
    dummy_input = torch.randn(2, 3, 224, 224) # framework passes RGB
    logits = model(dummy_input)
    features = model.extract_features(dummy_input)
    
    print(f"Logits shape: {logits.shape}")       # Expected: [2, 14]
    print(f"Features shape: {features.shape}")   # Expected: [2, 1024, 7, 7]
