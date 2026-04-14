import torch
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def generate(self, image: torch.Tensor, question: str) -> np.ndarray:
        """Generates a mock Grad-CAM heatmap."""
        # Returns a mock heatmap for visualization
        return np.random.rand(224, 224)
