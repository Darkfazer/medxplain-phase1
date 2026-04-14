from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseVisionEncoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

class MockVisionEncoder(BaseVisionEncoder):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Returns dummy features (batch_size, num_patches, hidden_size)
        return torch.rand(x.size(0), 196, self.hidden_size, device=x.device)

# TODO: Add ViT, ResNet, BiomedCLIP implementations here
