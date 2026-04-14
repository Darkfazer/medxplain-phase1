from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BaseTextDecoder(ABC, nn.Module):
    @abstractmethod
    def forward(self, context, questions):
        pass

class MockTextDecoder(BaseTextDecoder):
    def __init__(self, vocab_size=32000):
        super().__init__()
        # Mock projection layer
        self.proj = nn.Linear(768, vocab_size)

    def forward(self, context: torch.Tensor, questions: list) -> torch.Tensor:
        # Generate dummy logits
        batch_size = context.size(0)
        return self.proj(context.mean(dim=1)) # (batch_size, vocab_size)

# TODO: Add BioGPT, LLaMA implementations
