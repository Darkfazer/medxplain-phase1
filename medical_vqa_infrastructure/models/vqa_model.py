import torch
import torch.nn as nn
from typing import Dict, Any

from .vision_encoder import MockVisionEncoder
from .text_decoder import MockTextDecoder
from .fusion import CrossAttentionFusion

class MedicalVQAModel(nn.Module):
    """Complete VQA Model assembling vision, fusion, and decoder components."""
    def __init__(self, config: Dict[str, Any], mock_mode: bool = False):
        super().__init__()
        self.config = config
        self.mock_mode = mock_mode
        
        # Instantiate pluggable components
        if self.mock_mode:
            self.vision_encoder = MockVisionEncoder()
            self.text_decoder = MockTextDecoder()
        else:
            # TODO: Add factory instantiation based on config
            self.vision_encoder = MockVisionEncoder() 
            self.text_decoder = MockTextDecoder()
            
        self.fusion = CrossAttentionFusion()

    def forward(self, images: torch.Tensor, questions: list) -> torch.Tensor:
        vision_features = self.vision_encoder(images)
        # Note: robust implementation would process questions -> text_features first
        fused_features = self.fusion(vision_features, None)
        output_logits = self.text_decoder(fused_features, questions)
        return output_logits
