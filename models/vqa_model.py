"""
models/vqa_model.py
===================
BLIP-1 VQA model wrapper (Salesforce/blip-vqa-base).

This class is used for fine-tuning and matches the model loaded
by backend.py for inference, ensuring checkpoint compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import BlipForQuestionAnswering

import configs.config as cfg


class VQAModel(nn.Module):
    """Thin wrapper around BLIP-1 for VQA fine-tuning.

    Parameters
    ----------
    model_name   : HuggingFace model identifier (default: cfg.MODEL_NAME)
    freeze_vision: if True, freeze the vision encoder weights during training
    """

    def __init__(
        self,
        model_name: str = cfg.MODEL_NAME,
        freeze_vision: bool = True,
    ) -> None:
        super().__init__()
        self.model = BlipForQuestionAnswering.from_pretrained(model_name)
        self.model.to(cfg.DEVICE)

        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False

        # Allow TF32 on Ampere+ GPUs for faster matmul without precision loss
        if cfg.DEVICE == "cuda":
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ):
        """Forward pass returning a HuggingFace ModelOutput with a .loss field."""
        device = next(self.model.parameters()).device
        return self.model(
            pixel_values=pixel_values.to(device),
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            labels=labels.to(device) if labels is not None else None,
        )

    def generate(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int = cfg.MAX_LENGTH,
    ) -> torch.Tensor:
        """Generate answer token ids."""
        device = next(self.model.parameters()).device
        return self.model.generate(
            pixel_values=pixel_values.to(device),
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
            max_new_tokens=max_new_tokens,
        )

    def save_checkpoint(self, path: str) -> None:
        """Save model state dict to path."""
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str) -> None:
        """Load model state dict from path (uses cfg.DEVICE for mapping)."""
        state = torch.load(path, map_location=cfg.DEVICE)
        self.model.load_state_dict(state)
