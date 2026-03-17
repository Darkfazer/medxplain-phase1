import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import configs.config as cfg

class VQAModel(nn.Module):
    def __init__(self, model_name=cfg.MODEL_NAME, freeze_vision=True):
        super().__init__()
        # Load BLIP-2 model
        # Using device_map="auto" can simplify placements if bitsandbytes is available, 
        # but let's stick to placing it on the cfg.DEVICE directly to avoid loading 
        # issues if accelerate is not perfectly configured.
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if cfg.DEVICE == "cuda" else torch.float32,
        )
        self.model.to(cfg.DEVICE)
        
        # Freeze the vision model to save memory during fine-tuning
        if freeze_vision:
            for param in self.model.vision_model.parameters():
                param.requires_grad = False
                
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        pixel_values = pixel_values.to(self.model.device)
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        if labels is not None:
            labels = labels.to(self.model.device)
            
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs
        
    def generate(self, pixel_values, input_ids, attention_mask, max_length=cfg.MAX_LENGTH):
        pixel_values = pixel_values.to(self.model.device)
        input_ids = input_ids.to(self.model.device)
        attention_mask = attention_mask.to(self.model.device)
        
        # Generation
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_length
        )
        return outputs
