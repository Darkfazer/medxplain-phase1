import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model

class Blip2VQAAdapter(nn.Module):
    """
    PyTorch Adapter for HuggingFace's BLIP-2 incorporating LoRA (Low-Rank Adaptation)
    for efficient medical feature fine-tuning on consumer GPUs.
    """
    def __init__(self, model_name: str = "Salesforce/blip2-opt-2.7b", 
                 use_lora: bool = True, 
                 lora_r: int = 16, 
                 lora_alpha: int = 32):
        super().__init__()
        
        self.use_lora = use_lora
        
        print(f"Loading Base BLIP-2 Sequence Model ({model_name})... This may take a moment.")
        # Device map 'auto' helps distribute across multiple GPUs if available natively, 
        # but for explicit DDP we usually load to RAM then cast to `.to(device)`
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            # load_in_8bit=True, # Optional: Un-comment if strictly running out of VRAM and bitsandbytes is installed
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        
        if self.use_lora:
            print("Injecting LoRA adapters into Q-Former and Language Model Attention layers...")
            # The prompt requests injecting LoRA specifically into the Q-Former.
            # In huggingface's BLIP-2 architecture, Q-Former attention layers use 'query' and 'value'.
            target_modules = ["query", "value"] 
            
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=target_modules
            )
            
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()
            
    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Straight passthrough to the underlying BLIP-2 forward function.
        Generates LM cross-entropy loss intrinsically if `labels` are provided.
        """
        # BLIP-2 automatically calculates loss internally if labels are passed.
        # It handles the visual projection mapping internally.
        outputs = self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

    def generate(self, pixel_values, input_ids, attention_mask, max_new_tokens=50):
        """
        Inference logic. Uses the LLM head to generate text autoregressively 
        conditioned on the image embeddings and the input question prompt.
        """
        outputs = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            early_stopping=True,
            num_beams=3 # Small beam search for more coherent diagnostic text
        )
        return outputs
        
    def save_checkpoint(self, path: str):
        """Saves only the LoRA weights, drastically reducing storage size."""
        if self.use_lora:
            self.model.save_pretrained(path) # Huggingface/PEFT handles this
        else:
            torch.save(self.model.state_dict(), path)
            
    def load_checkpoint(self, path: str):
        if self.use_lora:
            import os
            if os.path.exists(path):
                from peft import PeftModel
                print(f"Loading LoRA weights from {path}")
                # We have to re-wrap the base model with the loaded peft weights
                self.model = PeftModel.from_pretrained(self.model.base_model, path)
        else:
            self.model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    # Structural verification (Will require ~5GB of RAM just to instantiate structurally without 8bit)
    print("Initializing BLIP2 Adapter Structurally. It will throw an error if transformers/peft are missing.")
    try:
        model = Blip2VQAAdapter(use_lora=False) # Turning off lora for sheer structural mock test speed
        print("BLIP-2 Adapter architecture ready.")
    except ImportError as e:
        print(f"Missing dependency: {e}. Please ensure transformers and peft are installed via pip.")
