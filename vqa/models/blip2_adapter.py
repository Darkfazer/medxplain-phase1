import torch
import torch.nn as nn
from transformers import Blip2ForConditionalGeneration
from peft import LoraConfig, get_peft_model


class Blip2VQAAdapter(nn.Module):
    """
    PyTorch Adapter for HuggingFace's BLIP-2 incorporating LoRA (Low-Rank Adaptation)
    for efficient medical feature fine-tuning on consumer GPUs.
    """
    def __init__(
        self,
        model_name: str = "Salesforce/blip2-opt-2.7b",
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ):
        super().__init__()

        self.use_lora = use_lora

        print(f"Loading Base BLIP-2 Sequence Model ({model_name})... This may take a moment.")
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )

        if self.use_lora:
            print("Injecting LoRA adapters into Q-Former and Language Model Attention layers...")
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=0.05,
                bias="none",
                target_modules=["query", "value"],
            )
            self.model = get_peft_model(self.model, config)
            self.model.print_trainable_parameters()

    def forward(self, pixel_values, input_ids, attention_mask, labels=None):
        """
        Straight passthrough to the underlying BLIP-2 forward function.
        Computes LM cross-entropy loss internally if `labels` are provided.
        """
        return self.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def generate(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        max_new_tokens: int = 50,
        min_new_tokens: int = 1,
        num_beams: int = 3,
        do_sample: bool = False,
        repetition_penalty: float = 1.2,
        early_stopping: bool = True,
    ):
        """
        Inference: generate text autoregressively conditioned on image + prompt.
        All generation kwargs are explicit parameters so callers get a clear
        signature and no unexpected-keyword-argument errors.
        """
        return self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            early_stopping=early_stopping,
        )

    def save_checkpoint(self, path: str):
        """Saves only the LoRA weights, drastically reducing storage size."""
        if self.use_lora:
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path: str):
        import os
        if self.use_lora:
            if os.path.exists(path):
                from peft import PeftModel
                print(f"Loading LoRA weights from {path}")
                self.model = PeftModel.from_pretrained(self.model.base_model, path)
        else:
            self.model.load_state_dict(torch.load(path))


if __name__ == "__main__":
    print("Initializing BLIP2 Adapter structurally...")
    try:
        model = Blip2VQAAdapter(use_lora=False)
        print("BLIP-2 Adapter architecture ready.")
    except ImportError as e:
        print(f"Missing dependency: {e}.")