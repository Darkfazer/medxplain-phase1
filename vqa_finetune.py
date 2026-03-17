import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoProcessor
from vqa.models.blip2_adapter import Blip2VQAAdapter
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def finetune_vqa(train_loader, val_loader, epochs=5):
    print("--- VQA-RAD LoRA Fine-Tuning Setup ---")
    
    # 1. Load BLIP-2 with LoRA
    # Using 'opt-2.7b' as requested. We set use_lora=True to only train the attention heads.
    model = Blip2VQAAdapter(use_lora=True).to(DEVICE)
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    
    # 2. Freeze Vision Encoder initially (already handled by LoRA config in Blip2Adapter)
    # Just to be safe:
    for name, param in model.model.vision_model.named_parameters():
        param.requires_grad = False
        
    print("Trainable parameters (LoRA + QFormer + Proj):")
    model.model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    
    best_closed_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(loop):
            # Assumes batch yields: images (PIL), questions (list of str), answers (list of str)
            images, questions, answers = batch
            
            # Proper prompt template
            prompts = [f"Question: {q}\nAnswer:" for q in questions]
            
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(DEVICE)
            # Tokenize targets
            labels = processor(text=answers, return_tensors="pt", padding=True).input_ids.to(DEVICE)
            labels[labels == processor.tokenizer.pad_token_id] = -100
            
            inputs["labels"] = labels
            
            optimizer.zero_grad()
            outputs = model.model(**inputs)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
            # Validation every 100 steps
            if step > 0 and step % 100 == 0:
                closed_acc, open_acc = validate(model, processor, val_loader)
                print(f"Step {step} | Closed Acc: {closed_acc:.2f} | Open Acc: {open_acc:.2f}")
                
                if closed_acc > best_closed_acc:
                    best_closed_acc = closed_acc
                    os.makedirs("experiments/vqa/checkpoints", exist_ok=True)
                    model.save_checkpoint(f"experiments/vqa/checkpoints/best_lora_model_step_{step}")
                    print(f"  -> Saved new best model (Closed Acc: {best_closed_acc:.2f})")
                    model.train() # Back to train mode
                    
        print(f"Epoch {epoch} finished. Avg Loss: {total_loss / len(train_loader):.4f}")

def validate(model, processor, val_loader):
    model.eval()
    closed_correct, closed_total = 0, 0
    open_correct, open_total = 0, 0
    
    # Exact Generation parameters recommended
    generation_config = {
        "max_length": 50,
        "min_length": 1,
        "do_sample": False,
        "num_beams": 3,
        "temperature": 1.0,
        "repetition_penalty": 1.2
    }
    
    with torch.no_grad():
        for batch in val_loader:
            images, questions, answers, answer_types = batch
            prompts = [f"Question: {q}\nAnswer:" for q in questions]
            inputs = processor(images=images, text=prompts, return_tensors="pt", padding=True).to(DEVICE)
            
            generated_ids = model.generate(
                pixel_values=inputs.pixel_values,
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                **generation_config
            )
            
            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            for p, a, t in zip(preds, answers, answer_types):
                # Clean Yes/No
                p = p.strip().lower()
                a = a.strip().lower()
                
                is_correct = 1 if p == a else 0
                
                if t == "CLOSED":
                    closed_correct += is_correct
                    closed_total += 1
                else:
                    open_correct += is_correct
                    open_total += 1
                    
    c_acc = closed_correct / max(closed_total, 1)
    o_acc = open_correct / max(open_total, 1)
    return c_acc, o_acc

if __name__ == "__main__":
    from data.dataset import get_dataloaders
    from transformers import AutoProcessor
    import configs.config as cfg
    
    print("VQA Fine-tuning Script initialized. Fetching DataLoaders...")
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    # Using small batch size for 2.7B param model
    train_dl, val_dl = get_dataloaders(processor, batch_size=2)
    finetune_vqa(train_dl, val_dl, epochs=13)
