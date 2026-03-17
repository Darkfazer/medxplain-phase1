import os
import torch
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from vqa.training.vqa_metrics import VQAMetrics

class CustomVQATrainer:
    """
    Handles Multi-modal language modeling fine-tuning for the MedicalCrossAttentionVQA model.
    """
    def __init__(self, model, tokenizer, optimizer, scheduler, device, config):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
        self.epochs = config.get('epochs', 10)
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 2)
        self.use_amp = config.get('mixed_precision', True)
        
        self.scaler = GradScaler() if self.use_amp else None
        self.metrics_tracker = VQAMetrics()
        self.best_bleu = -1.0

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{self.epochs}] Training (Custom VQA)")
        for i, batch in enumerate(loop):
            encoding, q_strs, a_strs, t_strs = batch
            
            pixel_values = encoding["pixel_values"].to(self.device).float()
            input_ids = encoding["input_ids"].to(self.device)
            attention_mask = encoding["attention_mask"].to(self.device)
            labels = encoding["labels"].to(self.device)

            if self.use_amp:
                with autocast():
                    outputs = self.model(
                        pixel_values=pixel_values,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss / self.grad_accum_steps
            else:
                outputs = self.model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss / self.grad_accum_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            loop.set_postfix(loss=loss.item() * self.grad_accum_steps)

        return total_loss / len(train_loader)

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        self.metrics_tracker.reset()
        
        loop = tqdm(val_loader, desc=f"Epoch [{epoch}/{self.epochs}] Validation (Custom VQA)")
        with torch.no_grad():
            for batch in loop:
                encoding, q_strs, a_strs, t_strs = batch
                
                pixel_values = encoding["pixel_values"].to(self.device).float()
                
                # Autoregressive generation loop
                if self.use_amp:
                    with autocast():
                        generated_ids = self.model.generate(
                            pixel_values=pixel_values,
                            start_token_id=self.tokenizer.bos_token_id or 2,
                            end_token_id=self.tokenizer.eos_token_id or 2,
                            max_length=20,
                            device=self.device
                        )
                else:
                    generated_ids = self.model.generate(
                        pixel_values=pixel_values,
                        start_token_id=self.tokenizer.bos_token_id or 2,
                        end_token_id=self.tokenizer.eos_token_id or 2,
                        max_length=20,
                        device=self.device
                    )
                
                b_size = pixel_values.shape[0]
                generated_texts = []
                for b_idx in range(b_size):
                    gen = self.tokenizer.decode(generated_ids[b_idx], skip_special_tokens=True).strip()
                    # Strip out prompts if echoed back
                    if q_strs[b_idx] in gen: gen = gen.replace(f"Question: {q_strs[b_idx]} Answer:", "").strip()
                    generated_texts.append(gen)
                
                for gen_text, gt_text, q_type in zip(generated_texts, a_strs, t_strs):
                    self.metrics_tracker.update(gen_text, gt_text, q_type)
                    
        res = self.metrics_tracker.compute()
        print(f"Epoch {epoch} Eval | Closed Acc: {res['Accuracy_Closed']:.4f} | Open BLEU: {res['BLEU_Open']:.4f}")
        return res

    def fit(self, train_loader, val_loader, save_dir="experiments/vqa/checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            metrics = self.evaluate(val_loader, epoch)
            
            current_bleu = metrics.get('BLEU_All', 0)
            if current_bleu > self.best_bleu:
                self.best_bleu = current_bleu
                save_path = os.path.join(save_dir, "best_custom_fusion.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved new best Custom Fusion VQA model with BLEU: {current_bleu:.4f}")
