import os
import torch
from torch.utils.data import DataLoader
import logging

class Trainer:
    """Handles Phase 1 & Phase 2 training with DDP, AMP."""
    def __init__(self, model: torch.nn.Module, dataloader: DataLoader, optimizer, loss_fn, device: str = 'cuda'):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        
        for batch in self.dataloader:
            images = batch['images'].to(self.device)
            # answers processing skipped for mock brevity
            
            with torch.cuda.amp.autocast(enabled=(self.device == 'cuda')):
                outputs = self.model(images, batch['questions'])
                # mock target
                targets = torch.randint(0, 2, outputs.shape, device=self.device, dtype=torch.float32)
                loss = self.loss_fn(outputs, targets)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
            
        return total_loss / max(1, len(self.dataloader))
