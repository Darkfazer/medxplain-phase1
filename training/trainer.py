import os
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from training.metrics import ClinicalMetrics
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter

class BaseTrainer:
    """
    Unified training loop for any medical classification model.
    Handles Multi-GPU, Mixed Precision (AMP), gradient accumulation, and model-specific metrics.
    """
    def __init__(self, model, criterion, optimizer, scheduler, device, class_names, config):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.class_names = class_names
        
        # Hyperparameters
        self.epochs = config.get('epochs', 20)
        self.grad_accum_steps = config.get('gradient_accumulation_steps', 1)
        self.use_amp = config.get('mixed_precision', True)
        self.patience = config.get('early_stopping_patience', 5)
        
        # AMP Scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Metrics Tracker
        self.metrics_tracker = ClinicalMetrics(class_names)
        
        # Logging
        self.writer = SummaryWriter(log_dir=os.path.join("experiments", "logs"))
        self.best_val_auc = -1.0
        self.counter = 0

    def get_metrics_tracker(self):
        # We instantiate a new metrics obj for evaluation per epoch to keep clean state
        return ClinicalMetrics(self.class_names)

    def train_one_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        
        # Progressive Resizing
        # try:
        #     from training.augmentations import get_progressive_resizing_transforms
        #     train_loader.dataset.transform = get_progressive_resizing_transforms(epoch, self.epochs)
        #     print(f"Applying progressive resizing for Epoch {epoch}")
        # except Exception as e:
        #     pass

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{self.epochs}] Training")
        for i, (images, targets) in enumerate(loop):
            images, targets = images.to(self.device), targets.to(self.device).float()

            # Label Smoothing quick win
            epsilon = 0.1
            targets = targets * (1 - epsilon) + 0.5 * epsilon

            # Forward + AMP
            if self.use_amp:
                with autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets) / self.grad_accum_steps
            else:
                logits = self.model(images)
                loss = self.criterion(logits, targets) / self.grad_accum_steps

            # Backward
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient Flow check
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm < 0.01:
                pass # Silently handle, logged in tensorboard
            elif total_norm > 100:
                pass # Silently handle, logged in tensorboard

            self.writer.add_scalar('Gradients/Norm', total_norm, epoch * len(train_loader) + i)

            # Optimizer Step (Gradient Accumulation)
            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == len(train_loader):
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps
            loop.set_postfix(loss=loss.item() * self.grad_accum_steps, grad_norm=total_norm)

        avg_loss = total_loss / len(train_loader)
        self.writer.add_scalar('Loss/Train', avg_loss, epoch)
        return avg_loss

    def evaluate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0.0
        tracker = self.get_metrics_tracker()
        tracker.reset()
        
        loop = tqdm(val_loader, desc=f"Epoch [{epoch}/{self.epochs}] Validation")
        with torch.no_grad():
            for images, targets in loop:
                images, targets = images.to(self.device), targets.to(self.device).float()

                if self.use_amp:
                    with autocast():
                        logits = self.model(images)
                        loss = self.criterion(logits, targets)
                else:
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                preds = torch.sigmoid(logits)
                tracker.update(targets, preds)

                loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(val_loader)
        metrics = tracker.compute()
        
        # Logging
        self.writer.add_scalar('Loss/Val', avg_loss, epoch)
        for k, v in metrics.items():
            if not torch.isnan(torch.tensor(v)):
                self.writer.add_scalar(f'Val_Metrics/{k}', v, epoch)

        print(f"Epoch {epoch} Val Loss: {avg_loss:.4f} | Macro AUC: {metrics['auc_macro']:.4f}")
        return avg_loss, metrics

    def fit(self, train_loader, val_loader, save_dir="experiments/results/checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        
        # SWA Setup
        swa_model = AveragedModel(self.model)
        swa_start = max(1, self.epochs - 5)
        swa_scheduler = SWALR(self.optimizer, swa_lr=0.05) if self.epochs > 5 else None

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, metrics = self.evaluate(val_loader, epoch)
            
            # Overfitting Diagnostic
            if train_loss < val_loss * 0.5:
                print(f"[DIAGNOSTIC] OVERFITTING DETECTED! Train Loss ({train_loss:.4f}) << Val Loss ({val_loss:.4f}). Consider adding more dropout or augmentation.")
            elif train_loss > 0.8 and val_loss > 0.8:
                print(f"[DIAGNOSTIC] UNDERFITTING DETECTED! Both losses are high. Consider increasing model capacity or learning rate.")
                
            # Scheduler Step
            if epoch >= swa_start and swa_scheduler is not None:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                print(f"[SWA] Updating Averaged Weights for Epoch {epoch}")
            else:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

            # Checkpoint Saving
            val_auc = metrics.get('auc_macro', 0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.counter = 0
                save_path = os.path.join(save_dir, "best_model.pth")
                self.model.save_checkpoint(save_path)
                print(f"Saved new best model with Macro AUC: {val_auc:.4f}")
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
                    
        # Update batch norm statistics for SWA
        if swa_scheduler is not None:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
            swa_path = os.path.join(save_dir, "swa_best_model.pth")
            torch.save(swa_model.module.state_dict(), swa_path)
            print(f"Saved SWA model to {swa_path}")
            
        self.writer.close()
        return self.best_val_auc
