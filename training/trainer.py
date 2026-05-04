"""
training/trainer.py
====================
Unified training loop for medical image classification models.

Fixes applied vs. original:
  - torch.amp API (not deprecated torch.cuda.amp)
  - Gradient clipping actually called
  - Gradient norm computed after backward, before step
  - save_checkpoint uses torch.save fallback if method absent
  - SWA only activated when epochs > 5
"""
from __future__ import annotations

import logging
import os

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast   # ← updated API (PyTorch ≥ 2.4)
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from training.metrics import ClinicalMetrics

log = logging.getLogger(__name__)


class BaseTrainer:
    """Unified training loop for any BaseMedicalClassifier.

    Supports:
      - Multi-GPU via nn.DataParallel (wrap model before passing in)
      - Automatic Mixed Precision (AMP)
      - Gradient accumulation
      - Gradient norm logging + clipping
      - Stochastic Weight Averaging (SWA)
      - Early stopping

    Parameters
    ----------
    model       : the classifier (nn.Module)
    criterion   : loss function
    optimizer   : e.g. AdamW
    scheduler   : LR scheduler or None
    device      : torch.device
    class_names : list of pathology names
    config      : dict with optional keys:
                    epochs, gradient_accumulation_steps, mixed_precision,
                    early_stopping_patience, max_grad_norm
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: torch.device,
        class_names: list[str],
        config: dict,
    ) -> None:
        self.model     = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device    = device
        self.class_names = class_names

        self.epochs         = config.get("epochs", 20)
        self.grad_accum     = config.get("gradient_accumulation_steps", 1)
        self.use_amp        = config.get("mixed_precision", True)
        self.patience       = config.get("early_stopping_patience", 5)
        self.max_grad_norm  = config.get("max_grad_norm", 1.0)

        # GradScaler – new torch.amp API
        self.scaler = GradScaler("cuda") if (self.use_amp and device.type == "cuda") else None

        self.metrics_tracker = ClinicalMetrics(class_names)
        self.writer          = SummaryWriter(log_dir=os.path.join("experiments", "logs"))
        self.best_val_auc    = -1.0
        self.counter         = 0

    # ── Training epoch ────────────────────────────────────────────────────────
    def train_one_epoch(self, train_loader, epoch: int) -> float:
        """Run one full training epoch.  Returns mean batch loss."""
        self.model.train()
        total_loss = 0.0
        self.optimizer.zero_grad()

        loop = tqdm(train_loader, desc=f"Epoch [{epoch}/{self.epochs}] Train")
        for i, (images, targets) in enumerate(loop):
            images  = images.to(self.device)
            targets = targets.to(self.device).float()

            # Label smoothing
            eps     = 0.1
            targets = targets * (1 - eps) + 0.5 * eps

            # Forward + AMP
            device_str = self.device.type
            ctx = autocast(device_str) if self.use_amp else torch.inference_mode.__class__.__mro__[0]
            if self.use_amp:
                with autocast(device_str):
                    logits = self.model(images)
                    loss   = self.criterion(logits, targets) / self.grad_accum
            else:
                logits = self.model(images)
                loss   = self.criterion(logits, targets) / self.grad_accum

            # Backward
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient norm (computed AFTER backward, BEFORE step)
            total_norm = 0.0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            self.writer.add_scalar("Gradients/Norm", total_norm, epoch * len(train_loader) + i)

            # Gradient accumulation step
            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(train_loader):
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            loop.set_postfix(loss=f"{loss.item() * self.grad_accum:.4f}",
                             grad_norm=f"{total_norm:.3f}")

        avg_loss = total_loss / len(train_loader)
        self.writer.add_scalar("Loss/Train", avg_loss, epoch)
        return avg_loss

    # ── Validation epoch ──────────────────────────────────────────────────────
    def evaluate(self, val_loader, epoch: int) -> tuple[float, dict]:
        """Run validation.  Returns (mean_loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        tracker = ClinicalMetrics(self.class_names)
        tracker.reset()

        device_str = self.device.type
        loop = tqdm(val_loader, desc=f"Epoch [{epoch}/{self.epochs}] Val")
        with torch.no_grad():
            for images, targets in loop:
                images  = images.to(self.device)
                targets = targets.to(self.device).float()

                if self.use_amp and device_str == "cuda":
                    with autocast(device_str):
                        logits = self.model(images)
                        loss   = self.criterion(logits, targets)
                else:
                    logits = self.model(images)
                    loss   = self.criterion(logits, targets)

                total_loss += loss.item()
                preds = torch.sigmoid(logits)
                tracker.update(targets, preds)
                loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(val_loader)
        metrics  = tracker.compute()

        self.writer.add_scalar("Loss/Val", avg_loss, epoch)
        for k, v in metrics.items():
            if not (isinstance(v, float) and (v != v)):  # skip NaN
                self.writer.add_scalar(f"Val_Metrics/{k}", v, epoch)

        log.info("Epoch %d  val_loss=%.4f  macro_AUC=%.4f",
                 epoch, avg_loss, metrics.get("auc_macro", 0))
        return avg_loss, metrics

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(
        self,
        train_loader,
        val_loader,
        save_dir: str = "experiments/results/checkpoints",
    ) -> float:
        """Train for self.epochs epochs with early stopping.

        Returns the best validation AUC achieved.
        """
        os.makedirs(save_dir, exist_ok=True)

        use_swa = self.epochs > 5
        swa_model     = AveragedModel(self.model) if use_swa else None
        swa_start     = max(1, self.epochs - 5)
        swa_scheduler = SWALR(self.optimizer, swa_lr=0.05) if use_swa else None

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss, metrics = self.evaluate(val_loader, epoch)

            # Overfitting / underfitting diagnostics
            if train_loss < val_loss * 0.5:
                log.warning(
                    "[DIAGNOSTIC] Possible OVERFITTING: train=%.4f << val=%.4f",
                    train_loss, val_loss,
                )
            elif train_loss > 0.8 and val_loss > 0.8:
                log.warning(
                    "[DIAGNOSTIC] Possible UNDERFITTING: train=%.4f  val=%.4f",
                    train_loss, val_loss,
                )

            # Scheduler step
            if use_swa and epoch >= swa_start:
                swa_model.update_parameters(self.model)
                swa_scheduler.step()
                log.info("[SWA] Updated averaged weights at epoch %d", epoch)
            else:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif self.scheduler is not None:
                    self.scheduler.step()

            # Checkpoint on improved AUC
            val_auc = metrics.get("auc_macro", 0.0)
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.counter = 0
                save_path = os.path.join(save_dir, "best_model.pth")
                if hasattr(self.model, "save_checkpoint"):
                    self.model.save_checkpoint(save_path)
                else:
                    torch.save(self.model.state_dict(), save_path)
                log.info("Saved new best model  AUC=%.4f → %s", val_auc, save_path)
            else:
                self.counter += 1
                if self.counter >= self.patience:
                    log.info("Early stopping triggered at epoch %d", epoch)
                    break

        # Finalise SWA
        if use_swa and swa_model is not None:
            torch.optim.swa_utils.update_bn(train_loader, swa_model, device=self.device)
            swa_path = os.path.join(save_dir, "swa_model.pth")
            torch.save(swa_model.module.state_dict(), swa_path)
            log.info("SWA model saved → %s", swa_path)

        self.writer.close()
        return self.best_val_auc
