"""
calibrated_inference.py
========================
Temperature scaling and class-specific threshold calibration wrapper
for TorchXRayVision DenseNet121.

Fixes vs. original:
  - Absolute import (no sys.path hack needed; run from project root or install)
  - LF line endings
  - load_checkpoint: map_location respects cfg.DEVICE
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

import configs.config as cfg
from models.cnn_models.densenet_adapter import DenseNetAdapter

log = logging.getLogger(__name__)


class TemperatureScaler(nn.Module):
    """Learn a single scalar temperature on a calibration set.

    After fitting, ``model_with_temperature(logits) / temperature`` gives
    better-calibrated probabilities.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model       = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.model(x)
        return logits / self.temperature

    def fit(
        self,
        val_loader,
        lr: float = 0.01,
        max_iter: int = 50,
    ) -> None:
        """Optimise temperature on the validation set using NLL loss.

        Parameters
        ----------
        val_loader : DataLoader yielding (images, labels) batches
        lr         : learning rate for L-BFGS
        max_iter   : maximum optimiser iterations
        """
        self.model.eval()
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(cfg.DEVICE)
                all_logits.append(self.model(images).cpu())
                all_labels.append(labels.cpu())

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS(
            [self.temperature], lr=lr, max_iter=max_iter
        )

        def _closure() -> torch.Tensor:
            optimizer.zero_grad()
            scaled = logits / self.temperature
            loss   = criterion(scaled, labels.float())
            loss.backward()
            return loss

        optimizer.step(_closure)
        log.info("Calibrated temperature: %.4f", self.temperature.item())


class CalibratedClassifier:
    """Wrap DenseNetAdapter with temperature scaling + per-class thresholds.

    Parameters
    ----------
    num_classes     : number of output classes
    checkpoint_path : optional path to a saved DenseNetAdapter state_dict
    """

    DEFAULT_THRESHOLDS: Dict[str, float] = {
        "Atelectasis":       0.45,
        "Cardiomegaly":      0.50,
        "Effusion":          0.45,
        "Infiltration":      0.45,
        "Mass":              0.40,
        "Nodule":            0.40,
        "Pneumonia":         0.45,
        "Pneumothorax":      0.40,
        "Consolidation":     0.45,
        "Edema":             0.45,
        "Emphysema":         0.40,
        "Fibrosis":          0.40,
        "Pleural_Thickening":0.40,
        "Hernia":            0.30,
    }

    def __init__(
        self,
        num_classes: int = 14,
        checkpoint_path: Optional[str] = None,
    ) -> None:
        self.device = torch.device(cfg.DEVICE)
        base = DenseNetAdapter(num_classes=num_classes).to(self.device)

        if checkpoint_path and Path(checkpoint_path).exists():
            base.load_checkpoint(checkpoint_path, device=self.device)
            log.info("Loaded checkpoint: %s", checkpoint_path)

        self.scaled_model = TemperatureScaler(base).to(self.device)
        self.thresholds   = self.DEFAULT_THRESHOLDS

    def calibrate(self, val_loader) -> None:
        """Calibrate temperature using a validation DataLoader."""
        self.scaled_model.fit(val_loader)

    def predict(
        self,
        image_tensor: torch.Tensor,
        class_names: Optional[list[str]] = None,
    ) -> Dict[str, float]:
        """Run calibrated inference on a single image tensor.

        Parameters
        ----------
        image_tensor : (1, C, H, W) tensor on device
        class_names  : optional list of class name strings

        Returns
        -------
        dict mapping class names to calibrated probabilities
        """
        self.scaled_model.eval()
        with torch.no_grad():
            logits = self.scaled_model(image_tensor.to(self.device))
            probs  = torch.sigmoid(logits).squeeze().cpu().numpy()

        num_c  = len(probs)
        names  = class_names or [f"class_{i}" for i in range(num_c)]
        return {name: float(prob) for name, prob in zip(names, probs)}

    def predict_with_thresholds(
        self,
        image_tensor: torch.Tensor,
        class_names: Optional[list[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return both probabilities and binary decisions using per-class thresholds.

        Returns
        -------
        dict: {class_name: {"prob": ..., "positive": True/False}}
        """
        prob_dict = self.predict(image_tensor, class_names)
        result: Dict[str, Dict[str, float]] = {}
        for name, prob in prob_dict.items():
            thresh = self.thresholds.get(name, 0.5)
            result[name] = {"prob": prob, "positive": float(prob >= thresh)}
        return result


if __name__ == "__main__":
    from transformers import BlipProcessor
    classifier = CalibratedClassifier(num_classes=14)
    dummy = torch.randn(1, 3, 224, 224)
    preds = classifier.predict(dummy)
    print("Calibrated predictions (first 5):")
    for name, prob in list(preds.items())[:5]:
        print(f"  {name}: {prob:.4f}")