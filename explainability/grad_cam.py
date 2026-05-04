"""
explainability/grad_cam.py
==========================
Single canonical Grad-CAM implementation using the pytorch-grad-cam library.

Two public APIs:
  MedicalGradCAM – for BaseMedicalClassifier (DenseNet, ResNet, etc.)
  generate_overlay – convenience function, returns an RGB overlay array

Fixes vs. original:
  - Removed deprecated `use_cuda` kwarg (not supported in grad-cam >= 1.4)
  - Model and tensors placed on device *before* GradCAM call
  - `extract_bounding_box` reshape-safe
"""
from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)


class MedicalGradCAM:
    """Grad-CAM wrapper for any BaseMedicalClassifier.

    Parameters
    ----------
    model        : classifier (nn.Module, already on device)
    target_layer : the nn.Module layer to hook (e.g. model.model.features.denseblock4)

    Example
    -------
    >>> cam = MedicalGradCAM(model, target_layer=model.model.features.denseblock4)
    >>> mask = cam.generate(input_tensor, target_class=3)
    >>> overlay = cam.overlay(original_image_float01, mask)
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        from pytorch_grad_cam import GradCAM   # imported here to keep module importable
                                               # even when grad-cam is not installed
        self.model        = model
        self.target_layer = target_layer
        # No use_cuda argument: device is inferred from the model parameters
        self._cam = GradCAM(model=self.model, target_layers=[self.target_layer])

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> np.ndarray:
        """Compute the Grad-CAM mask.

        Parameters
        ----------
        input_tensor : (1, C, H, W) tensor on the *same device as the model*
        target_class : pathology index (0-based)

        Returns
        -------
        np.ndarray of shape (H, W) with values in [0, 1]
        """
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets      = [ClassifierOutputTarget(target_class)]
        grayscale    = self._cam(input_tensor=input_tensor, targets=targets)
        return grayscale[0]   # (H, W)

    def overlay(
        self,
        img_float: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """Overlay the CAM mask on an RGB image.

        Parameters
        ----------
        img_float : (H, W, 3) float32 array with values in [0, 1]
        mask      : (H, W) float32 CAM mask in [0, 1]
        alpha     : CAM opacity (0 = invisible, 1 = fully opaque)

        Returns
        -------
        uint8 RGB numpy array
        """
        from pytorch_grad_cam.utils.image import show_cam_on_image
        img_float = img_float.astype(np.float32)
        if img_float.max() > 1.0:
            img_float = img_float / 255.0
        return show_cam_on_image(img_float, mask, use_rgb=True,
                                 colormap=cv2.COLORMAP_JET,
                                 image_weight=1.0 - alpha)

    def extract_bounding_box(
        self,
        mask: np.ndarray,
        threshold: float = 0.5,
    ) -> list[dict]:
        """Threshold the CAM mask and return bounding boxes of activated regions.

        Parameters
        ----------
        mask      : (H, W) CAM mask in [0, 1]
        threshold : binarisation threshold

        Returns
        -------
        list of dicts with keys x, y, w, h (pixel coordinates)
        """
        binary = (mask > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        bboxes: list[dict] = []
        min_area = mask.shape[0] * mask.shape[1] * 0.01
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w * h > min_area:
                bboxes.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
        return bboxes


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def generate_overlay(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int,
    original_image_uint8: np.ndarray,
) -> np.ndarray:
    """One-call Grad-CAM overlay generator.

    Parameters
    ----------
    model                : classifier on device
    target_layer         : layer to hook
    input_tensor         : (1, C, H, W) on device
    target_class         : pathology index
    original_image_uint8 : (H, W, 3) uint8 RGB – the *original* image to overlay on

    Returns
    -------
    uint8 RGB overlay numpy array (same spatial size as original_image_uint8)
    """
    cam  = MedicalGradCAM(model, target_layer)
    mask = cam.generate(input_tensor, target_class)

    h, w = original_image_uint8.shape[:2]
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

    return cam.overlay(original_image_uint8.astype(np.float32) / 255.0, mask)


if __name__ == "__main__":
    print("explainability/grad_cam.py – import OK")
