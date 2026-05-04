"""
training/losses.py
==================
Custom loss functions for multi-label medical image classification.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(labels_df) -> torch.Tensor:
    """Compute inverse-frequency class weights from a binary label DataFrame.

    Parameters
    ----------
    labels_df : pd.DataFrame of shape (N, num_classes) with binary 0/1 values

    Returns
    -------
    torch.Tensor of shape (num_classes,) with float32 weights
    """
    class_counts = labels_df.sum(axis=0).replace(0, 1)  # avoid /0
    total = len(labels_df)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32)


class WeightedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with per-class positive weighting.

    Particularly useful for medical multi-label tasks where positive
    examples are rare.

    Parameters
    ----------
    pos_weight : (num_classes,) tensor of positive weights, or None
    """

    def __init__(self, pos_weight: torch.Tensor | None = None) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)


class FocalLoss(nn.Module):
    """Focal Loss for multi-label classification.

    Down-weights well-classified examples so training focuses on hard ones.

    Parameters
    ----------
    alpha  : weighting factor for the rare class  (default 0.25)
    gamma  : focusing parameter (default 2.0)
    reduction : ``'mean'`` | ``'sum'`` | ``'none'``
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs    = torch.sigmoid(logits)
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t      = probs * targets + (1 - probs) * (1 - targets)
        loss     = bce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


if __name__ == "__main__":
    logits  = torch.randn(4, 14)
    targets = torch.randint(0, 2, (4, 14)).float()
    print(f"BCE Loss  : {WeightedBCEWithLogitsLoss()(logits, targets):.4f}")
    print(f"Focal Loss: {FocalLoss()(logits, targets):.4f}")
