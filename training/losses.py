import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_class_weights(labels_df):
    """Inverse frequency weighting"""
    class_counts = labels_df.sum(axis=0)
    total = len(labels_df)
    weights = total / (len(class_counts) * class_counts)
    return torch.tensor(weights.values, dtype=torch.float32)

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Standard BCEWithLogitsLoss modified to handle class imbalance via positive weights.
    Particularly useful for medical multi-label tasks where positives are rare.
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        """
        Args:
            pos_weight (Tensor): a weight of positive examples.
                Must be a vector with length equal to the number of classes.
        """
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.criterion(logits, targets)

class FocalLoss(nn.Module):
    """
    Focal Loss for Multi-label classification.
    Focuses training on hard examples by down-weighting well-classified ones.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # BCE components
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        
        # Focal terms
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce_loss * ((1 - p_t) ** self.gamma)
        
        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

if __name__ == "__main__":
    # Test script
    logits = torch.randn(4, 14) # Batch of 4, 14 classes
    targets = torch.randint(0, 2, (4, 14)).float()
    
    criterion1 = WeightedBCEWithLogitsLoss()
    criterion2 = FocalLoss()
    
    print(f"BCE Loss: {criterion1(logits, targets):.4f}")
    print(f"Focal Loss: {criterion2(logits, targets):.4f}")
