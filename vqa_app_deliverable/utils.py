import torch
import torch.nn.functional as F
import numpy as np
from config import Config

def apply_temperature_scaling(logits, temperature=Config.TEMPERATURE):
    """
    Applies temperature scaling to raw logits for probability calibration.
    """
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits)
    scaled_logits = logits / temperature
    return torch.sigmoid(scaled_logits).detach().cpu().numpy()

def apply_thresholds(probs, thresholds=Config.OPTIMIZED_THRESHOLDS):
    """
    Applies per-class optimal thresholds to get binary classifications.
    probs: numpy array of shape (N, NUM_CLASSES) or (NUM_CLASSES,)
    """
    if probs.ndim == 1:
        probs = np.expand_dims(probs, 0)
        
    predictions = np.zeros_like(probs, dtype=int)
    for i, cls_name in enumerate(Config.CLASS_NAMES):
        t = thresholds.get(cls_name, Config.DEFAULT_THRESHOLD)
        predictions[:, i] = (probs[:, i] >= t).astype(int)
        
    return predictions[0] if predictions.shape[0] == 1 else predictions

def weighted_binary_cross_entropy(y_pred, y_true, pos_weight=None):
    """
    Calculates weighted binary cross entropy loss.
    y_pred: logit predictions
    y_true: ground truth
    pos_weight: class specific weights
    """
    # Just a utility placeholder in case fine-tuning is triggered
    if pos_weight is None:
        pos_weight = torch.ones([Config.NUM_CLASSES]).to(Config.DEVICE)
    loss = F.binary_cross_entropy_with_logits(y_pred, y_true, pos_weight=pos_weight)
    return loss
