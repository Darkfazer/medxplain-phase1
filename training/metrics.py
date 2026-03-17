import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
import time

class ClinicalMetrics:
    """
    Computes clinical metrics for Multi-Label classification arrays.
    Tracks AUC-ROC, F1, Sensitivity, Specificity per class and macro/micro averages.
    """
    def __init__(self, class_names: list):
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()
        
    def reset(self):
        self.y_true = []
        self.y_score = []
        self.y_pred = []
        self.inference_times = []

    def update(self, y_true: torch.Tensor, y_score: torch.Tensor, threshold: float = 0.5):
        """
        Args:
            y_true (Tensor): Ground truth binary labels (B, num_classes)
            y_score (Tensor): Predicted probabilities (B, num_classes)
            threshold (float): Threshold to binarize scores into predictions.
        """
        start_time = time.time()
        
        # Move completely to CPU numpy
        self.y_true.append(y_true.detach().cpu().numpy())
        self.y_score.append(y_score.detach().cpu().numpy())
        
        y_p = (y_score.detach().cpu().numpy() >= threshold).astype(int)
        self.y_pred.append(y_p)
        
        self.inference_times.append(time.time() - start_time)

    def compute(self) -> dict:
        """
        Computes the final dictionary of clinical metrics.
        """
        y_t = np.vstack(self.y_true)
        y_s = np.vstack(self.y_score)
        y_p = np.vstack(self.y_pred)

        results = {}
        
        # 1. AUC-ROC
        # We calculate per-class AUC, then macro/micro
        aucs = []
        for i, class_name in enumerate(self.class_names):
            try:
                auc = roc_auc_score(y_t[:, i], y_s[:, i])
                aucs.append(auc)
                results[f'auc_{class_name}'] = auc
            except ValueError:
                # Occurs if a class only has 1 label in the batch/split
                aucs.append(np.nan)
                
        results['auc_macro'] = np.nanmean(aucs)
        try:
            results['auc_micro'] = roc_auc_score(y_t, y_s, average='micro')
        except ValueError:
            results['auc_micro'] = np.nan

        # 2. F1 Scores
        results['f1_macro'] = f1_score(y_t, y_p, average='macro', zero_division=0)
        results['f1_micro'] = f1_score(y_t, y_p, average='micro', zero_division=0)
        
        # 3. Sensitivity / Specificity per class
        sensitivities = []
        specificities = []
        for i, class_name in enumerate(self.class_names):
            tn, fp, fn, tp = confusion_matrix(y_t[:, i], y_p[:, i], labels=[0, 1]).ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            sensitivities.append(sens)
            specificities.append(spec)
            
            results[f'sens_{class_name}'] = sens
            results[f'spec_{class_name}'] = spec
            
        results['sens_macro'] = np.mean(sensitivities)
        results['spec_macro'] = np.mean(specificities)
        
        # 4. Exact Match Ratio (Strict Accuracy)
        results['accuracy_exact'] = accuracy_score(y_t, y_p)

        # 5. Timing
        results['avg_inference_time_ms'] = np.mean(self.inference_times) * 1000

        return results

if __name__ == "__main__":
    # Test script
    metrics = ClinicalMetrics(class_names=["Atelectasis", "Cardiomegaly"])
    y_true = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_score = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.9], [0.1, 0.2]])
    metrics.update(y_true, y_score)
    res = metrics.compute()
    print("Test Metrics Computed:")
    for k, v in res.items():
        if "auc" in k or "macro" in k:
            print(f"{k}: {v:.4f}")
