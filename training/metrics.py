"""
training/metrics.py
====================
Clinical metrics tracker for multi-label chest X-ray classification.
Computes AUC-ROC, F1, Sensitivity, Specificity per class and macro averages.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


class ClinicalMetrics:
    """Accumulate predictions over an epoch then compute clinical metrics.

    Parameters
    ----------
    class_names : list of pathology/class names (length == num_classes)
    """

    def __init__(self, class_names: list[str]) -> None:
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.reset()

    def reset(self) -> None:
        self._y_true: list[np.ndarray] = []
        self._y_score: list[np.ndarray] = []
        self._y_pred: list[np.ndarray] = []
        self._times: list[float] = []

    def update(
        self,
        y_true: torch.Tensor,
        y_score: torch.Tensor,
        threshold: float = 0.5,
    ) -> None:
        """Accumulate one batch of predictions.

        Parameters
        ----------
        y_true  : (B, num_classes) binary ground-truth tensor
        y_score : (B, num_classes) predicted-probability tensor
        threshold : binarisation threshold for binary predictions
        """
        t0 = time.perf_counter()
        yt = y_true.detach().cpu().numpy()
        ys = y_score.detach().cpu().numpy()
        yp = (ys >= threshold).astype(int)
        self._y_true.append(yt)
        self._y_score.append(ys)
        self._y_pred.append(yp)
        self._times.append(time.perf_counter() - t0)

    def compute(self) -> dict[str, float]:
        """Compute and return all clinical metrics as a flat dict."""
        y_t = np.vstack(self._y_true)
        y_s = np.vstack(self._y_score)
        y_p = np.vstack(self._y_pred)
        results: dict[str, float] = {}

        # AUC-ROC per class + macro / micro
        aucs: list[float] = []
        for i, name in enumerate(self.class_names):
            try:
                auc = roc_auc_score(y_t[:, i], y_s[:, i])
                results[f"auc_{name}"] = auc
                aucs.append(auc)
            except ValueError:
                aucs.append(float("nan"))
        results["auc_macro"] = float(np.nanmean(aucs))
        try:
            results["auc_micro"] = float(roc_auc_score(y_t, y_s, average="micro"))
        except ValueError:
            results["auc_micro"] = float("nan")

        # F1
        results["f1_macro"] = float(f1_score(y_t, y_p, average="macro",  zero_division=0))
        results["f1_micro"] = float(f1_score(y_t, y_p, average="micro",  zero_division=0))

        # Sensitivity / Specificity per class + macro averages
        sens_list, spec_list = [], []
        for i, name in enumerate(self.class_names):
            cm = confusion_matrix(y_t[:, i], y_p[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            results[f"sens_{name}"] = float(sens)
            results[f"spec_{name}"] = float(spec)
            sens_list.append(sens)
            spec_list.append(spec)
        results["sens_macro"] = float(np.mean(sens_list))
        results["spec_macro"] = float(np.mean(spec_list))

        # Exact match accuracy
        results["accuracy_exact"] = float(accuracy_score(y_t, y_p))

        # Inference timing (average per update call, in ms)
        results["avg_inference_time_ms"] = float(np.mean(self._times) * 1000)

        return results


if __name__ == "__main__":
    metrics = ClinicalMetrics(["Atelectasis", "Cardiomegaly"])
    y_true  = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]])
    y_score = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.8, 0.9], [0.1, 0.2]])
    metrics.update(y_true, y_score)
    res = metrics.compute()
    for k, v in res.items():
        if any(t in k for t in ("auc", "macro")):
            print(f"{k}: {v:.4f}")
