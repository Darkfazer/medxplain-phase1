import torch
import torch.nn.functional as F

class TemperatureScaling:
    """Calibrates confidences using Temperature Scaling (T=1.263)."""
    def __init__(self, temperature: float = 1.263):
        self.temperature = temperature

    def calibrate(self, logits: torch.Tensor) -> torch.Tensor:
        return F.softmax(logits / self.temperature, dim=-1)
