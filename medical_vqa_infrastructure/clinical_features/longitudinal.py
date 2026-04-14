import torch

class LongitudinalAnalyzer:
    """Calculates difference between two sequential imaging studies."""
    def compare(self, img_current: torch.Tensor, img_prior: torch.Tensor) -> torch.Tensor:
        return torch.abs(img_current - img_prior)
