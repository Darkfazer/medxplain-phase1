import torch

def get_optimizer(model: torch.nn.Module, lr: float = 1e-4):
    """Returns scheduled optimizer with weight decay."""
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
