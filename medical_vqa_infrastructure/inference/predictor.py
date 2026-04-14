import torch
import time

class VQAPredictor:
    """Single image-question prediction with latency monitoring."""
    def __init__(self, model, device: str = 'cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

    def predict(self, image: torch.Tensor, question: str) -> dict:
        start_time = time.time()
        image = image.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(image, [question])
            
        latency = (time.time() - start_time) * 1000
        return {
            "prediction": "Mock Output",
            "confidence": 0.95,
            "latency_ms": latency
        }
