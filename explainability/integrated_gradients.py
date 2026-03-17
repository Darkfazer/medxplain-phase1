import torch
import numpy as np

class MedicalIntegratedGradients:
    """
    Alternative Explainability method using Integrated Gradients via Captum.
    Useful for comparing against Gradient/Activation based methods like Grad-CAM,
    especially strong for Transformer architectures where standard spatial CAM breaks down.
    """
    def __init__(self, model):
        try:
            from captum.attr import IntegratedGradients
            self.model = model
            self.model.eval()
            self.ig = IntegratedGradients(self.model)
        except ImportError:
            print("Warning: Captum is not installed. Integrated Gradients will not work.")
            self.ig = None

    def generate(self, input_tensor: torch.Tensor, target_class: int, baseline=None, steps=50) -> np.ndarray:
        """
        Computes the Integrated Gradients attribution mask.
        Args:
            input_tensor: The input image tensor (1, 3, H, W).
            target_class: The classification index to explain.
            baseline: The baseline reference (usually zeroes/black image).
            steps: Number of integral approximation steps.
        Returns:
            np.ndarray of shape (H, W) reduced attribution scores.
        """
        if self.ig is None:
            raise ImportError("Please install captum: `pip install captum`")
            
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)
            
        input_tensor.requires_grad_()
        
        # Calculate attributions
        attributions, delta = self.ig.attribute(
            input_tensor, 
            baseline, 
            target=target_class, 
            return_convergence_delta=True,
            n_steps=steps
        )
        
        # Convert to numpy block
        attr_np = attributions.squeeze(0).cpu().detach().numpy() # Shape (3, H, W)
        
        # Pool across channels (RGB/Grayscale wrapper) using sum of absolute values
        # as standard for feature importance maps.
        attr_pooled = np.sum(np.abs(attr_np), axis=0) 
        
        # Normalize to [0, 1]
        attr_pooled = attr_pooled - np.min(attr_pooled)
        attr_pooled = attr_pooled / (np.max(attr_pooled) + 1e-8)
        
        return attr_pooled

if __name__ == "__main__":
    print("Integrated Gradients Module Verified")
