import torch
import cv2
import numpy as np

class BLIP2GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        # HuggingFace standard transformers might output a tuple
        if isinstance(output, tuple):
            self.activations = output[0].detach()
        else:
            self.activations = output.detach()
            
    def save_gradient(self, module, grad_input, grad_output):
        # HuggingFace passes gradients backwards
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, pixel_values, input_ids, attention_mask):
        self.model.eval()
        self.model.zero_grad()
        
        # Require grad on pixel_values
        pixel_values.requires_grad_(True)
        
        # Force a forward pass
        outputs = self.model.model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Output language model logits
        logits = outputs.logits
        # We compute gradients with respect to the maximum logit of the first predicted token
        target = logits[0, -1, :].max()
        target.backward()
        
        if self.gradients is None or self.activations is None:
            raise ValueError("Hooks did not capture gradients or activations. Check target layer.")
            
        # Calculate Grad-CAM
        # self.gradients: [batch, seq_len, dim]
        # self.activations: [batch, seq_len, dim]
        # For ViT, token 0 is CLS, 1: is spatial.
        
        grad = self.gradients[0, 1:, :] # [seq, dim]
        act = self.activations[0, 1:, :] # [seq, dim]
        
        weights = torch.mean(grad, dim=0) # [dim]
        cam = torch.matmul(act, weights) # [seq]
        cam = torch.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        # Reshape to 2D
        seq_len = cam.shape[0]
        size = int(np.sqrt(seq_len))
        # Default for 224x224 and patch_size 14 is 16x16=256
        if size * size != seq_len:
            # Fallback if dimensions are weird (e.g., using a different resolution)
            size = int(np.sqrt(seq_len + 1))
            
        cam = cam.view(size, size).cpu().numpy()
        
        # Resize to 224x224 (the image input resolution)
        cam = cv2.resize(cam, (224, 224))
        return cam

def overlay_cam(image_np, cam, alpha=0.5):
    """
    Overlays a Grad-CAM heatmap onto an image.
    Args:
        image_np (np.ndarray): Original image array (H, W, 3), RGB format.
        cam (np.ndarray): Grad-CAM heatmap (H, W), values in [0, 1].
    Returns:
        np.ndarray: Overlay image.
    """
    # Create colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) # OpenCV uses BGR natively
    heatmap = np.float32(heatmap) / 255
    
    # Normalize original image
    if image_np.max() > 1.0:
        image_np = np.float32(image_np) / 255
    else:
        image_np = np.float32(image_np)
        
    cam_overlay = heatmap * alpha + image_np * (1 - alpha)
    cam_overlay = cam_overlay / np.max(cam_overlay)
    return np.uint8(255 * cam_overlay)
