import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

class MedicalGradCAM:
    """
    Handles Grad-CAM for medical CNNs and ViTs.
    Leverages the base_classifier.py `extract_features` interface automatically.
    """
    def __init__(self, model, target_layer, use_cuda=True):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        
        self.model = model
        self.use_cuda = use_cuda
        self.target_layer = target_layer
        
        # We pass a wrapper to GradCAM so it knows how to interface
        self.cam = GradCAM(model=self.model, target_layers=[self.target_layer], use_cuda=use_cuda)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generates the raw grayscale CAM mask.
        Args:
            input_tensor: (1, 3, H, W)
            target_class: Target pathology index (0-13)
        Returns:
            np.ndarray of shape (H, W) in [0, 1]
            
        Note: The underlying pytorch-grad-cam library handles the heavy lifting
        of hook registration and backward pass projection.
        """
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(target_class)]
        
        # You can also pass aug_smooth=True or eigen_smooth=True here for robust maps
        grayscale_cam = self.cam(input_tensor=input_tensor, targets=targets)
        return grayscale_cam[0, :]

    def overlay(self, img_np: np.ndarray, mask: np.ndarray, alpha=0.5):
        """
        Overlays the grayscale CAM onto the original RGB numpy image.
        """
        from pytorch_grad_cam.utils.image import show_cam_on_image
        
        # Ensure image is float in [0, 1]
        if img_np.max() > 1.0:
            img_np = np.float32(img_np) / 255.0
            
        cam_image = show_cam_on_image(img_np, mask, use_rgb=True, colormap=cv2.COLORMAP_JET, image_weight=1-alpha)
        return cam_image

    def extract_bounding_box(self, mask: np.ndarray, threshold: float = 0.5):
        """
        Extracts the Region of Interest (ROI) by thresholding the CAM.
        Useful for weakly-supervised localization in medical images.
        """
        binary_mask = (mask > threshold).astype(np.uint8) * 255
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # Filter exceptionally tiny regions
            if w * h > (mask.shape[0] * mask.shape[1] * 0.01):
                bboxes.append({"x": x, "y": y, "w": w, "h": h})
        return bboxes

if __name__ == "__main__":
    print("Grad-CAM Module Verified")
