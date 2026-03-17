import torch
import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as T

class CounterfactualExplainer:
    def __init__(self, classifier_model, cam_generator, class_names=None):
        self.model = classifier_model
        self.cam_generator = cam_generator
        self.class_names = class_names or []
        self.device = next(self.model.parameters()).device
        
    def encode_image(self, image_tensor):
        """Extract latent representation."""
        self.model.eval()
        with torch.no_grad():
            features = self.model.model.features(image_tensor)
        return features
        
    def get_pathology_vector(self, class_idx):
        """Average difference between positive/negative examples (Mock vector)."""
        return torch.randn(1, 1024, 7, 7).to(self.device)
        
    def interpolate(self, original_latent, pathology_vector, alpha=1.0):
        """Move latent representation towards 'normal' direction."""
        return original_latent - (alpha * pathology_vector)
        
    def decode_to_image(self, latent_tensor):
        """Decode latent back to image."""
        # Unimplemented, fallback to saliency-guided used below
        return None
        
    def generate_heatmap(self, original_img, counterfactual_img):
        """Difference between original and counterfactual."""
        diff = np.abs(original_img.astype(np.float32) - counterfactual_img.astype(np.float32))
        diff = np.mean(diff, axis=-1)
        diff = diff / (diff.max() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * diff), cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
    def generate_saliency_guided_counterfactual(self, pil_image, target_class_idx):
        """Fallback method: Blur salient regions to simulate 'normal' tissue."""
        self.model.eval()
        
        # 1. Preprocess image
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
            
        image_tensor = transform(pil_image).unsqueeze(0).to(self.device)
        
        # 2. Get original prediction confidence
        with torch.no_grad():
            orig_logits = self.model(image_tensor)
            orig_conf = torch.sigmoid(orig_logits)[0, target_class_idx].item()
            
        # 3. Get Grad-CAM heatmap mask
        mask = self.cam_generator.generate(image_tensor, target_class=target_class_idx)
        
        # 4. Blur regions of high saliency
        img_np = np.array(pil_image.resize((224, 224)))
        blurred_img = cv2.GaussianBlur(img_np, (51, 51), 0)
        
        mask_3d = np.expand_dims(mask, axis=-1)
        counterfactual_np = (img_np * (1 - mask_3d) + blurred_img * mask_3d).astype(np.uint8)
        
        # 5. Get new confidence on counterfactual
        cf_tensor = transform(Image.fromarray(counterfactual_np)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            new_logits = self.model(cf_tensor)
            new_conf = torch.sigmoid(new_logits)[0, target_class_idx].item()
            
        # 6. Difference map
        diff_map = self.generate_heatmap(img_np, counterfactual_np)
        
        return Image.fromarray(counterfactual_np), Image.fromarray(diff_map), orig_conf, new_conf
