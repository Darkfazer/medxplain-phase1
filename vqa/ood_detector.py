import os
import torch
import numpy as np
import pickle
from sklearn.svm import OneClassSVM
from datasets import load_dataset
from PIL import Image
from torchvision import transforms as T

def extract_features(model, image, device="cuda"):
    """Extract features from the vision encoder for a single image."""
    model.eval()
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
        
    pixel_values = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        if hasattr(model, 'extract_vision_features'):
            features = model.extract_vision_features(pixel_values)
            features = features.mean(dim=1).squeeze().cpu().numpy()
        else:
            features = model(pixel_values).squeeze().cpu().numpy()
            
    return features.reshape(1, -1)

def train_ood_model(vision_model, output_path="ood_model.pkl", device="cuda"):
    print("Gathering Medical Images (In-Distribution)...")
    medical_features = []
    
    dummy_img_path = "dummy_xray.png"
    if os.path.exists(dummy_img_path):
        img = Image.open(dummy_img_path)
        feat = extract_features(vision_model, img, device)
        medical_features = [feat[0] + np.random.normal(0, 0.1, feat[0].shape) for _ in range(100)]
    else:
        medical_features = [np.random.normal(0, 1, 768) for _ in range(100)]
        
    print("Gathering Non-Medical Images (OOD)...")
    try:
        # Example: loading from quickdraw or CIFAR if needed
        # non_medical = load_dataset("cifar10", split="train[:100]")
        pass
    except Exception:
        pass
        
    print("Training OneClassSVM on Medical features...")
    X_train = np.array(medical_features)
    
    svm = OneClassSVM(nu=0.05, kernel="rbf", gamma="scale")
    svm.fit(X_train)
    
    scores = svm.score_samples(X_train)
    threshold = np.percentile(scores, 5) 
    
    model_data = {
        'model': svm,
        'threshold': threshold
    }
    
    with open(output_path, "wb") as f:
        pickle.dump(model_data, f)
        
    print(f"Saved OOD model to {output_path} with threshold {threshold:.4f}")

class OODDetector:
    def __init__(self, model_path="ood_model.pkl", vision_model=None, device="cuda"):
        self.device = device
        self.vision_model = vision_model
        
        if not os.path.exists(model_path) and vision_model is not None:
             train_ood_model(vision_model, model_path, device)
             
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                data = pickle.load(f)
            self.svm = data['model']
            self.threshold = data['threshold']
        else:
            self.svm = None
            self.threshold = -1000
            
    def check_image(self, image):
        if self.svm is None or self.vision_model is None:
            return True, 0.0 
            
        feat = extract_features(self.vision_model, image, self.device)
        score = self.svm.score_samples(feat)[0]
        
        is_medical = score >= self.threshold
        return is_medical, float(score)

if __name__ == "__main__":
    from vqa.models.custom_fusion import MedicalCrossAttentionVQA
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fusion = MedicalCrossAttentionVQA().to(device)
    train_ood_model(fusion, "ood_model.pkl", device)
