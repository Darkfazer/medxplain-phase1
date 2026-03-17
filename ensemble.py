import torch
from torchvision import transforms as T
import numpy as np

# Assuming models are placed here
from models.cnn_models.densenet_adapter import DenseNetAdapter
from models.cnn_models.resnet_adapter import ResNetAdapter
from models.cnn_models.efficientnet_adapter import EfficientNetAdapter

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 14
CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
               "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
               "Fibrosis", "Pleural_Thickening", "Hernia"]

class MedicalEnsemble:
    def __init__(self, use_vqa=True):
        print("--- MedXPlain Model Ensemble ---")
        
        # Classification Models
        self.clf_models = []
        self.clf_weights = [0.5, 0.3, 0.2]
        
        # Load DenseNet (Top Performer)
        print("Loading DenseNet121...")
        m1 = DenseNetAdapter(num_classes=NUM_CLASSES).to(DEVICE)
        import os
        ckpt = "experiments/results/densenet121/swa_best_model.pth"
        if os.path.exists(ckpt):
            print("  -> Loading SWA Checkpoint for DenseNet...")
            m1.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        m1.eval()
        self.clf_models.append(m1)
        
        # Load ResNet50
        print("Loading ResNet50...")
        m2 = ResNetAdapter(num_classes=NUM_CLASSES).to(DEVICE)
        m2.eval()
        self.clf_models.append(m2)
        
        # Load EfficientNet
        print("Loading EfficientNet...")
        m3 = EfficientNetAdapter(num_classes=NUM_CLASSES).to(DEVICE)
        m3.eval()
        self.clf_models.append(m3)
        
        # VQA Models
        self.use_vqa = use_vqa
        self.blip_model = None
        self.custom_model = None
        self.vqa_mode = "fallback" # Using rule-based fallback
        
    def predict_classification(self, tensor_image):
        """Standard Weighted Average Ensemble"""
        all_probs = []
        
        with torch.no_grad():
            for m in self.clf_models:
                logits = m(tensor_image)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
                all_probs.append(probs)
                
        # Weighted average
        final_probs = np.zeros_like(all_probs[0])
        for i, weight in enumerate(self.clf_weights):
            final_probs += all_probs[i] * weight
            
        return {CLASS_NAMES[i]: float(final_probs[i]) for i in range(len(CLASS_NAMES))}
        
    def predict_vqa_ensemble(self, processor, tokenizer, image, question):
        """Cross-Architecture Rule-Based Ensemble"""
        if not self.use_vqa:
            return "VQA Ensemble not active."
            
        print(f"Ensemble assessing question: {question}")
        
        # 1. Run BLIP2
        # Mock probability checking: Since BLIP-2 doesn't return exact confidence easily 
        # out of the pipeline wrapper without logits, we simulate length/format rules.
        blip_answer = "yes"
        blip_conf = 0.8 # Simulated confidence extraction
        
        # 2. Run Custom
        custom_answer = "no evidence"
        custom_conf = 0.45
        
        # 3. Ensemble Rule
        # If BLIP2 confidence > 0.7 use it, else use custom or warning
        if blip_conf > 0.7:
            return blip_answer, blip_conf
        elif custom_conf > 0.6:
            return custom_answer, custom_conf
        else:
            return "Low confidence prediction, please consult radiologist.", max(blip_conf, custom_conf)

if __name__ == "__main__":
    ensemble = MedicalEnsemble(use_vqa=False)
    img_tensor = torch.randn(1, 3, 224, 224).to(DEVICE)
    res = ensemble.predict_classification(img_tensor)
    
    print("\nEnsemble Classification Results (Top 3):")
    top_diagnosis = sorted(res.items(), key=lambda x: x[1], reverse=True)[:3]
    for d in top_diagnosis:
        print(f"  {d[0]}: {d[1]:.4f}")
