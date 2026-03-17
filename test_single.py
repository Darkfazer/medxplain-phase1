import os
import torch
from PIL import Image

def test_pipeline():
    print("--- End-to-End Single Sample Pipeline Debug ---")
    
    # 1. Load Image
    img_path = "dummy_xray.png"
    if not os.path.exists(img_path):
        print("Error: Target dummy_xray.png not found")
        return
        
    print(f"\n[Step 1] Loading image from {img_path}")
    img = Image.open(img_path).convert('RGB')
    print(f"Image properties: Size {img.size}, Mode {img.mode}")
    
    # 2. Phase 1 Processing
    print("\n[Step 2] Initializing Phase 1 Classification Native Functions")
    from vqa.app.demo import preprocess_image, PHASE1_MODEL, CLASS_NAMES
    
    input_tensor = preprocess_image(img)
    print(f"Preprocessed tensor shape: {input_tensor.shape}")
    
    with torch.no_grad():
        logits = PHASE1_MODEL(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
    print("Phase 1 Raw Probabilities Output:")
    results = {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}
    for k, v in results.items():
        if v > 0.3:
            print(f"  {k}: {v:.4f}")
            
    top_pred = sorted(results.items(), key=lambda x: x[1], reverse=True)[0]
    print(f"Selected Top Diagnosis: {top_pred}")
    
    # 3. Phase 2 VQA
    print("\n[Step 3] Sending to Custom Cross-Attention VQA")
    from vqa.app.demo import chat_vqa
    
    question = "Is there any evidence of pneumonia?"
    print(f"Test Question: '{question}'")
    
    answer = chat_vqa(img, question, "Custom (Cross-Attention Fusion)")
    print(f"Phase 2 Answer Generation: '{answer}'")
    
    print("\nEnd-to-End Pipeline Completed Setup. No crashes detected.")

if __name__ == "__main__":
    test_pipeline()
