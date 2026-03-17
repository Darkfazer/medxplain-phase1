import os
import json
import numpy as np
import pandas as pd
from PIL import Image

def analyze_nih_chestxray():
    print("--- 1. NIH ChestX-ray14 Data Quality Check ---")
    # For simulation, assume metadata is in data/NIH/Data_Entry_2017.csv
    csv_path = "data/NIH/Data_Entry_2017.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        print("Mocking NIH Metadata for analysis...")
        # Create mock data
        df = pd.DataFrame({
            'Image Index': [f'img_{i}.png' for i in range(1000)],
            'Finding Labels': np.random.choice(['No Finding', 'Atelectasis|Effusion', 'Cardiomegaly', 'Mass', 'Nodule', 'Effusion'], 1000)
        })
        
    print("Class Distribution:")
    all_labels = df['Finding Labels'].str.split('|').explode()
    dist = all_labels.value_counts()
    for k, v in dist.items():
        print(f"  {k}: {v} samples")
        
    missing = df['Finding Labels'].isna().sum()
    print(f"\nMissing Labels: {missing}")
    
    # Verify resolutions (simulate reading 50 images)
    print("\nVerifying Image Resolutions (sampling 50)...")
    resolutions = []
    means = []
    dummy_path = "dummy_xray.png"
    if os.path.exists(dummy_path):
        for _ in range(50):
            img = Image.open(dummy_path).convert('L')
            resolutions.append(img.size)
            means.append(np.mean(np.array(img)))
            
    if resolutions:
        unique_res = set(resolutions)
        print(f"Resolutions found: {unique_res}")
        if len(unique_res) > 1 or (224, 224) not in unique_res:
            print("  [WARNING] Images are not uniformly 224x224!")
        else:
            print("  [OK] All sampled images are 224x224.")
            
        avg_mean = np.mean(means)
        print(f"Average Mean Pixel intensity (0-255 scale): {avg_mean:.2f}")
        if avg_mean < 80 or avg_mean > 180:
             print("  [WARNING] Mean pixel intensity seems abnormal, check normalization.")
    
def analyze_vqa_rad():
    print("\n--- 2. VQA-RAD Data Quality Check ---")
    json_path = "data/VQA-RAD/dataset.json"
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
    else:
        print("Mocking VQA-RAD JSON for analysis...")
        data = [
            {"question": "Is there a fracture?", "answer": "yes", "answer_type": "CLOSED"},
            {"question": "What is the primary abnormality?", "answer": "pulmonary edema", "answer_type": "OPEN"},
            {"question": "Are the lungs clear?", "answer": "no", "answer_type": "CLOSED"},
            {"question": "Where is the mass located?", "answer": "left upper lobe", "answer_type": "OPEN"},
            {"question": "Is there a fracture?", "answer": "yes", "answer_type": "CLOSED"} # Duplicate
        ] * 100
        
    df = pd.DataFrame(data)
    
    print("Question Types:")
    if 'answer_type' in df.columns:
        counts = df['answer_type'].value_counts()
        for k, v in counts.items():
            print(f"  {k}: {v}")
            
    df['answer_length'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    print(f"\nAnswer length distribution:\n  Mean: {df['answer_length'].mean():.2f} words\n  Max: {df['answer_length'].max()} words")
    
    duplicates = df.duplicated(subset=['question']).sum()
    print(f"\nDuplicate Questions found: {duplicates}")
    if duplicates > 0:
        print("  [WARNING] High number of duplicate questions in VQA dataset.")

if __name__ == "__main__":
    analyze_nih_chestxray()
    analyze_vqa_rad()
