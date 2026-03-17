import sys
import os
import torch
from PIL import Image

sys.path.append(os.path.abspath('.'))
from vqa.app.demo import chat_vqa, predict_classification

print("--- Testing Classification ---")
img = Image.new('RGB', (224, 224), color='black')
res, heatmap = predict_classification(img)
print("Classification Top Diagnosis:", list(res.items())[0] if res else "None")

print("\n--- Testing Custom VQA Backend ---")
ans_custom = chat_vqa(img, 'Is there any abnormality?', 'Custom (Cross-Attention Fusion)')
print("Custom VQA Output (should be untrained random text):", ans_custom)

print("\nAll integration pathways successfully executed.")
