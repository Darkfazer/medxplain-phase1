import os
import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Blip2Processor

import configs.config as cfg
from models.vqa_model import VQAModel
from explainability.gradcam import BLIP2GradCAM, overlay_cam

print("Loading processor and model context...")

# Load models and weights lazily inside app to avoid immediate memory allocation 
# globally, or globally if preferred. For Gradio, global is usually fine.
processor = Blip2Processor.from_pretrained(cfg.MODEL_NAME)
model = VQAModel(freeze_vision=False) # Important: Need gradients for Grad-CAM!

# Load best weights if available
model_path = os.path.join(cfg.BASE_DIR, "experiments", "best_model.pth")
if os.path.exists(model_path):
    print(f"Loading trained weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
else:
    print("No trained weights found. Using baseline pre-trained model.")

model.eval()

# Select target layer for Grad-CAM
# For BLIP-2 EVA-CLIP vision model, the layers are in `model.model.vision_model.encoder.layers`
# We'll hook the `layer_norm1` of the last transformer layer.
target_layer = model.model.vision_model.encoder.layers[-1].layer_norm1
grad_cam = BLIP2GradCAM(model, target_layer)

def infer(image, question):
    if image is None or question == "":
        return "Please provide an image and a question.", None
        
    # Convert image to RGB if not already
    image = image.convert('RGB')
        
    # Standardize prompt
    prompt = f"Question: {question} Answer:"
    
    encoding = processor(images=image, text=prompt, return_tensors="pt")
    pixel_values = encoding.pixel_values.to(cfg.DEVICE)
    input_ids = encoding.input_ids.to(cfg.DEVICE)
    attention_mask = encoding.attention_mask.to(cfg.DEVICE)
    
    # 1. Generate text
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, input_ids, attention_mask)
        answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
    # 2. Generate Grad-CAM
    # Re-run forward with grads to get heatmap
    try:
        cam = grad_cam.generate_cam(pixel_values, input_ids, attention_mask)
        # Process the image to match the CAM dimensions
        # The input pixel_values are normalized. We should resize the *original* image
        image_np = np.array(image.resize((224, 224)))
        cam_image = overlay_cam(image_np, cam, alpha=0.5)
        out_image = Image.fromarray(cam_image)
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")
        out_image = image
    
    return answer, out_image

# Gradio Interface
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="Medical Image"), 
        gr.Textbox(label="Question", placeholder="e.g. Is there a fracture?")
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"), 
        gr.Image(type="pil", label="Grad-CAM Heatmap")
    ],
    title="⚕️ VQA-RAD with BLIP-2 & Grad-CAM",
    description=("Upload a radiology image and ask a question. "
                 "The model will generate an answer and highlight the regions "
                 "it focused on using Grad-CAM."),
    theme=gr.themes.Soft()
)

if __name__ == "__main__":
    demo.launch(share=False)
