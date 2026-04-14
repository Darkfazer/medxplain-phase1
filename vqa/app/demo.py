import os
import sys
import torch
from PIL import Image
import gradio as gr
import numpy as np
import torchvision.transforms as T

# Ensure project root is in path to load our custom modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.cnn_models.densenet_adapter import DenseNetAdapter
from explainability.grad_cam import MedicalGradCAM
from vqa.models.blip2_adapter import Blip2VQAAdapter
from transformers import AutoProcessor

# --- GLOBAL CONFIGURATION (MOCK/PLACEHOLDER FOR NOW) ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 14
CLASS_NAMES = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", 
               "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", 
               "Fibrosis", "Pleural_Thickening", "Hernia"]
PHASE1_CHECKPOINTS = [
    "experiments/results/densenet121/best_model.pth",
    "experiments/results/densenet121/swa_best_model.pth",
]

def load_phase1_model():
    """Loads a classification model from Phase 1 for predictions and Grad-CAM."""
    print("Loading Phase 1 Classifier...")
    model = DenseNetAdapter(num_classes=NUM_CLASSES)
    loaded_ckpt = None
    for ckpt in PHASE1_CHECKPOINTS:
        if not os.path.exists(ckpt):
            continue
        try:
            model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            loaded_ckpt = ckpt
            break
        except Exception as e:
            print(f"  -> Failed loading checkpoint {ckpt}: {e}")
    if loaded_ckpt is not None:
        print(f"  -> Loaded checkpoint: {loaded_ckpt}")
    else:
        print("  -> Warning: no checkpoint found, using random weights.")
    model.to(DEVICE)
    model.eval()
    return model
    
def load_phase2_model():
    """Loads the BLIP-2 VQA Model from Phase 2."""
    try:
        print("Loading Phase 2 BLIP-2...")
        model = Blip2VQAAdapter(use_lora=False) # Keep lora false if weights aren't trained yet to save time in demo
        # model.load_checkpoint("experiments/vqa/checkpoints/best_lora_model")
        model.to(DEVICE)
        model.eval()
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
        return model, processor
    except ImportError:
        print("HuggingFace dependencies missing. Phase 2 disabled in demo.")
        return None, None

# Try loading global instances to prevent reloading on every Gradio button click
def load_custom_fusion_model():
    """Loads the new Cross-Attention Fusion Model."""
    try:
        from vqa.models.custom_fusion import MedicalCrossAttentionVQA
        from transformers import AutoTokenizer
        print("Loading Custom Fusion VQA Model...")
        model = MedicalCrossAttentionVQA(vocab_size=50272).to(DEVICE)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
        return model, tokenizer
    except Exception as e:
        print(f"Failed to load custom fusion model: {e}")
        return None, None

PHASE1_MODEL = load_phase1_model()
PHASE2_MODEL, PROCESSOR = load_phase2_model()

def load_ensemble():
    try:
        from ensemble import MedicalEnsemble
        print("Loading Medical Ensemble (DenseNet, ResNet, EfficientNet)...")
        ens = MedicalEnsemble(use_vqa=False)
        return ens
    except Exception as e:
        print(f"Failed to load MedicalEnsemble: {e}")
        return None

PHASE1_ENSEMBLE = load_ensemble()
from vqa.ood_detector import OODDetector
CUSTOM_FUSION_MODEL, CUSTOM_TOKENIZER = load_custom_fusion_model()
OOD_DETECTOR = None
if CUSTOM_FUSION_MODEL is not None:
    print("Initializing OOD Detector...")
    OOD_DETECTOR = OODDetector(model_path="experiments/ood_model.pkl", vision_model=CUSTOM_FUSION_MODEL, device=DEVICE)

def preprocess_image(pil_img):
    """DenseNet validation-time preprocessing."""
    pil_img = pil_img.convert("RGB")
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0).to(DEVICE)

def get_classification_results(input_tensor):
    if PHASE1_ENSEMBLE is not None:
        return PHASE1_ENSEMBLE.predict_classification(input_tensor)

    with torch.no_grad():
        logits = PHASE1_MODEL(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
    return {CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))}

def predict_classification(image):
    if image is None: return "Please upload an image.", None

    input_tensor = preprocess_image(image)
    results = get_classification_results(input_tensor)

    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    filtered_results = {k: v for k, v in ranked if v >= 0.3}

    if len(filtered_results) < 3:
        for label, score in ranked:
            if label not in filtered_results:
                filtered_results[label] = score
            if len(filtered_results) >= 3:
                break

    if not any(score >= 0.3 for score in filtered_results.values()):
        filtered_results = {f"Low confidence: {k}": v for k, v in filtered_results.items()}

    top_diagnosis = ranked[0]
    
    top_index = CLASS_NAMES.index(top_diagnosis[0])
    target_layer = PHASE1_MODEL.model.features[-1] 
    
    try:
        cam_generator = MedicalGradCAM(model=PHASE1_MODEL, target_layer=target_layer, use_cuda=torch.cuda.is_available())
        mask = cam_generator.generate(input_tensor, target_class=top_index)
        
        numpy_img = np.array(image.resize((224, 224)))
        heatmap_img = cam_generator.overlay(numpy_img, mask, alpha=0.5)
    except Exception as e:
        print(f"GradCAM Failed natively in Demo (requires autograd to be enabled in strict environments): {e}")
        heatmap_img = np.array(image.resize((224, 224))) # fallback
        
    return filtered_results, heatmap_img

def chat_vqa(image, question, model_choice):
    if image is None or not question: return "Please upload an image and ask a question."
    
    # Check if confidence threshold is too low from Phase 1 prior to answering
    input_tensor = preprocess_image(image)
    results = get_classification_results(input_tensor)
        
    max_conf = max(results.values())
    low_conf_prefix = ""
    # Enforcing strict 30% confidence threshold gating logic
    if max_conf < 0.30:
        low_conf_prefix = "### ⚠️ RADIOLOGIST ALERT REQUIRED\nOverall model confidence is extremely low (<30%). The following AI answer may be a hallucination. Please consult a human expert.\n---\n"
        
    extracted_pathologies = ", ".join([p for p, v in results.items() if v > 0.5])
    if not extracted_pathologies:
        extracted_pathologies = "No significant abnormalities"
        
    try:
        from vqa.pubmed_retriever.context_builder import build_context
        retrieved_abstracts = build_context(question, results)
    except Exception as e:
        print(f"PubMed retrieval failed: {e}")
        retrieved_abstracts = "PubMed retrieval disabled or failed."

    input_text = f"""Medical Image Analysis Task

Relevant Literature:
{retrieved_abstracts}

Question: {question}
Image Findings: {extracted_pathologies}

Answer with citations:"""
    
    if "Custom" in model_choice:
        if CUSTOM_FUSION_MODEL is None: return "Custom VQA Model is not loaded."
        
        inputs = CUSTOM_TOKENIZER(input_text, return_tensors="pt").to(DEVICE)
        pixel_values = preprocess_image(image)
        
        with torch.no_grad():
            generated_ids = CUSTOM_FUSION_MODEL.generate(
                pixel_values=pixel_values,
                start_token_id=CUSTOM_TOKENIZER.bos_token_id or 2,
                end_token_id=CUSTOM_TOKENIZER.eos_token_id or 2,
                max_new_tokens=50,
                device=DEVICE,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
        res = CUSTOM_TOKENIZER.decode(generated_ids[0], skip_special_tokens=True).strip()
        # Clean prefix text if model echoed it (common untrained behavior)
        if input_text in res: res = res.replace(input_text, "").strip()
        return low_conf_prefix + res
        
    else:
        if PHASE2_MODEL is None: return "VQA Model (BLIP-2) is not loaded due to missing dependencies."
        
        encoding = PROCESSOR(images=image.convert("RGB"), text=input_text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            generated_ids = PHASE2_MODEL.generate(
                pixel_values=encoding["pixel_values"], 
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                max_new_tokens=50,
                do_sample=False,
                num_beams=3,
                temperature=1.0,
                repetition_penalty=1.2
            )
            
        res = PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Enforce Yes/No if question implies it
        if question.lower().startswith(('is', 'are', 'does', 'do', 'has', 'have')):
            if 'yes' in res.lower() and 'no ' not in res.lower():
                res = 'yes'
            elif 'no ' in res.lower() or 'not' in res.lower() or 'negative' in res.lower() or res.lower() == 'no':
                res = 'no'
                
        return low_conf_prefix + res

def check_ood_status(image):
    if image is None or OOD_DETECTOR is None:
        return "", gr.update(interactive=True)
    
    is_medical, score = OOD_DETECTOR.check_image(image)
    if not is_medical:
        warning = f"### ⚠️ Unknown input distribution. Model may not be reliable. (Score: {score:.2f})"
        return warning, gr.update(interactive=False)
    else:
        return f"### ✅ Valid Medical Image Detected (Score: {score:.2f})", gr.update(interactive=True)

def generate_counterfactual_demo(image):
    if image is None: return None, None, None, "Please upload an image."
    
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        logits = PHASE1_MODEL(input_tensor)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
    top_index = int(np.argmax(probs))
    top_class_name = CLASS_NAMES[top_index]
    
    target_layer = PHASE1_MODEL.model.features[-1]
    from explainability.grad_cam import MedicalGradCAM
    cam_generator = MedicalGradCAM(model=PHASE1_MODEL, target_layer=target_layer, use_cuda=torch.cuda.is_available())
    
    from explainability.counterfactual import CounterfactualExplainer
    cf_explainer = CounterfactualExplainer(PHASE1_MODEL, cam_generator, CLASS_NAMES)
    
    try:
        cf_img, diff_img, orig_conf, new_conf = cf_explainer.generate_saliency_guided_counterfactual(image, top_index)
        text = f"### What-If Analysis\n**If the highlighted {top_class_name} region looked more normal, confidence would drop from {orig_conf*100:.1f}% to {new_conf*100:.1f}%.**"
    except Exception as e:
        print(f"Counterfactual Generation failed: {e}")
        cf_img, diff_img, text = image, image, "Counterfactual generation failed."
        
    return image, cf_img, diff_img, text

# --- GRADIO UI DEFINITION ---
with gr.Blocks(title="MedXPlain: Diagnostics & VQA") as demo:
    gr.Markdown("# MedXPlain: MVP Dashboard")
    gr.Markdown("Upload a Chest X-Ray to receive 14-class pathology predictions alongside Grad-CAM explainability heatmaps. Then, chat with the AI about the X-Ray findings.")
    
    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="pil", label="Upload Chest X-Ray")
            btn_diagnose = gr.Button("Run Phase 1: Classification & Explanations", variant="primary")
            btn_counterfactual = gr.Button("Generate Counterfactual Explanation (What-If)")
            
        with gr.Column(scale=1):
            class_labels = gr.Label(num_top_classes=3, label="Pathology Probabilities")
            cam_output = gr.Image(label="Grad-CAM Heatmap (Top Prediction)")

    btn_diagnose.click(fn=predict_classification, inputs=img_input, outputs=[class_labels, cam_output])
    
    cf_text = gr.Markdown("")
    with gr.Row():
        cf_orig = gr.Image(label="Original X-Ray")
        cf_new = gr.Image(label="Simulated Healthy X-Ray")
        cf_diff = gr.Image(label="Difference Map (Changed Regions)")
        
    btn_counterfactual.click(
        fn=generate_counterfactual_demo, 
        inputs=img_input, 
        outputs=[cf_orig, cf_new, cf_diff, cf_text]
    )
    
    gr.Markdown("---")
    gr.Markdown("## Visual Question Answering (VQA-RAD)")
    ood_warning = gr.Markdown("")
    
    with gr.Row():
        with gr.Column(scale=3):
            model_choice = gr.Radio(
                choices=["Baseline (BLIP-2 Q-Former)", "Custom (Cross-Attention Fusion)"], 
                value="Baseline (BLIP-2 Q-Former)", 
                label="VQA Architecture"
            )
            q_input = gr.Textbox(label="Ask a question about the X-Ray (e.g. 'Is there cardiomegaly?')")
            btn_chat = gr.Button("Generate Answer")
        with gr.Column(scale=2):
            a_output = gr.Textbox(label="AI Answer", interactive=False)
            
    img_input.change(fn=check_ood_status, inputs=img_input, outputs=[ood_warning, btn_chat])
    btn_chat.click(fn=chat_vqa, inputs=[img_input, q_input, model_choice], outputs=a_output)

if __name__ == "__main__":
    demo.launch(share=False)
