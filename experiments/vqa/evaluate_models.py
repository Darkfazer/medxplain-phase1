import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.models.custom_fusion import MedicalCrossAttentionVQA
from vqa.models.blip2_adapter import Blip2VQAAdapter
from vqa.data.custom_vqa_dataset import CustomVQADataset
from vqa.data.vqa_rad_dataset import VQARADDataset
from vqa.training.vqa_metrics import VQAMetrics
from tqdm import tqdm

def evaluate_custom_model(val_loader, model, tokenizer, device):
    metrics = VQAMetrics()
    model.eval()
    
    loop = tqdm(val_loader, desc="Evaluating Custom Fusion Model")
    with torch.no_grad():
        for batch in loop:
            encoding, q_strs, a_strs, t_strs = batch
            pixel_values = encoding["pixel_values"].to(device).float()
            
            generated_ids = model.generate(
                pixel_values=pixel_values,
                start_token_id=tokenizer.bos_token_id or 2,
                end_token_id=tokenizer.eos_token_id or 2,
                max_length=20,
                device=device
            )
            
            b_size = pixel_values.shape[0]
            generated_texts = []
            for b_idx in range(b_size):
                gen = tokenizer.decode(generated_ids[b_idx], skip_special_tokens=True).strip()
                if q_strs[b_idx] in gen: gen = gen.replace(f"Question: {q_strs[b_idx]} Answer:", "").strip()
                generated_texts.append(gen)
                
            for gen_text, gt_text, q_type in zip(generated_texts, a_strs, t_strs):
                metrics.update(gen_text, gt_text, q_type)
                
    return metrics.compute()

def evaluate_baseline_model(val_loader, model, processor, device):
    metrics = VQAMetrics()
    model.eval()
    
    loop = tqdm(val_loader, desc="Evaluating Baseline BLIP-2 Model")
    with torch.no_grad():
        for batch in loop:
            encoding, q_strs, a_strs, t_strs = batch
            pixel_values = encoding["pixel_values"].to(device).float()
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20
            )
            
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for gen, gt, typ in zip(generated_texts, a_strs, t_strs):
                metrics.update(gen.strip(), gt, typ)
                
    return metrics.compute()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Head-to-Head VQA Evaluation ({device}) ---")
    
    # Custom Model Setup
    print("\nLoading Custom Fusion environment...")
    custom_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    custom_val_dataset = CustomVQADataset(data_path='mock.json', image_dir='mock_imgs')
    custom_val_loader = DataLoader(custom_val_dataset, batch_size=2, shuffle=False)
    
    custom_model = MedicalCrossAttentionVQA(vocab_size=len(custom_tokenizer)).to(device)
    # custom_model.load_state_dict(torch.load("experiments/vqa/checkpoints/best_custom_fusion.pth"))
    
    # Baseline Setup
    print("Loading Baseline BLIP-2 environment...")
    baseline_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    baseline_val_dataset = VQARADDataset(data_path='mock.json', image_dir='mock_imgs')
    baseline_val_loader = DataLoader(baseline_val_dataset, batch_size=2, shuffle=False)
    
    try:
        baseline_model = Blip2VQAAdapter(use_lora=False).to(device)
        baseline_available = True
    except ImportError:
        print("Skipping Baseline: Transformers/Peft dependencies missing.")
        baseline_available = False
        
    print("\n--- Running Evaluations ---")
    
    custom_metrics = evaluate_custom_model(custom_val_loader, custom_model, custom_tokenizer, device)
    
    if baseline_available:
        try:
             baseline_metrics = evaluate_baseline_model(baseline_val_loader, baseline_model, baseline_processor, device)
        except Exception as e:
             print(f"Baseline inference failed (ignoring for mock context): {e}")
             baseline_metrics = {"Accuracy_Closed": 0.0, "BLEU_Open": 0.0}
    else:
        baseline_metrics = {"Accuracy_Closed": 0.0, "BLEU_Open": 0.0}
        
    print("\n================ FINAL RESULTS ================")
    print(f"{'Metric':<20} | {'Baseline (BLIP-2)':<20} | {'Custom Fusion':<20}")
    print("-" * 65)
    
    keys = ["Accuracy_Closed", "Accuracy_Open", "Accuracy_All", "BLEU_Closed", "BLEU_Open", "BLEU_All"]
    for k in keys:
        b_val = baseline_metrics.get(k, 0)
        c_val = custom_metrics.get(k, 0)
        print(f"{k:<20} | {b_val:<20.4f} | {c_val:<20.4f}")
    print("===============================================")

if __name__ == "__main__":
    main()
