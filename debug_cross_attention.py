import torch
from vqa.models.custom_fusion import MedicalCrossAttentionVQA
from vqa.models.blip2_adapter import Blip2VQAAdapter
from transformers import AutoProcessor, AutoTokenizer
from PIL import Image

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading models...")
    
    # Custom Fusion Model
    custom_model = MedicalCrossAttentionVQA().to(device)
    custom_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    
    # BLIP2 Model
    try:
        blip_model = Blip2VQAAdapter(use_lora=False).to(device)
        blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    except Exception as e:
        print(f"Failed to load BLIP2: {e}")
        return

    image = Image.new('RGB', (224, 224), color='gray')
    question = "What is shown in this image?"
    prompt = f"Question: {question} Answer:"

    print("\n--- DEBUGGING SHAPES ---")
    out_lines = []
    
    # 1. BLIP2 Shapes
    print("BLIP2:")
    blip_inputs = blip_processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        # Step through BLIP2 to get shapes
        pixel_values = blip_inputs.pixel_values
        vision_outputs = blip_model.model.vision_model(pixel_values=pixel_values)
        out_lines.append(f"BLIP2 Vision Encoder Output: {vision_outputs.last_hidden_state.shape}")
        
        image_embeds = vision_outputs[0]
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        query_tokens = blip_model.model.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = blip_model.model.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
        )
        query_output = query_outputs[0]
        language_model_inputs = blip_model.model.language_projection(query_output)
        out_lines.append(f"BLIP2 Projection Layer Output: {language_model_inputs.shape}")
        
        # Cross attention output shape isn't directly exposed in standard BLIP2 generate without hook, 
        # but projection is the equivalent of what enters the LLM.
        
    # 2. Custom Fusion Shapes
    print("\nCustom Fusion:")
    custom_inputs = custom_tokenizer(prompt, return_tensors="pt").to(device)
    
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    pixel_values = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Vision encoder
        v_features = custom_model.vision_encoder(pixel_values)
        out_lines.append(f"Custom Vision Encoder Output: {v_features.shape}")
        
        b, c, h, w = v_features.shape
        v_features_flat = v_features.view(b, c, -1).permute(0, 2, 1)
        projected_patches = custom_model.vision_proj(v_features_flat)
        out_lines.append(f"Custom Projection Layer Output: {projected_patches.shape}")
        
        # Cross attention output
        batch_size, seq_length = custom_inputs.input_ids.shape
        positions = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_length)
        tgt = custom_model.text_embedding(custom_inputs.input_ids) + custom_model.positional_encoding(positions)
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
        
        output = custom_model.decoder(
            tgt=tgt,
            memory=projected_patches,
            tgt_mask=tgt_mask
        )
        out_lines.append(f"Custom Cross-Attention Output: {output.shape}")
        
        logits = custom_model.lm_head(output)
        out_lines.append(f"Custom Final Logits Shape: {logits.shape}")

    for line in out_lines:
        print(line)
        
    with open("debug_output.txt", "w") as f:
        f.write("\n".join(out_lines))
    print("\nSaved shapes to debug_output.txt")

if __name__ == "__main__":
    main()
