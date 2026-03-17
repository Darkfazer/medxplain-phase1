import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class MedicalCrossAttentionVQA(nn.Module):
    """
    Custom Multimodal Fusion architecture for Medical VQA.
    Uses uncompressed spatial representations from DenseNet121 and multi-head 
    cross-attention within a Transformer Decoder to dynamically attend to localized
    image regions during language generation.
    """
    def __init__(
        self, 
        vocab_size: int = 50272, # OPT standard vocab size
        embed_dim: int = 768, 
        num_heads: int = 8, 
        decoder_layers: int = 6,
        vision_hidden_dim: int = 1024 
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        # --- VISION ENCODER ---
        print("Initializing Custom Fusion Model...")
        print("Loading DenseNet121 parameters for spatial feature extraction...")
        model = densenet121(weights=DenseNet121_Weights.DEFAULT)
        # Features output shape: [Batch, 1024, H/32, W/32]
        self.vision_encoder = model.features
        self.vision_proj = nn.Linear(vision_hidden_dim, embed_dim)
        
        # --- TEXT & FUSION ---
        print("Setting up Text Embedding and Cross-Attention Decoder Layers...")
        self.text_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_encoding = nn.Embedding(2048, embed_dim)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            batch_first=True,
            norm_first=True 
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=decoder_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def extract_vision_features(self, pixel_values):
        # Forward pass through CNN
        v_features = self.vision_encoder(pixel_values) # [B, 1024, H', W']
        
        b, c, h, w = v_features.shape
        # Flatten spatial dims: [B, 1024, H'*W'] -> [B, H'*W', 1024]
        v_features = v_features.view(b, c, -1).permute(0, 2, 1) 
        
        # Project to transformer hidden dimension
        projected_patches = self.vision_proj(v_features)
        return projected_patches

    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """Teacher forcing forward pass."""
        print(f"DEBUG: input_ids shape: {input_ids.shape}")
        if attention_mask is not None:
            print(f"DEBUG: attention_mask shape: {attention_mask.shape}")
            
        try:
            from transformers import AutoTokenizer
            debug_tok = AutoTokenizer.from_pretrained("facebook/opt-125m")
            decoded_text = debug_tok.decode(input_ids[0, :10], skip_special_tokens=False)
            print(f"DEBUG: First 10 tokens decoded: {decoded_text}")
        except Exception:
            pass
            
        memory = self.extract_vision_features(pixel_values) 
        
        batch_size, seq_length = input_ids.shape
        positions = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, seq_length)
        
        tgt = self.text_embedding(input_ids) + self.positional_encoding(positions) 
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=input_ids.device)
        tgt_key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        
        output = self.decoder(
            tgt=tgt, 
            memory=memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        logits = self.lm_head(output)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # Flatten to [Batch*SeqLen-1, Vocab] and [Batch*SeqLen-1]
            loss = loss_fct(shift_logits.view(-1, self.lm_head.out_features), shift_labels.view(-1))
            
        class CustomOutput:
            pass
        ret = CustomOutput()
        ret.logits = logits
        ret.loss = loss
        return ret
        
    def generate(self, pixel_values, start_token_id, end_token_id=None, max_length=50, device="cpu",
                 temperature=1.0, do_sample=False, top_p=1.0, repetition_penalty=1.0, max_new_tokens=None):
        """Autoregressive generation for inference."""
        self.eval()
        memory = self.extract_vision_features(pixel_values)
        batch_size = pixel_values.shape[0]
        
        if max_new_tokens is not None:
            max_length = max_new_tokens
            
        generated_ids = torch.full((batch_size, 1), start_token_id, dtype=torch.long, device=device)
        
        with torch.no_grad():
            for _ in range(max_length):
                seq_length = generated_ids.shape[1]
                positions = torch.arange(0, seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, seq_length)
                
                tgt = self.text_embedding(generated_ids) + self.positional_encoding(positions)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_length, device=device)
                
                output = self.decoder(tgt=tgt, memory=memory, tgt_mask=tgt_mask)
                next_token_logits = self.lm_head(output[:, -1, :])
                
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token in generated_ids[i]:
                            if next_token_logits[i, token] < 0:
                                next_token_logits[i, token] *= repetition_penalty
                            else:
                                next_token_logits[i, token] /= repetition_penalty

                if do_sample:
                    next_token_logits = next_token_logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        for i in range(batch_size):
                            indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                            next_token_logits[i, indices_to_remove] = -float('Inf')
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)
                else:
                    next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(1)
                
                generated_ids = torch.cat([generated_ids, next_token_id], dim=1)
                
                if end_token_id is not None and (next_token_id == end_token_id).all():
                    break
                    
        return generated_ids

if __name__ == "__main__":
    # Dummy test to verify the forward and generate pass
    print("Testing MedicalCrossAttentionVQA with mock tensors...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MedicalCrossAttentionVQA().to(device)
    
    mock_images = torch.randn(2, 3, 224, 224).to(device)
    mock_text = torch.randint(0, 50272, (2, 15)).to(device)
    mock_labels = torch.randint(0, 50272, (2, 15)).to(device)
    
    print("Running forward pass (training mode)...")
    outputs = model(mock_images, mock_text, labels=mock_labels)
    print(f"Logits shape: {outputs.logits.shape} (Expected: 2, 15, 50272)")
    print(f"Loss returned: {outputs.loss.item()}")
    
    print("Running generation (inference mode)...")
    gen_ids = model.generate(mock_images, start_token_id=2, end_token_id=3, max_length=10, device=device)
    print(f"Generated IDs shape: {gen_ids.shape} (Expected: 2, <=11)")
    print("Test passed successfully.")
