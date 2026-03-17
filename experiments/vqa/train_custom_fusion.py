import os
import sys
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from vqa.models.custom_fusion import MedicalCrossAttentionVQA
from vqa.data.custom_vqa_dataset import CustomVQADataset
from vqa.training.custom_vqa_trainer import CustomVQATrainer

def get_optimizer_and_scheduler(model, lr=5e-5, epochs=10, steps_per_epoch=100):
    # Freezing Vision Encoder for phase 2 initial training to match BLIP2 frozen ViT strategies
    for param in model.vision_encoder.parameters():
        param.requires_grad = False
        
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs * steps_per_epoch,
        eta_min=1e-6
    )
    return optimizer, scheduler

def main():
    print("--- Initialize Custom Fusion Fine-Tuning ---")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Configuration
    config = {
        'epochs': 1, # Setting to 1 for dummy tests
        'batch_size': 2,
        'gradient_accumulation_steps': 1,
        'mixed_precision': torch.cuda.is_available(),
        'learning_rate': 1e-4,
        'train_data_path': 'mock.json',
        'val_data_path': 'mock.json',
        'image_dir': 'mock_imgs'
    }
    
    print("Initializing Datasets...")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
    train_dataset = CustomVQADataset(
        data_path=config['train_data_path'], 
        image_dir=config['image_dir'],
        split='train'
    )
    val_dataset = CustomVQADataset(
        data_path=config['val_data_path'], 
        image_dir=config['image_dir'],
        split='val'
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    print("Initializing Model...")
    model = MedicalCrossAttentionVQA(vocab_size=len(tokenizer)).to(device)
    
    optimizer, scheduler = get_optimizer_and_scheduler(
        model, 
        lr=config['learning_rate'], 
        epochs=config['epochs'], 
        steps_per_epoch=len(train_loader)
    )
    
    trainer = CustomVQATrainer(
        model=model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    print(f"Starting Training Loop for {config['epochs']} epochs...")
    trainer.fit(train_loader, val_loader, save_dir="experiments/vqa/checkpoints")
    print("Training Script Finished.")

if __name__ == "__main__":
    main()
