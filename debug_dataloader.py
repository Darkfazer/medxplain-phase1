import os
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from torchvision.datasets import FakeData

def get_debug_dataloader(batch_size=16):
    """
    Simulates the MedXPlain DataLoader.
    In actual code, this would import from data/dataset.py
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Mocking a dataset with 14 classes (but FakeData only outputs single ints, so we'll convert to multi-hot)
    dataset = FakeData(size=100, image_size=(3, 224, 224), num_classes=14, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

def analyze_batch():
    print("--- MedXPlain DataLoader Diagnostic ---")
    loader = get_debug_dataloader()
    
    for batch_idx, (images, labels) in enumerate(loader):
        print(f"\nPulled Batch {batch_idx + 1}")
        
        # FakeData labels are single ints, mock multi-hot for analysis
        multi_hot = torch.zeros(images.size(0), 14)
        for i, l in enumerate(labels):
            multi_hot[i, l] = 1.0
            
        print(f"1. Image Batch Shape: {images.shape}")
        print(f"2. Label Batch Shape: {multi_hot.shape}")
        
        # Verify normalization
        means = images.mean(dim=(0, 2, 3))
        stds = images.std(dim=(0, 2, 3))
        print(f"3. Normalization Verification:")
        print(f"   Batch Mean per channel (RGB): {means.numpy()} (Should be ~0)")
        print(f"   Batch Std per channel (RGB): {stds.numpy()} (Should be ~1)")
        
        if torch.abs(means.mean()) > 1.0 or torch.abs(stds.mean() - 1.0) > 1.0:
            print("   ⚠️ WARNING: Normalization seems incorrect! Features should be standardized.")
            
        # Label distribution
        class_sums = multi_hot.sum(dim=0).numpy()
        print(f"4. Label Distribution in Batch:")
        print(f"   {class_sums}")
        if np.max(class_sums) > (images.size(0) * 0.8):
            print("   ⚠️ WARNING: Highly imbalanced batch detected. Consider stratified sampling.")
            
        # Save sample grid
        save_dir = "debug_dataloader_samples"
        os.makedirs(save_dir, exist_ok=True)
        
        from torchvision.utils import save_image
        # Unnormalize for viewing
        inv_normalize = T.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        vis_imgs = inv_normalize(images)
        
        save_path = os.path.join(save_dir, 'batch_grid.png')
        save_image(vis_imgs, save_path, nrow=4)
        print(f"5. Saved visualized image batch grid to: {save_path}")
        
        # Check if labels match
        print("6. Sampled labels for first 3 images in batch:")
        for i in range(3):
            active_classes = np.where(multi_hot[i].numpy() == 1.0)[0]
            print(f"   Image {i}: Active classes = {active_classes}")
            
        break # Only check first batch

if __name__ == "__main__":
    analyze_batch()
