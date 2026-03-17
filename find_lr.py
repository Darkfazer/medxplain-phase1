import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim import Adam
from torchvision.datasets import FakeData
import torchvision.transforms as T

from models.cnn_models.densenet_adapter import DenseNetAdapter

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def find_learning_rate(model, train_loader, optimizer, criterion, init_value=1e-7, final_value=1e-1, num_batches=100):
    model.train()
    
    mult = (final_value / init_value) ** (1/num_batches)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    
    print(f"Finding Optimal Range (from {init_value} to {final_value} over {num_batches} batches)...")
    
    for images, labels in train_loader:
        batch_num += 1
        
        # Mocking multi-hot for fake data
        multi_hot = torch.zeros(images.size(0), 14).to(DEVICE)
        for i, l in enumerate(labels):
            multi_hot[i, l] = 1.0
            
        images = images.to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, multi_hot)
        
        # Compute smoothed loss
        avg_loss = 0.98 * avg_loss + 0.02 * loss.item()
        smoothed_loss = avg_loss / (1 - 0.98**batch_num)
        
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            print("Loss exploded, stopping early.")
            break
            
        if smoothed_loss < best_loss or batch_num == 1:
            best_loss = smoothed_loss
            
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        
        loss.backward()
        optimizer.step()
        
        print(f"[{batch_num}/{num_batches}] lr: {lr:.2e}, loss: {smoothed_loss:.4f}")
        
        lr *= mult
        optimizer.param_groups[0]['lr'] = lr
        
        if batch_num >= num_batches:
            break
            
    # Calculate gradients of the loss curve to find steepest descent
    smoothed_losses = smooth_curve(losses)
    gradients = np.gradient(smoothed_losses)
    min_grad_idx = np.argmin(gradients)
    optimal_lr = 10**log_lrs[min_grad_idx]
    
    print(f"\nOptimal Learning Rate found: {optimal_lr:.2e}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(log_lrs, smoothed_losses)
    plt.axvline(x=log_lrs[min_grad_idx], color='r', linestyle='--', label=f'Optimal LR: {optimal_lr:.2e}')
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder Range Test')
    plt.legend()
    plt.savefig('optimal_lr.png')
    
    with open('lr_value.txt', 'w') as f:
        f.write(f"{optimal_lr:.2e}")
        
    return optimal_lr

if __name__ == "__main__":
    print("--- Learning Rate Finder ---")
    model = DenseNetAdapter(num_classes=14).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=1e-7)
    criterion = nn.BCEWithLogitsLoss()
    
    # Use FakeData as a mock for the LR finder script
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = FakeData(size=2000, image_size=(3, 224, 224), num_classes=14, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    
    find_learning_rate(model, loader, optimizer, criterion)
