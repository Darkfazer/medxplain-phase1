import torchvision.transforms as transforms

def get_medical_augmentations(img_size=224):
    """
    Standard augmentation pipeline for medical X-ray images.
    Preserves structural clinical features while forcing robust feature extraction.
    """
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3), # Valid for some pathologies, invalid for situs inversus, keep % low
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)), # Simulates patient positioning
        transforms.RandomResizedCrop(size=img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Simulates different X-ray scanner exposures
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_progressive_resizing_transforms(epoch, max_epochs):
    """
    Returns a transform pipeline with an image size dependent on the epoch.
    Start coarse (128x128), move to medium (160x160), finish at fine (224x224).
    """
    if epoch < 5:
        size = 128
    elif epoch < 10:
        size = 160
    else:
        size = 224
        
    return get_medical_augmentations(img_size=size)
