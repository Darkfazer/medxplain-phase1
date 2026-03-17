import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import os

class VQARADDataset(Dataset):
    def __init__(self, json_file, img_dir, transform=None, split='train', train_ratio=0.8):
        # Load JSON data
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.img_dir = img_dir
        
        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform
        
        # Split data
        split_idx = int(len(self.data) * train_ratio)
        if split == 'train':
            self.data = self.data[:split_idx]
        else:
            self.data = self.data[split_idx:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get JSON entry
        item = self.data[idx]
        
        # Load image
        img_path = os.path.join(self.img_dir, item['image_name'])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get question and answer
        question = item['question']
        answer = item['answer']
        
        # Optional: get question type for analysis
        q_type = item.get('Question_type', '')
        
        return {
            'image': image,
            'question': question,
            'answer': answer,
            'question_type': q_type,
            'image_name': item['image_name']
        }

# Quick test
if __name__ == "__main__":
    # Test the dataset
    dataset = VQARADDataset(
        json_file='data/VQA_RAD Dataset.json',
        img_dir='data/VQA_RAD Images'
    )
    
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Question: {sample['question']}")
    print(f"Answer: {sample['answer']}")