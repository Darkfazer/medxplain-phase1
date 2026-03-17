import os
import sys
import argparse
import yaml
import torch

# Add project root to sys.path so 'models' and 'training' can be resolved
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_models.resnet_adapter import ResNetAdapter
from models.cnn_models.densenet_adapter import DenseNetAdapter
from models.cnn_models.efficientnet_adapter import EfficientNetAdapter
from models.transformer_models.vit_adapter import ViTAdapter
from models.transformer_models.swin_adapter import SwinAdapter
from models.transformer_models.deit_adapter import DeiTAdapter
from models.medical_specific.torchxray_adapter import TorchXRayAdapter
from models.medical_specific.medvit_adapter import MedViTAdapter
from training.trainer import BaseTrainer
from training.losses import FocalLoss

class WeightedEnsemble(torch.nn.Module):
    def __init__(self, models_dict, weights):
        super().__init__()
        self.models_dict = models_dict
        self.weights = weights
        
    def forward(self, x):
        ensemble_pred = 0
        total_weight = 0
        for name, weight in self.weights.items():
            if name in self.models_dict:
                logits = self.models_dict[name](x)
                preds = torch.sigmoid(logits)
                ensemble_pred += weight * preds
                total_weight += weight
        
        if total_weight > 0:
            ensemble_pred = ensemble_pred / total_weight
            
        # Inverse sigmoid so Benchmark's torch.sigmoid(logits) handles it properly
        ensemble_pred = torch.clamp(ensemble_pred, 1e-7, 1 - 1e-7)
        return torch.log(ensemble_pred / (1 - ensemble_pred))

# Note: In a real scenario, datasets would be imported from a data/ loader module.
# For structure demonstration, we mock the Datasets/Loaders here.

def parse_args():
    parser = argparse.ArgumentParser(description="Run Comparative Medical Classification")
    parser.add_argument('--models', nargs='+', default=['resnet50'], 
                        help="List of models to run (e.g., resnet50 densenet121 vit_base)")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dataset', type=str, default='nih')
    return parser.parse_args()

def load_config(path):
    with open(path, 'r') as f: return yaml.safe_load(f)

def get_model(model_name: str, num_classes: int):
    # Registry mapping
    registry = {
        'resnet50': ResNetAdapter,
        'densenet121': DenseNetAdapter,
        'efficientnet_b4': EfficientNetAdapter,
        'vit_base': lambda num_classes: ViTAdapter(num_classes=num_classes, model_name="vit_base_patch16_224"),
        'swin_base': lambda num_classes: SwinAdapter(num_classes=num_classes, model_name="swin_base_patch4_window7_224"),
        'deit_base': lambda num_classes: DeiTAdapter(num_classes=num_classes, model_name="deit_base_patch16_224"),
        'torchxray_densenet': TorchXRayAdapter,
        'medvit_base': MedViTAdapter
    }
    if model_name not in registry:
        raise ValueError(f"Model {model_name} not implemented yet in the factory.")
    
    return registry[model_name](num_classes=num_classes)

def main():
    args = parse_args()
    
    # Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load Configurations
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_cfg = load_config(os.path.join(base_dir, 'configs', 'dataset_configs.yaml'))
    model_cfg = load_config(os.path.join(base_dir, 'configs', 'model_configs.yaml'))
    
    ds_conf = dataset_cfg['datasets'][args.dataset]
    num_classes = ds_conf['num_classes']
    class_names = ds_conf['classes']
    
    print(f"Dataset: {args.dataset.upper()} | Classes: {num_classes}")
    
    # Mock DataLoaders with clear visual signals (Injecting patterns so models can actually learn)
    from torch.utils.data import DataLoader, TensorDataset, Dataset
    from torchvision import transforms as T
    
    print("Generating Mock Datasets with learnable class-specific signals...")
    # Base background noise
    x_train = torch.randn(100, 3, 224, 224) * 0.1
    y_train = torch.randint(0, 2, (100, num_classes)).float()
    
    # Inject a distinct bright rectangular signal for each active class
    for i in range(100):
        for c in range(num_classes):
            if y_train[i, c] == 1.0:
                row = c // 4
                col = c % 4
                # Add intense signal at specific coordinate
                x_train[i, :, row*50:row*50+40, col*50:col*50+40] = 3.0
                
    class AugmentedDataset(Dataset):
        def __init__(self, x, y, transform=None):
            self.x = x
            self.y = y
            self.transform = transform
        def __len__(self):
            return len(self.x)
        def __getitem__(self, idx):
            img = self.x[idx]
            if self.transform:
                img = self.transform(img)
            return img, self.y[idx]
            
    # Apply maximum data augmentations as requested to simulate generalization improvements
    train_aug = T.Compose([
        T.ToPILImage(),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor()
    ])
    
    train_dl = DataLoader(AugmentedDataset(x_train, y_train, transform=train_aug), batch_size=args.batch_size, shuffle=True)
    
    x_val = torch.randn(20, 3, 224, 224) * 0.1
    y_val = torch.randint(0, 2, (20, num_classes)).float()
    for i in range(20):
        for c in range(num_classes):
            if y_val[i, c] == 1.0:
                row = c // 4
                col = c % 4
                x_val[i, :, row*50:row*50+40, col*50:col*50+40] = 3.0
                
    val_dl = DataLoader(TensorDataset(x_val, y_val), batch_size=args.batch_size)

    # Dictionary strictly to hold initialized models to pass to benchmark later
    trained_models = {}

    for m_name in args.models:
        print(f"\n{'='*50}\nTraining {m_name.upper()}\n{'='*50}")
        model = get_model(m_name, num_classes).to(device)
        
        # Merge dict config
        cfg_dict = model_cfg['training_defaults'].copy()
        if m_name in model_cfg['models']:
            cfg_dict.update(model_cfg['models'][m_name])
            
        cfg_dict['epochs'] = args.epochs # Override with CLI
        cfg_dict['mixed_precision'] = torch.cuda.is_available()
        
        # Use FocalLoss for Medical Imbalance instead of BCE
        criterion = FocalLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg_dict.get('learning_rate', 1e-5))
        
        trainer = BaseTrainer(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            class_names=class_names,
            config=cfg_dict
        )
        
        save_dir = os.path.join(base_dir, "experiments", "results", m_name)
        trainer.fit(train_dl, val_dl, save_dir=save_dir)
        trained_models[m_name] = model

    # --- Ensemble Logic ---
    ensemble_weights = {'deit_base': 0.4, 'medvit_base': 0.3, 'resnet50': 0.3}
    missing_for_ensemble = [m for m in ensemble_weights.keys() if m not in trained_models]
    if not missing_for_ensemble:
        print("\nCreating Top-3 Weighted Ensemble Model...")
        ensemble_model = WeightedEnsemble(trained_models, ensemble_weights)
        trained_models['Top3_Ensemble'] = ensemble_model
    else:
        print(f"\nSkipping Ensemble creation (Requires {ensemble_weights.keys()})")

    print("\nTraining Phase Complete. Starting Benchmarking...")
    from evaluation.benchmark import Benchmark
    from evaluation.visualization import VisualizationTools
    
    benchmark = Benchmark(model_dict=trained_models, test_loader=val_dl, device=device, class_names=class_names)
    benchmark.run()
    
    csv_path = os.path.join(base_dir, "experiments", "results", f"benchmark_results_{args.dataset}.csv")
    benchmark.save_results(csv_path)
    
    try:
        VisualizationTools.plot_model_comparison_bar(csv_path, save_dir=os.path.join(base_dir, "experiments", "results"))
        print(f"Visualizations saved to {os.path.join(base_dir, 'experiments', 'results')}")
    except Exception as e:
        print("Visualization failed:", e)

if __name__ == "__main__":
    main()
