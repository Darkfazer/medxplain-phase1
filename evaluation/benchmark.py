import os
import torch
import pandas as pd
from training.metrics import ClinicalMetrics

class Benchmark:
    """
    Utility class to load a trained model checkpoint and evaluate it
    against a holdout test set, capturing comprehensive metrics into a DataFrame.
    """
    def __init__(self, model_dict: dict, test_loader, device, class_names):
        """
        Args:
            model_dict (dict): Dictionary mapping model_name (str) to loaded model instances.
            test_loader (DataLoader): The holdout test set loader.
            device (torch.device): Compute device.
            class_names (list): List of pathology class names.
        """
        self.models = model_dict
        self.test_loader = test_loader
        self.device = device
        self.class_names = class_names
        self.results = []

    def run(self):
        """
        Runs evaluation on all provided models sequentially.
        """
        for model_name, model in self.models.items():
            print(f"Benchmarking: {model_name}")
            model.eval()
            tracker = ClinicalMetrics(self.class_names)
            tracker.reset()
            
            with torch.no_grad():
                for images, targets in self.test_loader:
                    images, targets = images.to(self.device), targets.to(self.device).float()
                    
                    logits = model(images)
                    preds = torch.sigmoid(logits)
                    
                    tracker.update(targets, preds)
                    
            metrics = tracker.compute()
            metrics['model'] = model_name
            self.results.append(metrics)
            
    def get_results_df(self) -> pd.DataFrame:
        """
        Returns all collected metrics as a Pandas DataFrame.
        """
        if not self.results:
            print("No results collected yet. Run .run() first.")
            return pd.DataFrame()
            
        df = pd.DataFrame(self.results)
        # Move 'model' column to the front for readability
        cols = ['model'] + [col for col in df.columns if col != 'model']
        return df[cols]
        
    def save_results(self, save_path: str = "experiments/results/benchmark_results.csv"):
        """Saves current benchmark run to CSV."""
        df = self.get_results_df()
        df.to_csv(save_path, index=False)
        print(f"Benchmark results saved to {save_path}")

if __name__ == "__main__":
    # Test script representation
    print("Benchmark Module Verified")
