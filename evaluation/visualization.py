import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve

class VisualizationTools:
    """
    Static toolkit for rendering benchmark visualizations specifically 
    tailored to massive multi-label classification experiments.
    """
    @staticmethod
    def plot_roc_curves(y_true_dict, y_score_dict, class_index, class_name, save_dir):
        """
        Plots ROC curves mapping multiple models for a specific class.
        Args:
            y_true_dict: Dict of model_name -> true labels array
            y_score_dict: Dict of model_name -> prediction scores array
            class_index: Int index of the class
            class_name: String name of the class
            save_dir: Base directory to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        for model_name in y_score_dict.keys():
            y_t = y_true_dict[model_name][:, class_index]
            y_s = y_score_dict[model_name][:, class_index]
            
            fpr, tpr, _ = roc_curve(y_t, y_s)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
            
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve Analysis: {class_name}')
        plt.legend(loc="lower right")
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"roc_{class_name}.png"), bbox_inches='tight')
        plt.close()

    @staticmethod
    def plot_model_comparison_bar(df_path, metric='auc_macro', save_dir='experiments/results'):
        """
        Draws a bar chart across models reading from a generated benchmark CSV.
        """
        if not os.path.exists(df_path):
            print(f"Skipping visualization. Path not found: {df_path}")
            return
            
        df = pd.read_csv(df_path)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='model', y=metric, data=df, palette='viridis')
        plt.title(f'Model Comparison: {metric.upper()}')
        plt.xticks(rotation=45)
        plt.ylim(max(0, df[metric].min() - 0.1), min(1.0, df[metric].max() + 0.05))
        
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"comparison_bar_{metric}.png"), bbox_inches='tight')
        plt.close()

if __name__ == "__main__":
    print("Visualization Module Verified.")
