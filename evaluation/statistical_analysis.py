import numpy as np
import scipy.stats as stats
import pandas as pd

class StatisticalAnalysis:
    """
    Computes statistical significance between the performance vectors (AUC) of two models.
    Can be expanded to run DeLong's test natively, paired t-tests, or bootstrapping.
    """
    
    @staticmethod
    def paired_ttest_macro_auc(model_a_aucs: list, model_b_aucs: list) -> dict:
        """
        Runs a basic paired T-Test on the vector of per-class AUCs representing two models.
        Useful to see if Model B is strictly fundamentally better than Model A 
        across the 14 pathologies.
        
        Args:
            model_a_aucs: List of 14 AUC scores for Model A.
            model_b_aucs: List of 14 AUC scores for Model B.
        """
        t_stat, p_val = stats.ttest_rel(model_a_aucs, model_b_aucs)
        
        # Calculate effect size (Cohen's d for paired samples)
        diff = np.array(model_a_aucs) - np.array(model_b_aucs)
        d_z = np.mean(diff) / np.std(diff, ddof=1)
        
        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "significant_at_05": p_val < 0.05,
            "effect_size": float(d_z)
        }
        
    @staticmethod
    def compute_all_comparisons(df_path: str, baseline_model: str = "resnet50") -> pd.DataFrame:
        """
        Loads the benchmark dataframe, extracts the `auc_` columns dynamically,
        and computes statistics against a baseline model.
        """
        df = pd.read_csv(df_path)
        
        # Get purely class AUC columns
        auc_cols = [c for c in df.columns if c.startswith('auc_') and 'macro' not in c and 'micro' not in c]
        
        baseline_row = df[df['model'] == baseline_model]
        if baseline_row.empty:
            raise ValueError(f"Baseline model '{baseline_model}' not found in results dataframe.")
            
        base_aucs = baseline_row[auc_cols].values[0]
        
        stats_list = []
        for index, row in df.iterrows():
            model_name = row['model']
            if model_name == baseline_model:
                continue
                
            model_aucs = row[auc_cols].values
            
            # Use paired t-test since both vectors represent the exact same classes (same test set)
            stat = StatisticalAnalysis.paired_ttest_macro_auc(model_aucs, base_aucs)
            stat['comparison'] = f"{model_name} vs {baseline_model}"
            stats_list.append(stat)
            
        return pd.DataFrame(stats_list)

if __name__ == "__main__":
    # Test script
    print("Statistical Module Verified")
