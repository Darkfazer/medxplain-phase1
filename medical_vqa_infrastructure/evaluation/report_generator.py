import pandas as pd

def export_results(results_list: list, filepath="report.csv"):
    df = pd.DataFrame(results_list)
    df.to_csv(filepath, index=False)
