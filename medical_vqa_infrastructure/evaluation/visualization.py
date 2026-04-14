import matplotlib.pyplot as plt

def plot_roc_curve(fpr, tpr, auc_score, save_path="roc.png"):
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()
