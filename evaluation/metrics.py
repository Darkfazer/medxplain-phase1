import re
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Ensure nltk wordnet is available (though not strictly needed for basic BLEU, handy if expanding)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """Lowercase and remove punctuation."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())

def compute_accuracy(preds, labels):
    """Strict/Robust accuracy for closed-ended questions."""
    correct = 0
    for p, l in zip(preds, labels):
        p_clean = clean_text(p)
        l_clean = clean_text(l)
        # Check if the desired label is exactly predicted or simply contained
        if p_clean == l_clean or (l_clean in p_clean and len(l_clean) > 0):
            correct += 1
    return (correct / len(preds)) * 100 if preds else 0.0

def compute_bleu(preds, labels):
    """BLEU score for open-ended questions."""
    smoothie = SmoothingFunction().method4
    scores = []
    for p, l in zip(preds, labels):
        ref = [clean_text(l).split()]
        cand = clean_text(p).split()
        score = sentence_bleu(ref, cand, smoothing_function=smoothie)
        scores.append(score)
    return (sum(scores) / len(scores)) * 100 if scores else 0.0
        
def evaluate_batch(preds, labels, q_types):
    """
    Args:
        preds (list of str): Generated predictions.
        labels (list of str): Ground truth answers.
        q_types (list of str): Question types ("OPEN" or "CLOSED").
    Returns:
        acc (float), bleu (float): Specific scores for this batch.
    """
    closed_preds, closed_labels = [], []
    open_preds, open_labels = [], []
    
    for p, l, qtype in zip(preds, labels, q_types):
        if qtype == "CLOSED":
            closed_preds.append(p)
            closed_labels.append(l)
        else:
            open_preds.append(p)
            open_labels.append(l)
            
    acc = compute_accuracy(closed_preds, closed_labels) if closed_preds else 0.0
    bleu = compute_bleu(open_preds, open_labels) if open_preds else 0.0
    
    return {
        "closed_acc": acc,
        "open_bleu": bleu,
        "num_closed": len(closed_preds),
        "num_open": len(open_preds)
    }

def aggregate_metrics(metrics_list):
    """Aggregate a list of batch metric dictionaries."""
    total_acc = 0.0
    total_bleu = 0.0
    total_closed = 0
    total_open = 0
    
    for m in metrics_list:
        n_c = m["num_closed"]
        n_o = m["num_open"]
        total_acc += m["closed_acc"] * n_c
        total_bleu += m["open_bleu"] * n_o
        total_closed += n_c
        total_open += n_o
        
    final_acc = total_acc / total_closed if total_closed > 0 else 0.0
    final_bleu = total_bleu / total_open if total_open > 0 else 0.0
    
    return {
        "Accuracy (Closed)": final_acc,
        "BLEU (Open)": final_bleu
    }
