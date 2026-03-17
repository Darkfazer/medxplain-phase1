import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

class VQAMetrics:
    """
    Computes VQA specific metrics comparing generated text to ground truth.
    Provides Exact Match (useful for closed-ended questions like 'Yes' or 'No')
    and BLEU scores (useful for open-ended questions like 'What is the finding?').
    """
    def __init__(self):
        # We try to ensure punkt is available for standard word tokenization
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
            
        self.exact_matches = {"CLOSED": [], "OPEN": [], "ALL": []}
        self.bleu_scores = {"CLOSED": [], "OPEN": [], "ALL": []}
        
    def _normalize_text(self, text: str) -> str:
        """
        Lowercases, strips trailing/leading whitespaces, and removes heavy punctuation
        to provide a fairer evaluation of generative output.
        """
        text = text.lower().strip()
        # Remove trailing periods common in generated text
        if text.endswith('.'):
            text = text[:-1]
        text = re.sub(r'[^\w\s]', '', text)
        return text

    def update(self, generated_text: str, ground_truth: str, q_type: str = "OPEN"):
        """
        Args:
            generated_text: The decoded predicted string from the model.
            ground_truth: The true string from the dataset.
            q_type: "CLOSED" (yes/no based) or "OPEN" (descriptive based)
        """
        norm_gen = self._normalize_text(generated_text)
        norm_gt = self._normalize_text(ground_truth)
        
        # 1. Exact Match Accuracy Calculation
        is_exact = 1 if norm_gen == norm_gt else 0
        self.exact_matches["ALL"].append(is_exact)
        
        if q_type.upper() in ["CLOSED", "OPEN"]:
             self.exact_matches[q_type.upper()].append(is_exact)

        # 2. BLEU Score Calculation (Standard for open-ended VQA tasks)
        # We use a smoothing function because some medical answers are just 1-2 words
        # which breaks standard un-smoothed BLEU-4 calculations.
        smoothie = SmoothingFunction().method4
        
        reference = [nltk.word_tokenize(norm_gt)]
        hypothesis = nltk.word_tokenize(norm_gen)
        
        bleu = sentence_bleu(reference, hypothesis, smoothing_function=smoothie)
        
        self.bleu_scores["ALL"].append(bleu)
        if q_type.upper() in ["CLOSED", "OPEN"]:
            self.bleu_scores[q_type.upper()].append(bleu)

    def compute(self) -> dict:
        def safe_mean(lst):
            return sum(lst) / len(lst) if len(lst) > 0 else 0.0
            
        return {
            "Accuracy_All": safe_mean(self.exact_matches["ALL"]),
            "Accuracy_Closed": safe_mean(self.exact_matches["CLOSED"]),
            "Accuracy_Open": safe_mean(self.exact_matches["OPEN"]),
            "BLEU_All": safe_mean(self.bleu_scores["ALL"]),
            "BLEU_Closed": safe_mean(self.bleu_scores["CLOSED"]),
            "BLEU_Open": safe_mean(self.bleu_scores["OPEN"])
        }
        
    def reset(self):
        """Clears buffers for the next epoch."""
        self.exact_matches = {"CLOSED": [], "OPEN": [], "ALL": []}
        self.bleu_scores = {"CLOSED": [], "OPEN": [], "ALL": []}

if __name__ == "__main__":
    # Test
    metrics = VQAMetrics()
    metrics.update("Cardiomegaly", "cardiomegaly.", "OPEN")
    metrics.update("Yes, it is", "Yes", "CLOSED") # Won't exact match, but will have some BLEU
    res = metrics.compute()
    print("VQA Evaluation Metrics Test Run:")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")
