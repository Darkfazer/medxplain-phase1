"""
evaluate_metrics_spec.py
========================
Cahier des Charges–compliant evaluation suite for MedXplain.

Computes
--------
1. VQA Accuracy       – exact-match (case-normalised)
2. BLEU-4             – corpus-level BLEU (nltk)
3. BERTScore F1       – semantic similarity (bert-score library)
4. Clinical F1 (Macro)– per-pathology presence F1 using report_parser patterns

Prints a formatted report and returns a metrics dictionary.

Targets (from spec)
------------------
  VQA Accuracy    > 65%
  BLEU-4          > 0.25
  BERTScore F1    > 0.75
  Clinical F1     > 0.60

Usage
-----
    from evaluation.evaluate_metrics_spec import run_full_evaluation, print_report

    metrics = run_full_evaluation(
        predictions=["Yes", "There is a right pleural effusion."],
        references= ["Yes", "Small right pleural effusion present."],
    )
    print_report(metrics)

    # or from CLI:
    python evaluation/evaluate_metrics_spec.py \\
        --predictions_csv results/predictions.csv \\
        --references_csv  data/radiologist_annotations.csv

CSV format (both files):
    image_id, answer

Dependencies
------------
pip install nltk
pip install bert-score     # optional – BERTScore
pip install pandas         # optional – CSV loading

nltk data (first run only):
    python -c "import nltk; nltk.download('punkt')"
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------
try:
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    import nltk
    _NLTK_AVAILABLE = True
except ImportError:
    _NLTK_AVAILABLE = False
    logger.warning("nltk not installed. BLEU will be 0. pip install nltk")

try:
    from bert_score import score as bert_score_fn  # type: ignore
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False
    logger.warning("bert-score not installed. BERTScore will be 0. pip install bert-score")

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

from data_ingestion.report_parser import ReportParser, PATHOLOGY_CATALOGUE

# ---------------------------------------------------------------------------
# Targets from the Cahier des Charges
# ---------------------------------------------------------------------------
TARGETS: Dict[str, float] = {
    "vqa_accuracy":     0.65,
    "bleu4":            0.25,
    "bertscore_f1":     0.75,
    "clinical_f1_macro": 0.60,
}


# ---------------------------------------------------------------------------
# Data model for results
# ---------------------------------------------------------------------------

@dataclass
class EvaluationMetrics:
    vqa_accuracy:       float = 0.0
    bleu4:              float = 0.0
    bertscore_precision: float = 0.0
    bertscore_recall:   float = 0.0
    bertscore_f1:       float = 0.0
    clinical_f1_macro:  float = 0.0
    clinical_f1_per_pathology: Dict[str, float] = field(default_factory=dict)
    n_samples:          int   = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vqa_accuracy":            round(self.vqa_accuracy,        4),
            "bleu4":                   round(self.bleu4,               4),
            "bertscore_precision":     round(self.bertscore_precision,  4),
            "bertscore_recall":        round(self.bertscore_recall,     4),
            "bertscore_f1":            round(self.bertscore_f1,        4),
            "clinical_f1_macro":       round(self.clinical_f1_macro,   4),
            "clinical_f1_per_pathology": {
                k: round(v, 4) for k, v in self.clinical_f1_per_pathology.items()
            },
            "n_samples":               self.n_samples,
        }


# ---------------------------------------------------------------------------
# Per-metric implementations
# ---------------------------------------------------------------------------

def _normalise(text: str) -> str:
    """Lowercase, strip, collapse whitespace."""
    return " ".join(text.lower().strip().split())


# ── VQA Accuracy ─────────────────────────────────────────────────────────────

def compute_vqa_accuracy(
    predictions: List[str],
    references:  List[str],
) -> float:
    """
    Exact-match accuracy after case/whitespace normalisation.

    Parameters
    ----------
    predictions : Model-generated answer strings.
    references  : Ground-truth answer strings.
    """
    if not predictions:
        return 0.0
    correct = sum(
        1 for p, r in zip(predictions, references)
        if _normalise(p) == _normalise(r)
    )
    return correct / len(predictions)


# ── BLEU-4 ───────────────────────────────────────────────────────────────────

def compute_bleu4(
    predictions: List[str],
    references:  List[str],
) -> float:
    """
    Corpus-level BLEU-4 score.

    Uses NLTK's ``corpus_bleu`` with smoothing to handle short answers
    that would otherwise receive a BLEU of 0 due to 0-count n-grams.

    Parameters
    ----------
    predictions : List of prediction strings.
    references  : List of reference strings (same order as predictions).
    """
    if not _NLTK_AVAILABLE:
        return 0.0

    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception:
            pass

    try:
        from nltk.tokenize import word_tokenize
    except Exception:
        word_tokenize = str.split  # type: ignore

    # corpus_bleu expects list-of-list for references
    ref_tokens  = [[word_tokenize(_normalise(r))] for r in references]
    pred_tokens = [word_tokenize(_normalise(p))   for p in predictions]

    smoothing = SmoothingFunction().method1
    try:
        return float(corpus_bleu(ref_tokens, pred_tokens, smoothing_function=smoothing))
    except Exception as exc:
        logger.warning("BLEU-4 computation failed: %s", exc)
        return 0.0


# ── BERTScore ────────────────────────────────────────────────────────────────

def compute_bertscore(
    predictions: List[str],
    references:  List[str],
    lang:        str = "en",
    model_type:  Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    Compute BERTScore Precision / Recall / F1.

    Parameters
    ----------
    predictions : Model answer strings.
    references  : Ground-truth strings.
    lang        : Language code passed to bert-score.
    model_type  : Optional explicit model (e.g. "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract").
                  If None, bert-score selects a default for the given language.

    Returns
    -------
    (precision, recall, f1)  – scalar floats (macro-average).
    """
    if not _BERTSCORE_AVAILABLE:
        return 0.0, 0.0, 0.0
    try:
        kwargs: Dict[str, Any] = {"lang": lang, "verbose": False}
        if model_type:
            kwargs["model_type"] = model_type
        P, R, F = bert_score_fn(predictions, references, **kwargs)
        return float(P.mean()), float(R.mean()), float(F.mean())
    except Exception as exc:
        logger.warning("BERTScore computation failed: %s", exc)
        return 0.0, 0.0, 0.0


# ── Clinical F1 ──────────────────────────────────────────────────────────────

def _extract_pathology_flags(
    texts: List[str],
    parser: ReportParser,
) -> Dict[str, List[int]]:
    """
    For each text in ``texts`` and each catalogued pathology, return
    a binary flag (1 = present, 0 = absent/unknown).

    Returns
    -------
    Dict mapping pathology_key → List[int] of length len(texts).
    """
    pathology_keys = [key for key, _, _ in PATHOLOGY_CATALOGUE]
    result: Dict[str, List[int]] = {k: [] for k in pathology_keys}

    for text in texts:
        parsed = parser.parse(text)
        findings = parsed.get("findings", {})
        for key in pathology_keys:
            flag = int(findings.get(key, {}).get("present", False))
            result[key].append(flag)

    return result


def _f1_binary(
    pred_flags: List[int],
    ref_flags:  List[int],
) -> float:
    """Compute binary F1 from two lists of 0/1 flags."""
    tp = sum(p == 1 and r == 1 for p, r in zip(pred_flags, ref_flags))
    fp = sum(p == 1 and r == 0 for p, r in zip(pred_flags, ref_flags))
    fn = sum(p == 0 and r == 1 for p, r in zip(pred_flags, ref_flags))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_clinical_f1(
    predictions: List[str],
    references:  List[str],
) -> Tuple[float, Dict[str, float]]:
    """
    Compute per-pathology binary F1 and macro-average Clinical F1.

    Both ``predictions`` and ``references`` are free-text strings;
    the ``ReportParser`` is used to detect pathology presence in each.

    Parameters
    ----------
    predictions : Model answer strings.
    references  : Ground-truth answer/report strings.

    Returns
    -------
    (macro_f1, per_pathology_dict)
    """
    parser = ReportParser()

    pred_flags = _extract_pathology_flags(predictions, parser)
    ref_flags  = _extract_pathology_flags(references,  parser)

    per_path_f1: Dict[str, float] = {}
    # Use display names for reporting
    display_map = {key: display for key, display, _ in PATHOLOGY_CATALOGUE}

    for key in pred_flags:
        f1 = _f1_binary(pred_flags[key], ref_flags[key])
        display_name = display_map.get(key, key)
        per_path_f1[display_name] = round(f1, 4)

    # Macro average over pathologies that appear at least once in refs
    active_scores: List[float] = []
    for key in pred_flags:
        if any(v == 1 for v in ref_flags[key]):
            active_scores.append(_f1_binary(pred_flags[key], ref_flags[key]))

    macro_f1 = float(sum(active_scores) / len(active_scores)) if active_scores else 0.0
    return round(macro_f1, 4), per_path_f1


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    predictions: List[str],
    references:  List[str],
    bertscore_model: Optional[str] = None,
) -> EvaluationMetrics:
    """
    Run all four evaluation metrics and return a populated
    ``EvaluationMetrics`` object.

    Parameters
    ----------
    predictions     : List of model answer strings (same length as references).
    references      : List of ground-truth answer strings.
    bertscore_model : Optional HuggingFace model name for BERTScore.
                      If None, uses the default English model.
    """
    assert len(predictions) == len(references), (
        f"Length mismatch: {len(predictions)} predictions vs {len(references)} references"
    )

    logger.info("Running evaluation on %d samples …", len(predictions))

    metrics = EvaluationMetrics(n_samples=len(predictions))

    metrics.vqa_accuracy = compute_vqa_accuracy(predictions, references)
    logger.info("VQA Accuracy:   %.4f", metrics.vqa_accuracy)

    metrics.bleu4 = compute_bleu4(predictions, references)
    logger.info("BLEU-4:         %.4f", metrics.bleu4)

    bp, br, bf = compute_bertscore(predictions, references, model_type=bertscore_model)
    metrics.bertscore_precision = bp
    metrics.bertscore_recall    = br
    metrics.bertscore_f1        = bf
    logger.info("BERTScore F1:   %.4f", metrics.bertscore_f1)

    cf1_macro, cf1_per = compute_clinical_f1(predictions, references)
    metrics.clinical_f1_macro          = cf1_macro
    metrics.clinical_f1_per_pathology  = cf1_per
    logger.info("Clinical F1:    %.4f", metrics.clinical_f1_macro)

    return metrics


# ---------------------------------------------------------------------------
# Formatted output (matches spec exactly)
# ---------------------------------------------------------------------------

def _pass_fail(value: float, target: float) -> str:
    return "✓" if value >= target else "✗"


def print_report(metrics: EvaluationMetrics) -> None:
    """Print a formatted Cahier des Charges–style evaluation report."""
    acc      = metrics.vqa_accuracy
    bleu     = metrics.bleu4
    bscore   = metrics.bertscore_f1
    cf1      = metrics.clinical_f1_macro

    lines = [
        "=" * 40,
        "MEDXPLAIN EVALUATION REPORT",
        "=" * 40,
        f"VQA Accuracy:          {acc * 100:.1f}%  "
        f"(Target: >65%)  {_pass_fail(acc, TARGETS['vqa_accuracy'])}",
        f"BLEU-4:                {bleu:.2f}   "
        f"(Target: >0.25) {_pass_fail(bleu, TARGETS['bleu4'])}",
        f"BERTScore F1:          {bscore:.2f}   "
        f"(Target: >0.75) {_pass_fail(bscore, TARGETS['bertscore_f1'])}",
        f"Clinical F1 (Macro):   {cf1:.2f}   "
        f"(Target: >0.60) {_pass_fail(cf1, TARGETS['clinical_f1_macro'])}",
        "-" * 40,
        "Per-Pathology F1:",
    ]

    for pathology, f1_val in sorted(
        metrics.clinical_f1_per_pathology.items(),
        key=lambda x: x[1],
        reverse=True,
    ):
        lines.append(f"  {pathology:<26} {f1_val:.2f}")

    lines.append("=" * 40)

    report_str = "\n".join(lines)
    print(report_str)
    return report_str


# ---------------------------------------------------------------------------
# CSV loading helpers
# ---------------------------------------------------------------------------

def _load_answers_from_csv(
    csv_path: str,
    answer_col: str = "answer",
) -> List[str]:
    """Load a list of answer strings from a CSV file."""
    if _PANDAS_AVAILABLE:
        df = pd.read_csv(csv_path)
        if answer_col not in df.columns:
            raise ValueError(
                f"Column '{answer_col}' not found in {csv_path}. "
                f"Available: {list(df.columns)}"
            )
        return df[answer_col].astype(str).tolist()

    # stdlib fallback
    answers: List[str] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if answer_col not in row:
                raise ValueError(
                    f"Column '{answer_col}' not found in {csv_path}."
                )
            answers.append(row[answer_col])
    return answers


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="MedXplain Evaluation Suite (Cahier des Charges metrics)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--predictions_csv",
        required=True,
        help="CSV with a column 'answer' containing model predictions",
    )
    p.add_argument(
        "--references_csv",
        required=True,
        help="CSV with a column 'answer' (or 'radiologist_answer') containing ground truth",
    )
    p.add_argument(
        "--ref_col",
        default="answer",
        help="Column name for references in references_csv",
    )
    p.add_argument(
        "--pred_col",
        default="answer",
        help="Column name for predictions in predictions_csv",
    )
    p.add_argument(
        "--bertscore_model",
        default=None,
        help=(
            "Optional BERTScore model name, e.g. "
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        ),
    )
    p.add_argument(
        "--output_json",
        default=None,
        help="Optional path to save metrics as JSON",
    )
    return p


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _build_arg_parser().parse_args()

    predictions = _load_answers_from_csv(args.predictions_csv, args.pred_col)
    references  = _load_answers_from_csv(args.references_csv,  args.ref_col)

    metrics = run_full_evaluation(
        predictions=predictions,
        references=references,
        bertscore_model=args.bertscore_model,
    )
    print_report(metrics)

    if args.output_json:
        import json
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(metrics.to_dict(), fh, indent=2)
        logger.info("Metrics saved → %s", out_path)


if __name__ == "__main__":
    main()
