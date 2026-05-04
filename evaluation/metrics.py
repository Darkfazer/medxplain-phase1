"""
evaluation/metrics.py
=====================
VQA evaluation metrics: Accuracy (closed) and BLEU (open) questions.

Fixes vs. original:
  - Uses punkt_tab (NLTK >= 3.9 compatible)
  - Returns answer to caller (not just module-level side effect)
"""
from __future__ import annotations

import re
import logging

import nltk

log = logging.getLogger(__name__)

# Download required NLTK data (silent if already present)
for resource in ("punkt_tab", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{resource}")
        break
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and normalise whitespace."""
    return re.sub(r"[^\w\s]", "", text.lower().strip())


def compute_accuracy(preds: list[str], labels: list[str]) -> float:
    """Exact-match accuracy for closed-ended questions.

    A prediction is correct if it exactly matches the label OR if the
    (short) label string is contained within a longer prediction.

    Returns accuracy as a percentage in [0, 100].
    """
    if not preds:
        return 0.0
    correct = sum(
        1
        for p, l in zip(preds, labels)
        if (pc := clean_text(p)) == (lc := clean_text(l))
        or (lc and lc in pc)
    )
    return (correct / len(preds)) * 100.0


def compute_bleu(preds: list[str], labels: list[str]) -> float:
    """Sentence-level BLEU (smoothed) for open-ended questions.

    Returns mean BLEU score as a percentage in [0, 100].
    """
    if not preds:
        return 0.0
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu(
            [clean_text(l).split()],
            clean_text(p).split(),
            smoothing_function=smoothie,
        )
        for p, l in zip(preds, labels)
    ]
    return (sum(scores) / len(scores)) * 100.0


def evaluate_batch(
    preds: list[str],
    labels: list[str],
    q_types: list[str],
) -> dict:
    """Compute accuracy (closed) and BLEU (open) for one batch.

    Parameters
    ----------
    preds   : model-generated answers
    labels  : ground-truth answers
    q_types : per-sample question type, either ``'OPEN'`` or ``'CLOSED'``

    Returns
    -------
    dict with keys: closed_acc, open_bleu, num_closed, num_open
    """
    closed_p, closed_l, open_p, open_l = [], [], [], []
    for p, l, qt in zip(preds, labels, q_types):
        if qt == "CLOSED":
            closed_p.append(p); closed_l.append(l)
        else:
            open_p.append(p);   open_l.append(l)

    return {
        "closed_acc": compute_accuracy(closed_p, closed_l) if closed_p else 0.0,
        "open_bleu":  compute_bleu(open_p, open_l)         if open_p  else 0.0,
        "num_closed": len(closed_p),
        "num_open":   len(open_p),
    }


def aggregate_metrics(metrics_list: list[dict]) -> dict:
    """Aggregate per-batch metric dicts into dataset-level metrics.

    Weighted averages by number of samples per question type.
    """
    total_acc, total_bleu = 0.0, 0.0
    total_closed, total_open = 0, 0

    for m in metrics_list:
        nc, no = m["num_closed"], m["num_open"]
        total_acc  += m["closed_acc"] * nc
        total_bleu += m["open_bleu"]  * no
        total_closed += nc
        total_open   += no

    return {
        "Accuracy (Closed)": total_acc  / total_closed if total_closed else 0.0,
        "BLEU (Open)":       total_bleu / total_open   if total_open   else 0.0,
    }
