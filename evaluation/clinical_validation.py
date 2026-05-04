"""
evaluation/clinical_validation.py
===================================
Generate a clinical validation report for radiologist review.

Fixes vs. original:
  - tempfile.mktemp() → NamedTemporaryFile(delete=False) (safe)
  - Full graceful degradation when optional deps (pandas, reportlab, PIL) absent
  - Clean public API: generate_validation_report()
"""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports with graceful fallback
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    _PANDAS = True
except ImportError:
    _PANDAS = False

try:
    from sklearn.metrics import cohen_kappa_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False
    log.warning("scikit-learn not installed – install with: pip install scikit-learn")

try:
    from PIL import Image as PILImage
    _PIL = True
except ImportError:
    _PIL = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image as RLImage, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )
    from reportlab.lib import colors as rl_colors
    _REPORTLAB = True
except ImportError:
    _REPORTLAB = False
    log.warning("reportlab not installed – PDF export disabled: pip install reportlab")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _normalise(s: str) -> str:
    return " ".join(s.lower().strip().split())


def _match(pred: str, gold: str) -> bool:
    return _normalise(pred) == _normalise(gold)


def _kappa(preds: List[str], refs: List[str]) -> float:
    if _SKLEARN:
        try:
            return float(cohen_kappa_score(refs, preds))
        except Exception as exc:
            log.warning("cohen_kappa_score failed: %s – using accuracy proxy.", exc)
    matches = sum(1 for p, r in zip(preds, refs) if _match(p, r))
    return matches / max(len(preds), 1)


def _load_csv(csv_path: str) -> List[Dict[str, str]]:
    if _PANDAS:
        try:
            return pd.read_csv(csv_path).to_dict(orient="records")
        except Exception as exc:
            log.error("pandas CSV read failed: %s", exc)
    rows: List[Dict[str, str]] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except OSError as exc:
        log.error("Cannot open CSV %s: %s", csv_path, exc)
    return rows


def _load_image(path: str):  # → Optional[PILImage.Image]
    if not _PIL:
        log.error("Pillow not installed – cannot load images.")
        return None
    try:
        return PILImage.open(path).convert("RGB")
    except Exception as exc:
        log.warning("Cannot open image %s: %s", path, exc)
        return None


def _build_pdf(
    output_path: str,
    image_path: str,
    question: str,
    model_answer: str,
    confidence: float,
    radiologist_answer: str,
    gradcam_array: Optional[np.ndarray],
    sample_id: str,
) -> None:
    """Build a single-page PDF discrepancy report."""
    if not _REPORTLAB:
        log.warning("PDF skipped for %s – reportlab not installed.", sample_id)
        return

    doc    = SimpleDocTemplate(output_path, pagesize=A4)
    story  = []
    styles = getSampleStyleSheet()
    ts     = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    story.append(Paragraph("MedXplain Clinical Validation – Discrepancy Report", styles["Title"]))
    story.append(Spacer(1, 0.3 * cm))
    story.append(Paragraph(f"Sample ID: {sample_id} | Generated: {ts}", styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))
    story.append(Paragraph("<b>⚠ This report contains no PHI.</b>", styles["Normal"]))
    story.append(Spacer(1, 0.7 * cm))

    data = [
        ["Field",               "Value"],
        ["Question",            question],
        ["Model Answer",        model_answer],
        ["Confidence",          f"{confidence * 100:.1f}%"],
        ["Radiologist Answer",  radiologist_answer],
        ["Match",               "✗ DISCREPANCY"],
    ]
    tbl = Table(data, colWidths=[4 * cm, 13 * cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, 0),  rl_colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",     (0, 0), (-1, 0),  rl_colors.white),
        ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("ROWBACKGROUNDS",(0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#f0f4f8")]),
        ("GRID",          (0, 0), (-1, -1), 0.5, rl_colors.HexColor("#cccccc")),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.7 * cm))

    if os.path.isfile(image_path):
        story.append(Paragraph("Source Image (de-identified):", styles["Heading3"]))
        try:
            story.append(RLImage(image_path, width=8 * cm, height=8 * cm))
        except Exception as exc:
            story.append(Paragraph(f"[Image embed failed: {exc}]", styles["Normal"]))
        story.append(Spacer(1, 0.4 * cm))

    if gradcam_array is not None and _PIL:
        try:
            # Use NamedTemporaryFile (safe replacement for mktemp)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp_path = tmp.name
            PILImage.fromarray(gradcam_array.astype(np.uint8)).save(tmp_path)
            story.append(Paragraph("Grad-CAM Heatmap Overlay:", styles["Heading3"]))
            story.append(RLImage(tmp_path, width=8 * cm, height=8 * cm))
            os.unlink(tmp_path)
        except Exception as exc:
            story.append(Paragraph(f"[Grad-CAM embed failed: {exc}]", styles["Normal"]))

    doc.build(story)
    log.info("Discrepancy PDF saved: %s", output_path)


def _save_metrics_json(metrics: Dict[str, Any], output_dir: Path) -> None:
    ts  = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = output_dir / f"validation_metrics_{ts}.json"
    try:
        with out.open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)
        log.info("Metrics saved to %s", out)
    except OSError as exc:
        log.error("Could not save metrics JSON: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def generate_validation_report(
    model: Any,
    validation_csv: str,
    output_dir: str,
    image_base_dir: str = ".",
    max_discrepancies: int = 200,
) -> Dict[str, Any]:
    """Run full clinical validation against radiologist annotations.

    The ``model`` object must expose:
        ``generate_answer(image, question) → (str, float)``
        ``generate_gradcam(image) → (np.ndarray, any)``   (optional)

    Parameters
    ----------
    model             : VQA model instance
    validation_csv    : path to CSV with columns image_id, question, radiologist_answer
    output_dir        : directory for PDF discrepancy reports
    image_base_dir    : root directory to resolve image_id paths
    max_discrepancies : max PDF reports to generate

    Returns
    -------
    dict: accuracy, cohen_kappa, n_total, n_correct, n_discrepancies
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    rows = _load_csv(validation_csv)
    if not rows:
        log.error("No rows loaded from %s", validation_csv)
        return {"accuracy": 0.0, "cohen_kappa": 0.0,
                "n_total": 0, "n_correct": 0, "n_discrepancies": 0}

    predictions: List[str] = []
    references:  List[str] = []
    n_disc = 0

    for row in rows:
        img_id  = row.get("image_id", "").strip()
        question = row.get("question", "").strip()
        rad_ans  = row.get("radiologist_answer", "").strip()

        if not (img_id and question and rad_ans):
            log.warning("Skipping incomplete row: %s", row)
            continue

        image_path = str(Path(image_base_dir) / img_id)
        image = _load_image(image_path)
        if image is None:
            predictions.append(""); references.append(rad_ans)
            continue

        try:
            model_answer, confidence = model.generate_answer(image, question)
        except Exception as exc:
            log.error("Inference failed for %s: %s", img_id, exc)
            model_answer, confidence = "ERROR", 0.0

        predictions.append(model_answer)
        references.append(rad_ans)

        if not _match(model_answer, rad_ans) and n_disc < max_discrepancies:
            gradcam = None
            try:
                gradcam, _ = model.generate_gradcam(image)
            except Exception:
                pass

            safe_id  = "".join(c if c.isalnum() else "_" for c in img_id)
            pdf_path = str(out_path / f"discrepancy_{safe_id}.pdf")
            _build_pdf(
                output_path=pdf_path,
                image_path=image_path,
                question=question,
                model_answer=model_answer,
                confidence=float(confidence),
                radiologist_answer=rad_ans,
                gradcam_array=gradcam,
                sample_id=img_id,
            )
            n_disc += 1

    if not predictions:
        return {"accuracy": 0.0, "cohen_kappa": 0.0,
                "n_total": 0, "n_correct": 0, "n_discrepancies": n_disc}

    n_correct = sum(1 for p, r in zip(predictions, references) if _match(p, r))
    accuracy  = n_correct / len(predictions)
    kappa     = _kappa(predictions, references)

    metrics = {
        "accuracy":        round(accuracy, 4),
        "cohen_kappa":     round(kappa,    4),
        "n_total":         len(predictions),
        "n_correct":       n_correct,
        "n_discrepancies": n_disc,
    }
    log.info(
        "Validation complete | acc=%.3f | kappa=%.3f | disc=%d/%d",
        accuracy, kappa, n_disc, len(predictions),
    )
    _save_metrics_json(metrics, out_path)
    return metrics
