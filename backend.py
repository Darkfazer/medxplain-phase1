"""
MedXplain – Core Backend  (backend.py)
=======================================
Models
  • TorchXRayVision DenseNet121-res224-all  → classification
  • Salesforce/blip-vqa-base  (BLIP-1)      → Visual Question Answering
  • pytorch-grad-cam                         → Grad-CAM explainability
  • pydicom                                  → DICOM ingestion
  • scikit-image SSIM                        → Longitudinal comparison

Patient data is stored locally under  medxplain_db/
  reports_<pid>.json   – VQA / report history
  prior_<pid>.png      – most recent image for longitudinal diff

NOTE: DB_DIR is created lazily on first write, not at import time.
"""

from __future__ import annotations

import cv2
import json
import logging
import math
import time
import datetime
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

log = logging.getLogger("MedXplain.backend")

# ─────────────────────────────────────────────────────────────────────────────
#  Paths (no side-effects at import time)
# ─────────────────────────────────────────────────────────────────────────────
DB_DIR = Path("medxplain_db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

log.info("Backend device: %s", DEVICE)


def _ensure_db_dir() -> None:
    """Create the patient database directory on first use."""
    DB_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Lazy model cache
# ─────────────────────────────────────────────────────────────────────────────
_txrv: object = None
_blip_proc: object = None
_blip_model: object = None


def _load_txrv():
    """Load (once) TorchXRayVision DenseNet121."""
    global _txrv
    if _txrv is not None:
        return _txrv
    import torchxrayvision as xrv
    log.info("Loading TorchXRayVision DenseNet121…")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval().to(DEVICE)
    _txrv = model
    log.info("TorchXRayVision ready.")
    return _txrv


def _load_blip():
    """Load (once) BLIP-1 VQA model (Salesforce/blip-vqa-base)."""
    global _blip_proc, _blip_model
    if _blip_model is not None:
        return _blip_proc, _blip_model
    from transformers import BlipProcessor, BlipForQuestionAnswering
    log.info("Loading BLIP-VQA-base…")
    _blip_proc = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    _blip_model = BlipForQuestionAnswering.from_pretrained(
        "Salesforce/blip-vqa-base"
    ).to(DEVICE)
    _blip_model.eval()
    log.info("BLIP ready.")
    return _blip_proc, _blip_model


# ─────────────────────────────────────────────────────────────────────────────
#  DICOM ingestion
# ─────────────────────────────────────────────────────────────────────────────
def load_dicom(path: str) -> np.ndarray:
    """Load a DICOM file and return a uint8 RGB numpy array."""
    import pydicom
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
    arr = arr.clip(0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
#  Image pre-processing for TorchXRayVision
# ─────────────────────────────────────────────────────────────────────────────
def _preprocess_txrv(img_arr: np.ndarray) -> torch.Tensor:
    """Resize to 224×224, convert to grayscale float, normalise for TXRaV.

    Returns shape (1, 1, 224, 224) on DEVICE.
    """
    import torchxrayvision as xrv
    pil = Image.fromarray(img_arr).convert("L").resize((224, 224))
    arr = np.array(pil, dtype=np.float32)
    arr = xrv.datasets.normalize(arr, maxval=255, reshape=True)  # (1, 224, 224)
    return torch.from_numpy(arr).unsqueeze(0).to(DEVICE)         # (1, 1, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
#  Classification
# ─────────────────────────────────────────────────────────────────────────────
RISK_MAP = {
    "Pneumonia": "High", "COVID-19": "High", "Pleural Effusion": "Medium",
    "Cardiomegaly": "Medium", "Edema": "High", "Consolidation": "High",
    "Atelectasis": "Medium", "Pneumothorax": "High",
    "Infiltration": "Medium", "Mass": "High", "Nodule": "Medium",
    "Hernia": "Low", "Fibrosis": "Low", "Emphysema": "Low", "No Finding": "Low",
}


def classify_image(img_arr: np.ndarray, model_choice: str = "Ensemble") -> dict:
    """Run TorchXRayVision DenseNet121 and return classification results.

    Returns
    -------
    dict with keys:
        label       : str   – top-1 pathology
        confidence  : float – sigmoid probability for top-1
        interval    : list  – [lower, upper] 90% heuristic band
        risk        : str   – Low / Medium / High
        all_probs   : dict  – {pathology: probability} for all 14 classes
    """
    model = _load_txrv()
    tensor = _preprocess_txrv(img_arr)

    with torch.no_grad():
        logits = model(tensor)                           # (1, num_pathologies)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    pathologies = model.pathologies
    prob_dict = {p: float(probs[i]) for i, p in enumerate(pathologies)}

    top_idx = int(np.argmax(probs))
    label = pathologies[top_idx]
    conf = float(probs[top_idx])

    # Heuristic uncertainty band: ±1 standard deviation of sigmoid outputs,
    # clipped to [0, 1].  This is NOT a conformal interval; treat as a rough
    # uncertainty indicator only.
    std = float(probs.std())
    ci = [
        round(float(np.clip(conf - std, 0.0, 1.0)), 3),
        round(float(np.clip(conf + std, 0.0, 1.0)), 3),
    ]

    risk = RISK_MAP.get(label, "Medium")

    return {
        "label":      label,
        "confidence": conf,
        "interval":   ci,
        "risk":       risk,
        "all_probs":  prob_dict,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  VQA
# ─────────────────────────────────────────────────────────────────────────────
def answer_vqa(img_arr: np.ndarray, question: str, context: str = "") -> dict:
    """Run BLIP-1 VQA on the image + question (with optional context prefix).

    Confidence is computed as the mean length-normalised log-probability of
    the generated token sequence, then exponentiated to [0, 1].

    Returns
    -------
    dict with keys: answer (str), confidence (float in [0, 1])
    """
    proc, model = _load_blip()
    pil = Image.fromarray(img_arr).convert("RGB")
    full_q = f"{context}\n{question}".strip() if context else question

    inputs = proc(images=pil, text=full_q, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=64,
            output_scores=True,
            return_dict_in_generate=True,
        )
        tokens = out.sequences[0]
        answer = proc.decode(tokens, skip_special_tokens=True).strip()

        # Length-normalised mean log-probability → exponentiate to [0, 1]
        if out.scores:
            log_probs = [
                torch.log_softmax(s, dim=-1)[0, tok].item()
                for s, tok in zip(out.scores, tokens[tokens.shape[0] - len(out.scores):])
            ]
            mean_lp = sum(log_probs) / max(len(log_probs), 1)
            conf = float(math.exp(max(mean_lp, -10.0)))   # clamp to avoid tiny values
        else:
            conf = 0.0

    return {"answer": answer or "Unable to generate answer.", "confidence": conf}


# ─────────────────────────────────────────────────────────────────────────────
#  Grad-CAM  (uses shared explainability module)
# ─────────────────────────────────────────────────────────────────────────────
def generate_gradcam(img_arr: np.ndarray,
                     target_class_idx: Optional[int] = None) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlay using pytorch-grad-cam.

    Uses the last DenseBlock of TorchXRayVision DenseNet121.

    Parameters
    ----------
    img_arr          : uint8 RGB numpy array (H, W, 3)
    target_class_idx : pathology index to highlight; defaults to top-1

    Returns
    -------
    uint8 RGB overlay numpy array (same H×W as input)
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image

    model = _load_txrv()
    target_layer = model.features.denseblock4

    # Determine target class index
    if target_class_idx is None:
        tensor = _preprocess_txrv(img_arr)
        with torch.no_grad():
            probs = torch.sigmoid(model(tensor)).squeeze().cpu().numpy()
        target_class_idx = int(np.argmax(probs))

    # pytorch-grad-cam expects (1, C, H, W) float32 in [0, 1]
    # TorchXRayVision uses (1, 1, 224, 224) grayscale
    tensor_for_cam = _preprocess_txrv(img_arr)      # (1, 1, 224, 224)
    # Expand to 3-channel so show_cam_on_image works cleanly
    tensor_rgb = tensor_for_cam.repeat(1, 3, 1, 1).to(DEVICE)

    targets = [ClassifierOutputTarget(target_class_idx)]

    # GradCAM does NOT take use_cuda; device is determined by model/tensor placement
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=tensor_rgb, targets=targets)[0]  # (H, W)

    # Build RGB overlay on original image
    h, w = img_arr.shape[:2]
    cam_resized = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)

    img_float = img_arr.astype(np.float32) / 255.0
    overlay = show_cam_on_image(img_float, cam_resized, use_rgb=True)  # uint8 RGB
    return overlay


# ─────────────────────────────────────────────────────────────────────────────
#  Patient Database helpers
# ─────────────────────────────────────────────────────────────────────────────
def _report_path(pid: str) -> Path:
    return DB_DIR / f"reports_{pid}.json"


def _prior_path(pid: str) -> Path:
    return DB_DIR / f"prior_{pid}.png"


def load_reports(pid: str = "default") -> list[dict]:
    """Load stored report history for a patient (newest first)."""
    p = _report_path(pid)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def save_report_entry(pid: str, entry: dict) -> None:
    """Prepend a new entry to the patient's report history (max 20 kept)."""
    _ensure_db_dir()
    reports = load_reports(pid)
    reports.insert(0, entry)
    _report_path(pid).write_text(
        json.dumps(reports[:20], indent=2), encoding="utf-8"
    )


def save_prior_image(pid: str, img_arr: np.ndarray) -> None:
    """Persist the current image as the prior baseline for this patient."""
    _ensure_db_dir()
    Image.fromarray(img_arr).save(_prior_path(pid))


def load_prior_image(pid: str) -> Optional[np.ndarray]:
    """Load the stored prior image for longitudinal comparison, or None."""
    p = _prior_path(pid)
    if p.exists():
        return np.array(Image.open(p).convert("RGB"))
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  Feature 1 – Report-Aware context string
# ─────────────────────────────────────────────────────────────────────────────
def build_report_context(pid: str = "default", n: int = 3) -> str:
    """Return a context prefix derived from the last n stored reports."""
    reports = load_reports(pid)[:n]
    if not reports:
        return ""
    lines = ["[PRIOR REPORTS]"]
    for r in reports:
        lines.append(
            f"  {r.get('date','?')}: {r.get('prediction','?')}. "
            f"VQA: \"{r.get('vqa_answer','')}\""
        )
    lines.append("[END PRIOR REPORTS]")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature 2 – Context-Aware vitals/labs context string
# ─────────────────────────────────────────────────────────────────────────────
def build_vitals_context(bp: str, hr: str, temp: str,
                         spo2: str, wbc: str, crp: str) -> str:
    """Build a structured context prefix from validated vitals/labs strings."""
    parts: list[str] = []
    if bp:   parts.append(f"BP={bp}")
    if hr:   parts.append(f"HR={hr}")
    if temp: parts.append(f"Temp={temp}°C")
    if spo2: parts.append(f"SpO2={spo2}%")
    vitals_str = ", ".join(parts) if parts else "N/A"

    lab_parts: list[str] = []
    if wbc: lab_parts.append(f"WBC={wbc}")
    if crp: lab_parts.append(f"CRP={crp}")
    labs_str = ", ".join(lab_parts) if lab_parts else "N/A"

    if vitals_str == "N/A" and labs_str == "N/A":
        return ""
    return f"[VITALS: {vitals_str}]\n[LABS: {labs_str}]"


# ─────────────────────────────────────────────────────────────────────────────
#  Feature 3 – Longitudinal comparison
# ─────────────────────────────────────────────────────────────────────────────
def longitudinal_compare(current: np.ndarray, pid: str = "default") -> str:
    """Compare the current image against the stored prior using SSIM.

    Returns a human-readable verdict string with SSIM score.
    """
    prior = load_prior_image(pid)
    if prior is None:
        return "_No prior image stored for this patient — baseline saved._"

    from skimage.metrics import structural_similarity as ssim
    from skimage.color import rgb2gray

    def _resize_gray(arr: np.ndarray) -> np.ndarray:
        return cv2.resize(rgb2gray(arr).astype(np.float32), (224, 224))

    score = float(ssim(_resize_gray(current), _resize_gray(prior), data_range=1.0))

    if score > 0.92:
        verdict = "✅ **No significant change** detected vs. prior study."
    elif score > 0.80:
        verdict = "🟡 **Mild progression** detected vs. prior study."
    else:
        verdict = "🔴 **Significant change detected** vs. prior study."

    return f"{verdict}  _(SSIM = {score:.3f})_"


# ─────────────────────────────────────────────────────────────────────────────
#  Feature 4 – One-Click Structured Report
# ─────────────────────────────────────────────────────────────────────────────
def build_structured_report(pid: str, cls: dict, vqa_answer: str,
                             diff_text: str, model_choice: str) -> str:
    """Generate a plain-text structured clinical report."""
    label = cls["label"]
    conf  = cls["confidence"]
    ci    = cls["interval"]
    risk  = cls["risk"]
    now   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = [
        "MEDXPLAIN CLINICAL REPORT",
        "=" * 44,
        f"Date     : {now}",
        f"Patient  : {pid}",
        f"Model    : {model_choice}",
        "",
        "FINDINGS",
        "-" * 44,
        f"Primary finding    : {label}",
        f"Confidence         : {conf:.1%}",
        f"Uncertainty band   : [{ci[0]:.3f} – {ci[1]:.3f}]",
        f"Risk Level         : {risk}",
        "",
        "IMPRESSION",
        "-" * 44,
        vqa_answer or f"Imaging findings consistent with {label}.",
        "",
        "DIFFERENTIAL DIAGNOSIS",
        "-" * 44,
        diff_text or "N/A",
        "",
        "RECOMMENDATION",
        "-" * 44,
        "Clinical correlation required.",
        "Follow-up imaging recommended in 4–6 weeks if acute findings present.",
        "",
        "─" * 44,
        "Report generated by MedXplain v1.0-beta.",
        "NOT for standalone clinical diagnosis.",
        "Always verify with a qualified radiologist or clinician.",
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature 5 – Differential Diagnosis
# ─────────────────────────────────────────────────────────────────────────────
def build_differential(all_probs: dict, top_k: int = 3) -> str:
    """Return a markdown string listing the top-k differentials."""
    ranked = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    lines: list[str] = []
    for i, (label, prob) in enumerate(ranked[:top_k], 1):
        bar_n = int(prob * 16)
        bar   = "█" * bar_n + "░" * (16 - bar_n)
        lines.append(f"{i}. **{label}** — {prob:.1%}  `{bar}`")
    return "\n".join(lines)
