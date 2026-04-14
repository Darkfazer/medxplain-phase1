"""
app_gradio.py  ─ MedXplain Hospital VQA Demo (Spec-Compliant Edition)
======================================================================
Fully hospital-ready Gradio interface aligned with Cahier des Charges UI spec:

Layout
──────
┌─ Upload column ─────────────────────────────────────────────────────────┐
│  gr.File  (.dcm / .png / .jpg)                                          │
│  gr.Textbox  Clinical Question                                           │
│  gr.Button   Analyze  [primary]                                          │
└─────────────────────────────────────────────────────────────────────────┘
┌─ Results column (scale=2) ──────────────────────────────────────────────┐
│  [Answer Textbox]  [Confidence Number]                                   │
│  [Uploaded Image]  [Grad-CAM Heatmap]                                   │
└─────────────────────────────────────────────────────────────────────────┘
┌─ gr.Accordion  "DICOM Metadata"  (open=False) ──────────────────────────┐
│  gr.JSON  Study Information                                              │
└─────────────────────────────────────────────────────────────────────────┘

Plus all original clinical feature tabs (Report-Aware, Context-Aware,
Longitudinal, Differential, One-Click Report) preserved.

Privacy
───────
Persistent top banner: 🔒 LOCAL PROCESSING ONLY – No data leaves this machine.

Launch
──────
    cd vqa_app_deliverable/
    python app_gradio.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _THIS_DIR.parent
for _p in (_THIS_DIR, _ROOT_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import gradio as gr
from PIL import Image as PILImage

from config import Config
from model_inference import MedicalVQAModel
from advanced_features import AdvancedClinicalFeatures

# ── Optional DICOM support ────────────────────────────────────────────────────
try:
    from data_ingestion.dicom_pipeline import DICOMLoader
    _DICOM_AVAILABLE = True
    _dicom_loader    = DICOMLoader(
        target_size=(Config.MAX_IMAGE_SIZE, Config.MAX_IMAGE_SIZE)
    )
except ImportError:
    _DICOM_AVAILABLE  = False
    _dicom_loader     = None

# ── Model initialisation ──────────────────────────────────────────────────────
print("[MedXplain] Initialising models …")
vqa_model       = MedicalVQAModel(use_mock=Config.USE_MOCK_MODEL)
advanced_feats  = AdvancedClinicalFeatures(vqa_model)
print("[MedXplain] Ready.")

# ── Accepted file types ───────────────────────────────────────────────────────
_FILE_TYPES = [".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
_DICOM_STATUS = (
    "✅ DICOM (.dcm) supported"
    if _DICOM_AVAILABLE
    else "⚠️ DICOM disabled – `pip install pydicom pylibjpeg`"
)


# ─────────────────────────────────────────────────────────────────────────────
# File loading helper
# ─────────────────────────────────────────────────────────────────────────────

def _load_file(
    file_obj: Any,
) -> Tuple[Optional[PILImage.Image], Dict[str, Any]]:
    """
    Accept a Gradio ``gr.File`` upload object and return:
      (PIL RGB Image, metadata_dict)

    Supports .dcm and standard raster formats.
    """
    if file_obj is None:
        return None, {}

    path: str = file_obj if isinstance(file_obj, str) else file_obj.name
    ext  = Path(path).suffix.lower()
    meta: Dict[str, Any] = {"source": ext.strip(".").upper() or "unknown"}

    # ── DICOM branch ─────────────────────────────────────────────────────────
    if ext == ".dcm":
        if not _DICOM_AVAILABLE:
            raise gr.Error(
                "DICOM support requires: pip install pydicom pylibjpeg"
            )
        study = _dicom_loader.load_study(path)
        if study is None or not study.images:
            raise gr.Error("Could not decode DICOM – file may be corrupted.")

        tensor = study.images[0]                     # (C, H, W) float32 [0,1]
        arr    = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
            pil = PILImage.fromarray(arr, mode="L").convert("RGB")
        else:
            pil = PILImage.fromarray(arr)

        meta = {
            "Source":        "DICOM",
            "Modality":      study.modality,
            "Patient ID":    f"{study.patient_id}  (pseudonymised)",
            "Study UID":     f"{study.study_uid}  (pseudonymised)",
            "Frames":        len(study.images),
            "Rows":          study.metadata.get("rows", "N/A"),
            "Columns":       study.metadata.get("columns", "N/A"),
            "Window Center": study.metadata.get("window_center", "N/A"),
            "Window Width":  study.metadata.get("window_width",  "N/A"),
            "Pixel Spacing": study.metadata.get("pixel_spacing_mm", "N/A"),
        }
        return pil, meta

    # ── Standard raster branch ───────────────────────────────────────────────
    try:
        pil = PILImage.open(path).convert("RGB")
        meta["dimensions"] = f"{pil.width} × {pil.height} px"
        return pil, meta
    except Exception as exc:
        raise gr.Error(f"Failed to open image: {exc}") from exc


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 – Primary VQA + Grad-CAM callback  (spec layout)
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    file_obj: Any,
    question: str,
) -> Tuple[str, float, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """
    Primary hospital workflow:
      (file, question) → (answer, confidence, preview_img, heatmap, metadata_dict)
    """
    if file_obj is None:
        return "⚠️ Please upload an image or DICOM file.", 0.0, None, None, {}

    pil_image, metadata = _load_file(file_obj)

    # Preview array for gr.Image (RGB uint8)
    preview_np = np.array(pil_image)

    # VQA inference
    answer, conf = vqa_model.generate_answer(pil_image, question or "")
    confidence   = round(float(conf) * 100, 1)

    # Grad-CAM
    heatmap_np: Optional[np.ndarray] = None
    try:
        heatmap_np, _ = vqa_model.generate_gradcam(pil_image)
    except Exception as exc:
        pass  # heatmap stays None; UI shows blank

    return answer, confidence, preview_np, heatmap_np, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Remaining tab callbacks (existing clinical features – file-upload aware)
# ─────────────────────────────────────────────────────────────────────────────

def tab_report_aware(file_obj, prior_report: str, question: str) -> str:
    if file_obj is None:
        return "⚠️ Please upload an image."
    pil, _ = _load_file(file_obj)
    answer, _ = advanced_feats.report_aware_answering(pil, prior_report, question)
    return answer


def tab_context_aware(file_obj, clinical_context: str, question: str) -> str:
    if file_obj is None:
        return "⚠️ Please upload an image."
    pil, _ = _load_file(file_obj)
    answer, _ = advanced_feats.context_aware_answering(pil, clinical_context, question)
    return answer


def tab_longitudinal(file_curr, file_prior, question: str) -> str:
    if file_curr is None or file_prior is None:
        return "⚠️ Please upload both current and prior images."
    curr, _ = _load_file(file_curr)
    prior, _ = _load_file(file_prior)
    answer, _ = advanced_feats.compare_longitudinal(curr, prior, question)
    return answer


def tab_differential(file_obj, question: str) -> str:
    if file_obj is None:
        return "⚠️ Please upload an image."
    pil, _ = _load_file(file_obj)
    top_3 = advanced_feats.differential_diagnosis(pil, question)
    return "".join(f"**{k}**: {v}\n\n" for k, v in top_3.items())


def tab_one_click(file_obj) -> str:
    if file_obj is None:
        return "⚠️ Please upload an image."
    pil, _ = _load_file(file_obj)
    return advanced_feats.generate_one_click_report(pil)


# ─────────────────────────────────────────────────────────────────────────────
# Gradio layout  (spec-compliant)
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="MedXplain - Hospital VQA") as demo:

    # ── Persistent privacy banner ─────────────────────────────────────────────
    gr.Markdown("## 🏥 MedXplain: Medical Visual Question Answering")
    gr.Markdown(
        "🔒 **LOCAL PROCESSING ONLY** – No data leaves this machine. "
        "HIPAA-compliant design.  "
        f"&nbsp;&nbsp;|&nbsp;&nbsp; {_DICOM_STATUS}"
    )

    with gr.Tabs():

        # ── Tab 1 – Primary VQA (spec layout) ────────────────────────────────
        with gr.TabItem("🔬 VQA Analysis"):
            with gr.Row():
                # Left column: inputs
                with gr.Column(scale=1):
                    image_input   = gr.File(
                        label="Upload Medical Image",
                        file_types=_FILE_TYPES,
                    )
                    question_input = gr.Textbox(
                        label="Clinical Question",
                        placeholder="e.g., Is there a pleural effusion?",
                    )
                    submit_btn = gr.Button("▶ Analyze", variant="primary")

                # Right column: outputs
                with gr.Column(scale=2):
                    with gr.Row():
                        answer_output     = gr.Textbox(
                            label="Answer", interactive=False
                        )
                        confidence_output = gr.Number(
                            label="Confidence (%)", precision=1, interactive=False
                        )
                    with gr.Row():
                        image_display   = gr.Image(
                            label="Uploaded Image",
                            type="numpy",
                            interactive=False,
                        )
                        heatmap_display = gr.Image(
                            label="Grad-CAM Explanation",
                            type="numpy",
                            interactive=False,
                        )

            # DICOM metadata accordion (outside the row, full width)
            with gr.Accordion("📋 DICOM Metadata", open=False):
                metadata_output = gr.JSON(label="Study Information")

            submit_btn.click(
                fn=analyze,
                inputs=[image_input, question_input],
                outputs=[
                    answer_output,
                    confidence_output,
                    image_display,
                    heatmap_display,
                    metadata_output,
                ],
            )

        # ── Tab 2 – Report-Aware ──────────────────────────────────────────────
        with gr.TabItem("📄 Report-Aware"):
            with gr.Row():
                with gr.Column():
                    t2_file = gr.File(label="Upload Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t2_rep  = gr.Textbox(label="Prior Radiology Report", lines=4)
                    t2_qst  = gr.Textbox(label="Question",
                                         placeholder="How does this compare?")
                    t2_btn  = gr.Button("▶ Analyze", variant="primary")
                with gr.Column():
                    t2_ans = gr.Textbox(label="Answer", interactive=False)
            t2_btn.click(tab_report_aware, [t2_file, t2_rep, t2_qst], t2_ans)

        # ── Tab 3 – Context-Aware ─────────────────────────────────────────────
        with gr.TabItem("🧪 Context-Aware"):
            with gr.Row():
                with gr.Column():
                    t3_file = gr.File(label="Upload Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t3_ctx  = gr.Textbox(label="Clinical Notes / Vitals / Labs",
                                         lines=4)
                    t3_qst  = gr.Textbox(label="Question",
                                         placeholder="Most likely diagnosis given labs?")
                    t3_btn  = gr.Button("▶ Analyze", variant="primary")
                with gr.Column():
                    t3_ans = gr.Textbox(label="Answer", interactive=False)
            t3_btn.click(tab_context_aware, [t3_file, t3_ctx, t3_qst], t3_ans)

        # ── Tab 4 – Longitudinal ──────────────────────────────────────────────
        with gr.TabItem("📅 Longitudinal View"):
            with gr.Row():
                with gr.Column():
                    t4_curr = gr.File(label="Current Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t4_prior= gr.File(label="Prior Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t4_qst  = gr.Textbox(label="Question",
                                         placeholder="Is the nodule growing?")
                    t4_btn  = gr.Button("▶ Compare", variant="primary")
                with gr.Column():
                    t4_ans = gr.Textbox(label="Comparative Analysis",
                                        interactive=False)
            t4_btn.click(tab_longitudinal, [t4_curr, t4_prior, t4_qst], t4_ans)

        # ── Tab 5 – Differential Diagnosis ───────────────────────────────────
        with gr.TabItem("⚕️ Differential Diagnosis"):
            with gr.Row():
                with gr.Column():
                    t5_file = gr.File(label="Upload Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t5_qst  = gr.Textbox(label="Question",
                                         placeholder="Differential diagnoses?")
                    t5_btn  = gr.Button("▶ Rank Diagnoses", variant="primary")
                with gr.Column():
                    t5_ans = gr.Markdown("### Results will appear here")
            t5_btn.click(tab_differential, [t5_file, t5_qst], t5_ans)

        # ── Tab 6 – One-Click Report ─────────────────────────────────────────
        with gr.TabItem("📝 One-Click Report"):
            with gr.Row():
                with gr.Column():
                    t6_file = gr.File(label="Upload Image / DICOM",
                                      file_types=_FILE_TYPES)
                    t6_btn  = gr.Button("▶ Generate Draft Report",
                                        variant="primary")
                with gr.Column():
                    t6_ans = gr.Textbox(label="Auto-Generated Draft Report",
                                        lines=12, interactive=False)
            t6_btn.click(tab_one_click, [t6_file], t6_ans)


if __name__ == "__main__":
    demo.launch(server_port=Config.GRADIO_PORT, server_name="0.0.0.0", share=False)
