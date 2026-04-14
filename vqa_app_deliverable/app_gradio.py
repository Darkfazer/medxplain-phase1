"""
app_gradio.py  (Hospital-Ready Edition)
========================================
Enhanced MedXplain Gradio application with:
  - DICOM (.dcm) upload support via the data_ingestion pipeline
  - DICOM metadata display in a collapsible panel
  - Grad-CAM heatmap shown side-by-side with the VQA answer
  - Model confidence score (0-100 %)
  - On-screen privacy / data-locality notice
  - All existing clinical feature tabs preserved

Launch:
    cd vqa_app_deliverable/
    python app_gradio.py
"""

import sys
import os

# Allow imports from both the deliverable dir and the project root
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gradio as gr
import numpy as np
from PIL import Image as PILImage

from config import Config
from model_inference import MedicalVQAModel
from advanced_features import AdvancedClinicalFeatures

# ---------------------------------------------------------------------------
# Optional DICOM support
# ---------------------------------------------------------------------------
try:
    from data_ingestion.dicom_pipeline import DICOMLoader
    _DICOM_AVAILABLE = True
    _dicom_loader = DICOMLoader(target_size=(Config.MAX_IMAGE_SIZE, Config.MAX_IMAGE_SIZE))
except ImportError:
    _DICOM_AVAILABLE = False
    _dicom_loader = None

# ---------------------------------------------------------------------------
# Global model initialisation
# ---------------------------------------------------------------------------
print("[MedXplain] Initialising models …")
vqa_model    = MedicalVQAModel(use_mock=Config.USE_MOCK_MODEL)
advanced_feats = AdvancedClinicalFeatures(vqa_model)
print("[MedXplain] Ready.")

# ---------------------------------------------------------------------------
# DICOM loading helper
# ---------------------------------------------------------------------------

def _load_uploaded_file(file_obj) -> tuple[PILImage.Image, dict]:
    """
    Accept a Gradio file upload object (path string) and return:
    (PIL Image ready for VQA, metadata dict for display).

    Supports .dcm and standard image formats (.png, .jpg, .jpeg, .bmp, .tiff).
    """
    if file_obj is None:
        return None, {}

    path: str = file_obj if isinstance(file_obj, str) else file_obj.name
    ext = os.path.splitext(path)[-1].lower()
    metadata_info: dict = {}

    if ext == ".dcm":
        if not _DICOM_AVAILABLE:
            raise gr.Error(
                "DICOM support requires pydicom: pip install pydicom pylibjpeg"
            )
        study = _dicom_loader.load_study(path)
        if study is None or not study.images:
            raise gr.Error("Could not decode DICOM file. File may be corrupted.")

        # Use the first frame as the representative image
        tensor = study.images[0]  # (C, H, W) float32 in [0,1]
        arr    = (tensor.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        if arr.shape[-1] == 1:
            arr = np.squeeze(arr, axis=-1)
            pil_img = PILImage.fromarray(arr, mode="L").convert("RGB")
        else:
            pil_img = PILImage.fromarray(arr)

        metadata_info = {
            "Source":       "DICOM",
            "Modality":     study.modality,
            "Patient ID":   study.patient_id + " (pseudonymised)",
            "Study UID":    study.study_uid  + " (pseudonymised)",
            "Frames":       len(study.images),
            **study.metadata,
        }
        return pil_img, metadata_info

    else:
        pil_img = PILImage.open(path).convert("RGB")
        metadata_info = {"Source": ext.strip(".").upper()}
        return pil_img, metadata_info


def _format_metadata(meta: dict) -> str:
    """Format metadata dict into a readable markdown table."""
    if not meta:
        return "*No metadata available.*"
    rows = "\n".join(f"| **{k}** | {v} |" for k, v in meta.items())
    return f"| Field | Value |\n|---|---|\n{rows}"


# ---------------------------------------------------------------------------
# Tab 1 – VQA + Grad-CAM + Confidence (hospital-ready)
# ---------------------------------------------------------------------------

def tab1_vqa_heatmap(file_obj, question: str):
    """Main hospital tab: DICOM/image → answer + confidence + Grad-CAM."""
    if file_obj is None:
        return "⚠️ Please upload an image or DICOM file.", None, 0.0, "*No file uploaded.*"

    try:
        image, metadata = _load_uploaded_file(file_obj)
    except gr.Error:
        raise
    except Exception as exc:
        return f"❌ File load error: {exc}", None, 0.0, "*Load failed.*"

    answer, conf = vqa_model.generate_answer(image, question)
    gradcam_img, _ = vqa_model.generate_gradcam(image)

    confidence_pct = round(float(conf) * 100, 1)
    meta_md = _format_metadata(metadata)

    return answer, gradcam_img, confidence_pct, meta_md


# ---------------------------------------------------------------------------
# Existing tab callbacks (unchanged logic, updated signature for file upload)
# ---------------------------------------------------------------------------

def tab2_report_aware(file_obj, prior_report: str, question: str):
    if file_obj is None:
        return "⚠️ Please upload an image."
    image, _ = _load_uploaded_file(file_obj)
    answer, _ = advanced_feats.report_aware_answering(image, prior_report, question)
    return answer


def tab3_context_aware(file_obj, clinical_context: str, question: str):
    if file_obj is None:
        return "⚠️ Please upload an image."
    image, _ = _load_uploaded_file(file_obj)
    answer, _ = advanced_feats.context_aware_answering(image, clinical_context, question)
    return answer


def tab4_longitudinal(file_curr, file_prior, question: str):
    if file_curr is None or file_prior is None:
        return "⚠️ Please upload both current and prior images."
    image_curr, _ = _load_uploaded_file(file_curr)
    image_prior, _ = _load_uploaded_file(file_prior)
    answer, _ = advanced_feats.compare_longitudinal(image_curr, image_prior, question)
    return answer


def tab5_differential(file_obj, question: str):
    if file_obj is None:
        return "⚠️ Please upload an image."
    image, _ = _load_uploaded_file(file_obj)
    top_3 = advanced_feats.differential_diagnosis(image, question)
    return "".join(f"**{k}**: {v}\n\n" for k, v in top_3.items())


def tab6_one_click(file_obj):
    if file_obj is None:
        return "⚠️ Please upload an image."
    image, _ = _load_uploaded_file(file_obj)
    return advanced_feats.generate_one_click_report(image)


# ---------------------------------------------------------------------------
# Privacy notice (shown at top of every page)
# ---------------------------------------------------------------------------
_PRIVACY_NOTICE = """
> 🔒 **Privacy & Data Locality Notice**
> This application runs **entirely on your local machine or on-premises server**.
> No images, questions, or results are transmitted to any external server or cloud service.
> All DICOM metadata is pseudonymised before display.
> Compliant with HIPAA / GDPR data-minimisation principles when used in an air-gapped environment.
"""

_DICOM_NOTE = (
    "✅ DICOM (.dcm) upload supported"
    if _DICOM_AVAILABLE
    else "⚠️ DICOM upload disabled – install pydicom (`pip install pydicom pylibjpeg`)"
)

# ---------------------------------------------------------------------------
# Gradio layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="MedXplain VQA – Hospital Edition") as demo:

    gr.Markdown("# 🏥 MedXplain: Clinical VQA System — Hospital Edition")
    gr.Markdown(_PRIVACY_NOTICE)
    gr.Markdown(
        f"**Supported formats:** `.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.dcm` &nbsp;&nbsp;|&nbsp;&nbsp; {_DICOM_NOTE}"
    )

    with gr.Tabs():

        # ── Tab 1: VQA + Grad-CAM (hospital primary workflow) ─────────────
        with gr.TabItem("🔬 VQA + Heatmap"):
            with gr.Row():
                with gr.Column(scale=1):
                    t1_file = gr.File(
                        label="Upload Image or DICOM (.dcm, .png, .jpg …)",
                        file_types=[".dcm", ".png", ".jpg", ".jpeg", ".bmp", ".tiff"],
                    )
                    t1_qst = gr.Textbox(
                        label="Clinical Question",
                        placeholder="e.g. Is there a pleural effusion?",
                    )
                    t1_btn = gr.Button("▶ Analyse", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        t1_ans  = gr.Textbox(label="VQA Answer", interactive=False)
                        t1_conf = gr.Number(label="Confidence (%)", precision=1, interactive=False)
                    t1_cam = gr.Image(type="numpy", label="Grad-CAM Heatmap Overlay")
                    with gr.Accordion("📋 DICOM Metadata", open=False):
                        t1_meta = gr.Markdown("*Upload a DICOM file to view metadata.*")

            t1_btn.click(
                fn=tab1_vqa_heatmap,
                inputs=[t1_file, t1_qst],
                outputs=[t1_ans, t1_cam, t1_conf, t1_meta],
            )

        # ── Tab 2: Report-Aware ───────────────────────────────────────────
        with gr.TabItem("📄 Report-Aware"):
            with gr.Row():
                with gr.Column():
                    t2_file = gr.File(label="Upload Image / DICOM",
                                      file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t2_rep  = gr.Textbox(label="Prior Radiology Report", lines=4)
                    t2_qst  = gr.Textbox(label="Question",
                                         placeholder="How does this compare to the prior?")
                    t2_btn  = gr.Button("▶ Analyse", variant="primary")
                with gr.Column():
                    t2_ans = gr.Textbox(label="Answer", interactive=False)
            t2_btn.click(tab2_report_aware, [t2_file, t2_rep, t2_qst], t2_ans)

        # ── Tab 3: Context-Aware ──────────────────────────────────────────
        with gr.TabItem("🧪 Context-Aware"):
            with gr.Row():
                with gr.Column():
                    t3_file = gr.File(label="Upload Image / DICOM",
                                      file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t3_ctx  = gr.Textbox(label="Clinical Notes / Vitals / Labs", lines=4)
                    t3_qst  = gr.Textbox(label="Question",
                                         placeholder="What is the most likely diagnosis?")
                    t3_btn  = gr.Button("▶ Analyse", variant="primary")
                with gr.Column():
                    t3_ans = gr.Textbox(label="Answer", interactive=False)
            t3_btn.click(tab3_context_aware, [t3_file, t3_ctx, t3_qst], t3_ans)

        # ── Tab 4: Longitudinal ───────────────────────────────────────────
        with gr.TabItem("📅 Longitudinal View"):
            with gr.Row():
                with gr.Column():
                    t4_file1 = gr.File(label="Current Image / DICOM",
                                       file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t4_file2 = gr.File(label="Prior Image / DICOM",
                                       file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t4_qst   = gr.Textbox(label="Question",
                                          placeholder="Is the nodule growing?")
                    t4_btn   = gr.Button("▶ Compare", variant="primary")
                with gr.Column():
                    t4_ans = gr.Textbox(label="Comparative Analysis", interactive=False)
            t4_btn.click(tab4_longitudinal, [t4_file1, t4_file2, t4_qst], t4_ans)

        # ── Tab 5: Differential Diagnosis ─────────────────────────────────
        with gr.TabItem("⚕️ Differential Diagnosis"):
            with gr.Row():
                with gr.Column():
                    t5_file = gr.File(label="Upload Image / DICOM",
                                      file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t5_qst  = gr.Textbox(label="Question",
                                         placeholder="What are the differential diagnoses?")
                    t5_btn  = gr.Button("▶ Rank Diagnoses", variant="primary")
                with gr.Column():
                    t5_ans = gr.Markdown("### Results will appear here")
            t5_btn.click(tab5_differential, [t5_file, t5_qst], t5_ans)

        # ── Tab 6: One-Click Report ───────────────────────────────────────
        with gr.TabItem("📝 One-Click Report"):
            with gr.Row():
                with gr.Column():
                    t6_file = gr.File(label="Upload Image / DICOM",
                                      file_types=[".dcm", ".png", ".jpg", ".jpeg"])
                    t6_btn  = gr.Button("▶ Generate Draft Report", variant="primary")
                with gr.Column():
                    t6_ans = gr.Textbox(label="Auto-Generated Draft Report",
                                        lines=12, interactive=False)
            t6_btn.click(tab6_one_click, [t6_file], t6_ans)


if __name__ == "__main__":
    demo.launch(server_port=Config.GRADIO_PORT, server_name="0.0.0.0")
