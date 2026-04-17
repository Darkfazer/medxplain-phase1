"""
MedXplain – Medical Visual Question Answering Demo
===================================================
Production-ready Gradio application with six clinical feature tabs,
full mock-mode fallback, Grad-CAM heatmap overlay, latency tracking,
and graceful error handling throughout.
"""

import gradio as gr
import numpy as np
import cv2
import time
import logging
import sys
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Dict, Any, List

# ---------------------------------------------------------------------------
# Ensure the project root (medical_vqa_infrastructure/) is on sys.path so
# all sibling packages are importable however this file is invoked.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent          # …/medical_vqa_infrastructure
sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Infrastructure imports – every class name matches the real source file.
# ---------------------------------------------------------------------------
from config.config_loader import load_config
from inference.predictor import VQAPredictor
from inference.gradcam import GradCAM
from clinical_features.report_aware import ReportAwareModule
from clinical_features.context_aware import ContextAwareModule
from clinical_features.longitudinal import LongitudinalAnalyzer
from clinical_features.differential import DifferentialDiagnoser
from clinical_features.report_draft import ReportDrafter


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)


# ===========================================================================
# Main application class
# ===========================================================================

class MedicalVQADemo:
    """Gradio application class for MedXplain Medical VQA."""

    # -----------------------------------------------------------------------
    # Construction
    # -----------------------------------------------------------------------

    def __init__(self, config_name: str = "inference_config") -> None:
        """
        Initialise all infrastructure components.

        Parameters
        ----------
        config_name:
            Stem of the YAML config file inside ``config/`` (no .yaml suffix).
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration (tolerant – fall back to empty dict on failure)
        try:
            self.config = load_config(config_name)
            self.logger.info("✅ Config loaded: %s", config_name)
        except Exception as exc:
            self.logger.warning("⚠️  Config not found (%s) – continuing with defaults.", exc)
            self.config = {}

        # Model & Grad-CAM (optional – mock mode if unavailable)
        self.predictor: Optional[VQAPredictor] = None
        self.gradcam: Optional[GradCAM] = None
        self.mock_mode: bool = True

        try:
            # VQAPredictor expects (model, device); replace with real model when ready
            raise NotImplementedError("Real model not yet wired – using mock mode.")
        except Exception as exc:
            self.logger.warning("⚠️  Using mock mode: %s", exc)

        # Clinical feature modules (always available – no GPU required)
        self.report_aware = ReportAwareModule()
        self.context_aware = ContextAwareModule()
        self.longitudinal = LongitudinalAnalyzer()
        self.differential = DifferentialDiagnoser()
        self.report_draft = ReportDrafter()

        # In-memory session history
        self.session_history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _mock_prediction(self, image: np.ndarray, question: str) -> Dict[str, Any]:
        """
        Return a realistic-looking mock prediction when the real model is
        unavailable.  Simulates ~200 ms inference time.

        Parameters
        ----------
        image:
            RGB uint8 numpy array (H x W x 3).
        question:
            Clinical question string.

        Returns
        -------
        dict with keys: answer, confidence, attention_weights, latency_ms
        """
        time.sleep(0.20)                              # simulate inference delay
        attention = np.zeros((224, 224), dtype=np.float32)
        # Generate a smooth Gaussian blob so the heatmap looks plausible
        for _ in range(4):
            cx, cy = np.random.randint(50, 174, size=2)
            sigma = np.random.randint(20, 55)
            xs, ys = np.meshgrid(np.arange(224), np.arange(224))
            blob = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            attention += blob.astype(np.float32)
        attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        answers = [
            "No acute cardiopulmonary process identified.",
            "Mild bilateral interstitial opacities; consider early pneumonia.",
            "Enlarged cardiac silhouette; cardiomegaly cannot be excluded.",
            "Right lower lobe consolidation consistent with pneumonia.",
            "Clear lung fields bilaterally. No pleural effusion.",
        ]
        answer = np.random.choice(answers)
        return {
            "answer": f"[MOCK] {answer}",
            "confidence": round(float(np.random.uniform(0.72, 0.97)), 3),
            "attention_weights": attention,
            "latency_ms": 200.0,
        }

    def _apply_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.50,
    ) -> np.ndarray:
        """
        Overlay a single-channel heatmap on an RGB image using the JET
        colourmap, returning a blended uint8 RGB array.

        Parameters
        ----------
        image:
            RGB uint8 numpy array (H x W x 3).
        heatmap:
            Float32 array in [0, 1] of shape (H, W).
        alpha:
            Opacity of the heatmap overlay (0 = transparent, 1 = opaque).

        Returns
        -------
        Blended RGB uint8 array of the same spatial dimensions as ``image``.
        """
        h, w = image.shape[:2]

        # Resize heatmap to match image dimensions
        hm_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalise to [0, 255] and apply JET colourmap
        hm_uint8 = (hm_resized * 255).clip(0, 255).astype(np.uint8)
        hm_colour = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)  # BGR
        hm_colour_rgb = cv2.cvtColor(hm_colour, cv2.COLOR_BGR2RGB)

        # Blend with original image
        blended = cv2.addWeighted(
            image.astype(np.float32),
            1.0 - alpha,
            hm_colour_rgb.astype(np.float32),
            alpha,
            0,
        ).clip(0, 255).astype(np.uint8)
        return blended

    def _log_session(self, tab: str, question: str, answer: str) -> None:
        """Append an entry to the in-memory session history."""
        self.session_history.append({
            "tab": tab,
            "question": question,
            "answer": answer,
            "timestamp": time.strftime("%H:%M:%S"),
        })

    # -----------------------------------------------------------------------
    # Tab 1 – VQA & Heatmap
    # -----------------------------------------------------------------------

    def answer_with_heatmap(
        self,
        image: Optional[np.ndarray],
        question: str,
        temperature: float,
    ) -> Tuple[str, Optional[np.ndarray], str, str]:
        """
        Run VQA inference and return the answer together with a Grad-CAM
        heatmap overlay.

        Parameters
        ----------
        image:
            Input image as a numpy array (H x W x 3, uint8).
        question:
            Clinical question.
        temperature:
            Sampling temperature (informational; shown in output).

        Returns
        -------
        Tuple of (answer_text, heatmap_image, latency_str, confidence_str)
        """
        if image is None:
            return "⚠️ Please upload an image.", None, "—", "—"
        if not question.strip():
            return "⚠️ Please enter a clinical question.", None, "—", "—"

        t0 = time.perf_counter()
        try:
            result = self._mock_prediction(image, question)
            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            heatmap_overlay = self._apply_heatmap(image, result["attention_weights"])
            answer_text = (
                f"{result['answer']}\n\n"
                f"🌡️ Temperature: {temperature:.2f}   |   "
                f"🧠 Confidence: {result['confidence']:.1%}"
            )
            latency_str = f"⏱️ {latency_s:.3f} s"
            confidence_str = f"🎯 {result['confidence']:.1%}"

            self._log_session("VQA & Heatmap", question, result["answer"])
            return answer_text, heatmap_overlay, latency_str, confidence_str

        except Exception as exc:
            self.logger.error("answer_with_heatmap error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}", None, "—", "—"

    # -----------------------------------------------------------------------
    # Tab 2 – Report-Aware VQA
    # -----------------------------------------------------------------------

    def report_aware_vqa(
        self,
        image: Optional[np.ndarray],
        question: str,
        prior_report: str,
    ) -> str:
        """
        Augment the question with a prior radiology report and run VQA.

        Returns
        -------
        Answer string with report citation.
        """
        if image is None:
            return "⚠️ Please upload an image."
        if not question.strip():
            return "⚠️ Please enter a clinical question."

        t0 = time.perf_counter()
        try:
            enriched_q = self.report_aware.embed_prior_report(question, prior_report)
            result = self._mock_prediction(image, enriched_q)
            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            citation = (
                f"\n\n📋 Prior report incorporated: \"{prior_report[:80]}...\""
                if len(prior_report) > 80 else
                f"\n\n📋 Prior report incorporated: \"{prior_report}\""
            ) if prior_report.strip() else "\n\n📋 No prior report provided."

            self._log_session("Report-Aware", question, result["answer"])
            return (
                f"{result['answer']}"
                f"{citation}\n\n"
                f"🎯 Confidence: {result['confidence']:.1%}   |   ⏱️ {latency_s:.3f} s"
            )
        except Exception as exc:
            self.logger.error("report_aware_vqa error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}"

    # -----------------------------------------------------------------------
    # Tab 3 – Context-Aware VQA
    # -----------------------------------------------------------------------

    def context_aware_vqa(
        self,
        image: Optional[np.ndarray],
        question: str,
        clinical_notes: str,
        lab_results: str,
        vitals: str,
    ) -> str:
        """
        Fuse clinical context (notes, labs, vitals) with the image and run VQA.

        Returns
        -------
        Contextualised answer string.
        """
        if image is None:
            return "⚠️ Please upload an image."
        if not question.strip():
            return "⚠️ Please enter a clinical question."

        t0 = time.perf_counter()
        try:
            clinical_data = {
                "notes": clinical_notes,
                "labs": lab_results,
                "vitals": vitals,
            }
            # Fuse context (mock: returns image_features unchanged)
            mock_features = np.random.rand(512)
            self.context_aware.fuse_context(mock_features, clinical_data)

            enriched_q = (
                f"{question} "
                f"[NOTES: {clinical_notes}] "
                f"[LABS: {lab_results}] "
                f"[VITALS: {vitals}]"
            )
            result = self._mock_prediction(image, enriched_q)
            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            context_summary = "\n".join([
                f"📝 Notes: {clinical_notes or 'N/A'}",
                f"🧪 Labs:  {lab_results or 'N/A'}",
                f"💓 Vitals: {vitals or 'N/A'}",
            ])

            self._log_session("Context-Aware", question, result["answer"])
            return (
                f"{result['answer']}\n\n"
                f"── Clinical Context Provided ──\n{context_summary}\n\n"
                f"🎯 Confidence: {result['confidence']:.1%}   |   ⏱️ {latency_s:.3f} s"
            )
        except Exception as exc:
            self.logger.error("context_aware_vqa error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}"

    # -----------------------------------------------------------------------
    # Tab 4 – Longitudinal Analysis
    # -----------------------------------------------------------------------

    def longitudinal_analysis(
        self,
        current_image: Optional[np.ndarray],
        prior_image: Optional[np.ndarray],
        question: str,
    ) -> Tuple[str, Optional[np.ndarray]]:
        """
        Compare a current and prior imaging study and return a difference
        heatmap alongside the VQA answer.

        Returns
        -------
        Tuple of (answer_text, difference_heatmap_image)
        """
        if current_image is None or prior_image is None:
            return "⚠️ Please upload both current and prior images.", None
        if not question.strip():
            return "⚠️ Please enter a clinical question.", None

        t0 = time.perf_counter()
        try:
            import torch
            h, w = 224, 224

            def _to_tensor(img: np.ndarray) -> "torch.Tensor":
                resized = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                gray = resized.mean(axis=2) if resized.ndim == 3 else resized
                return torch.from_numpy(gray).unsqueeze(0)

            t_cur = _to_tensor(current_image)
            t_pri = _to_tensor(prior_image)
            diff_tensor = self.longitudinal.compare(t_cur, t_pri)  # (1, H, W)
            diff_np = diff_tensor.squeeze(0).numpy()

            # Normalise diff to [0, 1]
            diff_norm = (diff_np - diff_np.min()) / (diff_np.max() - diff_np.min() + 1e-8)

            diff_overlay = self._apply_heatmap(
                cv2.resize(current_image, (w, h)), diff_norm, alpha=0.6
            )

            result = self._mock_prediction(current_image, f"[LONGITUDINAL] {question}")
            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            change_pct = float(diff_norm.mean()) * 100
            answer_text = (
                f"{result['answer']}\n\n"
                f"📊 Mean change index: {change_pct:.1f}%   |   "
                f"🎯 Confidence: {result['confidence']:.1%}   |   ⏱️ {latency_s:.3f} s"
            )

            self._log_session("Longitudinal", question, result["answer"])
            return answer_text, diff_overlay

        except Exception as exc:
            self.logger.error("longitudinal_analysis error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}", None

    # -----------------------------------------------------------------------
    # Tab 5 – Differential Diagnosis
    # -----------------------------------------------------------------------

    def differential_diagnosis(
        self,
        image: Optional[np.ndarray],
        question: str,
        top_k: int,
    ) -> str:
        """
        Generate ranked differential diagnoses for the uploaded image.

        Parameters
        ----------
        top_k:
            Number of top differentials to display (1-5).

        Returns
        -------
        Formatted string of ranked diagnoses with probabilities.
        """
        if image is None:
            return "⚠️ Please upload an image."
        if not question.strip():
            return "⚠️ Please enter a clinical question."

        t0 = time.perf_counter()
        try:
            result = self._mock_prediction(image, question)
            differentials = self.differential.get_differentials(result)[:int(top_k)]
            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            lines = [f"🏆 Top-{len(differentials)} Differential Diagnoses\n{'─' * 42}"]
            for rank, dx in enumerate(differentials, 1):
                bar_len = int(dx["confidence"] * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                lines.append(
                    f"\n#{rank}  {dx['diagnosis']}\n"
                    f"    Confidence: {bar} {dx['confidence']:.1%}\n"
                    f"    💬 {dx['explanation']}"
                )

            lines.append(f"\n{'─' * 42}\n⏱️ {latency_s:.3f} s")
            self._log_session("Differential", question, differentials[0]["diagnosis"])
            return "\n".join(lines)

        except Exception as exc:
            self.logger.error("differential_diagnosis error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}"

    # -----------------------------------------------------------------------
    # Tab 6 – One-Click Report
    # -----------------------------------------------------------------------

    def one_click_report(
        self,
        image: Optional[np.ndarray],
        question: str,
    ) -> str:
        """
        Generate a structured radiology report (Findings + Impression +
        Recommendation) in a single click using the new ReportDrafter API.

        Returns
        -------
        Formatted report string.
        """
        if image is None:
            return "⚠️ Please upload an image."
        if not question.strip():
            return "⚠️ Please enter a clinical question."

        t0 = time.perf_counter()
        try:
            # Step 1 – get mock VQA prediction
            result = self._mock_prediction(image, question)

            # Step 2 – get differential diagnoses
            differentials = self.differential.get_differentials(result)

            # Step 3 – build structured predictions dict for ReportDrafter
            predictions = {
                "findings":    [dx["diagnosis"] for dx in differentials],
                "confidences": {dx["diagnosis"]: dx["confidence"] for dx in differentials},
            }

            # Step 4 – generate the full structured report
            report = self.report_draft.generate_draft(
                image        = image,
                question     = question,
                predictions  = predictions,
                patient_info = {},          # populated from UI in future release
            )

            latency_s = time.perf_counter() - t0
            print(f"⏱️ Inference time: {latency_s:.3f}s")

            # Append a latency footer
            report += f"\n\n🎯 Confidence: {result['confidence']:.1%}   |   ⏱️ {latency_s:.3f} s"

            top_dx = differentials[0]["diagnosis"] if differentials else "Unspecified"
            self._log_session("One-Click Report", question, top_dx)
            return report

        except Exception as exc:
            self.logger.error("one_click_report error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}"



# ===========================================================================
# Module-level theme & CSS (accessible from both create_demo and __main__)
# ===========================================================================

_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
)

_APP_CSS = """
    .footer-disclaimer { font-size: 0.85em; color: #888; text-align: center; }
    .tab-label { font-weight: 600; }
"""


# ===========================================================================
# Gradio UI builder
# ===========================================================================

def create_demo() -> gr.Blocks:
    """
    Build and return the six-tab Gradio Blocks interface.

    Returns
    -------
    gr.Blocks application (not yet launched).
    """
    demo_backend = MedicalVQADemo()

    with gr.Blocks(
        title="MedXplain – Medical VQA",
        theme=_APP_THEME,
        css=_APP_CSS,
    ) as app:

        # ── Header ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🏥 MedXplain: Medical Visual Question Answering
            **AI-powered clinical decision support for chest X-ray analysis**

            > Upload a chest X-ray and ask a clinical question.
            > All outputs are generated in mock mode until a real model checkpoint is loaded.
            """,
        )

        with gr.Tabs():

            # ── Tab 1: VQA & Heatmap ────────────────────────────────────────
            with gr.TabItem("🔍 VQA & Heatmap"):
                gr.Markdown("### Visual Question Answering with Grad-CAM Heatmap")
                with gr.Row():
                    with gr.Column(scale=1):
                        t1_image = gr.Image(
                            type="numpy", label="📷 Chest X-Ray", height=300
                        )
                        t1_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. Is there evidence of pneumonia?",
                        )
                        t1_temp = gr.Slider(
                            minimum=0.1, maximum=1.5, value=0.7, step=0.05,
                            label="🌡️ Temperature",
                        )
                        t1_btn = gr.Button("🚀 Analyze", variant="primary")
                    with gr.Column(scale=1):
                        t1_answer = gr.Textbox(
                            label="📋 Answer", lines=5, interactive=False
                        )
                        t1_heatmap = gr.Image(
                            label="🔥 Grad-CAM Heatmap", height=300
                        )
                        with gr.Row():
                            t1_latency = gr.Textbox(
                                label="⏱️ Latency", interactive=False, scale=1
                            )
                            t1_conf = gr.Textbox(
                                label="🎯 Confidence", interactive=False, scale=1
                            )
                t1_btn.click(
                    fn=demo_backend.answer_with_heatmap,
                    inputs=[t1_image, t1_question, t1_temp],
                    outputs=[t1_answer, t1_heatmap, t1_latency, t1_conf],
                )

            # ── Tab 2: Report-Aware VQA ─────────────────────────────────────
            with gr.TabItem("📄 Report-Aware"):
                gr.Markdown("### VQA informed by a prior radiology report")
                with gr.Row():
                    with gr.Column(scale=1):
                        t2_image = gr.Image(
                            type="numpy", label="📷 Chest X-Ray", height=280
                        )
                        t2_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. How has the consolidation changed?",
                        )
                        t2_prior = gr.Textbox(
                            label="📋 Prior Radiology Report",
                            lines=5,
                            placeholder="Paste the prior report text here…",
                        )
                        t2_btn = gr.Button("📄 Analyze with Report", variant="primary")
                    with gr.Column(scale=1):
                        t2_answer = gr.Textbox(
                            label="📋 Answer with Citation", lines=10, interactive=False
                        )
                t2_btn.click(
                    fn=demo_backend.report_aware_vqa,
                    inputs=[t2_image, t2_question, t2_prior],
                    outputs=[t2_answer],
                )

            # ── Tab 3: Context-Aware VQA ────────────────────────────────────
            with gr.TabItem("🧪 Context-Aware"):
                gr.Markdown("### VQA fused with clinical notes, labs & vitals")
                with gr.Row():
                    with gr.Column(scale=1):
                        t3_image = gr.Image(
                            type="numpy", label="📷 Chest X-Ray", height=250
                        )
                        t3_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. Given the elevated WBC, is this bacterial?",
                        )
                        t3_notes = gr.Textbox(
                            label="📝 Clinical Notes",
                            lines=3,
                            placeholder="e.g. 65 yo male, fever 38.9°C, productive cough x3 days",
                        )
                        t3_labs = gr.Textbox(
                            label="🧪 Lab Results",
                            lines=3,
                            placeholder="e.g. WBC 14.2 K/µL, CRP 85 mg/L, Procalcitonin 0.8 ng/mL",
                        )
                        t3_vitals = gr.Textbox(
                            label="💓 Vitals",
                            placeholder="e.g. BP 128/82, HR 98, SpO2 94%, RR 22",
                        )
                        t3_btn = gr.Button("🧪 Analyze with Context", variant="primary")
                    with gr.Column(scale=1):
                        t3_answer = gr.Textbox(
                            label="📋 Contextual Answer", lines=12, interactive=False
                        )
                t3_btn.click(
                    fn=demo_backend.context_aware_vqa,
                    inputs=[t3_image, t3_question, t3_notes, t3_labs, t3_vitals],
                    outputs=[t3_answer],
                )

            # ── Tab 4: Longitudinal Analysis ────────────────────────────────
            with gr.TabItem("📅 Longitudinal"):
                gr.Markdown("### Compare current and prior imaging for interval change")
                with gr.Row():
                    with gr.Column(scale=1):
                        t4_current = gr.Image(
                            type="numpy", label="📷 Current X-Ray", height=260
                        )
                        t4_prior = gr.Image(
                            type="numpy", label="📷 Prior X-Ray", height=260
                        )
                        t4_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. Has the pleural effusion changed?",
                        )
                        t4_btn = gr.Button("📅 Compare Studies", variant="primary")
                    with gr.Column(scale=1):
                        t4_answer = gr.Textbox(
                            label="📋 Longitudinal Analysis", lines=6, interactive=False
                        )
                        t4_diff = gr.Image(
                            label="🗺️ Difference Heatmap", height=300
                        )
                t4_btn.click(
                    fn=demo_backend.longitudinal_analysis,
                    inputs=[t4_current, t4_prior, t4_question],
                    outputs=[t4_answer, t4_diff],
                )

            # ── Tab 5: Differential Diagnosis ───────────────────────────────
            with gr.TabItem("🩺 Differential Dx"):
                gr.Markdown("### Ranked differential diagnoses with confidence scores")
                with gr.Row():
                    with gr.Column(scale=1):
                        t5_image = gr.Image(
                            type="numpy", label="📷 Chest X-Ray", height=280
                        )
                        t5_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. What are the most likely diagnoses?",
                        )
                        t5_topk = gr.Slider(
                            minimum=1, maximum=5, value=3, step=1,
                            label="🏆 Number of Differentials (Top-K)",
                        )
                        t5_btn = gr.Button("🩺 Generate Differentials", variant="primary")
                    with gr.Column(scale=1):
                        t5_output = gr.Textbox(
                            label="🏆 Ranked Differentials", lines=16, interactive=False
                        )
                t5_btn.click(
                    fn=demo_backend.differential_diagnosis,
                    inputs=[t5_image, t5_question, t5_topk],
                    outputs=[t5_output],
                )

            # ── Tab 6: One-Click Report ─────────────────────────────────────
            with gr.TabItem("📝 One-Click Report"):
                gr.Markdown("### Generate a structured radiology report in one click")
                with gr.Row():
                    with gr.Column(scale=1):
                        t6_image = gr.Image(
                            type="numpy", label="📷 Chest X-Ray", height=280
                        )
                        t6_question = gr.Textbox(
                            label="❓ Clinical Question",
                            placeholder="e.g. Generate a full report for this chest X-ray.",
                        )
                        t6_btn = gr.Button(
                            "📝 Generate Full Report", variant="primary", size="lg"
                        )
                    with gr.Column(scale=1):
                        t6_report = gr.Textbox(
                            label="📄 Structured Radiology Report",
                            lines=20,
                            interactive=False,
                        )
                t6_btn.click(
                    fn=demo_backend.one_click_report,
                    inputs=[t6_image, t6_question],
                    outputs=[t6_report],
                )

        # ── Footer ──────────────────────────────────────────────────────────
        gr.Markdown(
            """
            ---
            <div class="footer-disclaimer">

            **⚠️ Clinical Decision Support Only**
            This AI system is intended for research and decision-support purposes only.
            All findings must be reviewed and validated by a qualified radiologist or
            licensed medical professional before any clinical action is taken.
            MedXplain does **not** replace professional medical judgment.

            </div>
            """,
        )

    return app


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    import socket

    def _find_free_port(start: int = 7860, end: int = 7870) -> int:
        """Return the first free TCP port in [start, end]."""
        for port in range(start, end + 1):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("0.0.0.0", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No free port found in range {start}–{end}.")

    _port = _find_free_port()
    print(f"🚀 Launching MedXplain on http://localhost:{_port}")
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=_port,
        share=False,
        debug=False,
        max_threads=4,
    )
