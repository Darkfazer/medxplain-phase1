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
# Force UTF-8 on Windows consoles (cp1252 cannot encode emoji used in logs).
# ---------------------------------------------------------------------------
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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

    # -----------------------------------------------------------------------
    # Context-Aware helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_value(text: str, label: str) -> str:
        """
        Extract the first numeric value that immediately follows *label* in
        *text*, e.g. ``_extract_value("WBC 14.2, ...", "WBC")`` → ``"14.2"``.
        Returns ``"N/A"`` if the label is not found.
        """
        import re
        pattern = rf"{re.escape(label)}[:\s]*([\d]+\.?[\d]*)"
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1) if m else "N/A"

    def _mock_context_aware_answer(
        self,
        clinical_notes: str,
        lab_results: str,
        vitals: str,
    ) -> Tuple[str, float]:
        """
        Generate an intelligent mock VQA answer driven entirely by the supplied
        clinical context (notes, labs, vitals).

        Returns
        -------
        Tuple of (answer_text, confidence_0_to_100)
        """
        import re
        import time as _time

        notes_lc  = clinical_notes.lower()
        labs_lc   = lab_results.lower()
        vitals_lc = vitals.lower()

        # ── Parse vitals ────────────────────────────────────────────────────
        has_fever   = False
        has_hypoxia = False
        has_tachy_hr = False
        has_tachy_rr = False

        fever_val = self._extract_value(vitals, "Temperature")
        if fever_val == "N/A":
            fever_val = self._extract_value(vitals, "Temp")
        try:
            if float(fever_val) > 38.0:
                has_fever = True
        except ValueError:
            pass
        if "fever" in notes_lc or "fever" in vitals_lc:
            has_fever = True
        # Accept bare "38.x" or "39.x" patterns anywhere in notes/vitals
        if re.search(r"\b3[89]\.\d", notes_lc + vitals_lc):
            has_fever = True

        spo2_val = self._extract_value(vitals, "SpO2")
        try:
            if float(spo2_val) < 95:
                has_hypoxia = True
        except ValueError:
            pass

        hr_val = self._extract_value(vitals, "HR")
        if hr_val == "N/A":
            hr_val = self._extract_value(vitals, "Heart Rate")
        try:
            if float(hr_val) > 100:
                has_tachy_hr = True
        except ValueError:
            pass

        rr_val = self._extract_value(vitals, "RR")
        if rr_val == "N/A":
            rr_val = self._extract_value(vitals, "Respiratory Rate")
        try:
            if float(rr_val) > 20:
                has_tachy_rr = True
        except ValueError:
            pass

        # ── Parse labs ──────────────────────────────────────────────────────
        wbc_elevated    = False
        procal_elevated = False
        procal_high     = False   # >0.5 → highly likely bacterial
        crp_elevated    = False
        neutro_elevated = False

        wbc_val = self._extract_value(lab_results, "WBC")
        try:
            if float(wbc_val) > 11.0:
                wbc_elevated = True
        except ValueError:
            pass

        neutro_val = self._extract_value(lab_results, "Neutrophils")
        if neutro_val == "N/A":
            neutro_val = self._extract_value(lab_results, "Neutrophil")
        try:
            if float(neutro_val) > 70:
                neutro_elevated = True
        except ValueError:
            pass

        procal_val = self._extract_value(lab_results, "Procalcitonin")
        try:
            pv = float(procal_val)
            if pv > 0.25:
                procal_elevated = True
            if pv > 0.5:
                procal_high = True
        except ValueError:
            pass

        crp_val = self._extract_value(lab_results, "CRP")
        try:
            if float(crp_val) > 10:
                crp_elevated = True
        except ValueError:
            pass

        # ── Parse clinical notes keywords ────────────────────────────────────
        has_cough      = "cough" in notes_lc
        has_productive = "productive" in notes_lc
        has_sob        = any(k in notes_lc for k in ["shortness of breath", "dyspnea", "breathless"])
        has_crackles   = any(k in notes_lc for k in ["crackles", "rales"])

        # ── Build formatted lab/vital summaries ────────────────────────────
        def _fmt_labs() -> str:
            parts = []
            if wbc_val != "N/A":
                parts.append(f"WBC {wbc_val}{' ↑' if wbc_elevated else ''}")
            if neutro_val != "N/A":
                parts.append(f"Neutrophils {neutro_val}%{' ↑' if neutro_elevated else ''}")
            if crp_val != "N/A":
                parts.append(f"CRP {crp_val}{' ↑' if crp_elevated else ''}")
            if procal_val != "N/A":
                parts.append(f"Procalcitonin {procal_val}{' ↑' if procal_elevated else ''}")
            return ", ".join(parts) if parts else (lab_results or "N/A")

        def _fmt_vitals() -> str:
            parts = []
            if fever_val != "N/A":
                parts.append(f"Temp {fever_val}°C{' ↑' if has_fever else ''}")
            if spo2_val != "N/A":
                parts.append(f"SpO2 {spo2_val}%{' ↓' if has_hypoxia else ''}")
            if hr_val != "N/A":
                parts.append(f"HR {hr_val}{' ↑' if has_tachy_hr else ''}")
            if rr_val != "N/A":
                parts.append(f"RR {rr_val}{' ↑' if has_tachy_rr else ''}")
            return ", ".join(parts) if parts else (vitals or "N/A")

        # ── Decision logic – most-specific pattern first ─────────────────────
        answer: str
        confidence: float

        # Pattern A: strong bacterial pneumonia signal
        if procal_high and wbc_elevated and has_fever:
            severity = "MODERATE to SEVERE" if has_hypoxia else "MILD to MODERATE"
            hypoxia_note = (
                f"\n• Hypoxia (SpO2 {spo2_val}%) → suggests significant disease burden"
                if has_hypoxia else ""
            )
            answer = (
                f"FINDINGS CONSISTENT WITH BACTERIAL PNEUMONIA.\n\n"
                f"Key supporting clinical data:\n"
                f"• Elevated WBC ({wbc_val}) with neutrophilia ({neutro_val}%) → bacterial pattern\n"
                f"• Elevated Procalcitonin ({procal_val} ng/mL) → highly specific for bacterial pneumonia\n"
                f"• Fever ({fever_val}°C) and productive cough → active infectious process"
                f"{hypoxia_note}\n\n"
                f"Impression: {severity} community-acquired pneumonia.\n\n"
                f"Recommendation: Community-acquired pneumonia coverage with antibiotics.\n"
                f"Follow-up chest X-ray in 4–6 weeks to ensure resolution."
            )
            confidence = 94.0

        # Pattern B: moderate bacterial signal (elevated Procalcitonin but WBC borderline)
        elif procal_elevated and has_fever and (wbc_elevated or neutro_elevated or crp_elevated):
            answer = (
                f"FINDINGS CONSISTENT WITH BACTERIAL PNEUMONIA.\n\n"
                f"Key supporting clinical data:\n"
                f"• Procalcitonin elevated ({procal_val} ng/mL) → bacterial etiology likely\n"
                f"• Inflammatory markers elevated (WBC {wbc_val}, CRP {crp_val}) → active infection\n"
                f"• Fever ({fever_val}°C) → systemic infectious response\n\n"
                f"Recommendation: Antibiotic therapy indicated. Clinical correlation advised."
            )
            confidence = 87.0

        # Pattern C: hypoxia + fever + elevated WBC – severity focus
        elif has_hypoxia and has_fever and wbc_elevated:
            answer = (
                f"MODERATE TO SEVERE PNEUMONIA.\n\n"
                f"Hypoxia (SpO2 {spo2_val}%) combined with fever ({fever_val}°C) and leukocytosis "
                f"(WBC {wbc_val}) suggests significant pulmonary infection with impaired gas exchange.\n\n"
                f"Recommendation: Assess need for supplemental oxygen. Hospitalisation likely required. "
                f"Empiric broad-spectrum antibiotics recommended."
            )
            confidence = 89.0

        # Pattern D: fever + cough but normal Procalcitonin → viral/atypical
        elif has_fever and (has_cough or has_sob) and not procal_elevated:
            procal_note = (
                f"Procalcitonin {procal_val} ng/mL (normal) argues against typical bacterial aetiology."
                if procal_val != "N/A"
                else "Procalcitonin not available; viral aetiology cannot be excluded."
            )
            answer = (
                f"Possible VIRAL PNEUMONIA or ATYPICAL PNEUMONIA.\n\n"
                f"{procal_note}\n"
                f"Clinical correlation with viral panel (COVID-19, Influenza, RSV, Mycoplasma) recommended.\n\n"
                f"Recommendation: Consider antiviral therapy if influenza confirmed. "
                f"Watchful waiting appropriate if well-oxygenated."
            )
            confidence = 78.0

        # Pattern E: shortness of breath + hypoxia + normal WBC (no infection)
        elif has_hypoxia and has_sob and not has_fever and not wbc_elevated:
            answer = (
                f"Possible PULMONARY EMBOLISM or INTERSTITIAL LUNG DISEASE.\n\n"
                f"Hypoxia (SpO2 {spo2_val}%) in the absence of fever or significant leukocytosis "
                f"raises concern for non-infectious aetiology.\n\n"
                f"Recommendation: Obtain D-dimer; consider CT pulmonary angiography if pre-test "
                f"probability is moderate-to-high. High-resolution CT may be warranted for ILD evaluation."
            )
            confidence = 65.0

        # Pattern F: crackles on exam + fever + elevated WBC → CAP
        elif has_crackles and has_fever and wbc_elevated:
            answer = (
                f"COMMUNITY-ACQUIRED PNEUMONIA.\n\n"
                f"Clinical findings (crackles, fever {fever_val}°C, WBC {wbc_val}) "
                f"are consistent with community-acquired pneumonia.\n\n"
                f"Recommendation: Antibiotics and follow-up chest X-ray in 4–6 weeks."
            )
            confidence = 85.0

        # Pattern G: no significant abnormalities
        elif not has_fever and not wbc_elevated and not has_hypoxia and not procal_elevated:
            answer = "No acute cardiopulmonary process identified. Clinical and laboratory parameters within normal limits."
            confidence = 88.0

        # Fallback: some abnormalities but no clear pattern
        else:
            flags = []
            if has_fever:
                flags.append(f"fever ({fever_val}°C)")
            if wbc_elevated:
                flags.append(f"leukocytosis (WBC {wbc_val})")
            if has_hypoxia:
                flags.append(f"hypoxia (SpO2 {spo2_val}%)")
            if crp_elevated:
                flags.append(f"elevated CRP ({crp_val})")
            flag_str = ", ".join(flags) if flags else "non-specific findings"
            answer = (
                f"Indeterminate findings. Clinical context suggests {flag_str}.\n\n"
                f"Recommendation: Clinical correlation and follow-up imaging recommended."
            )
            confidence = 62.0

        return answer, confidence

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
            print(f"[VQA] Inference time: {latency_s:.3f}s")

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
            print(f"[Report-Aware] Inference time: {latency_s:.3f}s")

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
        Fuse clinical context (notes, labs, vitals) with the image and run
        an intelligent mock VQA that analyses the clinical picture.

        The answer is driven by parsed symptoms, lab values, and vitals rather
        than being a random generic response.

        Returns
        -------
        Contextualised answer string with clinical context summary.
        """
        if image is None:
            return "⚠️ Please upload an image."
        if not question.strip():
            return "⚠️ Please enter a clinical question."

        t0 = time.perf_counter()
        try:
            clinical_data = {
                "notes": clinical_notes,
                "labs":  lab_results,
                "vitals": vitals,
            }

            # Fuse context (mock: passes image_features through unchanged)
            mock_features = np.random.rand(512)
            self.context_aware.fuse_context(mock_features, clinical_data)

            # ── Intelligent context-aware answer ────────────────────────────
            has_any_context = any(
                s.strip() for s in [clinical_notes, lab_results, vitals]
            )

            if has_any_context:
                answer, confidence_pct = self._mock_context_aware_answer(
                    clinical_notes, lab_results, vitals
                )
                # Simulate inference latency
                time.sleep(0.15)
            else:
                # No clinical context supplied – fall back to generic mock
                base = self._mock_prediction(image, question)
                answer        = base["answer"]
                confidence_pct = base["confidence"] * 100

            latency_s = time.perf_counter() - t0
            print(f"[Context-Aware] Inference time: {latency_s:.3f}s")

            # ── Build formatted lab / vital summary lines ────────────────────
            def _summary_line(label: str, val: str, unit: str = "") -> str:
                return f"{label}: {val}{unit}" if val.strip() else f"{label}: N/A"

            wbc_str    = self._extract_value(lab_results, "WBC")
            neutro_str = self._extract_value(lab_results, "Neutrophils")
            crp_str    = self._extract_value(lab_results, "CRP")
            procal_str = self._extract_value(lab_results, "Procalcitonin")
            temp_str   = self._extract_value(vitals, "Temperature") or self._extract_value(vitals, "Temp")
            spo2_str   = self._extract_value(vitals, "SpO2")

            def _flag(raw: str, threshold: float, direction: str = "high") -> str:
                try:
                    v = float(raw)
                    if direction == "high" and v > threshold:
                        return f"{raw} ↑"
                    if direction == "low" and v < threshold:
                        return f"{raw} ↓"
                except ValueError:
                    pass
                return raw

            labs_fmt = ", ".join(filter(None, [
                f"WBC {_flag(wbc_str, 11.0)}".strip()      if wbc_str    != "N/A" else "",
                f"Neutrophils {_flag(neutro_str, 70.0)}%"  if neutro_str != "N/A" else "",
                f"CRP {_flag(crp_str, 10.0)}".strip()      if crp_str    != "N/A" else "",
                f"Procalcitonin {_flag(procal_str, 0.25)}" if procal_str != "N/A" else "",
            ])) or (lab_results or "N/A")

            vitals_fmt = ", ".join(filter(None, [
                f"Temp {_flag(temp_str, 38.0)}°C"           if temp_str  != "N/A" else "",
                f"SpO2 {_flag(spo2_str, 95.0, 'low')}%"    if spo2_str  != "N/A" else "",
            ])) or (vitals or "N/A")

            # Condense notes to a brief summary
            notes_display = clinical_notes.strip()[:120] + (
                "…" if len(clinical_notes.strip()) > 120 else ""
            ) if clinical_notes.strip() else "N/A"

            context_summary = (
                f"📝 Notes: {notes_display}\n"
                f"🧪 Labs:  {labs_fmt}\n"
                f"💓 Vitals: {vitals_fmt}"
            )

            self._log_session("Context-Aware", question, answer)
            return (
                f"{answer}\n\n"
                f"── Clinical Context Used ──\n"
                f"{context_summary}\n\n"
                f"🎯 Confidence: {confidence_pct:.0f}%   |   ⏱️ {latency_s:.2f} s"
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
            print(f"[Longitudinal] Inference time: {latency_s:.3f}s")

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
            print(f"[Differential] Inference time: {latency_s:.3f}s")

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
            print(f"[Report] Inference time: {latency_s:.3f}s")

            # Append a latency footer
            report += f"\n\n🎯 Confidence: {result['confidence']:.1%}   |   ⏱️ {latency_s:.3f} s"

            top_dx = differentials[0]["diagnosis"] if differentials else "Unspecified"
            self._log_session("One-Click Report", question, top_dx)
            return report

        except Exception as exc:
            self.logger.error("one_click_report error: %s", exc, exc_info=True)
            return f"❌ Error: {exc}"


    # -----------------------------------------------------------------------
    # Integrated Dashboard – single master method that runs ALL modules
    # -----------------------------------------------------------------------

    def integrated_analyze(
        self,
        current_image:   Optional[np.ndarray],
        prior_image:     Optional[np.ndarray],
        question:        str,
        prior_report:    str,
        clinical_notes:  str,
        lab_results:     str,
        vitals:          str,
        temperature:     float,
        top_k:           int,
    ) -> Tuple[str, Optional[np.ndarray], str, str, str, str, Optional[np.ndarray], str, str]:
        """
        Run ALL five clinical modules in one call and return every output.

        Data flow
        ---------
        current_image + prior_report + context  →  enriched VQA prediction
        enriched prediction                     →  differentials
        current + prior images                  →  longitudinal diff heatmap
        all of the above                        →  structured one-click report

        Returns (10 values, matching the 10 Gradio output components)
        -------
        0  vqa_answer       str
        1  heatmap          np.ndarray | None
        2  latency_str      str
        3  confidence_str   str
        4  report_aware_out str
        5  context_out      str
        6  long_heatmap     np.ndarray | None
        7  long_answer      str
        8  diff_out         str
        9  full_report      str
        """
        # Guard: need at least the current image and a question
        if current_image is None:
            empty = "⚠️ Please upload a current chest X-ray."
            return empty, None, "—", "—", empty, empty, None, empty, empty, empty
        if not question.strip():
            empty = "⚠️ Please enter a clinical question."
            return empty, None, "—", "—", empty, empty, None, empty, empty, empty

        t_total = time.perf_counter()

        # ── Step 1: enrich question with prior report ────────────────────────
        enriched_q = question
        report_aware_out = "📋 No prior report provided."
        if prior_report.strip():
            enriched_q = self.report_aware.embed_prior_report(question, prior_report)
            snippet = prior_report[:120] + ("…" if len(prior_report) > 120 else "")
            report_aware_out = (
                f"📋 Prior report incorporated into prompt.\n\n"
                f"Excerpt: \"{ snippet }\""
            )

        # ── Step 2: enrich question with clinical context ────────────────────
        context_out = "🧪 No clinical context provided."
        has_context = any([clinical_notes.strip(), lab_results.strip(), vitals.strip()])
        if has_context:
            enriched_q += (
                f" [NOTES: {clinical_notes}]"
                f" [LABS: {lab_results}]"
                f" [VITALS: {vitals}]"
            )
            context_out = "\n".join([
                f"📝 Notes:  {clinical_notes  or 'N/A'}",
                f"🧪 Labs:   {lab_results    or 'N/A'}",
                f"💓 Vitals: {vitals         or 'N/A'}",
            ])

        # ── Step 3: VQA prediction (with enriched question) ──────────────────
        try:
            result = self._mock_prediction(current_image, enriched_q)
            lat_vqa = result["latency_ms"] / 1000
            heatmap_overlay = self._apply_heatmap(
                current_image, result["attention_weights"]
            )
            vqa_answer = (
                f"{result['answer']}\n\n"
                f"🌡️ Temperature: {temperature:.2f}   |   "
                f"🧠 Confidence: {result['confidence']:.1%}"
            )
            confidence_str = f"🎯 {result['confidence']:.1%}"
            latency_str    = f"⏱️ {lat_vqa:.3f} s"
        except Exception as exc:
            self.logger.error("integrated VQA error: %s", exc, exc_info=True)
            vqa_answer      = f"❌ VQA error: {exc}"
            heatmap_overlay = None
            confidence_str  = "—"
            latency_str     = "—"
            result          = {"confidence": 0.0, "attention_weights": np.zeros((224, 224))}

        # ── Step 4: longitudinal (only if prior image provided) ───────────────
        long_heatmap = None
        long_answer  = "📅 No prior X-ray uploaded – longitudinal analysis skipped."
        if prior_image is not None:
            try:
                import torch
                h, w = 224, 224

                def _to_tensor(img: np.ndarray) -> "torch.Tensor":
                    resized = cv2.resize(img, (w, h)).astype(np.float32) / 255.0
                    gray    = resized.mean(axis=2) if resized.ndim == 3 else resized
                    return torch.from_numpy(gray).unsqueeze(0)

                diff_t   = self.longitudinal.compare(_to_tensor(current_image),
                                                      _to_tensor(prior_image))
                diff_np  = diff_t.squeeze(0).numpy()
                diff_norm = (
                    (diff_np - diff_np.min())
                    / (diff_np.max() - diff_np.min() + 1e-8)
                )
                long_heatmap = self._apply_heatmap(
                    cv2.resize(current_image, (w, h)), diff_norm, alpha=0.6
                )
                change_pct = float(diff_norm.mean()) * 100
                long_answer = (
                    f"📊 Mean change index: {change_pct:.1f}%\n"
                    f"Interval change detected between current and prior study."
                )
            except Exception as exc:
                self.logger.error("integrated longitudinal error: %s", exc, exc_info=True)
                long_answer = f"❌ Longitudinal error: {exc}"

        # ── Step 5: differential diagnosis ───────────────────────────────────
        try:
            differentials = self.differential.get_differentials(result)[:int(top_k)]
            lines = [f"🏆 Top-{len(differentials)} Differential Diagnoses\n{'─' * 38}"]
            for rank, dx in enumerate(differentials, 1):
                bar = "█" * int(dx["confidence"] * 20) + "░" * (20 - int(dx["confidence"] * 20))
                lines.append(
                    f"\n#{rank}  {dx['diagnosis']}\n"
                    f"    {bar} {dx['confidence']:.1%}\n"
                    f"    💬 {dx['explanation']}"
                )
            diff_out = "\n".join(lines)
        except Exception as exc:
            self.logger.error("integrated differential error: %s", exc, exc_info=True)
            diff_out      = f"❌ Differential error: {exc}"
            differentials = []

        # ── Step 6: one-click report using ALL context ────────────────────────
        try:
            predictions = {
                "findings":    [dx["diagnosis"] for dx in differentials],
                "confidences": {dx["diagnosis"]: dx["confidence"] for dx in differentials},
                "comparison":  (
                    long_answer
                    if prior_image is not None
                    else "No prior study available for comparison."
                ),
            }
            if prior_report.strip():
                predictions["impression"] = (
                    f"{differentials[0]['diagnosis']} – informed by prior report context."
                    if differentials else "No active disease identified."
                )

            full_report = self.report_draft.generate_draft(
                image        = current_image,
                question     = question,
                predictions  = predictions,
                patient_info = {},
            )
        except Exception as exc:
            self.logger.error("integrated report error: %s", exc, exc_info=True)
            full_report = f"❌ Report error: {exc}"

        total_s = time.perf_counter() - t_total
        print(f"⏱️ [Integrated] Pipeline total: {total_s:.3f}s")
        self._log_session("Integrated Dashboard", question,
                          differentials[0]["diagnosis"] if differentials else "Unknown")

        return (
            vqa_answer,
            heatmap_overlay,
            latency_str,
            confidence_str,
            report_aware_out,
            context_out,
            long_heatmap,
            long_answer,
            diff_out,
            full_report,
        )



# ===========================================================================
# Module-level theme & CSS  –  ChatGPT-style redesign
# ===========================================================================

_APP_THEME = gr.themes.Base(
    primary_hue="emerald",
    secondary_hue="teal",
    neutral_hue="zinc",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
)

_APP_CSS = """
/* ═══════════════════════════════════════════════════════════════
   GLOBAL RESET & TOKENS
═══════════════════════════════════════════════════════════════ */
:root {
  --sidebar-w: 260px;
  --bg-main:   #212121;
  --bg-sidebar:#171717;
  --bg-card:   #2f2f2f;
  --bg-input:  #404040;
  --accent:    #10a37f;
  --accent-h:  #0d8c6d;
  --text-main: #ececec;
  --text-muted:#8e8ea0;
  --border:    #3f3f3f;
  --radius-lg: 16px;
  --radius-md: 10px;
  --radius-sm: 6px;
  --shadow:    0 4px 24px rgba(0,0,0,0.35);
}
body, .gradio-container {
  background: var(--bg-main) !important;
  color: var(--text-main) !important;
  font-family: 'Inter', sans-serif !important;
  margin: 0; padding: 0;
}

/* ── Hide default Gradio chrome ─────────────────────────────── */
footer { display:none !important; }
.contain { max-width:100% !important; padding:0 !important; }

/* ═══════════════════════════════════════════════════════════════
   SIDEBAR
═══════════════════════════════════════════════════════════════ */
.sidebar {
  width: var(--sidebar-w);
  min-width: var(--sidebar-w);
  background: var(--bg-sidebar);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 16px 10px;
  gap: 4px;
  height: 100vh;
  position: sticky;
  top: 0;
  overflow-y: auto;
}
.sidebar-logo {
  font-size: 1.1rem;
  font-weight: 700;
  color: var(--text-main);
  padding: 8px 12px 20px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 8px;
  letter-spacing: -0.3px;
}
.sidebar-logo span { color: var(--accent); }
.sidebar-section {
  font-size: 0.65rem;
  font-weight: 600;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  padding: 8px 12px 4px;
}
.nav-btn {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  background: transparent !important;
  border: none !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-muted) !important;
  font-size: 0.87rem !important;
  font-weight: 500 !important;
  padding: 9px 12px !important;
  cursor: pointer !important;
  text-align: left !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
  box-shadow: none !important;
}
.nav-btn:hover {
  background: rgba(255,255,255,0.06) !important;
  color: var(--text-main) !important;
}
.nav-btn.active {
  background: rgba(16,163,127,0.15) !important;
  color: var(--accent) !important;
  font-weight: 600 !important;
}
.nav-icon { font-size: 1rem; width: 20px; text-align: center; }

/* ═══════════════════════════════════════════════════════════════
   MAIN CONTENT AREA
═══════════════════════════════════════════════════════════════ */
.main-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 24px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-main);
}
.topbar-title {
  font-size: 1rem;
  font-weight: 600;
  color: var(--text-main);
}
.mode-badge {
  font-size: 0.72rem;
  background: rgba(16,163,127,0.18);
  color: var(--accent);
  border: 1px solid rgba(16,163,127,0.35);
  border-radius: 20px;
  padding: 3px 10px;
  font-weight: 600;
}

/* ═══════════════════════════════════════════════════════════════
   CHAT / OUTPUT PANEL
═══════════════════════════════════════════════════════════════ */
.chat-area {
  flex: 1;
  overflow-y: auto;
  padding: 28px 10% 0;
  display: flex;
  flex-direction: column;
  gap: 20px;
}
/* Gradio Chatbot overrides */
.gr-chatbot {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.message.user > div {
  background: var(--bg-input) !important;
  border-radius: var(--radius-lg) !important;
  border-bottom-right-radius: 4px !important;
  color: var(--text-main) !important;
  max-width: 78% !important;
  align-self: flex-end !important;
  padding: 12px 18px !important;
  font-size: 0.9rem !important;
  line-height: 1.6 !important;
}
.message.bot > div {
  background: var(--bg-card) !important;
  border-radius: var(--radius-lg) !important;
  border-bottom-left-radius: 4px !important;
  color: var(--text-main) !important;
  max-width: 82% !important;
  align-self: flex-start !important;
  padding: 14px 20px !important;
  font-size: 0.88rem !important;
  line-height: 1.7 !important;
  border: 1px solid var(--border) !important;
}

/* ═══════════════════════════════════════════════════════════════
   INPUT STRIP (bottom)
═══════════════════════════════════════════════════════════════ */
.input-strip {
  padding: 16px 10%;
  background: var(--bg-main);
  border-top: 1px solid var(--border);
}
.input-box textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  color: var(--text-main) !important;
  font-size: 0.92rem !important;
  padding: 14px 18px !important;
  resize: none !important;
}
.input-box textarea:focus {
  border-color: var(--accent) !important;
  outline: none !important;
  box-shadow: 0 0 0 2px rgba(16,163,127,0.2) !important;
}
.send-btn {
  background: var(--accent) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  color: #fff !important;
  font-weight: 600 !important;
  font-size: 0.88rem !important;
  padding: 12px 22px !important;
  cursor: pointer !important;
  transition: background 0.2s ease !important;
  height: 48px !important;
}
.send-btn:hover { background: var(--accent-h) !important; }

/* ═══════════════════════════════════════════════════════════════
   RIGHT INPUT PANEL
═══════════════════════════════════════════════════════════════ */
.right-panel {
  width: 320px;
  min-width: 280px;
  background: var(--bg-sidebar);
  border-left: 1px solid var(--border);
  padding: 16px 14px;
  overflow-y: auto;
  height: 100vh;
}
.panel-section-title {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 14px 0 6px;
}

/* ═══════════════════════════════════════════════════════════════
   GRADIO COMPONENT OVERRIDES
═══════════════════════════════════════════════════════════════ */
label, .gr-form > label, .gr-input-label {
  color: var(--text-muted) !important;
  font-size: 0.78rem !important;
  font-weight: 500 !important;
}
.gr-box, .gr-panel, .gr-block {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
}
textarea, input[type=text] {
  background: var(--bg-input) !important;
  color: var(--text-main) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
}
/* Tab bar */
.tab-nav { background: var(--bg-sidebar) !important; border-bottom: 1px solid var(--border) !important; }
.tab-nav button {
  color: var(--text-muted) !important;
  background: transparent !important;
  border-radius: 0 !important;
  font-size: 0.82rem !important;
  padding: 10px 16px !important;
  border-bottom: 2px solid transparent !important;
}
.tab-nav button.selected {
  color: var(--accent) !important;
  border-bottom: 2px solid var(--accent) !important;
}
/* Buttons */
.gr-button-primary {
  background: var(--accent) !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  color: #fff !important;
  font-weight: 600 !important;
  transition: background 0.2s ease !important;
}
.gr-button-primary:hover { background: var(--accent-h) !important; }
/* Sliders */
input[type=range] { accent-color: var(--accent) !important; }
/* Textbox outputs */
.output-textbox textarea {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-main) !important;
  font-size: 0.85rem !important;
  line-height: 1.65 !important;
}
/* Images */
.gr-image { border-radius: var(--radius-md) !important; overflow: hidden; }

/* ═══════════════════════════════════════════════════════════════
   METRIC CHIPS
═══════════════════════════════════════════════════════════════ */
.metric-chip textarea {
  text-align: center !important;
  font-weight: 700 !important;
  font-size: 1rem !important;
  color: var(--accent) !important;
  background: rgba(16,163,127,0.10) !important;
  border: 1px solid rgba(16,163,127,0.30) !important;
  border-radius: var(--radius-md) !important;
}

/* ═══════════════════════════════════════════════════════════════
   WELCOME CARD
═══════════════════════════════════════════════════════════════ */
.welcome-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 32px;
  max-width: 640px;
  margin: 40px auto;
  text-align: center;
}
.welcome-card h2 { font-size: 1.5rem; margin-bottom: 8px; color: var(--text-main); }
.welcome-card p  { color: var(--text-muted); font-size: 0.9rem; line-height: 1.6; }
.welcome-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-top: 24px;
  text-align: left;
}
.welcome-tile {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 14px;
  font-size: 0.82rem;
  color: var(--text-muted);
  cursor: pointer;
  transition: background 0.15s ease;
}
.welcome-tile:hover { background: rgba(255,255,255,0.08); }
.welcome-tile strong { display:block; color: var(--text-main); margin-bottom: 4px; font-size: 0.88rem; }

/* ═══════════════════════════════════════════════════════════════
   DISCLAIMER FOOTER
═══════════════════════════════════════════════════════════════ */
.footer-disclaimer {
  font-size: 0.72rem;
  color: var(--text-muted);
  text-align: center;
  padding: 8px 0 4px;
  border-top: 1px solid var(--border);
}
"""


# ===========================================================================
# Gradio UI builder  –  ChatGPT-style redesign
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

            # ── Tab 7: Integrated Clinical Dashboard ────────────────────────
            with gr.TabItem("🏥 Integrated Dashboard"):
                gr.Markdown(
                    """### 🏥 Integrated Clinical Dashboard
                    All five clinical modules run together — share a single image, question and
                    clinical context, and receive every output at once.
                    """
                )

                with gr.Row(equal_height=False):

                    # ── LEFT: Input panel (30%) ──────────────────────────────
                    with gr.Column(scale=3, min_width=280):
                        gr.Markdown("#### 📥 Inputs")

                        with gr.Group():
                            gr.Markdown("**🖼️ Images**")
                            dash_cur  = gr.Image(type="numpy", label="Current X-Ray",
                                                 height=220)
                            dash_prior = gr.Image(type="numpy",
                                                  label="Prior X-Ray (optional – for longitudinal)",
                                                  height=180)

                        with gr.Group():
                            gr.Markdown("**❓ Clinical Question**")
                            dash_q = gr.Textbox(
                                placeholder="e.g. Is there evidence of pneumonia?",
                                lines=2, show_label=False,
                            )

                        with gr.Group():
                            gr.Markdown("**📄 Prior Radiology Report**")
                            dash_report = gr.Textbox(
                                placeholder="Paste prior report text here…",
                                lines=4, show_label=False,
                            )

                        with gr.Group():
                            gr.Markdown("**🧪 Clinical Context**")
                            dash_notes  = gr.Textbox(label="Clinical Notes",
                                                     lines=2,
                                                     placeholder="e.g. Fever 38.9°C, productive cough")
                            dash_labs   = gr.Textbox(label="Lab Results",
                                                     lines=2,
                                                     placeholder="e.g. WBC 14.2, CRP 85")
                            dash_vitals = gr.Textbox(label="Vitals",
                                                     placeholder="e.g. BP 128/82, SpO2 94%")

                        with gr.Group():
                            gr.Markdown("**⚙️ Settings**")
                            dash_temp = gr.Slider(0.1, 1.5, value=0.7, step=0.05,
                                                  label="🌡️ Temperature")
                            dash_topk = gr.Slider(1, 5, value=3, step=1,
                                                  label="🏆 Top-K Differentials")

                        dash_btn = gr.Button(
                            "🚀 Analyze All Modules", variant="primary", size="lg"
                        )

                    # ── RIGHT: Output panel (70%) ────────────────────────────
                    with gr.Column(scale=7):
                        gr.Markdown("#### 📊 Integrated Outputs")

                        with gr.Row():
                            with gr.Column(scale=2):
                                gr.Markdown("**🔍 VQA Answer**")
                                dash_answer = gr.Textbox(
                                    label="Answer", lines=4, interactive=False
                                )
                                with gr.Row():
                                    dash_lat  = gr.Textbox(label="⏱️ Latency",
                                                           interactive=False, scale=1)
                                    dash_conf = gr.Textbox(label="🎯 Confidence",
                                                           interactive=False, scale=1)
                            with gr.Column(scale=3):
                                dash_heatmap = gr.Image(label="🔥 Grad-CAM Heatmap",
                                                        height=200)

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**📄 Report-Aware Context**")
                                dash_ra_out = gr.Textbox(
                                    label="Prior Report Status",
                                    lines=3, interactive=False,
                                )
                            with gr.Column():
                                gr.Markdown("**🧪 Clinical Context Summary**")
                                dash_ctx_out = gr.Textbox(
                                    label="Context Data",
                                    lines=3, interactive=False,
                                )

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("**📅 Longitudinal Analysis**")
                                dash_long_ans = gr.Textbox(
                                    label="Change Summary",
                                    lines=2, interactive=False,
                                )
                                dash_long_map = gr.Image(
                                    label="🗺️ Difference Heatmap", height=200
                                )
                            with gr.Column():
                                gr.Markdown("**🩺 Differential Diagnosis**")
                                dash_diff_out = gr.Textbox(
                                    label="Ranked Differentials",
                                    lines=10, interactive=False,
                                )

                        gr.Markdown("**📝 Integrated One-Click Report**")
                        dash_full_report = gr.Textbox(
                            label="Complete Structured Report",
                            lines=22, interactive=False,
                        )

                dash_btn.click(
                    fn=demo_backend.integrated_analyze,
                    inputs=[
                        dash_cur, dash_prior, dash_q,
                        dash_report,
                        dash_notes, dash_labs, dash_vitals,
                        dash_temp, dash_topk,
                    ],
                    outputs=[
                        dash_answer, dash_heatmap, dash_lat, dash_conf,
                        dash_ra_out,
                        dash_ctx_out,
                        dash_long_map, dash_long_ans,
                        dash_diff_out,
                        dash_full_report,
                    ],
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
    print(f"[MedXplain] Launching on http://localhost:{_port}")
    app = create_demo()
    app.launch(
        server_name="0.0.0.0",
        server_port=_port,
        share=False,
        debug=False,
        max_threads=4,
    )
