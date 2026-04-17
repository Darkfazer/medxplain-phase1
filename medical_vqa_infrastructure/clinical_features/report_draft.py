"""
clinical_features/report_draft.py
==================================
Module 5 – One-Click Report Generator for MedXplain.

Generates a professionally formatted, structured radiology report from
model predictions.  Works in full mock mode (no GPU / model required).

Usage
-----
from clinical_features.report_draft import ReportDrafter

drafter = ReportDrafter()
report  = drafter.generate_draft(
    image        = xray_numpy,
    question     = "Generate a full report",
    predictions  = {"findings": [...], "confidences": {...}},
    patient_info = {"gender": "F", "study_date": "1/5/2016"},
)
print(report)
"""

from __future__ import annotations

import random
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Findings templates – one dict per condition
# ---------------------------------------------------------------------------

_NORMAL_TEMPLATES: Dict[str, str] = {
    "trachea":   "Midline trachea. No deviation or narrowing.",
    "cardiac":   "Heart size is normal. Aorta and visualized mediastinal "
                 "structures appear unremarkable.",
    "lungs":     "No parenchymal infiltration, consolidation, or mass "
                 "identified. Lungs are clear.",
    "pleura":    "No pleural effusion or pneumothorax identified.",
    "bones":     "Osseous structures are intact. No acute bony pathology.",
    "comparison":"No significant interval change compared with prior study.",
}

_CONDITION_TEMPLATES: Dict[str, Dict[str, str]] = {
    "pneumonia": {
        "trachea":   "Trachea is midline.",
        "cardiac":   "Cardiac silhouette is within normal limits.",
        "lungs":     "Right lower lobe consolidation identified, consistent "
                     "with pneumonia. Air bronchograms present.",
        "pleura":    "Small reactive right pleural effusion cannot be excluded.",
        "bones":     "No acute osseous abnormality.",
        "comparison":"New consolidation compared with prior study.",
        "impression":"Right lower lobe pneumonia.",
        "recommendation": (
            "Antibiotic therapy as clinically appropriate.\n"
            "  Follow-up chest X-ray in 4–6 weeks to confirm resolution.\n"
            "  CT chest if no clinical improvement after 4 weeks."
        ),
    },
    "pleural_effusion": {
        "trachea":   "Trachea deviated contralateral to effusion.",
        "cardiac":   "Cardiac silhouette difficult to assess due to effusion.",
        "lungs":     "Opacification of the right lower hemithorax with "
                     "meniscus sign consistent with pleural effusion.",
        "pleura":    "Moderate right pleural effusion. Left pleural space clear.",
        "bones":     "Osseous structures appear intact.",
        "comparison":"Increase in effusion size compared with prior study.",
        "impression":"Moderate right pleural effusion.",
        "recommendation": (
            "Clinical correlation recommended to determine aetiology.\n"
            "  Diagnostic thoracentesis may be considered.\n"
            "  Follow-up imaging after drainage."
        ),
    },
    "cardiomegaly": {
        "trachea":   "Trachea midline.",
        "cardiac":   "Cardiothoracic ratio exceeds 0.5, consistent with "
                     "cardiomegaly. Prominent pulmonary vasculature.",
        "lungs":     "Bilateral perihilar haziness suggestive of pulmonary "
                     "venous hypertension.",
        "pleura":    "Bilateral small pleural effusions noted.",
        "bones":     "No acute osseous lesion.",
        "comparison":"Worsening cardiomegaly compared with prior.",
        "impression":"Cardiomegaly with signs of cardiac decompensation.",
        "recommendation": (
            "Urgent cardiology review recommended.\n"
            "  Echocardiogram to assess cardiac function.\n"
            "  Consider diuretic therapy."
        ),
    },
    "rib_fractures": {
        "trachea":   "Trachea is midline.",
        "cardiac":   "Heart size is normal.",
        "lungs":     "Underlying lung parenchyma clear.",
        "pleura":    "No pneumothorax identified. Small haemothorax cannot "
                     "be excluded.",
        "bones":     "Right-sided rib fractures noted at ribs 3, 4, and 5. "
                     "No acute displacement.",
        "comparison":"Fractures appear chronic; no interval change.",
        "impression":"Chronic right-sided rib fractures – stable.",
        "recommendation": (
            "Adequate analgesia for pain management.\n"
            "  Rule out underlying pulmonary contusion.\n"
            "  No immediate follow-up indicated."
        ),
    },
    "atelectasis": {
        "trachea":   "Trachea shifted towards the affected side.",
        "cardiac":   "Mediastinum shifted ipsilaterally.",
        "lungs":     "Left lower lobe linear atelectasis noted. Volume loss "
                     "in the left lower zone.",
        "pleura":    "No pleural effusion.",
        "bones":     "Osseous structures intact.",
        "comparison":"New atelectasis compared with prior study.",
        "impression":"Left lower lobe atelectasis.",
        "recommendation": (
            "Incentive spirometry and physiotherapy encouraged.\n"
            "  Repeat chest X-ray in 24–48 hours.\n"
            "  Bronchoscopy if atelectasis persists."
        ),
    },
    "pulmonary_edema": {
        "trachea":   "Trachea midline.",
        "cardiac":   "Cardiomegaly present. Widened vascular pedicle.",
        "lungs":     "Bilateral perihilar alveolar opacities in a 'bat-wing' "
                     "distribution. Interstitial Kerley B lines present.",
        "pleura":    "Bilateral small pleural effusions.",
        "bones":     "No acute osseous finding.",
        "comparison":"Significantly worsened compared with prior.",
        "impression":"Acute pulmonary oedema.",
        "recommendation": (
            "Urgent medical review.\n"
            "  IV diuresis and haemodynamic monitoring.\n"
            "  Repeat CXR after treatment to assess response."
        ),
    },
    "mass_nodule": {
        "trachea":   "Trachea midline.",
        "cardiac":   "Heart size within normal limits.",
        "lungs":     "Solitary pulmonary nodule identified in the right upper "
                     "lobe measuring approximately 2.1 cm. Margins irregular.",
        "pleura":    "No pleural effusion.",
        "bones":     "No lytic or sclerotic lesions.",
        "comparison":"New finding – not present on prior study.",
        "impression":"Right upper lobe pulmonary nodule – indeterminate.",
        "recommendation": (
            "Urgent CT chest with contrast recommended.\n"
            "  PET-CT if CT confirms suspicious features.\n"
            "  Respiratory / oncology referral."
        ),
    },
}

# Conditions mapped from common prediction labels
_LABEL_TO_CONDITION: Dict[str, str] = {
    "pneumonia":        "pneumonia",
    "consolidation":    "pneumonia",
    "pleural effusion": "pleural_effusion",
    "effusion":         "pleural_effusion",
    "cardiomegaly":     "cardiomegaly",
    "rib fracture":     "rib_fractures",
    "atelectasis":      "atelectasis",
    "pulmonary edema":  "pulmonary_edema",
    "edema":            "pulmonary_edema",
    "mass":             "mass_nodule",
    "nodule":           "mass_nodule",
    "normal":           None,   # handled separately
}

# Recommendation for normal finding
_NORMAL_RECOMMENDATION = (
    "No immediate follow-up indicated.\n"
    "  Routine clinical follow-up as scheduled."
)


# ===========================================================================
# ReportDrafter
# ===========================================================================

class ReportDrafter:
    """
    Module 5 – One-Click Structured Radiology Report Generator.

    Produces a complete, professionally formatted radiology report from
    model predictions, image, and optional patient demographics.

    Works in full mock mode if no real predictions are supplied.
    """

    # Width of the decorative border lines
    _WIDE  = "═" * 59
    _THIN  = "─" * 59

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Initialise the report drafter.

        Parameters
        ----------
        config:
            Optional configuration dictionary.  Supported keys:
            - ``exam_type`` (str): e.g. ``"Chest X-ray (AP/PA)"``
            - ``institution`` (str): institution name shown in the header
        """
        config = config or {}
        self.exam_type    = config.get("exam_type", "Chest X-ray (AP/PA)")
        self.institution  = config.get("institution", "MedXplain")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_draft(
        self,
        image:        Optional[Union[np.ndarray, Any]] = None,
        question:     str                              = "Generate a full report",
        predictions:  Optional[Dict[str, Any]]         = None,
        patient_info: Optional[Dict[str, str]]         = None,
    ) -> str:
        """
        Generate a complete structured radiology report.

        Parameters
        ----------
        image:
            Chest X-ray as a numpy array or PIL Image (used for validation only;
            not re-processed here — that happens upstream in the VQA pipeline).
        question:
            Clinical question or indication (used as Clinical History).
        predictions:
            Dictionary from the VQA model.  Supported keys:

            - ``findings``    (List[str])         – list of finding labels
            - ``confidences`` (Dict[str, float])  – label → confidence (0–1)
            - ``impression``  (str)               – pre-formed impression text
            - ``comparison``  (str)               – comparison to prior study
        patient_info:
            Patient demographics.  Supported keys:
            name, patient_id, dob, gender, study_date, referring_physician,
            clinical_history.

        Returns
        -------
        str
            Complete formatted report ready for display.
        """
        self._validate_image(image)

        patient_info = patient_info or {}
        predictions  = predictions  or {}

        # Resolve primary condition from predictions
        condition, confidence_map = self._resolve_condition(predictions)

        # Build each report section
        header      = self._format_header()
        patient_sec = self._format_patient_info(patient_info, question)
        findings    = self._format_findings(condition, confidence_map, predictions)
        impression  = self._format_impression(condition, predictions)
        recommend   = self._format_recommendation(condition)
        disclaimer  = self._format_disclaimer()

        return "\n".join([
            self._WIDE,
            header,
            self._WIDE,
            "",
            patient_sec,
            "",
            self._WIDE,
            "FINDINGS",
            self._THIN,
            findings,
            "",
            self._WIDE,
            "IMPRESSION",
            self._THIN,
            impression,
            "",
            self._WIDE,
            "RECOMMENDATION",
            self._THIN,
            recommend,
            "",
            self._THIN,
            disclaimer,
            self._THIN,
        ])

    # ------------------------------------------------------------------
    # Section formatters (private)
    # ------------------------------------------------------------------

    def _format_header(self) -> str:
        """Return the report title block with timestamp."""
        ts = self._get_timestamp()
        return (
            f"  🏥 {self.institution.upper()} RADIOLOGY REPORT\n"
            f"  Generated: {ts}"
        )

    def _format_patient_info(
        self,
        info: Dict[str, str],
        question: str,
    ) -> str:
        """
        Format the PATIENT INFORMATION and CLINICAL HISTORY sections.

        Parameters
        ----------
        info:
            Patient demographics dict.
        question:
            Fallback for clinical history if not supplied in ``info``.
        """
        def _get(key: str, default: str = "Not specified") -> str:
            return str(info.get(key, default)).strip() or default

        history = _get("clinical_history", question.strip() or "Not specified")

        lines = [
            "PATIENT INFORMATION",
            f"  NAME:            {_get('name')}",
            f"  PATIENT NUMBER:  {_get('patient_id')}",
            f"  DATE OF BIRTH:   {_get('dob')}",
            f"  GENDER:          {_get('gender')}",
            f"  STUDY DATE:      {_get('study_date', 'Current')}",
            f"  REF. PHYSICIAN:  {_get('referring_physician')}",
            f"  EXAM:            {self.exam_type}",
            "",
            "CLINICAL HISTORY",
            f"  {history}",
        ]
        return "\n".join(lines)

    def _format_findings(
        self,
        condition: Optional[str],
        confidence_map: Dict[str, float],
        predictions: Dict[str, Any],
    ) -> str:
        """
        Build the structured FINDINGS section.

        Uses condition-specific templates.  Falls back to normal templates
        when no condition is detected.

        Parameters
        ----------
        condition:
            Resolved condition key (e.g. ``"pneumonia"``), or ``None`` for normal.
        confidence_map:
            Label → confidence mapping from predictions.
        predictions:
            Raw predictions dict (may contain custom ``findings`` list).
        """
        tmpl = (
            _CONDITION_TEMPLATES.get(condition, _NORMAL_TEMPLATES)
            if condition else _NORMAL_TEMPLATES
        )

        # Merge keys so we always have all subsections
        sections: Dict[str, str] = {**_NORMAL_TEMPLATES, **tmpl}

        # Override comparison if provided in predictions
        if "comparison" in predictions:
            sections["comparison"] = str(predictions["comparison"])

        def _bullet(text: str, conf_label: Optional[str] = None) -> str:
            """Format a single bullet point, optionally with a confidence bar."""
            line = f"  • {text}"
            if conf_label and conf_label in confidence_map:
                bar = self._confidence_bar(confidence_map[conf_label])
                pct = int(confidence_map[conf_label] * 100)
                line += f"\n    {bar} {pct}%"
            return line

        # Gather any extra custom findings from predictions
        extra_findings = [
            f for f in predictions.get("findings", [])
            if f.lower() not in _LABEL_TO_CONDITION
        ]

        parts = [
            "TRACHEA & AIRWAYS:",
            _bullet(sections["trachea"]),
            "",
            "CARDIAC & MEDIASTINUM:",
            _bullet(sections["cardiac"]),
            "",
            "LUNG PARENCHYMA:",
            _bullet(sections["lungs"], conf_label=condition),
            "",
            "PLEURA:",
            _bullet(sections["pleura"]),
            "",
            "BONES & SOFT TISSUES:",
            _bullet(sections["bones"]),
        ]

        # Append any additional custom findings
        if extra_findings:
            parts += ["", "ADDITIONAL FINDINGS:"]
            for f in extra_findings:
                parts.append(_bullet(f))

        # Comparison / prior study
        parts += [
            "",
            "COMPARISON:",
            _bullet(sections.get("comparison", _NORMAL_TEMPLATES["comparison"])),
        ]

        return "\n".join(parts)

    def _format_impression(
        self,
        condition: Optional[str],
        predictions: Dict[str, Any],
    ) -> str:
        """
        Format the IMPRESSION section.

        Parameters
        ----------
        condition:
            Resolved condition key, or ``None`` for normal.
        predictions:
            Raw predictions dict (may contain ``impression`` key).
        """
        # Prefer explicit impression from predictions
        if "impression" in predictions:
            text = str(predictions["impression"])
        elif condition and condition in _CONDITION_TEMPLATES:
            text = _CONDITION_TEMPLATES[condition]["impression"]
        else:
            text = "No active cardiopulmonary disease."

        # Indent every line inside the section
        indented = "\n".join(f"  {ln}" for ln in text.splitlines())
        return indented

    def _format_recommendation(self, condition: Optional[str]) -> str:
        """
        Generate clinical recommendations from the resolved condition.

        Parameters
        ----------
        condition:
            Resolved condition key, or ``None`` for normal.
        """
        if condition and condition in _CONDITION_TEMPLATES:
            text = _CONDITION_TEMPLATES[condition]["recommendation"]
        else:
            text = _NORMAL_RECOMMENDATION

        indented = "\n".join(f"  {ln}" for ln in text.splitlines())
        return indented

    def _format_disclaimer(self) -> str:
        """Return the mandatory AI-generated disclaimer block."""
        return (
            "⚠️  AI-GENERATED REPORT - MUST BE REVIEWED BY\n"
            "   A QUALIFIED RADIOLOGIST BEFORE CLINICAL USE"
        )

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------

    def _resolve_condition(
        self,
        predictions: Dict[str, Any],
    ) -> tuple[Optional[str], Dict[str, float]]:
        """
        Determine the primary condition from predictions.

        The highest-confidence recognised finding wins.
        Returns ``(condition_key, confidence_map)``.

        In mock mode (empty predictions), returns a random realistic condition.
        """
        raw_findings:    List[str]         = predictions.get("findings",    [])
        confidences_in:  Dict[str, float]  = predictions.get("confidences", {})

        if not raw_findings and not confidences_in:
            return self._mock_condition()

        # Map each finding label to a known condition key with its confidence
        matched: List[tuple[float, str]] = []
        for label in raw_findings:
            key = _LABEL_TO_CONDITION.get(label.lower())
            if key is None:           # explicitly "normal"
                continue
            if key:
                conf = confidences_in.get(label, confidences_in.get(key, 0.80))
                matched.append((conf, key))

        if not matched:
            return None, confidences_in   # all findings were "normal"

        # Pick highest confidence condition
        matched.sort(reverse=True)
        primary_condition = matched[0][1]

        # Rebuild confidence map keyed by condition
        conf_map: Dict[str, float] = {}
        for conf, key in matched:
            conf_map[key] = conf

        return primary_condition, conf_map

    def _mock_condition(self) -> tuple[Optional[str], Dict[str, float]]:
        """
        Return a random realistic condition for mock mode.

        Occasionally returns ``None`` (normal X-ray) to keep variety.
        """
        pool = [
            ("pneumonia",       0.85),
            ("pleural_effusion",0.78),
            ("cardiomegaly",    0.82),
            ("rib_fractures",   0.91),
            ("atelectasis",     0.73),
            ("pulmonary_edema", 0.88),
            ("mass_nodule",     0.76),
            (None,              1.00),   # normal
        ]
        condition, conf = random.choice(pool)
        confidence_map  = {condition: conf} if condition else {}
        return condition, confidence_map

    @staticmethod
    def _confidence_bar(confidence: float, width: int = 20) -> str:
        """
        Render a Unicode block progress bar for a confidence value.

        Parameters
        ----------
        confidence:
            Float in ``[0, 1]``.
        width:
            Total number of bar characters.

        Returns
        -------
        str
            e.g. ``"██████████████░░░░░░"``
        """
        filled = round(confidence * width)
        return "█" * filled + "░" * (width - filled)

    @staticmethod
    def _validate_image(image: Any) -> None:
        """
        Validate that the supplied image is a non-empty numpy array or None.

        Parameters
        ----------
        image:
            Image to validate.

        Raises
        ------
        ValueError
            If a non-None value is provided but is clearly not an image.
        """
        if image is None:
            return
        if isinstance(image, np.ndarray):
            if image.ndim not in (2, 3):
                raise ValueError(
                    f"Image array must be 2-D (grayscale) or 3-D (RGB), "
                    f"got shape {image.shape}."
                )
        # PIL Images and other objects pass through silently

    @staticmethod
    def _get_timestamp() -> str:
        """Return the current date-time as a formatted string."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
