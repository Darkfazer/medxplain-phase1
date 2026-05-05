"""
pdf_generator.py
================
Professional radiology-style PDF report generation for MedXplain.
Uses reportlab for clean, multi-page PDFs with proper margins and headings.
"""
from __future__ import annotations

import base64
import io
import os
import tempfile
from datetime import datetime
from typing import Optional

from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, PageBreak, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY

PAGE_WIDTH, PAGE_HEIGHT = letter
MARGIN = 0.75 * inch


def _b64_to_pil(data_uri: str) -> Optional[Image.Image]:
    if not data_uri:
        return None
    try:
        raw = data_uri.split(",", 1)[1] if "," in data_uri and data_uri.startswith("data:") else data_uri
        return Image.open(io.BytesIO(base64.b64decode(raw))).convert("RGB")
    except Exception:
        return None


def _safe(text: str) -> str:
    return (text or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def generate_structured_pdf(
    patient_id: str = "default",
    image_base64: str = "",
    gradcam_base64: str = "",
    question: str = "",
    answer: str = "",
    report_context: str = "",
    mode: str = "Standard",
    classification: dict | None = None,
    clinical_question: str = "",
) -> io.BytesIO:
    """Generate a professional radiology-style PDF report."""

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=MARGIN,
        leftMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    title_style = ParagraphStyle(
        "RadTitle",
        parent=styles["Heading1"],
        fontSize=18,
        leading=22,
        textColor=HexColor("#1B8B8B"),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
    )
    heading_style = ParagraphStyle(
        "RadHeading",
        parent=styles["Heading2"],
        fontSize=12,
        leading=15,
        textColor=HexColor("#1B8B8B"),
        spaceAfter=6,
        spaceBefore=10,
        fontName="Helvetica-Bold",
    )
    subheading_style = ParagraphStyle(
        "RadSubHeading",
        parent=styles["Heading3"],
        fontSize=10,
        leading=13,
        textColor=HexColor("#374151"),
        spaceAfter=4,
        spaceBefore=8,
        fontName="Helvetica-Bold",
    )
    body_style = ParagraphStyle(
        "RadBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    label_style = ParagraphStyle(
        "RadLabel",
        parent=styles["BodyText"],
        fontSize=9,
        leading=12,
        textColor=HexColor("#6b7280"),
        spaceAfter=2,
    )

    story: list = []

    # ── Header ──
    story.append(Paragraph("MEDXPLAIN RADIOLOGY REPORT", title_style))
    story.append(Spacer(1, 4))

    meta_data = [
        ["Patient ID:", _safe(patient_id), "Date:", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Mode:", _safe(mode), "System:", "MedXplain v1.0"],
    ]
    meta_table = Table(meta_data, colWidths=[1.1 * inch, 2.4 * inch, 0.9 * inch, 2.4 * inch])
    meta_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#374151")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 8))

    # ── Images ──
    img_pil = _b64_to_pil(image_base64)
    cam_pil = _b64_to_pil(gradcam_base64)
    if img_pil or cam_pil:
        image_row = []
        if img_pil:
            max_w, max_h = 230, 170
            scale = min(max_w / img_pil.width, max_h / img_pil.height, 1.0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                img_pil.save(tmp.name, format="JPEG", quality=90)
                image_row.append(RLImage(tmp.name, width=img_pil.width * scale, height=img_pil.height * scale))
        if cam_pil:
            max_w, max_h = 230, 170
            scale = min(max_w / cam_pil.width, max_h / cam_pil.height, 1.0)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
                cam_pil.save(tmp.name, format="JPEG", quality=90)
                image_row.append(RLImage(tmp.name, width=cam_pil.width * scale, height=cam_pil.height * scale))
        if image_row:
            img_table = Table([image_row], colWidths=[3.2 * inch] * len(image_row))
            img_table.setStyle(TableStyle([
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ]))
            story.append(img_table)
            story.append(Spacer(1, 4))
            captions = []
            if img_pil:
                captions.append("Uploaded Image")
            if cam_pil:
                captions.append("Grad-CAM Overlay")
            if len(captions) == 2:
                cap_table = Table([[Paragraph(captions[0], label_style), Paragraph(captions[1], label_style)]],
                                  colWidths=[3.2 * inch, 3.2 * inch])
                cap_table.setStyle(TableStyle([
                    ("ALIGN", (0, 0), (0, 0), "CENTER"),
                    ("ALIGN", (1, 0), (1, 0), "CENTER"),
                ]))
                story.append(cap_table)
            else:
                story.append(Paragraph(captions[0] if captions else "", label_style))
            story.append(Spacer(1, 8))

    # ── Classification ──
    if classification:
        story.append(Paragraph("CLASSIFICATION", heading_style))
        cls_data = []
        for key, value in classification.items():
            if key == "probabilities" and isinstance(value, dict):
                continue
            cls_data.append([f"<b>{_safe(key.replace('_', ' ').title())}:</b>", _safe(str(value))])
        if cls_data:
            cls_table = Table(cls_data, colWidths=[1.5 * inch, 4.3 * inch])
            cls_table.setStyle(TableStyle([
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#374151")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.append(cls_table)
        # probabilities as sub-table
        probs = classification.get("probabilities")
        if isinstance(probs, dict):
            prob_rows = []
            for lbl, pr in sorted(probs.items(), key=lambda x: x[1], reverse=True)[:5]:
                bar_n = int(pr * 16)
                bar = "█" * bar_n + "░" * (16 - bar_n)
                prob_rows.append([_safe(lbl), f"{pr:.1%}", _safe(bar)])
            if prob_rows:
                story.append(Paragraph("Top Probabilities", subheading_style))
                prob_table = Table(prob_rows, colWidths=[2.2 * inch, 1.0 * inch, 2.6 * inch])
                prob_table.setStyle(TableStyle([
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("TEXTCOLOR", (0, 0), (-1, -1), HexColor("#374151")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ]))
                story.append(prob_table)
        story.append(Spacer(1, 4))

    # ── Technique ──
    story.append(Paragraph("TECHNIQUE", heading_style))
    story.append(Paragraph(
        "Post IV and oral contrast fine slice imaging of the chest, abdomen, and pelvis. "
        "Images were reconstructed in axial, coronal, and sagittal planes. "
        "Windowing optimised for soft tissue, lung, and bone assessment.",
        body_style,
    ))
    story.append(Spacer(1, 4))

    # ── Comparison ──
    story.append(Paragraph("COMPARISON", heading_style))
    story.append(Paragraph("None available." if not report_context else _safe(report_context), body_style))
    story.append(Spacer(1, 4))

    # ── Clinical Context (Doctor mode) ──
    if clinical_question:
        story.append(Paragraph("CLINICAL CONTEXT", heading_style))
        story.append(Paragraph("<b>Clinical Question:</b> " + _safe(clinical_question), body_style))
        story.append(Spacer(1, 4))

    # ── Findings ──
    story.append(Paragraph("FINDINGS", heading_style))
    story.append(Paragraph("<b>Chest:</b>", subheading_style))
    if classification:
        chest_text = (
            f"The primary imaging finding is <b>{_safe(classification.get('label', 'N/A'))}</b> "
            f"with model confidence <b>{classification.get('confidence', 0):.1%}</b>. "
            f"No additional acute intrathoracic abnormality is described by the AI system."
        )
    else:
        chest_text = "No chest imaging analysis performed."
    story.append(Paragraph(chest_text, body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("<b>Abdomen / Pelvis:</b>", subheading_style))
    story.append(Paragraph(
        "No dedicated abdominal or pelvic imaging assessment was performed in this study. "
        "Correlation with dedicated abdominal imaging is recommended if clinically indicated.",
        body_style,
    ))
    story.append(Spacer(1, 4))

    # ── VQA / Answer ──
    if question or answer:
        story.append(Paragraph("QUESTION & ANSWER", heading_style))
        if question:
            story.append(Paragraph(f"<b>Question:</b> {_safe(question)}", body_style))
        if answer:
            story.append(Paragraph(f"<b>Answer:</b> {_safe(answer)}", body_style))
        story.append(Spacer(1, 4))

    # ── Conclusion ──
    story.append(Paragraph("CONCLUSION", heading_style))
    conclusion_parts = []
    if classification:
        conclusion_parts.append(
            f"Imaging findings are most consistent with <b>{_safe(classification.get('label', 'N/A'))}</b> "
            f"(confidence {classification.get('confidence', 0):.1%})."
        )
    if answer:
        conclusion_parts.append(f"AI-assisted interpretation: {_safe(answer)}")
    if not conclusion_parts:
        conclusion_parts.append("No imaging-based conclusion generated.")
    story.append(Paragraph(" ".join(conclusion_parts), body_style))
    story.append(Spacer(1, 4))

    # ── Recommendation ──
    story.append(Paragraph("RECOMMENDATION", heading_style))
    story.append(Paragraph(
        "1. Clinical correlation with patient history, symptoms, and laboratory findings is essential.<br/>"
        "2. If acute findings are present, follow-up imaging in 4–6 weeks is recommended.<br/>"
        "3. This report was generated by an AI assistant and must be verified by a qualified radiologist or clinician.<br/>"
        "4. Not for standalone clinical diagnosis.",
        body_style,
    ))
    story.append(Spacer(1, 12))

    # ── Footer disclaimer ──
    story.append(Table([[Paragraph(
        "<font size='7' color='#9ca3af'>"
        "This report was generated automatically by MedXplain v1.0. "
        "It is intended for decision-support only and does not replace professional medical judgment. "
        "HIPAA &amp; GDPR compliant design.</font>",
        ParagraphStyle("Disclaimer", alignment=TA_CENTER, fontSize=7, textColor=HexColor("#9ca3af"))
    )]], colWidths=[5.8 * inch]))

    doc.build(story)
    buf.seek(0)
    return buf
