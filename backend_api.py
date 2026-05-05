"""
MedXplain – FastAPI REST server  (backend_api.py)
=================================================
Run with:  python backend_api.py
           uvicorn backend_api:app --host 0.0.0.0 --port 8000

Endpoints:
  POST /api/analyze   – main inference endpoint
  GET  /api/history   – patient report history
  GET  /               – serves medxplain_ui.html
"""
import sys
import io
import base64
import json
import logging
import cv2
import os
import re
import asyncio
from pathlib import Path

import requests

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Load environment variables from .env file (if present)
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger("MedXplain.api")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import numpy as np
from PIL import Image
import uvicorn

import backend as B

# ─────────────────────────────────────────────────────────────────────────────
#  App + CORS
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="MedXplain API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition"],
    max_age=600,
)

HTML_FILE = Path(__file__).parent / "medxplain_ui.html"


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> str:
    
    """Serve the React-based HTML frontend."""
    try:
        if not HTML_FILE.exists():
            log.error(f"medxplain_ui.html not found at {HTML_FILE}")
            raise HTTPException(status_code=500, detail="medxplain_ui.html not found on server")
        content = HTML_FILE.read_text(encoding="utf-8")
        log.info(f"Serving medxplain_ui.html ({len(content)} bytes)")
        return content
    except Exception as e:
        log.error(f"Error serving HTML: {e}")
        raise HTTPException(status_code=500, detail=f"Error serving HTML: {str(e)}")


@app.get("/health")
async def health_check() -> dict:
    """Lightweight endpoint for frontend connectivity verification."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/ping")
async def ping() -> dict:
    """Simple ping endpoint to verify server is alive."""
    return {"status": "alive", "timestamp": asyncio.get_event_loop().time()}


# ─────────────────────────────────────────────────────────────────────────────
#  Model name normalisation
# ─────────────────────────────────────────────────────────────────────────────
_DO_CLS_NAMES = {
    "Ensemble (Classification + VQA)",
    "TorchXRayVision (Classification only)",
    "MedXplain-V1",
    "MedXplain-V2 (Research)",
}
_DO_VQA_NAMES = {
    "Ensemble (Classification + VQA)",
    "BLIP-VQA (Visual Q&A only)",
    "MedXplain-V1",
    "MedXplain-V2 (Research)",
    "MedXplain-Lite",
}


def _should_classify(model_choice: str) -> bool:
    return model_choice in _DO_CLS_NAMES


def _should_vqa(model_choice: str) -> bool:
    return model_choice in _DO_VQA_NAMES


def _strip_data_uri(data_uri: str) -> str:
    """Return only the base64 payload, stripping the data URI prefix."""
    if not data_uri:
        return ""
    if "," in data_uri:
        return data_uri.split(",", 1)[1]
    return data_uri


def _img_to_b64(arr: np.ndarray) -> str:
    """Return base64 JPEG data URI from numpy RGB array."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", rgb)
    b64 = base64.b64encode(buf).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _thumbnail_to_b64(arr: np.ndarray, max_size: int = 320) -> str:
    h, w = arr.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        thumb = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        thumb = arr
    return _img_to_b64(thumb)


def _validate_clinical_param(name: str, value: str) -> str:
    if not value:
        return ""
    s = value.strip()
    if re.fullmatch(r"\d{2,3}/\d{2,3}", s):
        return s
    if re.fullmatch(r"\d{2,3}", s):
        return s
    if re.fullmatch(r"\d{2}\.?\d*", s):
        return s
    if re.fullmatch(r"\d{2,3}%?", s):
        return s
    return ""


def _parse_clinical_params(raw: str) -> dict:
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


class ReportPayload(BaseModel):
    patient_id: str = "default"
    image_base64: str = ""
    question: str = ""
    answer: str = ""
    gradcam_base64: str = ""
    classification: dict | None = None
    mode: str = "Standard"
    report_context: str = ""
    clinical_question: str = ""


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/analyze
# ─────────────────────────────────────────────────────────────────────────────
def _is_doctor_mode(mode: str) -> bool:
    return (mode or "").strip().lower() in {"doctor", "doctor assistant", "doctor_assistant"}


@app.post("/api/analyze")
async def analyze(
    image:        UploadFile = File(default=None),
    question:     str = Form(default=""),
    mode:         str = Form(default="Standard"),
    model_choice: str = Form(default="Ensemble (Classification + VQA)"),
    patient_id:   str = Form(default="default"),
    # Advanced feature flags sent as "true"/"false" strings from JS
    feat_report:  str = Form(default="false"),
    feat_context: str = Form(default="false"),
    feat_longit:  str = Form(default="false"),
    feat_oneclik: str = Form(default="false"),
    feat_diff:    str = Form(default="false"),
    # Vitals / labs (optional; validated before use)
    bp:   str = Form(default=""),
    hr:   str = Form(default=""),
    temp: str = Form(default=""),
    spo2: str = Form(default=""),
    wbc:  str = Form(default=""),
    crp:  str = Form(default=""),
    gradcam_strength: str = Form(default="0.45"),
    temperature: str = Form(default="1.0"),
    max_tokens: str = Form(default="64"),
    report_context: str = Form(default=""),
    clinical_params: str = Form(default=""),
    reformulate: str = Form(default="true"),  # Always true
    # Doctor-mode clinical question (simplified - only this field)
    clinical_question: str = Form(default=""),
):
    """Main inference endpoint.

    Accepts a multipart form with an optional image (JPEG/PNG/DICOM) and
    clinical parameters.  Returns a JSON payload with classification,
    VQA, Grad-CAM (base64), differential, longitudinal, and report fields.
    In Doctor Assistant mode, an image is optional; the system can generate
    a structured clinical note purely from text fields via LLM.
    """
    def _flag(s: str) -> bool:
        return str(s).lower() == "true"

    doctor_mode = _is_doctor_mode(mode)
    has_image = image is not None and image.filename

    # ── Validate clinical inputs ──────────────────────────────────────────────
    bp   = _validate_clinical_param("bp",   bp)
    hr   = _validate_clinical_param("hr",   hr)
    temp = _validate_clinical_param("temp", temp)
    spo2 = _validate_clinical_param("spo2", spo2)
    wbc  = _validate_clinical_param("wbc",  wbc)
    crp  = _validate_clinical_param("crp",  crp)
    params = _parse_clinical_params(clinical_params)
    bp   = bp   or str(params.get("bp", "")).strip()
    hr   = hr   or str(params.get("hr", "")).strip()
    temp = temp or str(params.get("temp", "")).strip()
    spo2 = spo2 or str(params.get("spo2", "")).strip()
    wbc  = wbc  or str(params.get("wbc", "")).strip()
    crp  = crp  or str(params.get("crp", "")).strip()

    # ── Load image (optional) ────────────────────────────────────────────────
    arr: np.ndarray | None = None
    image_thumbnail = ""
    if has_image:
        raw = await image.read()
        filename = (image.filename or "").lower()
        if filename.endswith(".dcm") or image.content_type in (
            "application/dicom", "application/octet-stream"
        ):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".dcm") as tmp:
                tmp.write(raw)
                tmp_path = tmp.name
            try:
                arr = B.load_dicom(tmp_path)
            except Exception as e:
                raise HTTPException(400, f"Cannot decode DICOM: {e}")
            finally:
                os.unlink(tmp_path)
        else:
            try:
                pil = Image.open(io.BytesIO(raw)).convert("RGB")
                arr = np.array(pil)
            except Exception as e:
                raise HTTPException(400, f"Cannot decode image: {e}")
        image_thumbnail = _thumbnail_to_b64(arr)

    result: dict = {}
    do_cls = _should_classify(model_choice) and has_image
    do_vqa = _should_vqa(model_choice) and bool(question.strip()) and has_image

    # ── Doctor Assistant mode: text-only clinical note ────────────────────
    if doctor_mode and not has_image:
        doctor_note = await asyncio.to_thread(
            B.generate_doctor_note,
            clinical_question=clinical_question or question,
            vitals=B.build_vitals_context(bp, hr, temp, spo2, wbc, crp),
        )
        result.update({
            "answer": doctor_note,
            "vqa_answer": doctor_note,
            "vqa_conf": 0.0,
            "mode": mode,
            "model_choice": model_choice,
            "patient_id": patient_id,
            "image_base64": image_thumbnail,
            "report": doctor_note,
            "gradcam_b64": "",
            "gradcam_image": "",
            "differential": [],
            "longitudinal": "",
            "classification": None,
            "question": question.strip(),
            "clinical_params": {"bp": bp, "hr": hr, "temp": temp, "spo2": spo2, "wbc": wbc, "crp": crp},
            "doctor_fields": {
                "clinical_question": clinical_question or question,
            },
        })
        # Persist to DB
        import time as _time
        B.save_report_entry(patient_id, {
            "date": _time.strftime("%Y-%m-%d"),
            "prediction": "Doctor Assistant (text-only)",
            "vqa_answer": doctor_note,
            "report": doctor_note,
            "question": question.strip(),
            "mode": mode,
            "model_choice": model_choice,
            "image_base64": image_thumbnail,
            "result": result,
        })
        return result

    # ── Require image for Standard mode ──────────────────────────────────────
    if not has_image:
        raise HTTPException(400, "No image provided. In Standard mode, an image is required.")

    # ── Classification ────────────────────────────────────────────────────────
    cls = None
    if do_cls:
        cls = B.classify_image(arr, model_choice)
        result.update({
            "label":      cls["label"],
            "confidence": round(cls["confidence"], 4),
            "ci":         cls["interval"],
            "risk":       cls["risk"],
        })

    # ── Build VQA context ─────────────────────────────────────────────────────
    ctx_parts: list[str] = []
    if _flag(feat_report):
        rc = B.build_report_context(patient_id)
        if rc:
            ctx_parts.append(rc)
    if report_context.strip():
        ctx_parts.append(f"[CLINICAL REPORT SNIPPET]\n{report_context.strip()}")
    if _flag(feat_context):
        vc = B.build_vitals_context(bp, hr, temp, spo2, wbc, crp)
        if vc:
            ctx_parts.append(vc)
    if cls:
        ctx_parts.append(f"[CLASSIFICATION: {cls['label']} {cls['confidence']:.1%}]")
    # Add doctor-mode clinical question to context
    if doctor_mode and clinical_question.strip():
        ctx_parts.append(f"[CLINICAL QUESTION]\n{clinical_question.strip()}")
    context_str = "\n".join(ctx_parts)

    # ── Medical Query Reformulation ─────────────────────────────────────────
    original_question = question.strip()
    reformulated_question = original_question
    if _flag(reformulate) and LLM_ENABLED and original_question:
        reformulated_question = await reformulate_question(original_question)
        log.info("Reformulated question: %r → %r", original_question, reformulated_question)

    # ── VQA ───────────────────────────────────────────────────────────────────
    vqa_answer, vqa_conf = "", 0.0
    if do_vqa:
        vqa = B.answer_vqa(arr, reformulated_question.strip(), context_str)
        vqa_answer = B.format_answer_for_mode(vqa["answer"], mode, cls, context_str)
        vqa_conf   = round(vqa["confidence"], 4)
    result["vqa_answer"] = vqa_answer
    result["vqa_conf"]   = vqa_conf
    result["original_question"] = original_question
    result["reformulated_question"] = (
        reformulated_question if reformulated_question != original_question else ""
    )

    # ── Grad-CAM / explainability overlay ────────────────────────────────────
    cam_arr = B.generate_explainability_overlay(arr)
    gradcam_b64 = _img_to_b64(cam_arr)
    result["gradcam_b64"] = gradcam_b64
    result["gradcam_image"] = gradcam_b64

    # ── Differential diagnosis ───────────────────────────────────────────────
    differential: list[dict] = []
    if _flag(feat_diff) and cls and cls.get("all_probs"):
        ranked = sorted(
            cls["all_probs"].items(), key=lambda x: x[1], reverse=True
        )
        differential = [
            {"label": lbl, "prob": round(prob, 4)}
            for lbl, prob in ranked[:3]
        ]
    result["differential"] = differential

    # ── Longitudinal ──────────────────────────────────────────────────────────
    longitudinal = ""
    if _flag(feat_longit):
        longitudinal = B.longitudinal_compare(arr, patient_id)
    # Always update baseline silently
    B.save_prior_image(patient_id, arr)
    result["longitudinal"] = longitudinal

    # ── One-Click Report ──────────────────────────────────────────────────────
    report_txt = ""
    if _flag(feat_oneclik) and cls:
        diff_text = "\n".join(
            f"{i + 1}. {d['label']} — {d['prob']:.1%}"
            for i, d in enumerate(differential)
        ) if differential else ""
        report_txt = B.build_structured_report(
            pid=patient_id,
            cls=cls,
            vqa_answer=vqa_answer,
            diff_text=diff_text,
            model_choice=model_choice,
        )
    result["report"] = report_txt
    result["answer"] = vqa_answer
    result["classification"] = cls if cls else None
    result["question"] = question.strip()
    result["mode"] = mode
    result["model_choice"] = model_choice
    result["image_base64"] = image_thumbnail
    result["report_context"] = report_context.strip()
    result["clinical_params"] = {"bp": bp, "hr": hr, "temp": temp, "spo2": spo2, "wbc": wbc, "crp": crp}
    result["reformulate_enabled"] = _flag(reformulate) and LLM_ENABLED
    result["doctor_fields"] = {
        "clinical_question": clinical_question or question,
    }

    # ── Persist to patient DB ─────────────────────────────────────────────────
    import time as _time
    B.save_report_entry(patient_id, {
        "date":       _time.strftime("%Y-%m-%d"),
        "prediction": cls["label"] if cls else "VQA only",
        "vqa_answer": vqa_answer,
        "report":     report_txt,
        "question":   question.strip(),
        "mode":       mode,
        "model_choice": model_choice,
        "image_base64": image_thumbnail,
        "result":     result,
    })

    return result


@app.post("/predict")
async def predict(
    file:         UploadFile = File(default=None),
    question:     str = Form(default=""),
    mode:         str = Form(default="Standard"),
    model_choice: str = Form(default="Ensemble (Classification + VQA)"),
    patient_id:   str = Form(default="default"),
    feat_report:  str = Form(default="false"),
    feat_context: str = Form(default="false"),
    feat_longit:  str = Form(default="false"),
    feat_oneclik: str = Form(default="false"),
    feat_diff:    str = Form(default="false"),
    bp:   str = Form(default=""),
    hr:   str = Form(default=""),
    temp: str = Form(default=""),
    spo2: str = Form(default=""),
    wbc:  str = Form(default=""),
    crp:  str = Form(default=""),
    gradcam_strength: str = Form(default="0.45"),
    temperature: str = Form(default="1.0"),
    max_tokens: str = Form(default="64"),
    report_context: str = Form(default=""),
    clinical_params: str = Form(default=""),
    reformulate: str = Form(default="true"),  # Always true, ignored by backend
    clinical_question: str = Form(default=""),
):
    """Compatibility endpoint for clients that post `file` + `question`."""
    return await analyze(
        image=file,
        question=question,
        mode=mode,
        model_choice=model_choice,
        patient_id=patient_id,
        feat_report=feat_report,
        feat_context=feat_context,
        feat_longit=feat_longit,
        feat_oneclik=feat_oneclik,
        feat_diff=feat_diff,
        bp=bp,
        hr=hr,
        temp=temp,
        spo2=spo2,
        wbc=wbc,
        crp=crp,
        gradcam_strength=gradcam_strength,
        temperature=temperature,
        max_tokens=max_tokens,
        report_context=report_context,
        clinical_params=clinical_params,
        reformulate=reformulate,
        clinical_question=clinical_question,
    )


# ─────────────────────────────────────────────────────────────────────────────
#  LLM Query Reformulation (Gemini 1.5 Flash)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
LLM_ENABLED = bool(GEMINI_API_KEY)

if not LLM_ENABLED:
    log.warning("LLM query reformulation DISABLED — set GEMINI_API_KEY env var to enable.")


REFORMULATION_PROMPT = (
    "You are a medical NLP assistant. Rewrite the user's informal question "
    "into a precise, clinically appropriate query suitable for a medical VQA system. "
    "Preserve the original intent but improve terminology. Return ONLY the rewritten question, "
    "nothing else.\n\n"
    "User question: {question}\n\n"
    "Rewritten question:"
)


async def reformulate_question(question: str) -> str:
    """Send the question to Gemini 1.5 Flash and return the rewritten query."""
    if not LLM_ENABLED:
        return question

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{
            "parts": [{"text": REFORMULATION_PROMPT.format(question=question)}]
        }],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 128},
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            text = (
                candidates[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if text:
                log.info("LLM reformulation successful")
                return text
    except Exception as exc:
        log.warning("LLM reformulation failed: %s", exc)

    return question  # fallback to original


@app.post("/generate_report")
async def generate_report(payload: ReportPayload):
    """Generate a professional radiology-style PDF report from the last UI prediction."""
    try:
        from pdf_generator import generate_structured_pdf
    except ImportError as exc:
        raise HTTPException(500, "pdf_generator module is required for PDF generation") from exc

    pdf_buf = generate_structured_pdf(
        patient_id=payload.patient_id,
        image_base64=payload.image_base64,
        gradcam_base64=payload.gradcam_base64,
        question=payload.question,
        answer=payload.answer,
        report_context=payload.report_context,
        mode=payload.mode,
        classification=payload.classification,
        clinical_question=payload.clinical_question,
    )
    headers = {"Content-Disposition": "attachment; filename=medxplain_report.pdf"}
    return StreamingResponse(pdf_buf, media_type="application/pdf", headers=headers)


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/history
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(patient_id: str = "default") -> list:
    """Return stored report history for a patient."""
    return B.load_reports(patient_id)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)