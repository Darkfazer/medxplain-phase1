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
import logging
import re
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s – %(message)s",
)
log = logging.getLogger("MedXplain.api")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
)

HTML_FILE = Path(__file__).parent / "medxplain_ui.html"


@app.get("/", response_class=HTMLResponse)
async def serve_ui() -> str:
    """Serve the React-based HTML frontend."""
    return HTML_FILE.read_text(encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
#  Model name normalisation
#  Maps the names used in medxplain_ui.html → internal mode strings
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


# ─────────────────────────────────────────────────────────────────────────────
#  Input validation helpers
# ─────────────────────────────────────────────────────────────────────────────
_NUMERIC_RE = re.compile(r"^[\d./\s]+$")


def _validate_clinical_param(name: str, value: str) -> str:
    """Return the value as-is if it looks numeric/fractional, else empty string."""
    v = value.strip()
    if not v:
        return ""
    if _NUMERIC_RE.match(v):
        return v
    log.warning("Clinical parameter '%s' has non-numeric value '%s' – ignored.", name, v)
    return ""


# ─────────────────────────────────────────────────────────────────────────────
#  Helper: numpy array → base64 PNG data-URI
# ─────────────────────────────────────────────────────────────────────────────
def _img_to_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────────────────────────────────────
#  POST /api/analyze
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/api/analyze")
async def analyze(
    image:        UploadFile = File(...),
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
):
    """Main inference endpoint.

    Accepts a multipart form with an image (JPEG/PNG/DICOM) and optional
    clinical parameters.  Returns a JSON payload with classification,
    VQA, Grad-CAM (base64), differential, longitudinal, and report fields.
    """
    def _flag(s: str) -> bool:
        return str(s).lower() == "true"

    # ── Validate clinical inputs ──────────────────────────────────────────────
    bp   = _validate_clinical_param("bp",   bp)
    hr   = _validate_clinical_param("hr",   hr)
    temp = _validate_clinical_param("temp", temp)
    spo2 = _validate_clinical_param("spo2", spo2)
    wbc  = _validate_clinical_param("wbc",  wbc)
    crp  = _validate_clinical_param("crp",  crp)

    # ── Load image (JPEG/PNG or DICOM) ───────────────────────────────────────
    raw = await image.read()
    filename = (image.filename or "").lower()
    arr: np.ndarray

    if filename.endswith(".dcm") or image.content_type in (
        "application/dicom", "application/octet-stream"
    ):
        # DICOM: save to a temp file and hand off to backend.load_dicom
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

    result: dict = {}
    do_cls = _should_classify(model_choice)
    do_vqa = _should_vqa(model_choice) and bool(question.strip())

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
    if _flag(feat_context):
        vc = B.build_vitals_context(bp, hr, temp, spo2, wbc, crp)
        if vc:
            ctx_parts.append(vc)
    if cls:
        ctx_parts.append(f"[CLASSIFICATION: {cls['label']} {cls['confidence']:.1%}]")
    context_str = "\n".join(ctx_parts)

    # ── VQA ───────────────────────────────────────────────────────────────────
    vqa_answer, vqa_conf = "", 0.0
    if do_vqa:
        vqa = B.answer_vqa(arr, question.strip(), context_str)
        vqa_answer = vqa["answer"]
        vqa_conf   = round(vqa["confidence"], 4)
    result["vqa_answer"] = vqa_answer
    result["vqa_conf"]   = vqa_conf

    # ── Grad-CAM / explainability overlay ────────────────────────────────────
    # Always return an image: model-backed Grad-CAM when available, otherwise
    # a deterministic fallback heatmap.
    cam_arr = B.generate_explainability_overlay(arr)
    gradcam_b64 = _img_to_b64(cam_arr)
    result["gradcam_b64"] = gradcam_b64
    result["gradcam_image"] = gradcam_b64

    # ── Differential diagnosis ────────────────────────────────────────────────
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
        "result":     result,
    })

    return result


@app.post("/predict")
async def predict(
    file:         UploadFile = File(...),
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
    )


# ─────────────────────────────────────────────────────────────────────────────
#  GET /api/history
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/history")
async def get_history(patient_id: str = "default") -> list:
    """Return stored report history for a patient."""
    return B.load_reports(patient_id)


@app.get("/reports")
async def get_reports(patient_id: str = "default") -> list:
    """Compatibility alias for frontend history."""
    return B.load_reports(patient_id)


@app.get("/report/{patient_id}")
async def get_report(patient_id: str = "default") -> list:
    """Return the full report list for one patient."""
    return B.load_reports(patient_id)


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "backend_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )
