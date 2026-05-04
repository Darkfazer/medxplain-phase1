"""
MedXplain – Gradio UI  (app.py)
================================
Run with:  python app.py

Uses backend.py directly (no HTTP round-trip).
"""
import sys
import time
import logging
import tempfile
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

import backend as B

APP_VER    = "1.0.0"
PATIENT_ID = "default"

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────
MODELS = [
    "Ensemble (Classification + VQA)",
    "TorchXRayVision (Classification only)",
    "BLIP-VQA (Visual Q&A only)",
]
RISK_ICON = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


# ─────────────────────────────────────────────────────────────────────────────
#  Core handler
# ─────────────────────────────────────────────────────────────────────────────
def handle_send(
    image, model_choice, mode, question,
    feat_report, feat_context, feat_longitudinal, feat_oneclik, feat_diff,
    bp, hr, temp, spo2, wbc, crp,
    sess,
):
    """Main analysis handler – calls backend.py directly, returns updated state.

    Returns
    -------
    tuple: (new_chat_messages, sess, cam_update, report_dl_update, question_reset)
    """
    if image is None:
        return (
            [{"role": "assistant", "content": "⚠️ Please upload a medical image first."}],
            sess,
            gr.update(visible=False),
            gr.update(visible=False),
            "",
        )

    arr = np.array(image.convert("RGB"))

    # ── Classification ────────────────────────────────────────────────────────
    do_classify = model_choice in (
        "Ensemble (Classification + VQA)", "TorchXRayVision (Classification only)"
    )
    cls = None
    if do_classify:
        try:
            cls = B.classify_image(arr, model_choice)
        except Exception as e:
            return (
                [{"role": "assistant", "content": f"❌ Classification error: {e}"}],
                sess, gr.update(visible=False), gr.update(visible=False), "",
            )

    # ── Build VQA context ─────────────────────────────────────────────────────
    context_parts: list[str] = []
    if feat_report:
        rc = B.build_report_context(PATIENT_ID)
        if rc:
            context_parts.append(rc)
    if feat_context:
        vc = B.build_vitals_context(bp, hr, temp, spo2, wbc, crp)
        if vc:
            context_parts.append(vc)
    if cls:
        context_parts.append(
            f"[CLASSIFICATION: {cls['label']} {cls['confidence']:.1%}]"
        )
    context_str = "\n".join(context_parts)

    # ── VQA ───────────────────────────────────────────────────────────────────
    do_vqa = (
        model_choice in ("Ensemble (Classification + VQA)", "BLIP-VQA (Visual Q&A only)")
        and question and question.strip()
    )
    vqa = None
    if do_vqa:
        try:
            vqa = B.answer_vqa(arr, question.strip(), context_str)
        except Exception as e:
            return (
                [{"role": "assistant", "content": f"❌ VQA error: {e}"}],
                sess, gr.update(visible=False), gr.update(visible=False), "",
            )

    # ── Longitudinal ──────────────────────────────────────────────────────────
    long_text = ""
    if feat_longitudinal:
        long_text = B.longitudinal_compare(arr, PATIENT_ID)
    B.save_prior_image(PATIENT_ID, arr)

    # ── Differential ─────────────────────────────────────────────────────────
    diff_text = ""
    if feat_diff and cls and cls.get("all_probs"):
        diff_text = B.build_differential(cls["all_probs"], top_k=3)

    # ── One-Click Report ──────────────────────────────────────────────────────
    report_dl_update = gr.update(visible=False)
    report_txt = ""
    if feat_oneclik and cls:
        report_txt = B.build_structured_report(
            pid=PATIENT_ID,
            cls=cls,
            vqa_answer=vqa["answer"] if vqa else "",
            diff_text=diff_text,
            model_choice=model_choice,
        )
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=".txt",
            prefix="medxplain_report_", mode="w", encoding="utf-8",
        )
        tmp.write(report_txt)
        tmp.close()
        report_dl_update = gr.update(value=tmp.name, visible=True)

    # ── Persist ───────────────────────────────────────────────────────────────
    B.save_report_entry(PATIENT_ID, {
        "date":       time.strftime("%Y-%m-%d"),
        "prediction": cls["label"] if cls else "VQA only",
        "vqa_answer": vqa["answer"] if vqa else "",
        "report":     report_txt,
    })

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    cam_update = gr.update(visible=False)
    if mode == "Doctor Assistant" and do_classify and cls:
        try:
            cam_arr = B.generate_gradcam(arr)
            cam_update = gr.update(value=Image.fromarray(cam_arr), visible=True)
        except Exception as e:
            logging.warning("Grad-CAM failed: %s", e)

    # ── Build response markdown ───────────────────────────────────────────────
    rows = ["**🩻 MedXplain Analysis**", ""]

    if cls:
        label, conf, ci, risk = (
            cls["label"], cls["confidence"], cls["interval"], cls["risk"]
        )
        rows += [
            "| | |", "|---|---|",
            f"| **Prediction** | {label} |",
            f"| **Confidence** | {conf:.1%} |",
            f"| **Uncertainty band** | [{ci[0]:.3f} – {ci[1]:.3f}] |",
            f"| **Risk** | {RISK_ICON.get(risk, '⚪')} {risk} |",
            "",
        ]

    if vqa:
        rows += [
            "---",
            f"**Q:** _{question.strip()}_", "",
            f"**A:** {vqa['answer']}",
            f"**VQA confidence:** {vqa['confidence']:.1%}", "",
        ]

    if diff_text:
        rows += ["---", "**🔬 Differential Diagnosis**", "", diff_text, ""]

    if long_text:
        rows += ["---", f"**📅 Longitudinal:** {long_text}", ""]

    if mode == "Doctor Assistant":
        rows += ["---", "**🔥 Grad-CAM** heatmap displayed above.", ""]

    if feat_oneclik and report_txt:
        rows += [
            "---",
            "**📄 One-Click Report** generated — download link below.", "",
        ]

    rows += [
        "---",
        f"🛡️ PHI auto-redaction active · 🏥 {model_choice} · v{APP_VER}",
    ]

    response = "\n".join(rows)

    # Build messages list for Gradio Chatbot (type="messages")
    messages = []
    if question.strip():
        messages.append({"role": "user", "content": question.strip()})
    messages.append({"role": "assistant", "content": response})

    # Update session history
    label_for_hist = cls["label"] if cls else "VQA Query"
    sess = sess or []
    sess.insert(0, {
        "label": label_for_hist,
        "date":  time.strftime("%m/%d/%Y"),
        "mode":  mode,
    })

    return messages, sess, cam_update, report_dl_update, ""


def render_hist(sess: list) -> str:
    if not sess:
        return "_No previous sessions yet._"
    return "\n\n".join(
        f"🗂 **{s['label']}**  \n"
        f"<span style='color:#94a3b8;font-size:.75rem'>"
        f"{s['date']} · {s['mode']}</span>"
        for s in sess[:10]
    )


# ─────────────────────────────────────────────────────────────────────────────
#  CSS
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
*,*::before,*::after{box-sizing:border-box;}
body,.gradio-container{
  font-family:'Inter',system-ui,-apple-system,sans-serif !important;
  background:#f4f5f7 !important; color:#111827 !important;
  margin:0 !important; padding:0 !important;
}
footer,.built-with{display:none !important;}
.contain{max-width:100% !important; padding:0 !important;}
#shell{display:flex !important; min-height:100vh; align-items:stretch;}
#sidebar{
  width:280px !important; min-width:280px !important; max-width:280px !important;
  background:#fff !important; border-right:1px solid #e5e7eb !important;
  padding:24px 16px 16px !important;
  display:flex !important; flex-direction:column !important;
  gap:0 !important; align-items:stretch !important; min-height:100vh;
}
#sb-logo .markdown-text h2{font-size:1.25rem !important;font-weight:700 !important;color:#111827 !important;margin:0 0 2px !important;}
#sb-logo .markdown-text p{font-size:.72rem !important;color:#6b7280 !important;margin:0 !important;}
#sb-badge .markdown-text p{
  display:inline-block !important; background:#f0fdf4 !important;
  color:#15803d !important; border:1px solid #bbf7d0 !important;
  border-radius:999px !important; padding:2px 10px !important;
  font-size:.72rem !important; font-weight:500 !important; margin:8px 0 10px !important;
}
#sb-model label{font-size:.68rem !important;font-weight:600 !important;text-transform:uppercase !important;letter-spacing:.05em !important;color:#9ca3af !important;}
#sb-model select,#sb-model .wrap-inner{border:1px solid #e5e7eb !important;border-radius:7px !important;background:#fff !important;font-size:.83rem !important;color:#111827 !important;}
#btn-new button{
  background:#fff !important; color:#374151 !important;
  border:1.5px solid #d1d5db !important; border-radius:8px !important;
  font-weight:600 !important; font-size:.85rem !important;
  width:100% !important; padding:9px 0 !important;
  cursor:pointer !important; transition:all .15s !important; margin:2px 0 12px !important;
}
#btn-new button:hover{background:#eff6ff !important;border-color:#2563eb !important;color:#2563eb !important;}
#sb-hist-label .markdown-text p{font-size:.68rem !important;font-weight:700 !important;text-transform:uppercase !important;letter-spacing:.07em !important;color:#9ca3af !important;margin:0 0 6px !important;}
#sb-history .markdown-text p{font-size:.82rem !important;padding:6px 8px !important;border-radius:7px !important;margin:2px 0 !important;}
#sb-spacer{flex:1 !important;}
#sb-footer .markdown-text p{font-size:.67rem !important;color:#d1d5db !important;margin:0 !important;}
#main{flex:1 !important;display:flex !important;flex-direction:column !important;align-items:center !important;background:#f4f5f7 !important;padding:0 !important;overflow-y:auto !important;}
#main-inner{width:100% !important;max-width:680px !important;display:flex !important;flex-direction:column !important;align-items:center !important;padding:36px 24px 80px !important;}
#upload-box{width:100% !important;max-width:540px !important;margin:0 auto 12px !important;}
#feat-row{width:100% !important;max-width:540px !important;margin:0 auto 12px !important;gap:6px !important;flex-wrap:wrap !important;}
#mode-wrap{width:100% !important;max-width:540px !important;margin:0 auto 12px !important;}
#chatbox{width:100% !important;max-width:540px !important;margin:0 auto 12px !important;}
#chatbox .message.bot{background:#fff !important;border:1px solid #e5e7eb !important;border-radius:12px !important;font-size:.88rem !important;line-height:1.65 !important;color:#111827 !important;max-width:100% !important;box-shadow:0 1px 3px rgba(0,0,0,.06) !important;}
#chatbox .message.user{background:#2563eb !important;color:#fff !important;border-radius:12px !important;font-size:.88rem !important;max-width:80% !important;}
#cambox{width:100% !important;max-width:540px !important;margin:0 auto 12px !important;}
#report-dl{width:100% !important;max-width:540px !important;margin:0 auto 8px !important;}
#input-row{width:100% !important;max-width:540px !important;margin:0 auto !important;}
#btn-send button{background:#2563eb !important;color:#fff !important;border:none !important;border-radius:10px !important;font-size:1rem !important;font-weight:600 !important;padding:10px 18px !important;cursor:pointer !important;min-width:52px !important;}
#disclaimer .markdown-text p{font-size:.7rem !important;color:#d1d5db !important;text-align:center !important;margin-top:16px !important;border-top:1px solid #e5e7eb !important;padding-top:12px !important;line-height:1.5 !important;}
"""

# ─────────────────────────────────────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="MedXplain – Medical AI", css=CSS) as demo:

    sess_state = gr.State([])

    with gr.Row(elem_id="shell", equal_height=False):

        # ── SIDEBAR ──────────────────────────────────────────────────────────
        with gr.Column(elem_id="sidebar", scale=0, min_width=280):
            gr.Markdown("## 🩻 MedXplain\nMulti-Modal Medical AI", elem_id="sb-logo")
            gr.Markdown("● Live Model", elem_id="sb-badge")

            model_dd = gr.Dropdown(
                MODELS, value=MODELS[0],
                label="Model", elem_id="sb-model", interactive=True,
            )
            gr.Markdown("---")
            btn_new = gr.Button("＋  New Report", elem_id="btn-new")
            gr.Markdown("HISTORY", elem_id="sb-hist-label")
            hist_md = gr.Markdown("_No previous sessions yet._", elem_id="sb-history")
            gr.HTML('<div id="sb-spacer" style="flex:1"></div>')
            gr.Markdown(f"v{APP_VER} · HIPAA / GDPR compliant", elem_id="sb-footer")

        # ── MAIN ─────────────────────────────────────────────────────────────
        with gr.Column(elem_id="main", scale=1):
            with gr.Column(elem_id="main-inner"):

                img_input = gr.Image(
                    label="Upload medical image (X-ray, CT, MRI, Ultrasound, Pathology)",
                    type="pil",
                    sources=["upload", "clipboard"],
                    elem_id="upload-box",
                    height=148,
                )

                # Clinical Context
                with gr.Accordion("🩺 Clinical Context (vitals & labs)", open=False):
                    gr.Markdown(
                        "_Enabled automatically when Context-Aware feature is on._"
                    )
                    with gr.Row():
                        bp   = gr.Textbox(label="BP (mmHg)",  placeholder="120/80", scale=1)
                        hr   = gr.Textbox(label="HR (bpm)",   placeholder="72",     scale=1)
                        temp = gr.Textbox(label="Temp (°C)",  placeholder="37.2",   scale=1)
                        spo2 = gr.Textbox(label="SpO2 (%)",   placeholder="98",     scale=1)
                    with gr.Row():
                        wbc = gr.Textbox(label="WBC (×10⁹/L)", placeholder="7.5", scale=1)
                        crp = gr.Textbox(label="CRP (mg/L)",   placeholder="2.1",  scale=1)

                # Feature toggles
                with gr.Row(elem_id="feat-row"):
                    feat_report = gr.Checkbox(label="📋 Report-Aware",     value=False)
                    feat_ctx    = gr.Checkbox(label="🧪 Context-Aware",    value=False)
                    feat_long   = gr.Checkbox(label="📅 Longitudinal",     value=False)
                    feat_click  = gr.Checkbox(label="📄 One-Click Report", value=False)
                    feat_diff   = gr.Checkbox(label="🔬 Differential Dx",  value=False)

                # Mode
                mode = gr.Radio(
                    ["Standard", "Doctor Assistant"],
                    value="Standard", label="Mode", elem_id="mode-wrap",
                )

                # Chat (type="messages" is the Gradio 4+ API)
                chatbot = gr.Chatbot(
                    value=[],
                    height=340,
                    show_label=False,
                    elem_id="chatbox",
                    type="messages",
                    render_markdown=True,
                    visible=False,
                )

                # Grad-CAM output
                cam_out = gr.Image(
                    label="🔥 Grad-CAM Explainability",
                    type="pil",
                    interactive=False,
                    visible=False,
                    elem_id="cambox",
                )

                # Report download
                report_dl = gr.File(
                    label="📄 Download Clinical Report",
                    visible=False,
                    elem_id="report-dl",
                )

                # Input row
                with gr.Row(elem_id="input-row"):
                    question = gr.Textbox(
                        placeholder="Describe symptoms or ask a clinical question…",
                        lines=2, max_lines=5, show_label=False, scale=9,
                    )
                    btn_send = gr.Button(
                        "➤", variant="primary", elem_id="btn-send", scale=1
                    )

                gr.Markdown(
                    "⚠️ MedXplain is a research tool. Verify with a clinician. "
                    "Not for standalone diagnosis.",
                    elem_id="disclaimer",
                )

    # ── EVENT WIRING ─────────────────────────────────────────────────────────
    send_ins = [
        img_input, model_dd, mode, question,
        feat_report, feat_ctx, feat_long, feat_click, feat_diff,
        bp, hr, temp, spo2, wbc, crp,
        sess_state,
    ]

    def _send(*args):
        msgs, sess, cam_upd, rep_upd, q_out = handle_send(*args)
        return (
            gr.update(value=msgs, visible=bool(msgs)),
            sess,
            cam_upd,
            rep_upd,
            render_hist(sess),
            q_out,
        )

    def _new(sess):
        return (
            gr.update(value=[], visible=False),
            sess,
            gr.update(visible=False),
            gr.update(visible=False),
            None,
            "",
            render_hist(sess),
        )

    send_outs = [chatbot, sess_state, cam_out, report_dl, hist_md, question]

    btn_send.click(fn=_send, inputs=send_ins, outputs=send_outs)
    question.submit(fn=_send, inputs=send_ins, outputs=send_outs)
    btn_new.click(
        fn=_new,
        inputs=[sess_state],
        outputs=[chatbot, sess_state, cam_out, report_dl, img_input, question, hist_md],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
