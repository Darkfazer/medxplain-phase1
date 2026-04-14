import gradio as gr
from config import Config
from model_inference import MedicalVQAModel
from advanced_features import AdvancedClinicalFeatures

# Initialize models and feature wrappers globally to persist weights in memory
print("Initializing models...")
vqa_model = MedicalVQAModel(use_mock=Config.USE_MOCK_MODEL)
advanced_feats = AdvancedClinicalFeatures(vqa_model)

# Functions for Gradio UI Callbacks
def tab1_vqa_heatmap(image, question):
    if image is None:
        return "Please upload an image.", None
    answer, conf = vqa_model.generate_answer(image, question)
    gradcam_img, _ = vqa_model.generate_gradcam(image)
    return answer, gradcam_img

def tab2_report_aware(image, prior_report, question):
    if image is None:
        return "Please upload an image."
    answer, conf = advanced_feats.report_aware_answering(image, prior_report, question)
    return answer

def tab3_context_aware(image, clinical_context, question):
    if image is None:
        return "Please upload an image."
    answer, conf = advanced_feats.context_aware_answering(image, clinical_context, question)
    return answer

def tab4_longitudinal(image_curr, image_prior, question):
    if image_curr is None or image_prior is None:
        return "Please upload both images."
    answer, conf = advanced_feats.compare_longitudinal(image_curr, image_prior, question)
    return answer

def tab5_differential(image, question):
    if image is None:
        return "Please upload an image."
    top_3 = advanced_feats.differential_diagnosis(image, question)
    response = ""
    for k, v in top_3.items():
        response += f"**{k}**: {v}\n\n"
    return response

def tab6_one_click(image):
    if image is None:
        return "Please upload an image."
    return advanced_feats.generate_one_click_report(image)

# Gradio Block UI
with gr.Blocks(title="MedXplain VQA - Phase 2") as demo:
    gr.Markdown("# MedXplain: Clinical VQA System")
    gr.Markdown("An interactive system for Chest X-Ray analysis with Explainable AI.")
    
    with gr.Tabs():
        # Tab 1
        with gr.TabItem("VQA & Heatmaps"):
            with gr.Row():
                with gr.Column():
                    t1_img = gr.Image(type="pil", label="Upload X-Ray")
                    t1_qst = gr.Textbox(label="Clinical Question", placeholder="e.g., Is there cardiomegaly?")
                    t1_btn = gr.Button("Analyze")
                with gr.Column():
                    t1_ans = gr.Textbox(label="VQA Answer")
                    t1_cam = gr.Image(type="numpy", label="Grad-CAM Visualization")
            t1_btn.click(fn=tab1_vqa_heatmap, inputs=[t1_img, t1_qst], outputs=[t1_ans, t1_cam])
            
        # Tab 2
        with gr.TabItem("Report-Aware"):
            with gr.Row():
                with gr.Column():
                    t2_img = gr.Image(type="pil", label="Upload Current X-Ray")
                    t2_rep = gr.Textbox(label="Prior Radiology Report", lines=4)
                    t2_qst = gr.Textbox(label="Question", placeholder="How does this compare?")
                    t2_btn = gr.Button("Analyze")
                with gr.Column():
                    t2_ans = gr.Textbox(label="Answer")
            t2_btn.click(fn=tab2_report_aware, inputs=[t2_img, t2_rep, t2_qst], outputs=[t2_ans])
            
        # Tab 3
        with gr.TabItem("Context-Aware"):
            with gr.Row():
                with gr.Column():
                    t3_img = gr.Image(type="pil", label="Upload X-Ray")
                    t3_ctx = gr.Textbox(label="Clinical Notes / Vitals / Labs", lines=4)
                    t3_qst = gr.Textbox(label="Question", placeholder="What is the most likely diagnosis given labs?")
                    t3_btn = gr.Button("Analyze")
                with gr.Column():
                    t3_ans = gr.Textbox(label="Answer")
            t3_btn.click(fn=tab3_context_aware, inputs=[t3_img, t3_ctx, t3_qst], outputs=[t3_ans])
            
        # Tab 4
        with gr.TabItem("Longitudinal View"):
            with gr.Row():
                with gr.Column():
                    t4_img1 = gr.Image(type="pil", label="Current X-Ray")
                    t4_img2 = gr.Image(type="pil", label="Prior X-Ray")
                    t4_qst  = gr.Textbox(label="Question", placeholder="Is the nodule growing?")
                    t4_btn  = gr.Button("Compare")
                with gr.Column():
                    t4_ans = gr.Textbox(label="Comparative Analysis")
            t4_btn.click(fn=tab4_longitudinal, inputs=[t4_img1, t4_img2, t4_qst], outputs=[t4_ans])
            
        # Tab 5
        with gr.TabItem("Differential Diagnosis"):
            with gr.Row():
                with gr.Column():
                    t5_img = gr.Image(type="pil", label="Upload X-Ray")
                    t5_qst = gr.Textbox(label="Question", placeholder="What are the differential diagnoses?")
                    t5_btn = gr.Button("Rank Diagnoses")
                with gr.Column():
                    t5_ans = gr.Markdown("### Results will appear here")
            t5_btn.click(fn=tab5_differential, inputs=[t5_img, t5_qst], outputs=[t5_ans])
            
        # Tab 6
        with gr.TabItem("One-Click Report"):
            with gr.Row():
                with gr.Column():
                    t6_img = gr.Image(type="pil", label="Upload X-Ray")
                    t6_btn = gr.Button("Generate Structured Draft")
                with gr.Column():
                    t6_ans = gr.Textbox(label="Auto-Generated Draft Report", lines=10)
            t6_btn.click(fn=tab6_one_click, inputs=[t6_img], outputs=[t6_ans])

if __name__ == "__main__":
    demo.launch(server_port=Config.GRADIO_PORT, server_name="0.0.0.0")
