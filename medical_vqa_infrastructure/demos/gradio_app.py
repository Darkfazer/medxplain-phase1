import gradio as gr

def dummy_inference(image, question):
    return "Mock Diagnosis: Normal", "Confidence: 0.95"

with gr.Blocks() as demo:
    gr.Markdown("# Medical VQA System")
    with gr.Row():
        img_input = gr.Image(type="pil")
        question_input = gr.Textbox(label="Clinical Question")
        
    out_answer = gr.Textbox(label="Answer")
    out_conf = gr.Textbox(label="Confidence")
    
    btn = gr.Button("Analyze")
    btn.click(fn=dummy_inference, inputs=[img_input, question_input], outputs=[out_answer, out_conf])

if __name__ == "__main__":
    demo.launch()
