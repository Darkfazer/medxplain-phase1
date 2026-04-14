# MedXplain - Phase 2 VQA Application

This directory contains the production-ready code for the Medical Visual Question Answering (VQA) system Phase 2.

## Features Included
1. **Interactive Demo**: A robust Gradio application (`app_gradio.py`) demonstrating VQA, Grad-CAM overlays, and sub-second mock latency requirements.
2. **Advanced Clinical Modules**:
   - `Report-Aware Answering`: Correlates past reports with current contexts.
   - `Context-Aware Answering`: Ingests vitals and labs into the NLP pipeline.
   - `Longitudinal View`: Fuses consecutive time-point images to isolate differences.
   - `One-Click Reporting`: Automatically generates finding/impression structured reports.
   - `Differential Diagnosis`: Summarizes and ranks the top 3 underlying etiologies.
3. **Evaluation Suite**: (`evaluation.py`) generates AUC, Accuracy, BLEU, ROC curves, calibration errors mapping directly to standard output tables.

## Quickstart

### 1. Installation
This isolated application leverages HuggingFace `transformers`, `gradio`, and PyTorch alongside `grad-cam`.
```bash
pip install -r requirements.txt
```

### 2. Prepare Weights
Update the `Config.USE_MOCK_MODEL` mapping located within `config.py`.
- If set to `True`, the system runs a fast, dummy ViT backbone for seamless UI demonstration without heavy GPU requirements.
- To map real weights, flip the configuration parameter to `False` and assign your best `.pth`/transformer directories in `model_inference.py` (load_model hook).

### 3. Launch Demo
```bash
python app_gradio.py
```
A local webserver will launch at `http://0.0.0.0:7860/` allowing you full interactive access.

### 4. Run Evaluations
Use the `run_evaluation_suite` in `evaluation.py` and pass your batch arrays to produce high-resolution `roc_curve.png`, calibration plots, and the final CSV.
