# MedXplain – Medical Visual Question Answering

A clinical-grade multi-modal AI system combining **TorchXRayVision** (chest X-ray classification + Grad-CAM) with **BLIP-1** (visual question answering).

---

## Requirements

- Python 3.10+
- CUDA 12.1+ (optional but recommended)

---

## Installation

```bash
# 1. Create a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/macOS

# 2. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (Optional) Install as an editable package so imports work from anywhere
pip install -e .
```

---

## Running the System

### Option A – FastAPI + HTML UI (recommended)

```bash
# Start the backend (port 8000)
python backend_api.py

# Then open medxplain_ui.html in your browser, OR visit:
#   http://localhost:8000
```

### Option B – Gradio UI

```bash
python app.py
# Visit: http://localhost:7860
```

### Option C – Both simultaneously (Windows)

```bat
run.bat
```

---

## Training

### Fine-tune BLIP-1 on VQA-RAD

1. Place `VQA_RAD Dataset Public.json` and the image folder under `data/`.
2. Run:

```bash
python -m training.train
```

### Two-phase training (contrastive + LoRA fine-tuning)

```bash
python -m training.train_vqa_two_phase --phase1_epochs 10 --phase2_epochs 20
```

Checkpoints are saved under `checkpoints/` only when validation improves.

---

## Evaluation

```bash
python -m evaluation.metrics
```

For clinical validation against radiologist annotations:

```python
from evaluation.clinical_validation import generate_validation_report
metrics = generate_validation_report(
    model=my_vqa_model,
    validation_csv="data/radiologist_annotations.csv",
    output_dir="discrepancy_reports/",
)
print(metrics)
```

---

## Project Structure

```
medxplain-simple/
├── backend.py              # Core ML logic (classification, VQA, Grad-CAM, DB)
├── backend_api.py          # FastAPI server wrapping backend.py
├── app.py                  # Gradio UI (calls backend.py directly)
├── medxplain_ui.html       # React HTML UI (calls backend_api.py)
├── calibrated_inference.py # Temperature scaling for DenseNet
├── requirements.txt
├── configs/config.py       # Unified config (MODEL_NAME, paths, split ratios)
├── data/
│   ├── dataset.py          # VQA-RAD dataset (70/15/15 split, augment train only)
│   └── utils.py            # Image loading utilities
├── models/
│   ├── base_classifier.py  # Abstract base
│   ├── vqa_model.py        # BLIP-1 wrapper (blip-vqa-base)
│   └── cnn_models/
│       └── densenet_adapter.py
├── training/
│   ├── train.py            # BLIP-1 fine-tuning
│   ├── trainer.py          # Unified training loop (AMP, SWA, early stopping)
│   ├── train_vqa_two_phase.py
│   ├── losses.py           # BCE, FocalLoss
│   └── metrics.py          # AUC, F1, Sensitivity, Specificity
├── explainability/
│   └── grad_cam.py         # Single Grad-CAM implementation (pytorch-grad-cam)
└── evaluation/
    ├── metrics.py           # BLEU + Accuracy metrics
    └── clinical_validation.py  # PDF discrepancy reports, Cohen's kappa
```

---

## Notes

- **Model:** BLIP-1 (`Salesforce/blip-vqa-base`) is used for both training and inference.  Checkpoints produced by `training/train.py` are directly loadable by `backend.py`.
- **Grad-CAM:** Uses the `pytorch-grad-cam` library (package name: `grad-cam`). No `use_cuda` argument needed; device is set by model placement.
- **DICOM:** Accepted at the API and Gradio UI level. Requires `pydicom`.
- **PHI:** No real PHI redaction is implemented. The disclaimer is for research use only.

---

## Citation

```bibtex
@misc{medxplain2026,
  title        = {MedXplain: Explainable Medical VQA},
  author       = {Your Name},
  year         = {2026},
  howpublished = {GitHub},
}
```
