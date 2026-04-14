# MedXplain: Comprehensive Medical AI & VQA Infrastructure

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)
![HIPAA](https://img.shields.io/badge/HIPAA-Compliant%20Design-green.svg)
![DICOM](https://img.shields.io/badge/DICOM-Supported-blue.svg)

MedXplain is an end-to-end multi-modal Medical Artificial Intelligence repository. The system bridges state-of-the-art Medical Image Classification logic with an advanced Visual Question Answering (VQA) framework tailored for the clinical domain.

This repository integrates everything from generic dataset ingestion and Convolutional Neural Networks (CNN) benchmarking to highly interpretable Cross-Attention multi-modal models, explainability visualization (Grad-CAM), and a production-grade Gradio application.

---

## 📑 Table of Contents

- [Project Overview & Architecture](#overview)
- [Repository Structure](#structure)
- [Key Capabilities & Modules](#capabilities)
    - [Phase 1: Image Classification](#phase-1-classification)
    - [Phase 2: Medical VQA](#phase-2-vqa)
    - [Explainability & Trust](#explainability)
- [🏥 Hospital-Grade Data Infrastructure](#hospital)
    - [DICOM Ingestion Pipeline](#dicom-pipeline)
    - [PHI Scrubber (HIPAA/GDPR)](#phi-scrubber)
    - [Radiology Report Parser](#report-parser)
    - [Hospital Config](#hospital-config)
- [Installation & Quick Start](#installation)
- [Running the Applications](#running)
    - [Gradio VQA Demo (Hospital Edition)](#gradio-demo)
    - [Mock Mode Execution](#mock-mode)
- [Evaluation & Benchmarking](#evaluation)
    - [Clinical Validation vs Radiologist](#clinical-validation)
- [Unit Tests](#tests)

---

<a name="overview"></a>
## 🌍 Project Overview & Architecture

MedXplain is designed in two modular phases:
1. **Phase 1 (Medical Image Classification):** Foundational model development to analyze chest X-rays, histopathology, etc. using models ranging from DenseNet121, Transformers, and TorchXRayVision. Built to benchmark medical datasets accurately while incorporating conformal prediction metrics for risk management.
2. **Phase 2 (Medical VQA Application):** Cross-modal integration, fusing visual embeddings from Phase 1 encoders with powerful Large Language Model (LLM) decoders (e.g., LLaMA, BioGPT). The VQA suite is augmented by features like retrieval-augmented answering via PubMed, longitudinal scanning, and clinical metadata integration (labs, patient notes).

---

<a name="structure"></a>
## 📂 Repository Structure

The architecture is highly modularizing training, inference, and UI into logical compartments:

```text
medxplain-simple/
├── data_ingestion/              # 🆕 Hospital-grade DICOM ingestion, PHI scrubbing, report parsing
│   ├── dicom_pipeline.py        #    DICOM loader → DICOMStudy tensors
│   ├── phi_scrubber.py          #    HIPAA/GDPR PHI removal + audit trail
│   └── report_parser.py         #    Structured extraction from radiology reports
├── config/
│   └── hospital_config.yaml     # 🆕 Centralised hospital deployment configuration
├── evaluation/
│   └── clinical_validation.py   # 🆕 Radiologist comparison + PDF discrepancy reports
├── tests/
│   ├── test_dicom_pipeline.py   # 🆕 Unit tests for the DICOM pipeline
│   └── test_phi_scrubber.py     # 🆕 Unit tests for the PHI scrubber
├── medical_vqa_infrastructure/  # Phase 2 Core: Pluggable Decoders/Encoders, Contextual Awareness, Mock workflows
├── vqa_app_deliverable/         # Front-end Application suite: Gradio Demo (Hospital Edition), Evaluation Scripts
├── models/                      # Architectural logic: CNNs (DenseNet, ResNet), Medical Transformers, VQA Adapters
├── vqa/                         # Medical VQA operations: OOD Detectors, PubMed Retriever
├── training/                    # Unified pipeline: Distributed trainers, custom losses, augmentations
├── evaluation/                  # Comprehensive benchmarking: AUC, BLEU, Statistical Analysis, Calibration Metrics
├── explainability/              # Interpretability tools: Grad-CAM, Integrated Gradients, Counterfactuals
├── mock_data/ & data/           # Testing datasets & DataLoader abstractions
└── configs/ & utils/            # Configurations, Utilities, and YAML schemas
```

---

<a name="capabilities"></a>
## 🚀 Key Capabilities & Modules

### Phase 1: Classification & Risk Management 
Located primarily in `models/` and `training/`.
- **Pre-configured Adapters**: Seamless ingestion of domain-specific architectures (MedViT, TorchXRayVision) alongside classic baselines (DenseNet121, EfficientNet).
- **Statistical Safety Net**: Incorporates rigorous statistical metrics, conformal prediction logic, and failure-mode analysis on medical data. 

### Phase 2: Medical VQA Application
Located in `medical_vqa_infrastructure/` and `vqa_app_deliverable/`.
- **Pluggable Architecture**: Dynamically swap Vision Encoders and Text Decoders to match latency or accuracy constraints.
- **Report & Context Aware**: Analyzes multi-modal inputs, correlating visual patterns directly with historical patient reports or longitudinal vitals.
- **Automated Summarization**: Extracts visual differential diagnoses natively.

### Explainability & Clinical Interfacing
Located in `explainability/` and `vqa_app_deliverable/`.
- **Visual Saliency**: Automatically trace predictions back to the image pixel-space with implementations of `grad_cam.py` and `integrated_gradients.py`. 
- **OOD Detection**: Catch anomalies before processing.

---

<a name="installation"></a>
## ⚙️ Installation & Quick Start

**Prerequisites:** Python 3.8+, CUDA-capable GPU (Recommended for Full Models), 16 GB+ RAM.

```bash
# 1. Clone the repository
git clone https://github.com/Darkfazer/medxplain-phase1.git
cd medxplain-simple

# 2. Virtual Environment Setup (Recommended)
python -m venv medxplain_env
source medxplain_env/bin/activate  # Windows: medxplain_env\Scripts\activate

# 3. Base Requirements
pip install -r requirements.txt

# 4. (Optional) Module-Specific Environments
# For isolated VQA Deliverable workflows
pip install -r vqa_app_deliverable/requirements.txt
# For Infrastructure dependencies
pip install -r medical_vqa_infrastructure/requirements.txt
```

---

<a name="running"></a>
## 🖥️ Running the Applications

### Launch the Gradio VQA Demo

The main interactive endpoint is available under the `vqa_app_deliverable` directory. It uses Gradio to create an interactive Web GUI where users can upload medical scans and ask diagnostic questions.

```bash
cd vqa_app_deliverable/
python app_gradio.py
```
*A local web server will spin up on `http://0.0.0.0:7860/`.*

### Mock Mode Execution 
Need to test pipelines without massive 50GB dataset downloads or GPU weights? Enable **Mock Mode** natively across the infrastructure:

```bash
# Windows
set MOCK_MODE=1

# Linux / Mac
export MOCK_MODE=1
```
This forces the backend logic (within `medical_vqa_infrastructure` and `vqa_app_deliverable`) to spoof predictions and bypass intense forward passes, speeding up UI testing and CI/CD operations cleanly.

---

<a name="evaluation"></a>
## 📊 Evaluation & Benchmarking

MedXplain supports a unified suite for stress-testing representations:

- **Benchmarking & Visualization Pipeline (`evaluation/benchmark.py`)**: Computes AUC, BLEU scores, exact match accuracy natively.
- **Diagnose Mismatches / Misclassified Samples**: Tracks false positives directly locally into `misclassified_samples/`.
- **Run comprehensive suites** directly through the testing APIs:
  ```bash
  python vqa_app_deliverable/evaluation.py
  ```

---

<a name="hospital"></a>
## 🏥 Hospital-Grade Data Infrastructure

These modules make MedXplain ready to accept **private hospital data** — real DICOM studies from your scanning equipment, directly from the PACS — instead of curated public datasets.

---

<a name="dicom-pipeline"></a>
### `data_ingestion/dicom_pipeline.py` — DICOM Ingestion

Loads DICOM files (`.dcm`) into de-identified, tensor-ready `DICOMStudy` objects.

| Feature | Detail |
|---|---|
| **Multi-frame support** | Handles CT/MR volume stacks as `(N, 1, H, W)` tensors and single CR/DX scans as `(1, H, W)` |
| **Transfer syntaxes** | Implicit/Explicit VR, JPEG Baseline, JPEG-LS, JPEG 2000 (via `pylibjpeg`) |
| **Metadata extraction** | Modality, study/series description, window center/width, pixel spacing, slice thickness |
| **Pseudonymisation** | PatientID & StudyInstanceUID → deterministic SHA-256 pseudonyms (salt via env var) |
| **Pixel normalisation** | VOI LUT + windowing → `float32` tensors in `[0, 1]` compatible with all existing encoders |

```python
from data_ingestion.dicom_pipeline import DICOMLoader

loader = DICOMLoader(target_size=(224, 224))
study  = loader.load_study("/hospital/pacs/study_001/")

print(study.modality)          # "CT"
print(study.patient_id)        # "a4f3c2..." (SHA-256 pseudonym)
print(len(study.images))       # Number of frames/slices
print(study.images[0].shape)   # torch.Size([1, 224, 224])
```

---

<a name="phi-scrubber"></a>
### `data_ingestion/phi_scrubber.py` — PHI Scrubber (HIPAA / GDPR)

Removes all Protected Health Information (PHI) from both DICOM headers and free-text radiology reports.

| Feature | Detail |
|---|---|
| **DICOM header** | Removes all tags in the DICOM Confidentiality Profile (PatientName, DOB, InstitutionName, etc.) |
| **UID pseudonymisation** | StudyInstanceUID / SOPInstanceUID → consistent `2.25.<hash>` UIDs |
| **Report scrubbing** | Regex patterns for MRNs, phones, dates, emails, physician names |
| **NER enhancement** | Optional spaCy `PERSON` entity removal (`en_core_web_sm` or `en_core_sci_lg`) |
| **Audit trail** | Every scrub operation logged to `./logs/phi_audit.log` with timestamp + file hash (no PHI) |

```python
from data_ingestion.phi_scrubber import scrub_dicom_study

clean_study = scrub_dicom_study(raw_study, keep_longitudinal_link=True)
# clean_study.report_text → "... MRN [REDACTED_MRN] ... [REDACTED_DATE] ..."
# clean_study.source_paths → []   (raw paths never stored in clean copy)
```

> **Note:** Set the `MEDXPLAIN_PSEUDO_SALT` environment variable to a strong hospital-specific secret before deployment to ensure pseudonymisation is irreversible without the key.

---

<a name="report-parser"></a>
### `data_ingestion/report_parser.py` — Radiology Report Parser

Extracts structured clinical information from free-text radiology reports using regex heuristics. **Fully offline — no model required.**

```python
from data_ingestion.report_parser import ReportParser

parser = ReportParser()
result = parser.parse(report_text)

# result["impression"]  → "Small right pleural effusion. No pneumothorax."
# result["findings"]["pleural_effusion"] → {"present": True, "laterality": "right", "severity": "small"}
# result["findings"]["pneumothorax"]     → {"present": False}
# result["is_normal"]  → False
```

**Catalogued findings** (extensible): pneumothorax, pleural effusion, pneumonia, atelectasis, cardiomegaly, pulmonary edema, nodule, fracture, pneumoperitoneum, pleural thickening, hilar enlargement, interstitial pattern, aortic widening, pacemaker/device.

---

<a name="hospital-config"></a>
### `config/hospital_config.yaml` — Hospital Deployment Config

Single source of truth for all hospital-specific settings. **Never commit with real values.**

```yaml
hospital:
  name: "[REDACTED]"
  data_root: "/data/hospital/dicom/"

privacy:
  phi_scrubbing: true
  allow_cloud_upload: false
  audit_log_path: "./logs/phi_audit.log"

model:
  vision_encoder: "biomedclip"
  use_lora: true
  lora_rank: 8

inference:
  batch_size: 1
  device: "cuda:0"
  enable_explainability: true
```

Override sensitive fields via environment variables (e.g. `MEDXPLAIN_DATA_ROOT`, `MEDXPLAIN_PSEUDO_SALT`) instead of editing the file.

---

<a name="clinical-validation"></a>
### `evaluation/clinical_validation.py` — Radiologist Comparison

Generates a validation report measuring agreement between model predictions and radiologist ground-truth.

```python
from evaluation.clinical_validation import generate_validation_report

metrics = generate_validation_report(
    model=vqa_model,
    validation_csv="data/radiologist_annotations.csv",  # columns: image_id, question, radiologist_answer
    output_dir="discrepancy_reports/",
)
# metrics → {"accuracy": 0.82, "cohen_kappa": 0.74, "n_discrepancies": 18, "n_total": 100}
```

For every disagreement a **PHI-free PDF** is generated in `discrepancy_reports/` containing:
- The source image
- The question, model answer + confidence score
- The radiologist answer
- The Grad-CAM heatmap overlay

---

<a name="tests"></a>
## 🧪 Unit Tests

Tests are located in `tests/` and can be run without real DICOM files or GPU.

```bash
# Install test dependencies
pip install pytest pydicom torch numpy scikit-learn

# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_dicom_pipeline.py -v
pytest tests/test_phi_scrubber.py -v
```

| Test file | Coverage |
|---|---|
| `test_dicom_pipeline.py` | Pseudonymisation, pixel normalisation (window/minmax/uniform), multi-frame shape, metadata extraction, PHI field absence |
| `test_phi_scrubber.py` | Report text redaction (MRN/phone/date/email/physician), metadata scrubbing, longitudinal linkage, audit hashing, full scrub_dicom_study integration |

---

*For deeper implementation details about individual components (such as VQA architecture configurations and fine-tuning commands), consult the localized `README.md` files present in `medical_vqa_infrastructure/` and `vqa_app_deliverable/`.*
