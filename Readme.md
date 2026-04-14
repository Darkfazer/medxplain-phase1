# MedXplain: Comprehensive Medical AI & VQA Infrastructure

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

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
- [Installation & Quick Start](#installation)
- [Running the Applications](#running)
    - [Gradio VQA Demo](#gradio-demo)
    - [Mock Mode Execution](#mock-mode)
- [Evaluation & Benchmarking](#evaluation)

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
├── medical_vqa_infrastructure/  # Phase 2 Core: Pluggable Decoders/Encoders, Contextual Awareness, Mock workflows
├── vqa_app_deliverable/         # Front-end Application suite: Gradio Demo, Evaluation Scripts
├── models/                      # Architectural logic: CNNs (DenseNet, ResNet), Medical Transformers, VQA Adapters
├── vqa/                         # Medical VQA operations: OOD (Out-of-Distribution) Detectors, PubMed Retriever 
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

*For deeper implementation details about individual components (such as VQA architecture configurations and fine-tuning commands), consult the localized `README.md` files present in `medical_vqa_infrastructure/` and `vqa_app_deliverable/`.*
