# Medical VQA Infrastructure

A production-ready, modular PyTorch infrastructure for Medical Visual Question Answering.

## Core Features
1. **Pluggable Architecture**: Easily swap Vision Encoders (ViT, ResNet, BiomedCLIP) and Text Decoders (BioGPT, LLaMA).
2. **Clinical Focus**: Integrated explainability (Grad-CAM), Temperature Calibration (T=1.263), and Context/Longitudinal awareness.
3. **Mock Mode testing**: Run `export MOCK_MODE=1` to simulate the entire pipeline without real datasets or model weights.

## Getting Started

```bash
pip install -r requirements.txt
cp .env.example .env
pytest tests/
```

## Structure
- `models/`: Encoders, decoders, and cross-attention fusion logic.
- `data/`: Extensible PyTorch Dataset & Transforms.
- `training/`: AMP, Distributed Data Parallel (DDP), Custom losses.
- `evaluation/`: AUC/Accuracy/BLEU implementations & viz.
- `demos/`: Gradio and Streamlit interfaces.
