# MedXplain-Phase1: Medical Visual Question Answering (VQA) System

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)

MedXplain is a Medical Visual Question Answering (VQA) system designed to answer questions about medical images. This phase-1 implementation includes complete training, evaluation, and explainability pipelines.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training Pipeline](#training-pipeline)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Explainability](#explainability)
- [Debugging Tools](#debugging-tools)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)

---

## Features

- Multi-modal VQA model combining vision and language understanding
- Cross-attention mechanisms for image-question interaction
- Comprehensive training pipeline with fine-tuning support
- Model explainability tools for medical AI interpretability
- Ensemble learning capabilities
- Extensive debugging and diagnostic tools
- Real-time demo application

---

## Installation

**Prerequisites:** Python 3.8+, CUDA-capable GPU (recommended), 16 GB+ RAM

```bash
# 1. Clone the repository
git clone https://github.com/Darkfazer/medxplain-phase1.git
cd medxplain-phase1

# 2. Create a virtual environment (optional but recommended)
python -m venv medxplain_env
source medxplain_env/bin/activate  # Windows: medxplain_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Dataset Preparation

### Required Structure

```
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── annotations/
│   ├── vqa_train.json
│   ├── vqa_val.json
│   └── vqa_test.json
└── questions/
    ├── questions_train.json
    ├── questions_val.json
    └── questions_test.json
```

### Dataset Configuration

Update `configs/dataset_config.yaml`:

```yaml
data:
  image_dir: "path/to/your/images"
  annotation_dir: "path/to/annotations"
  question_dir: "path/to/questions"
  batch_size: 32
  num_workers: 4
```

### Data Quality Check

```bash
python check_data_quality.py --data_dir ./data
```

---

## Training Pipeline

### Quick Start

```bash
python vqa_finetune.py
```

### Basic Training with Custom Config

```bash
python vqa_finetune.py \
    --config ./configs/training_config.yaml \
    --epochs 50 \
    --batch_size 32 \
    --learning_rate 1e-4
```

### Fine-tuning a Pretrained Model

```bash
python vqa_finetune.py \
    --mode finetune \
    --pretrained_model ./checkpoints/baseline_model.pth \
    --freeze_backbone False \
    --target_modules "cross_attention,classifier"
```

### Targeted Training

```bash
python run_target_training.py \
    --target "cross_attention" \
    --dataset "medical" \
    --epochs 30 \
    --lr_scheduler cosine
```

### Learning Rate Finder

```bash
python find_lr.py \
    --model vqa_model \
    --start_lr 1e-7 \
    --end_lr 10 \
    --num_iter 100
```

### Ensemble Training

```bash
python ensemble.py \
    --n_models 5 \
    --strategy bagging \
    --save_dir ./ensemble_models
```

### Windows

```bat
run_pipeline.bat --mode train --config configs/training_config.yaml
```

---

## Configuration

`configs/training_config.yaml`:

```yaml
model:
  name: "medvqa_transformer"
  vision_encoder: "resnet50"
  text_encoder: "biobert"
  hidden_size: 768
  num_attention_heads: 12

training:
  optimizer: "adamw"
  learning_rate: 1e-4
  weight_decay: 0.01
  scheduler:
    name: "cosine"
    warmup_steps: 1000

data:
  image_size: 224
  max_question_length: 64
  answer_vocab_size: 1000

logging:
  log_dir: "./logs"
  save_interval: 5
  eval_interval: 1
```

---

## Evaluation

### Basic Evaluation

```bash
python -m evaluation.evaluate \
    --checkpoint ./checkpoints/best_model.pth \
    --test_data ./data/test
```

### Generate Accuracy Report

```bash
python diagnose_accuracy.py \
    --model_path ./checkpoints/best_model.pth \
    --output ./accuracy_report.txt \
    --detailed True
```

### Analyze VQA Mismatches

```bash
python analyze_vqa_mismatch.py \
    --predictions ./results/predictions.json \
    --ground_truth ./data/annotations/vqa_test.json \
    --output ./mismatch_analysis.json
```

---

## Explainability

### Generate Model Explanations

```bash
python -m explainability.generate_explanations \
    --model ./checkpoints/best_model.pth \
    --image ./sample_images/chest_xray.jpg \
    --question "What abnormality is present?" \
    --method grad_cam \
    --output ./explanations/
```

### Cross-Attention Visualization

```bash
python debug_cross_attention.py \
    --model ./checkpoints/best_model.pth \
    --sample ./data/sample.json \
    --visualize True
```

### Quick Demo

```bash
python app.py --port 8501 --model ./checkpoints/best_model.pth
```

Then open your browser to `http://localhost:8501`

---

## Debugging Tools

### DataLoader Debug

```bash
python debug_dataloader.py \
    --config ./configs/dataset_config.yaml \
    --num_batches 10 \
    --check_shapes True
```

### Single Sample Test

```bash
python test_single.py \
    --image ./test_image.jpg \
    --question "What is shown in this image?" \
    --model ./checkpoints/best_model.pth
```

### RAG Test

```bash
python test_rag.py \
    --query "pneumonia findings" \
    --retriever ./models/retriever.pth \
    --generator ./models/generator.pth
```

---

## Project Structure

```
medxplain-phase1/
├── configs/            # Configuration files
├── models/             # Model architectures
├── training/           # Training scripts
├── evaluation/         # Evaluation metrics and tools
├── explainability/     # Model interpretability
├── utils/              # Helper functions
├── experiments/        # Experiment tracking
├── vqa/                # Core VQA implementation
├── app.py              # Demo application
├── vqa_finetune.py     # Main training script
├── requirements.txt    # Dependencies
└── *.py                # Various utility scripts
```

> Checkpoints are saved to `./checkpoints/` by default. Logs are stored in `./logs/` for TensorBoard visualization. All scripts support `--help` for detailed usage. For distributed training, use `torch.distributed.launch`.

---

## Troubleshooting

**CUDA Out of Memory**

Reduce batch size or use gradient accumulation:

```bash
python vqa_finetune.py --batch_size 16 --accumulation_steps 2
```

**Data Loading Errors**

Verify dataset structure, check file permissions, and run:

```bash
python check_data_quality.py
```

**Poor Model Performance**

Run the learning rate finder, check data quality, and try targeted training:

```bash
python find_lr.py
python run_target_training.py
```

**Training Instability**

Reduce learning rate, add gradient clipping, and increase warmup steps in `configs/training_config.yaml`.

---

## License

[Add your license information here]

## Acknowledgments

[Add acknowledgments here]
