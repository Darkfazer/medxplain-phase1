import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "VQA_RAD Image Folder"
JSON_PATH = DATA_DIR / "VQA_RAD Dataset Public.json"

# Model
MODEL_NAME = "Salesforce/blip2-opt-2.7b"

# Training Hyperparameters
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.05
MAX_LENGTH = 50

# Device
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Random Seed
SEED = 42
