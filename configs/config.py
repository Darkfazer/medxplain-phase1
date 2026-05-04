"""
MedXplain – unified configuration  (configs/config.py)
=======================================================
All training and inference code imports from here.
Uses BLIP-1 (blip-vqa-base) to be consistent with backend.py.
"""
from pathlib import Path
import torch

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
IMAGE_DIR = DATA_DIR / "VQA_RAD Image Folder"
JSON_PATH = DATA_DIR / "VQA_RAD Dataset Public.json"

# ── Model  (BLIP-1 – matches backend.py) ─────────────────────────────────────
MODEL_NAME = "Salesforce/blip-vqa-base"

# ── Training hyperparameters ──────────────────────────────────────────────────
# BATCH_SIZE: keep small (≤4) on consumer GPUs to avoid cuBLAS VRAM errors
BATCH_SIZE    = 2
EPOCHS        = 10
LEARNING_RATE = 5e-5
WEIGHT_DECAY  = 0.05
MAX_LENGTH    = 32   # token length; keep ≤32 to avoid cuBLAS dimension issues

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)
