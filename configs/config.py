"""
MedXplain – unified configuration  (configs/config.py)
=======================================================
All training and inference code imports from here.
Uses BLIP-1 (blip-vqa-base) to be consistent with backend.py.
"""
from pathlib import Path
import os

os.environ.setdefault("CUBLASLT_DISABLE_TENSOR_CORE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
def _select_device() -> str:
    requested = os.environ.get("MEDXPLAIN_DEVICE", "cuda").strip().lower()
    require_cuda = os.environ.get("MEDXPLAIN_REQUIRE_CUDA", "1").strip().lower()
    require_cuda = require_cuda in {"1", "true", "yes", "on"}

    if requested in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            raise RuntimeError(
                "MEDXPLAIN_REQUIRE_CUDA=1 but PyTorch cannot see a CUDA GPU. "
                "Install a CUDA-enabled torch build and NVIDIA drivers, or set "
                "MEDXPLAIN_REQUIRE_CUDA=0 to allow CPU fallback."
            )
        return "cpu"

    if requested == "cpu":
        if require_cuda:
            raise RuntimeError(
                "MEDXPLAIN_DEVICE=cpu conflicts with MEDXPLAIN_REQUIRE_CUDA=1."
            )
        return "cpu"

    raise ValueError(
        "MEDXPLAIN_DEVICE must be 'cuda'/'gpu' or 'cpu', "
        f"got {requested!r}."
    )


DEVICE = _select_device()

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Dataset splits ────────────────────────────────────────────────────────────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)
