import os

os.environ.setdefault("CUBLASLT_DISABLE_TENSOR_CORE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

class Config:
    # Basic settings
    _requested_device = os.getenv("MEDXPLAIN_DEVICE", "cuda").strip().lower()
    if _requested_device in {"cuda", "gpu"} and torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif os.getenv("MEDXPLAIN_REQUIRE_CUDA", "1").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("CUDA GPU is required but PyTorch cannot see one.")
    else:
        DEVICE = torch.device("cpu")
    USE_MOCK_MODEL = True  # Toggle this to False when loading real weights
    
    # Model Paths
    VISION_ENCODER_PATH = os.getenv("VISION_ENCODER_PATH", "experiments/results/densenet121/best_model.pth")
    LANGUAGE_MODEL_PATH = os.getenv("LANGUAGE_MODEL_PATH", "experiments/results/vqa/best_llm.pth")
    FUSION_MODEL_PATH = os.getenv("FUSION_MODEL_PATH", "experiments/results/vqa/best_fusion.pth")
    
    # Dataset and Class Definitions
    NUM_CLASSES = 14
    CLASS_NAMES = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule',
        'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema',
        'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # Calibration and Optimization
    TEMPERATURE = 1.263
    DEFAULT_THRESHOLD = 0.5
    # Per-class optimized thresholds mapped from Phase 1 validation (placeholders can be updated)
    OPTIMIZED_THRESHOLDS = {
        'Atelectasis': 0.45,
        'Cardiomegaly': 0.52,
        'Effusion': 0.48,
        'Infiltration': 0.50,
        'Mass': 0.55,
        'Nodule': 0.50,
        'Pneumonia': 0.40,
        'Pneumothorax': 0.60,
        'Consolidation': 0.50,
        'Edema': 0.48,
        'Emphysema': 0.50,
        'Fibrosis': 0.50,
        'Pleural_Thickening': 0.50,
        'Hernia': 0.50
    }
    
    # Demo App Settings
    GRADIO_PORT = 7860
    MAX_IMAGE_SIZE = 224
