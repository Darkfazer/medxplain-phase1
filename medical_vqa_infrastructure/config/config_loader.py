import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from dotenv import load_dotenv

# Load environment variables (.env) globally
load_dotenv()

def get_mock_mode() -> bool:
    """Returns True if the infrastructure is configured for mocked offline tests."""
    # Ensure parsing works regardless of string casing
    return os.getenv("MOCK_MODE", "False").lower() == "true"

def load_config(config_name: str) -> DictConfig:
    """
    Dynamically load a specific yaml config (e.g., 'model_config', 'data_config').
    Paths are relative to the config directory.
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing configuration file at: {config_path}")
        
    return OmegaConf.load(config_path)
