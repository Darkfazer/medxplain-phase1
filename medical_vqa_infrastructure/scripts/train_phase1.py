import os
import sys
import argparse

# Ensure the project root (medical_vqa_infrastructure/) is on sys.path so that
# sibling packages like `training`, `models`, `evaluation`, etc. are importable
# regardless of whether the package was installed with `pip install -e .`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_mode", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=1)
    args = parser.parse_args()
    
    print(f"Running Phase 1 Training (Classification)... (Epochs: {args.num_epochs}, Mock Mode: {args.mock_mode})")
    if args.mock_mode:
        os.environ['MOCK_MODE'] = '1'

if __name__ == "__main__":
    main()
