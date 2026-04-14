import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock_mode", type=bool, default=False)
    args = parser.parse_args()
    if args.mock_mode:
        os.environ['MOCK_MODE'] = '1'
    print(f"Evaluating Medical VQA Checkpoint on Test Set... (Mock Mode: {args.mock_mode})")

if __name__ == "__main__":
    main()
