"""
training/train.py
=================
Fine-tune BLIP-1 (blip-vqa-base) on VQA-RAD.

Usage
-----
    # As a script (both work):
    python training/train.py --epochs 5
    python -m training.train --epochs 5

    Full CLI:
    python training/train.py --config configs/config.py --mode train --epochs 5

Consistent with backend.py which also loads blip-vqa-base.
"""
from __future__ import annotations

# ── Bootstrap: ensure project root is on sys.path so 'configs', 'data', etc.
#    are importable whether this file is run as a script OR as a module.
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import os

# ── cuBLAS 12.4 BF16 definitive fix ────────────────────────────────────────────
# Python 3.13 only has torch 2.6+cu124 on Windows.  cuBLAS 12.4 routes FP32
# F.linear calls through cublasLt with BF16 (computeType=77), crashing BLIP.
#
# CUBLASLT_DISABLE_TENSOR_CORE=1 forces legacy SGEMM (non-cublasLt path)
# and completely eliminates the BF16 routing bug on all Ampere/Ada GPUs.
# Must be set before any CUDA context is created (before torch is imported).
os.environ["CUBLASLT_DISABLE_TENSOR_CORE"] = "1"
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
# ──────────────────────────────────────────────────────────────────────────

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import BlipProcessor   # ← BLIP-1 processor

import configs.config as cfg
from data.dataset import get_dataloaders
from models.vqa_model import VQAModel
from evaluation.metrics import evaluate_batch, aggregate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)


def train(epochs: int = cfg.EPOCHS, batch_size: int = cfg.BATCH_SIZE) -> None:
    """Fine-tune VQAModel (BLIP-1) on VQA-RAD.

    Parameters
    ----------
    epochs     : number of training epochs (overrides configs/config.py)
    batch_size : batch size (overrides configs/config.py)
    """
    log.info("Loading BlipProcessor for %s …", cfg.MODEL_NAME)
    processor = BlipProcessor.from_pretrained(cfg.MODEL_NAME)

    log.info("Preparing dataloaders …")
    train_loader, val_loader, _ = get_dataloaders(processor, batch_size=batch_size)

    log.info("Initialising model …")
    model = VQAModel(freeze_vision=True)

    # ── FP32 GPU setup ──────────────────────────────────────────────────────────
    # CUBLASLT_DISABLE_TENSOR_CORE=1 (set above before torch import) forces
    # legacy SGEMM and bypasses the cublasLt BF16 heuristic entirely.
    device_to_use = cfg.DEVICE
    model = model.float().to(device_to_use)
    log.info("Training device: %s  (CUDA %s)", device_to_use,
             torch.version.cuda if device_to_use == "cuda" else "N/A")
    torch.set_float32_matmul_precision("highest")
    if device_to_use == "cuda":
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
    # ────────────────────────────────────────────────────────────────────

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = os.path.join(cfg.BASE_DIR, "experiments")
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")

    log.info("Starting training for %d epoch(s) …", epochs)
    for epoch in range(epochs):
        # ── Training ─────────────────────────────────────────────────────────
        model.model.train()
        total_train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch in loop:
            labels = batch["labels"]
            # Replace -100 (loss-mask sentinel) with pad_id=0 before passing
            # to BLIP's forward; the model ignores those positions in the loss
            # internally via its own masking, but the embedding lookup crashes
            # if it receives -100 as an index.
            decoder_labels = labels.clone()
            decoder_labels[decoder_labels == -100] = 0

            outputs = model(
                pixel_values=batch["pixel_values"],
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=decoder_labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        log.info("Epoch %d  train_loss=%.4f", epoch + 1, avg_train_loss)

        # ── Validation ───────────────────────────────────────────────────────
        model.model.eval()
        total_val_loss = 0.0
        all_metrics: list[dict] = []

        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]")
            for batch in loop:
                labels = batch["labels"]
                decoder_labels = labels.clone()
                decoder_labels[decoder_labels == -100] = 0

                outputs = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=decoder_labels,
                )
                total_val_loss += outputs.loss.item()

                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                preds = processor.batch_decode(generated_ids, skip_special_tokens=True)
                preds = [p.strip() for p in preds]

                batch_metrics = evaluate_batch(
                    preds, batch["answers"], batch["answer_types"]
                )
                all_metrics.append(batch_metrics)

        avg_val_loss = total_val_loss / len(val_loader)
        agg = aggregate_metrics(all_metrics)
        log.info("Epoch %d  val_loss=%.4f", epoch + 1, avg_val_loss)
        for k, v in agg.items():
            log.info("  %s: %.2f", k, v)

        # Save checkpoint only when val loss improves
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            model.save_checkpoint(os.path.join(save_dir, "best_vqa_model.pth"))
            log.info("Saved new best model (val_loss=%.4f)", best_loss)

        scheduler.step()

    log.info("Training complete. Best val_loss=%.4f", best_loss)


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune BLIP-1 on VQA-RAD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", default="configs/config.py",
        help="Path to config file (informational; settings live in configs/config.py)",
    )
    p.add_argument(
        "--mode", default="train", choices=["train", "eval"],
        help="'train' to fine-tune, 'eval' to evaluate only",
    )
    p.add_argument(
        "--epochs", type=int, default=cfg.EPOCHS,
        help="Number of training epochs (overrides configs/config.py)",
    )
    p.add_argument(
        "--batch_size", type=int, default=cfg.BATCH_SIZE,
        help="Batch size",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    log.info("Config  : %s", args.config)
    log.info("Mode    : %s", args.mode)
    log.info("Epochs  : %d", args.epochs)

    if args.mode == "train":
        train(epochs=args.epochs, batch_size=args.batch_size)
    else:
        log.info("Eval-only mode: loading best checkpoint and running validation.")
        # Evaluation-only path (extend as needed)
        log.warning("Eval-only mode not yet implemented; run with --mode train.")
