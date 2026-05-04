"""
training/train_vqa_two_phase.py
================================
Two-phase training pipeline for MedXplain.

Phase 1 – Vision-Language Alignment
    • Vision encoder + text decoder frozen.
    • Only the Cross-Attention Fusion layer is trained.
    • Loss: NT-Xent (contrastive).
    • Checkpoint: checkpoints/phase1_alignment.pt

Phase 2 – VQA Fine-tuning
    • Loads Phase 1 checkpoint.
    • Unfreezes last 2 blocks of vision encoder.
    • Injects LoRA adapters into attention layers.
    • Loss: CrossEntropyLoss over answer class index.
    • Checkpoint saved only when val accuracy improves.

Usage
-----
    python -m training.train_vqa_two_phase \\
        --phase1_epochs 10 --phase2_epochs 20 --batch_size 8

Fixes applied vs. original:
  - torch.amp API (not deprecated torch.cuda.amp)
  - best_val_loss actually gates Phase 1 checkpoint saving
  - best_val_acc gates Phase 2 checkpoint saving
  - Checkpoint saved on best, not unconditionally at epoch end
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import GradScaler, autocast          # ← updated API
from torch.utils.data import DataLoader, Dataset

# Bootstrap: add project root so 'configs', 'data', etc. are importable
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Optional dependencies – graceful fallback
# ---------------------------------------------------------------------------
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None                                      # type: ignore
    _TQDM_AVAILABLE = False

from medical_vqa_infrastructure.models.vision_encoder import MockVisionEncoder
from medical_vqa_infrastructure.models.text_decoder   import MockTextDecoder
from medical_vqa_infrastructure.models.fusion         import CrossAttentionFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_vqa_two_phase")


# ---------------------------------------------------------------------------
# Mock dataset
# ---------------------------------------------------------------------------
class MockVQADataset(Dataset):
    MOCK_QUESTIONS = [
        "Is there a pleural effusion?",
        "Is there cardiomegaly?",
        "Is there a pneumothorax?",
        "Are the lung fields clear?",
    ]
    MOCK_ANSWERS = ["Yes", "No", "Yes", "No"]

    def __init__(self, n_samples: int = 256, n_classes: int = 2) -> None:
        self.n_samples = n_samples
        self.n_classes = n_classes

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        q_idx = idx % len(self.MOCK_QUESTIONS)
        return {
            "image":    torch.randn(3, 224, 224),
            "question": self.MOCK_QUESTIONS[q_idx],
            "answer":   self.MOCK_ANSWERS[q_idx],
            "label":    torch.tensor(q_idx % self.n_classes, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper around nn.Linear.

    Only lora_A and lora_B are trained; the base weight is frozen.
    Initialised per the LoRA paper: A ~ Gaussian, B = 0 → zero initial update.
    """

    def __init__(self, base_layer: nn.Linear, rank: int = 8, alpha: int = 16) -> None:
        super().__init__()
        self.base  = base_layer
        self.scale = alpha / rank
        in_f, out_f = base_layer.in_features, base_layer.out_features
        self.lora_A = nn.Linear(in_f,  rank,  bias=False)
        self.lora_B = nn.Linear(rank,  out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """Replace matching nn.Linear layers with LoRALinear wrappers (in-place)."""
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_module_names):
            parts  = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], LoRALinear(module, rank=rank, alpha=alpha))
            logger.debug("LoRA injected into: %s", name)
    return model


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent)
# ---------------------------------------------------------------------------
class NTXentLoss(nn.Module):
    """Normalised Temperature-scaled Cross Entropy loss for contrastive learning."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, img_emb: torch.Tensor, txt_emb: torch.Tensor) -> torch.Tensor:
        B = img_emb.size(0)
        img_emb = F.normalize(img_emb, dim=-1)
        txt_emb = F.normalize(txt_emb, dim=-1)
        logits  = torch.matmul(img_emb, txt_emb.T) / self.temperature
        labels  = torch.arange(B, device=img_emb.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------
class TwoPhaseVQAModel(nn.Module):
    """Full VQA model for two-phase training."""

    def __init__(
        self,
        vision_encoder: Optional[nn.Module] = None,
        text_decoder:   Optional[nn.Module] = None,
        fusion:         Optional[nn.Module] = None,
        hidden_size:    int = 768,
        num_classes:    int = 2,
    ) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder or MockVisionEncoder()
        self.text_decoder   = text_decoder   or MockTextDecoder()
        self.fusion         = fusion          or CrossAttentionFusion(hidden_size)
        self.image_proj     = nn.Linear(hidden_size, hidden_size)
        self.text_proj      = nn.Linear(hidden_size, hidden_size)
        self.answer_head    = nn.Linear(hidden_size, num_classes)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.vision_encoder(images)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)
        return self.image_proj(feats)

    def encode_text(self, questions: List[str]) -> torch.Tensor:
        B = len(questions)
        dummy = torch.zeros(B, 1, dtype=torch.long,
                            device=next(self.parameters()).device)
        feats = self.text_decoder(dummy, questions)
        feats = feats.mean(dim=1)
        if feats.shape[-1] != self.text_proj.in_features:
            feats = feats[..., : self.text_proj.in_features]
        return self.text_proj(feats)

    def forward_vqa(self, images: torch.Tensor, questions: List[str]) -> torch.Tensor:
        vis_feats = self.vision_encoder(images)
        if vis_feats.dim() == 2:
            vis_feats = vis_feats.unsqueeze(1)
        txt_feats = torch.zeros_like(vis_feats)
        fused     = self.fusion(vis_feats, txt_feats)
        return self.answer_head(fused.mean(dim=1))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wrap_iter(iterable, desc: str, total: Optional[int] = None) -> Iterator:
    if _TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


def freeze_module(m: nn.Module, label: str = "") -> None:
    for p in m.parameters():
        p.requires_grad = False
    if label:
        logger.info("Frozen: %s", label)


def unfreeze_module(m: nn.Module, label: str = "") -> None:
    for p in m.parameters():
        p.requires_grad = True
    if label:
        logger.info("Unfrozen: %s", label)


def unfreeze_last_n_layers(m: nn.Module, n: int, label: str = "") -> None:
    children = list(m.named_children())
    if len(children) <= n:
        unfreeze_module(m, label)
        return
    for name, child in children[-n:]:
        unfreeze_module(child, f"{label}.{name}" if label else name)


def trainable_param_count(m: nn.Module) -> Tuple[int, int]:
    total     = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return trainable, total


def load_config(config_path: str) -> Dict[str, Any]:
    if not _YAML_AVAILABLE:
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config not found: %s – using defaults.", config_path)
        return {}


def init_wandb(project: str, run_name: str, config: Dict[str, Any]) -> None:
    if not _WANDB_AVAILABLE:
        return
    try:
        wandb.init(project=project, name=run_name, config=config, reinit=True)
    except Exception as exc:
        logger.warning("WandB init failed: %s", exc)


def log_wandb(metrics: Dict[str, Any], step: int) -> None:
    if _WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.log(metrics, step=step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1
# ---------------------------------------------------------------------------
def run_phase1(
    model:          TwoPhaseVQAModel,
    dataloader:     DataLoader,
    val_loader:     DataLoader,
    args:           argparse.Namespace,
    device:         torch.device,
    checkpoint_dir: Path,
) -> str:
    """Train Cross-Attention Fusion with NT-Xent contrastive loss.

    Returns the path to the saved checkpoint.
    Checkpoint is saved only when validation loss improves.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 – Vision-Language Alignment")
    logger.info("=" * 60)

    freeze_module(model.vision_encoder, "vision_encoder")
    freeze_module(model.text_decoder,   "text_decoder")
    unfreeze_module(model.fusion,       "fusion")
    unfreeze_module(model.image_proj,   "image_proj")
    unfreeze_module(model.text_proj,    "text_proj")

    trainable, total = trainable_param_count(model)
    logger.info("Trainable: %d / %d (%.1f%%)", trainable, total, 100 * trainable / max(total, 1))

    model.to(device)
    criterion = NTXentLoss(temperature=0.07)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr, weight_decay=1e-4,
    )
    scaler    = GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs
    )

    global_step   = 0
    best_val_loss = float("inf")
    ckpt_path     = checkpoint_dir / "phase1_alignment.pt"

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in _wrap_iter(dataloader, f"P1 Epoch {epoch}/{args.phase1_epochs}"):
            images    = batch["image"].to(device)
            questions = batch["question"]

            device_str = device.type
            if scaler:
                with autocast(device_str):
                    loss = criterion(model.encode_image(images), model.encode_text(questions))
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model.encode_image(images), model.encode_text(questions))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            epoch_loss  += loss.item()
            global_step += 1
            log_wandb({"phase1/loss": loss.item(), "phase1/lr": optimizer.param_groups[0]["lr"]},
                      step=global_step)

            if global_step % args.val_every == 0:
                val_loss = _validate_phase1(model, val_loader, criterion, device, args)
                logger.info("[P1] step=%d  train=%.4f  val=%.4f", global_step, loss.item(), val_loss)
                log_wandb({"phase1/val_loss": val_loss}, step=global_step)

                # Save checkpoint only on improvement
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(
                        {"model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "epoch": epoch, "phase": 1, "val_loss": val_loss},
                        ckpt_path,
                    )
                    logger.info("Phase 1 checkpoint saved (val_loss=%.4f) → %s", val_loss, ckpt_path)

        avg = epoch_loss / max(1, len(dataloader))
        scheduler.step()
        logger.info("Phase 1 | Epoch %d/%d | avg_loss=%.4f", epoch, args.phase1_epochs, avg)

    # Ensure we always have a checkpoint (final epoch if no val improvement occurred)
    if not ckpt_path.exists():
        torch.save({"model_state_dict": model.state_dict(), "epoch": args.phase1_epochs, "phase": 1},
                   ckpt_path)
        logger.info("Phase 1 final checkpoint saved → %s", ckpt_path)

    return str(ckpt_path)


def _validate_phase1(model, val_loader, criterion, device, args) -> float:
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            images    = batch["image"].to(device)
            questions = batch["question"]
            device_str = device.type
            if args.amp and device.type == "cuda":
                with autocast(device_str):
                    loss = criterion(model.encode_image(images), model.encode_text(questions))
            else:
                loss = criterion(model.encode_image(images), model.encode_text(questions))
            total += loss.item(); n += 1
    model.train()
    return total / max(1, n)


# ---------------------------------------------------------------------------
# Phase 2
# ---------------------------------------------------------------------------
def run_phase2(
    model:          TwoPhaseVQAModel,
    dataloader:     DataLoader,
    val_loader:     DataLoader,
    args:           argparse.Namespace,
    device:         torch.device,
    checkpoint_dir: Path,
    phase1_ckpt:    str,
) -> str:
    """Fine-tune for VQA using CrossEntropyLoss.

    Checkpoint saved only when validation accuracy improves.
    Returns path to saved checkpoint.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 – VQA Fine-tuning")
    logger.info("=" * 60)

    ckpt = torch.load(phase1_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    freeze_module(model, "all")
    unfreeze_last_n_layers(model.vision_encoder, n=2, label="vision_encoder")
    unfreeze_module(model.fusion,      "fusion")
    unfreeze_module(model.answer_head, "answer_head")
    inject_lora(model.text_decoder, args.lora_targets, rank=args.lora_rank, alpha=args.lora_alpha)

    model.to(device)

    trainable, total = trainable_param_count(model)
    logger.info("Trainable: %d / %d (%.1f%%)", trainable, total, 100 * trainable / max(total, 1))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr, weight_decay=1e-4,
    )
    scaler    = GradScaler("cuda") if (args.amp and device.type == "cuda") else None
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.phase2_epochs)

    global_step  = 0
    best_val_acc = 0.0
    ckpt_path    = checkpoint_dir / "phase2_vqa_finetuned.pt"

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0

        for batch in _wrap_iter(dataloader, f"P2 Epoch {epoch}/{args.phase2_epochs}"):
            images    = batch["image"].to(device)
            labels    = batch["label"].to(device)
            questions = batch["question"]

            device_str = device.type
            if scaler:
                with autocast(device_str):
                    logits = model.forward_vqa(images, questions)
                    loss   = criterion(logits, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model.forward_vqa(images, questions)
                loss   = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds          = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total   += labels.size(0)
            epoch_loss    += loss.item()
            global_step   += 1

            log_wandb({"phase2/loss": loss.item(), "phase2/lr": optimizer.param_groups[0]["lr"],
                       "phase2/batch_acc": (preds == labels).float().mean().item()},
                      step=global_step)

            if global_step % args.val_every == 0:
                val_acc, val_loss, samples = _validate_phase2(model, val_loader, criterion, device, args)
                logger.info("[P2] step=%d  val_acc=%.3f  val_loss=%.4f", global_step, val_acc, val_loss)
                logger.info("  Samples: %s", samples[:3])
                log_wandb({"phase2/val_acc": val_acc, "phase2/val_loss": val_loss}, step=global_step)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(
                        {"model_state_dict": model.state_dict(),
                         "optimizer_state_dict": optimizer.state_dict(),
                         "epoch": epoch, "phase": 2, "best_val_acc": best_val_acc},
                        ckpt_path,
                    )
                    logger.info("Phase 2 checkpoint saved (val_acc=%.3f) → %s", val_acc, ckpt_path)

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_acc = epoch_correct / max(1, epoch_total)
        scheduler.step()
        logger.info("Phase 2 | Epoch %d/%d | loss=%.4f | acc=%.3f",
                    epoch, args.phase2_epochs, avg_loss, epoch_acc)

    if not ckpt_path.exists():
        torch.save({"model_state_dict": model.state_dict(), "epoch": args.phase2_epochs, "phase": 2},
                   ckpt_path)
        logger.info("Phase 2 final checkpoint saved → %s", ckpt_path)

    return str(ckpt_path)


def _validate_phase2(model, val_loader, criterion, device, args):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    samples: List[str] = []
    answer_vocab = ["Yes", "No"]

    with torch.no_grad():
        for batch in val_loader:
            images    = batch["image"].to(device)
            labels    = batch["label"].to(device)
            questions = batch["question"]
            device_str = device.type
            if args.amp and device.type == "cuda":
                with autocast(device_str):
                    logits = model.forward_vqa(images, questions)
                    loss   = criterion(logits, labels)
            else:
                logits = model.forward_vqa(images, questions)
                loss   = criterion(logits, labels)

            preds    = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item()

            if len(samples) < 5:
                for q, p_idx, g_idx in zip(questions, preds.tolist(), labels.tolist()):
                    p_str = answer_vocab[p_idx % len(answer_vocab)]
                    g_str = answer_vocab[g_idx % len(answer_vocab)]
                    samples.append(f"Q: {q!r}  Pred={p_str}  GT={g_str}")

    model.train()
    return (
        correct / max(1, total),
        total_loss / max(1, len(val_loader)),
        samples,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-phase VQA training for MedXplain",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config",         default="config/hospital_config.yaml")
    p.add_argument("--phase1_epochs",  type=int,   default=10)
    p.add_argument("--phase1_lr",      type=float, default=1e-4)
    p.add_argument("--phase2_epochs",  type=int,   default=20)
    p.add_argument("--phase2_lr",      type=float, default=5e-5)
    p.add_argument("--batch_size",     type=int,   default=8)
    p.add_argument("--num_workers",    type=int,   default=0,
                   help="0 = main process (Windows compatible)")
    p.add_argument("--amp",            action="store_true", default=True)
    p.add_argument("--val_every",      type=int,   default=50)
    p.add_argument("--lora_rank",      type=int,   default=8)
    p.add_argument("--lora_alpha",     type=int,   default=16)
    p.add_argument("--lora_targets",   nargs="+",  default=["query", "value", "key"])
    p.add_argument("--checkpoint_dir", default="checkpoints/")
    p.add_argument("--num_classes",    type=int,   default=2)
    p.add_argument("--n_train",        type=int,   default=512)
    p.add_argument("--n_val",          type=int,   default=64)
    p.add_argument("--wandb_project",  default="medxplain-vqa")
    p.add_argument("--wandb_run",      default="two-phase-training")
    p.add_argument("--no_wandb",       action="store_true")
    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    config = load_config(args.config)

    device = torch.device(
        config.get("inference", {}).get("device", "cuda")
        if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    if not args.no_wandb:
        init_wandb(args.wandb_project, args.wandb_run, vars(args))

    train_ds = MockVQADataset(n_samples=args.n_train, n_classes=args.num_classes)
    val_ds   = MockVQADataset(n_samples=args.n_val,   n_classes=args.num_classes)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=args.num_workers)

    model    = TwoPhaseVQAModel(num_classes=args.num_classes)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    p1 = run_phase1(model, train_loader, val_loader, args, device, ckpt_dir)
    p2 = run_phase2(model, train_loader, val_loader, args, device, ckpt_dir, p1)

    logger.info("Training complete.")
    logger.info("  Phase 1: %s", p1)
    logger.info("  Phase 2: %s", p2)

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
