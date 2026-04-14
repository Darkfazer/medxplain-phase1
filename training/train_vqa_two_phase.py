"""
train_vqa_two_phase.py
======================
Two-phase training pipeline for MedXplain Medical VQA, aligned with
Section 3.1 of the Cahier des Charges.

Phase 1 – Vision-Language Alignment
    • BiomedCLIP vision encoder + BioGPT decoder, both frozen.
    • Only the Cross-Attention Fusion layer (Component 3) is trained.
    • Loss: contrastive (NT-Xent / InfoNCE) between image & text embeddings.
    • Checkpoint saved as `phase1_alignment.pt`.

Phase 2 – VQA Fine-tuning
    • Loads Phase 1 checkpoint.
    • Unfreezes the last 2 transformer blocks of the vision encoder.
    • Applies LoRA adapters (rank 8) to the language model's attention layers.
    • Loss: CrossEntropy over answer token logits.
    • Checkpoint saved as `phase2_vqa_finetuned.pt`.

Features
    • WandB logging (optional, graceful fallback)
    • Gradient checkpointing
    • Automatic Mixed Precision (AMP)
    • Validation every N steps with sample predictions

Usage
-----
python training/train_vqa_two_phase.py \\
    --config config/hospital_config.yaml \\
    --phase1_epochs 10 \\
    --phase2_epochs 20 \\
    --batch_size 8

Dependencies
------------
pip install torch pyyaml tqdm
pip install wandb          # optional
pip install transformers   # for real BiomedCLIP / BioGPT loading
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Ensure project root is importable regardless of invocation directory
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Optional dependencies – graceful fallback
# ---------------------------------------------------------------------------
try:
    import wandb  # type: ignore
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False

try:
    from tqdm import tqdm  # type: ignore
    _TQDM_AVAILABLE = True
except ImportError:
    tqdm = None  # type: ignore
    _TQDM_AVAILABLE = False

# ---------------------------------------------------------------------------
# Project-internal imports
# ---------------------------------------------------------------------------
from medical_vqa_infrastructure.models.vision_encoder import MockVisionEncoder
from medical_vqa_infrastructure.models.text_decoder import MockTextDecoder
from medical_vqa_infrastructure.models.fusion import CrossAttentionFusion

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_vqa_two_phase")


# ---------------------------------------------------------------------------
# Mock dataset (replace with real MedVQA / DICOM dataset in production)
# ---------------------------------------------------------------------------

class MockVQADataset(Dataset):
    """
    Lightweight mock dataset for pipeline smoke-testing.

    Each item is a dict with:
        image:    float32 tensor (3, 224, 224)
        question: str
        answer:   str
        label:    int (answer class index)
    """

    MOCK_QUESTIONS = [
        "Is there a pleural effusion?",
        "Is there cardiomegaly?",
        "Is there a pneumothorax?",
        "Are the lung fields clear?",
    ]
    MOCK_ANSWERS = ["Yes", "No", "Yes", "No"]

    def __init__(self, n_samples: int = 256, n_classes: int = 2) -> None:
        self.n_samples  = n_samples
        self.n_classes  = n_classes

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
# LoRA – lightweight adapter for language model attention layers
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper around an existing ``nn.Linear`` layer.

    During training only the A and B matrices are updated; the original
    weight is kept frozen.

    Parameters
    ----------
    base_layer:  The frozen ``nn.Linear`` to adapt.
    rank:        LoRA rank (r).  Lower → fewer parameters.
    alpha:       Scaling factor.  Effective scale = alpha / rank.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: int = 16,
    ) -> None:
        super().__init__()
        self.base   = base_layer
        self.rank   = rank
        self.scale  = alpha / rank

        in_feat  = base_layer.in_features
        out_feat = base_layer.out_features

        self.lora_A = nn.Linear(in_feat,  rank,     bias=False)
        self.lora_B = nn.Linear(rank,     out_feat, bias=False)

        # Initialise A with random Gaussian, B with zeros (LoRA paper)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B.weight)

        # Freeze the original weight
        for param in self.base.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.scale * self.lora_B(self.lora_A(x))


def inject_lora(
    model: nn.Module,
    target_module_names: List[str],
    rank: int = 8,
    alpha: int = 16,
) -> nn.Module:
    """
    Recursively replace ``nn.Linear`` layers whose fully-qualified name
    contains any string in ``target_module_names`` with ``LoRALinear``
    wrappers.

    Returns the modified model (in-place mutation + return for chaining).
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(
            t in name for t in target_module_names
        ):
            # Walk the attribute path to find the parent and attribute name
            parts  = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            attr   = parts[-1]
            setattr(parent, attr, LoRALinear(module, rank=rank, alpha=alpha))
            logger.debug("LoRA injected into: %s", name)

    return model


# ---------------------------------------------------------------------------
# Contrastive loss (NT-Xent / InfoNCE) for Phase 1
# ---------------------------------------------------------------------------

class NTXentLoss(nn.Module):
    """
    Normalised Temperature-scaled Cross Entropy (NT-Xent) loss.

    Used in Phase 1 to align image and text embedding spaces.
    Expects two embedding tensors of equal shape ``(B, D)``.

    Parameters
    ----------
    temperature:  Softmax temperature τ.  Lower → sharper distribution.
    """

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self, image_embeds: torch.Tensor, text_embeds: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        image_embeds:  (B, D) L2-normalised image embeddings.
        text_embeds:   (B, D) L2-normalised text embeddings.
        """
        B = image_embeds.size(0)

        # L2-normalise
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds  = F.normalize(text_embeds,  dim=-1)

        # (B, B) cosine similarity matrix
        logits = torch.matmul(image_embeds, text_embeds.T) / self.temperature

        # Symmetric labels: diagonal is the positive pair
        labels = torch.arange(B, device=image_embeds.device)

        loss_i2t = F.cross_entropy(logits,   labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2.0


# ---------------------------------------------------------------------------
# Full VQA model assembled for two-phase training
# ---------------------------------------------------------------------------

class TwoPhaseVQAModel(nn.Module):
    """
    Full VQA model assembled from pluggable components, ready for
    two-phase training.

    Components
    ----------
    vision_encoder : extracts image embeddings  (B, N_vis, D)
    fusion         : cross-attention between image and text  (B, L, D)
    text_decoder   : generates answer token logits  (B, L, vocab_size)

    This class wraps the existing ``MockVisionEncoder`` / ``MockTextDecoder``
    / ``CrossAttentionFusion`` and can be swapped for real implementations
    (BiomedCLIP, BioGPT) by replacing the constructor arguments.
    """

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

        # Projection heads used ONLY in Phase 1 contrastive training
        self.image_proj = nn.Linear(hidden_size, hidden_size)
        self.text_proj  = nn.Linear(hidden_size, hidden_size)

        # Classification head for Phase 2 VQA
        self.answer_head = nn.Linear(hidden_size, num_classes)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """Return projected image embeddings (B, D) for contrastive loss."""
        feats = self.vision_encoder(images)   # (B, N, D)  or  (B, D)
        if feats.dim() == 3:
            feats = feats.mean(dim=1)          # global average pool → (B, D)
        return self.image_proj(feats)

    def encode_text(self, questions: List[str]) -> torch.Tensor:
        """Return projected text embeddings (B, D) for contrastive loss."""
        # MockTextDecoder returns (B, L, vocab) – we reduce it here
        # Replace with real tokeniser + encoder call for production.
        B = len(questions)
        dummy = torch.zeros(B, 1, dtype=torch.long, device=next(self.parameters()).device)
        feats = self.text_decoder(dummy, questions)  # (B, L, vocab)
        feats = feats.mean(dim=1)                     # (B, vocab)
        # Project to hidden_size
        if feats.shape[-1] != self.text_proj.in_features:
            feats = feats[..., : self.text_proj.in_features]
        return self.text_proj(feats)

    def forward_vqa(
        self,
        images:    torch.Tensor,
        questions: List[str],
    ) -> torch.Tensor:
        """Phase 2 forward: returns answer logits (B, num_classes)."""
        vis_feats  = self.vision_encoder(images)          # (B, N, D) or (B, D)
        if vis_feats.dim() == 2:
            vis_feats = vis_feats.unsqueeze(1)             # (B, 1, D)

        # Dummy text features matching vis_feats shape for fusion
        txt_feats  = torch.zeros_like(vis_feats)

        fused      = self.fusion(vis_feats, txt_feats)     # (B, L, D)
        pooled     = fused.mean(dim=1)                     # (B, D)
        return self.answer_head(pooled)                    # (B, num_classes)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML config file; return empty dict if YAML / file unavailable."""
    if not _YAML_AVAILABLE:
        logger.warning("PyYAML not installed – using defaults. pip install pyyaml")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except FileNotFoundError:
        logger.warning("Config file not found: %s – using defaults.", config_path)
        return {}


def _wrap_iter(
    iterable,
    desc: str,
    total: Optional[int] = None,
) -> Iterator:
    """Wrap an iterable with tqdm if available, else return as-is."""
    if _TQDM_AVAILABLE and tqdm is not None:
        return tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def init_wandb(
    project: str,
    run_name: str,
    config: Dict[str, Any],
) -> None:
    """Initialise a WandB run if the library is available."""
    if not _WANDB_AVAILABLE:
        logger.info("WandB not installed – logging disabled. pip install wandb")
        return
    try:
        wandb.init(project=project, name=run_name, config=config, reinit=True)
        logger.info("WandB run initialised: %s/%s", project, run_name)
    except Exception as exc:
        logger.warning("WandB init failed: %s – continuing without logging.", exc)


def log_wandb(metrics: Dict[str, Any], step: int) -> None:
    if _WANDB_AVAILABLE and wandb.run is not None:
        try:
            wandb.log(metrics, step=step)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def freeze_module(module: nn.Module, label: str = "") -> None:
    """Set requires_grad=False for all parameters in ``module``."""
    for p in module.parameters():
        p.requires_grad = False
    if label:
        logger.info("Frozen:   %s", label)


def unfreeze_module(module: nn.Module, label: str = "") -> None:
    """Set requires_grad=True for all parameters in ``module``."""
    for p in module.parameters():
        p.requires_grad = True
    if label:
        logger.info("Unfrozen: %s", label)


def unfreeze_last_n_layers(
    module: nn.Module,
    n: int,
    label: str = "",
) -> None:
    """
    Unfreeze the last ``n`` named children of ``module``.

    Works for sequential containers (ResNet, ViT blocks, etc.).
    Falls back to unfreezing the whole module if fewer than ``n``
    children are found.
    """
    children = list(module.named_children())
    if len(children) <= n:
        unfreeze_module(module, label)
        return

    for name, child in children[-n:]:
        unfreeze_module(child, f"{label}.{name}" if label else name)


def trainable_param_count(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params)."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


# ---------------------------------------------------------------------------
# Phase 1: Vision-Language Alignment
# ---------------------------------------------------------------------------

def run_phase1(
    model:       TwoPhaseVQAModel,
    dataloader:  DataLoader,
    val_loader:  DataLoader,
    args:        argparse.Namespace,
    device:      torch.device,
    checkpoint_dir: Path,
) -> str:
    """
    Train the Cross-Attention Fusion layer with a contrastive (NT-Xent) loss.

    Backbone (vision encoder + text decoder) is kept fully frozen.
    Only the fusion layer and projection heads receive gradient updates.

    Returns
    -------
    Path to the saved Phase 1 checkpoint.
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 – Vision-Language Alignment")
    logger.info("=" * 60)

    # Freeze backbones, unfreeze fusion + projections
    freeze_module(model.vision_encoder, "vision_encoder")
    freeze_module(model.text_decoder,   "text_decoder")
    unfreeze_module(model.fusion,        "fusion")
    unfreeze_module(model.image_proj,    "image_proj")
    unfreeze_module(model.text_proj,     "text_proj")

    trainable, total = trainable_param_count(model)
    logger.info(
        "Trainable: %d / %d params (%.1f%%)",
        trainable, total, 100 * trainable / max(total, 1),
    )

    model.to(device)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    criterion  = NTXentLoss(temperature=0.07)
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase1_lr,
        weight_decay=1e-4,
    )
    scaler     = GradScaler(enabled=args.amp and device.type == "cuda")
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase1_epochs
    )

    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, args.phase1_epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in _wrap_iter(dataloader, f"P1 Epoch {epoch}/{args.phase1_epochs}"):
            images    = batch["image"].to(device)
            questions = batch["question"]

            with autocast(enabled=args.amp and device.type == "cuda"):
                img_embeds = model.encode_image(images)
                txt_embeds = model.encode_text(questions)
                loss       = criterion(img_embeds, txt_embeds)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss  += loss.item()
            global_step += 1

            metrics = {
                "phase1/loss": loss.item(),
                "phase1/lr":   optimizer.param_groups[0]["lr"],
            }
            log_wandb(metrics, step=global_step)

            # In-training validation sample
            if global_step % args.val_every == 0:
                val_loss = _validate_phase1(model, val_loader, criterion, device, args)
                logger.info(
                    "[P1] step=%d  train_loss=%.4f  val_loss=%.4f",
                    global_step, loss.item(), val_loss,
                )
                log_wandb({"phase1/val_loss": val_loss}, step=global_step)

        avg_loss = epoch_loss / max(1, len(dataloader))
        scheduler.step()
        logger.info("Phase 1 | Epoch %d/%d | avg_loss=%.4f", epoch, args.phase1_epochs, avg_loss)

    # Save checkpoint
    ckpt_path = checkpoint_dir / "phase1_alignment.pt"
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                args.phase1_epochs,
            "phase":                1,
        },
        ckpt_path,
    )
    logger.info("Phase 1 checkpoint saved → %s", ckpt_path)
    return str(ckpt_path)


def _validate_phase1(
    model:      TwoPhaseVQAModel,
    val_loader: DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    args:       argparse.Namespace,
) -> float:
    """Run a short validation pass for Phase 1 and return mean loss."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    with torch.no_grad():
        for batch in val_loader:
            images    = batch["image"].to(device)
            questions = batch["question"]
            with autocast(enabled=args.amp and device.type == "cuda"):
                img_embeds = model.encode_image(images)
                txt_embeds = model.encode_text(questions)
                loss       = criterion(img_embeds, txt_embeds)
            total_loss += loss.item()
            n_batches  += 1
    model.train()
    return total_loss / max(1, n_batches)


# ---------------------------------------------------------------------------
# Phase 2: VQA Fine-tuning
# ---------------------------------------------------------------------------

def run_phase2(
    model:       TwoPhaseVQAModel,
    dataloader:  DataLoader,
    val_loader:  DataLoader,
    args:        argparse.Namespace,
    device:      torch.device,
    checkpoint_dir: Path,
    phase1_ckpt: str,
) -> str:
    """
    Fine-tune for VQA answering after loading Phase 1 weights.

    Changes from Phase 1:
      - Last 2 blocks of vision encoder unfrozen.
      - LoRA adapters injected into language model.
      - Loss: CrossEntropyLoss on answer class index.

    Returns
    -------
    Path to the saved Phase 2 checkpoint.
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 – VQA Fine-tuning")
    logger.info("=" * 60)

    # Load Phase 1 weights
    logger.info("Loading Phase 1 checkpoint: %s", phase1_ckpt)
    ckpt = torch.load(phase1_ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Freeze everything, then selectively unfreeze
    freeze_module(model, "all")
    unfreeze_last_n_layers(model.vision_encoder, n=2, label="vision_encoder")
    unfreeze_module(model.fusion,      "fusion")
    unfreeze_module(model.answer_head, "answer_head")

    # Inject LoRA into text decoder attention layers
    inject_lora(
        model.text_decoder,
        target_module_names=args.lora_targets,
        rank=args.lora_rank,
        alpha=args.lora_alpha,
    )
    # LoRA A/B matrices are trainable by default inside LoRALinear

    # Enable gradient checkpointing
    model.to(device)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)

    trainable, total = trainable_param_count(model)
    logger.info(
        "Trainable: %d / %d params (%.1f%%)",
        trainable, total, 100 * trainable / max(total, 1),
    )

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.phase2_lr,
        weight_decay=1e-4,
    )
    scaler     = GradScaler(enabled=args.amp and device.type == "cuda")
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.phase2_epochs
    )

    global_step = 0
    best_val_acc = 0.0

    for epoch in range(1, args.phase2_epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_correct = 0
        epoch_total   = 0

        for batch in _wrap_iter(dataloader, f"P2 Epoch {epoch}/{args.phase2_epochs}"):
            images  = batch["image"].to(device)
            labels  = batch["label"].to(device)
            questions = batch["question"]

            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = model.forward_vqa(images, questions)
                loss   = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            preds        = logits.argmax(dim=-1)
            epoch_correct += (preds == labels).sum().item()
            epoch_total   += labels.size(0)
            epoch_loss    += loss.item()
            global_step   += 1

            metrics = {
                "phase2/loss": loss.item(),
                "phase2/lr":   optimizer.param_groups[0]["lr"],
                "phase2/batch_acc": (preds == labels).float().mean().item(),
            }
            log_wandb(metrics, step=global_step)

            if global_step % args.val_every == 0:
                val_acc, val_loss, sample_preds = _validate_phase2(
                    model, val_loader, criterion, device, args
                )
                logger.info(
                    "[P2] step=%d  val_acc=%.3f  val_loss=%.4f",
                    global_step, val_acc, val_loss,
                )
                logger.info("  Sample predictions: %s", sample_preds[:3])
                log_wandb(
                    {"phase2/val_acc": val_acc, "phase2/val_loss": val_loss},
                    step=global_step,
                )
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

        avg_loss = epoch_loss / max(1, len(dataloader))
        epoch_acc = epoch_correct / max(1, epoch_total)
        scheduler.step()
        logger.info(
            "Phase 2 | Epoch %d/%d | avg_loss=%.4f | train_acc=%.3f",
            epoch, args.phase2_epochs, avg_loss, epoch_acc,
        )

    # Save final checkpoint
    ckpt_path = checkpoint_dir / "phase2_vqa_finetuned.pt"
    torch.save(
        {
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch":                args.phase2_epochs,
            "phase":                2,
            "best_val_acc":         best_val_acc,
        },
        ckpt_path,
    )
    logger.info("Phase 2 checkpoint saved → %s", ckpt_path)
    return str(ckpt_path)


def _validate_phase2(
    model:      TwoPhaseVQAModel,
    val_loader: DataLoader,
    criterion:  nn.Module,
    device:     torch.device,
    args:       argparse.Namespace,
) -> Tuple[float, float, List[str]]:
    """
    Run validation for Phase 2.

    Returns (accuracy, mean_loss, list_of_sample_prediction_strings).
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0
    sample_preds: List[str] = []
    answer_vocab = ["Yes", "No"]  # extend with real answer vocab

    with torch.no_grad():
        for batch in val_loader:
            images    = batch["image"].to(device)
            labels    = batch["label"].to(device)
            questions = batch["question"]

            with autocast(enabled=args.amp and device.type == "cuda"):
                logits = model.forward_vqa(images, questions)
                loss   = criterion(logits, labels)

            preds    = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item()

            # Collect a few human-readable predictions
            if len(sample_preds) < 5:
                for q, pred_idx, gt_idx in zip(
                    questions,
                    preds.tolist(),
                    labels.tolist(),
                ):
                    pred_str = answer_vocab[pred_idx % len(answer_vocab)]
                    gt_str   = answer_vocab[gt_idx   % len(answer_vocab)]
                    sample_preds.append(f"Q: {q!r}  Pred={pred_str}  GT={gt_str}")

    model.train()
    return (
        correct / max(1, total),
        total_loss / max(1, len(val_loader)),
        sample_preds,
    )


# ---------------------------------------------------------------------------
# Gradient checkpointing helper
# ---------------------------------------------------------------------------

def _enable_gradient_checkpointing(model: nn.Module) -> None:
    """
    Enable gradient checkpointing on models that support it
    (HuggingFace Transformer subclasses).  Silently skips otherwise.
    """
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled.")
    else:
        logger.debug(
            "Model class %s does not expose gradient_checkpointing_enable(); "
            "checkpointing skipped.",
            type(model).__name__,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-phase VQA training for MedXplain (Cahier des Charges §3.1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Config
    p.add_argument("--config", default="config/hospital_config.yaml",
                   help="Path to hospital_config.yaml")

    # Phase 1
    p.add_argument("--phase1_epochs", type=int,  default=10)
    p.add_argument("--phase1_lr",     type=float, default=1e-4)

    # Phase 2
    p.add_argument("--phase2_epochs", type=int,  default=20)
    p.add_argument("--phase2_lr",     type=float, default=5e-5)

    # Shared training
    p.add_argument("--batch_size",   type=int,  default=8)
    p.add_argument("--num_workers",  type=int,  default=0,
                   help="DataLoader workers (0 = main process for Windows compatibility)")
    p.add_argument("--amp",          action="store_true", default=True,
                   help="Automatic Mixed Precision")
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--val_every",    type=int,  default=50,
                   help="Validate every N gradient steps")

    # LoRA
    p.add_argument("--lora_rank",    type=int,  default=8)
    p.add_argument("--lora_alpha",   type=int,  default=16)
    p.add_argument(
        "--lora_targets",
        nargs="+",
        default=["query", "value", "key"],
        help="Substrings of layer names to inject LoRA into",
    )

    # Output
    p.add_argument("--checkpoint_dir", default="checkpoints/",
                   help="Directory for saved checkpoints")
    p.add_argument("--num_classes",  type=int, default=2)
    p.add_argument("--n_train",      type=int, default=512,
                   help="Mock training samples (ignored with real dataset)")
    p.add_argument("--n_val",        type=int, default=64)

    # WandB
    p.add_argument("--wandb_project", default="medxplain-vqa")
    p.add_argument("--wandb_run",     default="two-phase-training")
    p.add_argument("--no_wandb",      action="store_true",
                   help="Disable WandB even if installed")

    return p


def main() -> None:
    args   = build_arg_parser().parse_args()
    config = load_config(args.config)

    # Override args from YAML where present
    inf_cfg = config.get("inference", {})
    if "batch_size" in inf_cfg and "--batch_size" not in sys.argv:
        args.batch_size = inf_cfg["batch_size"]

    device = torch.device(
        inf_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu"
    )
    logger.info("Device: %s", device)

    # Init WandB
    if not args.no_wandb:
        init_wandb(args.wandb_project, args.wandb_run, vars(args))

    # Build dataloaders (swap MockVQADataset → real dataset here)
    train_ds = MockVQADataset(n_samples=args.n_train, n_classes=args.num_classes)
    val_ds   = MockVQADataset(n_samples=args.n_val,   n_classes=args.num_classes)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True,  num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # Build model
    model = TwoPhaseVQAModel(num_classes=args.num_classes)

    # Output directories
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1
    phase1_ckpt = run_phase1(model, train_loader, val_loader, args, device, ckpt_dir)

    # Phase 2
    phase2_ckpt = run_phase2(
        model, train_loader, val_loader, args, device, ckpt_dir, phase1_ckpt
    )

    logger.info("Training complete.")
    logger.info("  Phase 1 checkpoint: %s", phase1_ckpt)
    logger.info("  Phase 2 checkpoint: %s", phase2_ckpt)

    if _WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
