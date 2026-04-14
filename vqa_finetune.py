import os
import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vqa.models.blip2_adapter import Blip2VQAAdapter
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def finetune_vqa(train_loader, val_loader, epochs=5):
    print("--- VQA-RAD LoRA Fine-Tuning Setup ---")

    # 1. Load BLIP-2 with LoRA
    model = Blip2VQAAdapter(use_lora=True).to(DEVICE)

    # 2. Freeze Vision Encoder
    for param in model.model.vision_model.parameters():
        param.requires_grad = False

    print("Trainable parameters (LoRA + QFormer + Proj):")
    model.model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=5e-5,
    )

    best_closed_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in enumerate(loop):

            inputs = {
                "input_ids":      batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
                "pixel_values":   batch["pixel_values"].to(DEVICE),
                "labels":         batch["labels"].to(DEVICE),
            }

            optimizer.zero_grad()
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            # Validate every 100 steps
            if step > 0 and step % 100 == 0:
                closed_acc, open_acc = validate(model, val_loader)
                print(f"\nStep {step} | Closed Acc: {closed_acc:.4f} | Open Acc: {open_acc:.4f}")

                if closed_acc > best_closed_acc:
                    best_closed_acc = closed_acc
                    os.makedirs("experiments/vqa/checkpoints", exist_ok=True)
                    torch.save(
                        {
                            'epoch':                epoch,
                            'step':                 step,
                            'model_state_dict':     model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'closed_acc':           closed_acc,
                            'open_acc':             open_acc,
                        },
                        f"experiments/vqa/checkpoints/best_lora_model_step_{step}.pt",
                    )
                    print(f"  -> Saved new best model (Closed Acc: {best_closed_acc:.4f})")

                model.train()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Avg Loss: {avg_loss:.6f}")


def validate(model, val_loader):
    model.eval()
    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    closed_correct, closed_total = 0, 0
    open_correct,   open_total   = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            pixel_values   = batch["pixel_values"].to(DEVICE)
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            answers        = batch["answers"]
            answer_types   = batch.get("answer_types", ["CLOSED"] * len(answers))

            # Explicit kwargs matching the adapter's generate() signature
            generated_ids = model.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=50,
                min_new_tokens=1,
                num_beams=3,
                do_sample=False,
                repetition_penalty=1.2,
            )

            preds = processor.batch_decode(generated_ids, skip_special_tokens=True)

            for pred, true_answer, ans_type in zip(preds, answers, answer_types):
                pred        = pred.strip().lower()
                true_answer = true_answer.strip().lower()
                is_correct  = int(pred == true_answer)

                if ans_type == "CLOSED":
                    closed_correct += is_correct
                    closed_total   += 1
                else:
                    open_correct += is_correct
                    open_total   += 1

    c_acc = closed_correct / max(closed_total, 1)
    o_acc = open_correct   / max(open_total,   1)
    return c_acc, o_acc


if __name__ == "__main__":
    from data.dataset import get_dataloaders
    from transformers import AutoProcessor
    import configs.config as cfg

    print("VQA Fine-tuning Script initialized. Fetching DataLoaders...")
    print(f"Using device: {DEVICE}")

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    train_dl, val_dl = get_dataloaders(processor, batch_size=2)

    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}")

    # Debug: inspect batch structure safely
    print("\n=== Debug: Checking batch structure ===")
    sample_batch = next(iter(train_dl))
    print(f"Batch type: {type(sample_batch)}")
    if isinstance(sample_batch, dict):
        print(f"Batch keys: {list(sample_batch.keys())}")
        for key, value in sample_batch.items():
            if isinstance(value, torch.Tensor):
                if value.is_floating_point():
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                          f"min={value.min():.3f}, max={value.max():.3f}, mean={value.mean():.3f}")
                else:
                    print(f"  {key}: shape={value.shape}, dtype={value.dtype}, "
                          f"min={value.min()}, max={value.max()}")
            else:
                print(f"  {key}: {type(value).__name__}, length={len(value) if hasattr(value, '__len__') else 'N/A'}")

    # Verify input_ids and labels always match shape
    assert sample_batch['input_ids'].shape == sample_batch['labels'].shape, (
        f"Shape mismatch: input_ids {sample_batch['input_ids'].shape} "
        f"vs labels {sample_batch['labels'].shape}"
    )
    print("Shape check passed.")
    print("=====================================\n")

    finetune_vqa(train_dl, val_dl, epochs=13)