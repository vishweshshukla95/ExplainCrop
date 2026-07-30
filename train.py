"""
ExplainCrop — Training loop.

Building this in pieces. Piece 1: the multi-task loss function.

Three heads, three losses, combined with the weights from config.py:
  - cause_loss:      cross-entropy over (healthy/deficiency/disease/pest),
                      trained on ALL crops including disease-only ones
  - deficiency_loss:  cross-entropy over NPK classes, but ONLY counted for
                       samples where cause == nutrient_deficiency (that's
                       what the ignore_index=-100 in dataset.py's npk_idx
                       is for — PyTorch's cross_entropy skips those rows)
  - severity_loss:     heteroscedastic regression loss from models.py,
                        only counted for samples with a real severity label
"""

import torch
import torch.nn.functional as F

from config import CFG
from models import heteroscedastic_loss


def compute_losses(model_out: dict, cause_idx: torch.Tensor,
                    npk_idx: torch.Tensor, severity_target: torch.Tensor,
                    severity_mask: torch.Tensor) -> dict:
    """
    model_out: dict from CropConditionedViT.forward()
    cause_idx: (B,) ground truth cause labels
    npk_idx:   (B,) ground truth NPK labels, -100 where not applicable
               (PyTorch's cross_entropy ignores -100 by default)
    severity_target: (B,) weak severity score in [0,1] from severity.py
    severity_mask:   (B,) 1.0 where a severity label exists, 0.0 otherwise
                      (can't use -100 trick for regression, so we mask manually)
    """
    cause_loss = F.cross_entropy(model_out["cause_logits"], cause_idx)

    # Guard against nan: if EVERY sample in a batch has npk_idx=-100 (no
    # deficiency-labeled images in that batch — plausible given class
    # imbalance), cross_entropy's mean-reduction divides by zero -> nan,
    # which then contaminates the whole epoch's averaged loss.
    if (npk_idx != -100).any():
        deficiency_loss = F.cross_entropy(
            model_out["deficiency_logits"], npk_idx, ignore_index=-100)
    else:
        deficiency_loss = torch.tensor(0.0, device=model_out["deficiency_logits"].device)

    # Heteroscedastic severity loss, masked to only real labels.
    sev_mean = model_out["severity_mean"]
    sev_logvar = model_out["severity_logvar"]
    if severity_mask.sum() > 0:
        # Compute per-sample loss then average only over masked (labeled) rows.
        precision = torch.exp(-sev_logvar)
        per_sample = precision * (severity_target - sev_mean) ** 2 + sev_logvar
        severity_loss = (per_sample * severity_mask).sum() / severity_mask.sum()
    else:
        severity_loss = torch.tensor(0.0, device=sev_mean.device)

    w = CFG.loss_weights
    total = (w["cause"] * cause_loss
             + w["deficiency"] * deficiency_loss
             + w["severity"] * severity_loss)

    return {
        "total": total,
        "cause_loss": cause_loss,
        "deficiency_loss": deficiency_loss,
        "severity_loss": severity_loss,
    }


# ---------------------------------------------------------------------------
# Real training loop.
#
# Trains on the full manifest (all crops together, per the crop-conditioned
# design), saves the best checkpoint by validation total loss, and prints
# per-epoch metrics for cause/deficiency accuracy alongside the three losses.
# ---------------------------------------------------------------------------
import os
import time

from torch.utils.data import DataLoader

from dataset import ExplainCropDataset
from models import CropConditionedViT


def accuracy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> float:
    """Accuracy over rows where target != ignore_index. Returns -1.0 if no
    valid rows (e.g. a batch with no deficiency-labeled samples)."""
    mask = target != ignore_index
    if mask.sum() == 0:
        return -1.0
    preds = logits.argmax(dim=1)
    correct = (preds[mask] == target[mask]).sum().item()
    return correct / mask.sum().item()


def run_epoch(model, loader, optimizer, device, train: bool):
    model.train() if train else model.eval()

    totals = {"total": 0.0, "cause_loss": 0.0, "deficiency_loss": 0.0, "severity_loss": 0.0}
    cause_accs, defic_accs = [], []
    n_batches = 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, crop_idx, cause_idx, npk_idx, sev_idx in loader:
            images = images.to(device)
            crop_idx = crop_idx.to(device)
            cause_idx = cause_idx.to(device)
            npk_idx = npk_idx.to(device)
            # severity_target/mask: severity.py's weak labels aren't wired
            # into the manifest yet (dataset.py's `severity` column is
            # currently always "unlabeled" -> sev_idx=-100 for everything),
            # so the severity loss is effectively skipped this run. This is
            # a known gap, not a bug — flagged in the paper's limitations
            # alongside the weak-supervision caveat already in severity.py.
            severity_mask = (sev_idx != -100).float().to(device)
            severity_target = torch.zeros_like(severity_mask)  # unused while mask is all-zero

            if train:
                optimizer.zero_grad()

            out = model(images, crop_idx)
            losses = compute_losses(out, cause_idx, npk_idx, severity_target, severity_mask)

            if train:
                losses["total"].backward()
                optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_batches += 1

            cause_accs.append(accuracy(out["cause_logits"], cause_idx, ignore_index=-1))  # cause has no ignore
            defic_accs.append(accuracy(out["deficiency_logits"], npk_idx, ignore_index=-100))

    avg = {k: v / max(1, n_batches) for k, v in totals.items()}
    avg["cause_acc"] = sum(cause_accs) / len(cause_accs) if cause_accs else -1.0
    valid_defic = [a for a in defic_accs if a >= 0]
    avg["deficiency_acc"] = sum(valid_defic) / len(valid_defic) if valid_defic else -1.0
    return avg


def train_model(manifest_csv: str = None, cfg=CFG):
    manifest_csv = manifest_csv or os.path.join(cfg.data_root, "manifest.csv")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    train_ds = ExplainCropDataset(manifest_csv, split="train", image_size=cfg.image_size)
    val_ds = ExplainCropDataset(manifest_csv, split="val", image_size=cfg.image_size)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                               num_workers=cfg.num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers)

    model = CropConditionedViT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    os.makedirs(cfg.checkpoint_dir, exist_ok=True)
    best_val_loss = float("inf")
    start_epoch = 1

    ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_val_loss = checkpoint["val_loss"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from checkpoint: epoch {checkpoint['epoch']}, val_loss={best_val_loss:.4f}")

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, train=False)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{cfg.epochs} ({elapsed:.0f}s) | "
              f"train_loss={train_metrics['total']:.4f} cause_acc={train_metrics['cause_acc']:.3f} "
              f"defic_acc={train_metrics['deficiency_acc']:.3f} | "
              f"val_loss={val_metrics['total']:.4f} cause_acc={val_metrics['cause_acc']:.3f} "
              f"defic_acc={val_metrics['deficiency_acc']:.3f}")

        if val_metrics["total"] < best_val_loss:
            best_val_loss = val_metrics["total"]
            ckpt_path = os.path.join(cfg.checkpoint_dir, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint (val_loss={best_val_loss:.4f})")


if __name__ == "__main__":
    train_model()