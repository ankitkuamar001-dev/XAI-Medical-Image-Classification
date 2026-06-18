"""
============================================================
src/train.py — Training loop with early stopping, LR scheduling,
mixed precision, and TensorBoard logging
============================================================
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.utils import get_logger, save_checkpoint

logger = get_logger("train")


# ------------------------------------------------------------------ #
# Early Stopping
# ------------------------------------------------------------------ #
class EarlyStopping:
    """Halts training when val loss stops improving for `patience` epochs."""

    def __init__(self, patience: int = 7, delta: float = 1e-4):
        self.patience  = patience
        self.delta     = delta
        self.best_loss = np.inf
        self.counter   = 0
        self.best_state: dict = {}

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter   = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        return False


# ------------------------------------------------------------------ #
# Scheduler factory
# ------------------------------------------------------------------ #
def build_scheduler(optimizer, cfg: dict, num_epochs: int):
    name = cfg["training"].get("scheduler", "cosine")
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    elif name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    raise ValueError(f"Unknown scheduler: {name}")


# ------------------------------------------------------------------ #
# One epoch
# ------------------------------------------------------------------ #
def run_epoch(model, loader, criterion, optimizer, device,
              scaler, is_train=True) -> Tuple[float, float]:
    """Run one train or val epoch. Returns (avg_loss, accuracy)."""
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0

    # Try importing autocast/GradScaler (PyTorch 1.x vs 2.x)
    try:
        from torch.amp import autocast, GradScaler as GS
        amp_ctx = lambda: autocast("cuda", enabled=(scaler is not None and is_train))
    except ImportError:
        from torch.cuda.amp import autocast
        amp_ctx = lambda: autocast(enabled=(scaler is not None and is_train))

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc='Train' if is_train else 'Val', leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with amp_ctx():
                outputs = model(images)
                loss    = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct    += (outputs.argmax(dim=1) == labels).sum().item()
            total      += images.size(0)

    return total_loss / total, correct / total


# ------------------------------------------------------------------ #
# Main training function
# ------------------------------------------------------------------ #
def train_model(model, train_loader, val_loader, cfg, device,
                class_weights=None) -> Dict[str, List[float]]:
    """
    Full training pipeline:
      - Class-weighted CrossEntropyLoss (handles imbalanced data)
      - AdamW optimizer (decoupled weight decay)
      - LR scheduling (cosine / step / plateau)
      - Mixed precision AMP
      - Early stopping
      - TensorBoard logging
      - Best checkpoint saving
    """
    tr_cfg = cfg["training"]
    paths  = cfg["paths"]

    epochs   = tr_cfg["epochs"]
    use_amp  = tr_cfg.get("mixed_precision", True) and torch.cuda.is_available()
    patience = tr_cfg["early_stopping_patience"]

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=tr_cfg["learning_rate"],
        weight_decay=tr_cfg["weight_decay"],
    )

    scheduler  = build_scheduler(optimizer, cfg, epochs)

    try:
        from torch.amp import GradScaler
        scaler = GradScaler("cuda") if use_amp else None
    except Exception:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler() if use_amp else None

    writer     = SummaryWriter(log_dir=paths.get("tensorboard_dir", "runs/xai_medimage"))
    early_stop = EarlyStopping(patience=patience)

    history: Dict[str, List[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":  [],
    }
    best_val_acc = 0.0

    logger.info("=" * 60)
    logger.info(f"Starting training for {epochs} epochs (AMP={use_amp})")
    logger.info("=" * 60)

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer,
            device, scaler, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, optimizer,
            device, scaler=None, is_train=False
        )

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        writer.add_scalars("Loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Accuracy", {"train": train_acc,  "val": val_acc},  epoch)
        writer.add_scalar("LR", lr, epoch)

        logger.info(
            f"Epoch [{epoch:03d}/{epochs}] "
            f"Train {train_loss:.4f}/{train_acc:.4f} | "
            f"Val {val_loss:.4f}/{val_acc:.4f} | "
            f"LR={lr:.2e} | {time.time()-t0:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {"epoch": epoch, "model_state_dict": model.state_dict(),
                 "optimizer_state_dict": optimizer.state_dict(),
                 "val_acc": val_acc, "val_loss": val_loss},
                path=paths["best_model"],
            )
            logger.info(f"  ↳ Best model saved (val_acc={val_acc:.4f})")

        if early_stop(val_loss, model):
            logger.info(f"Early stopping at epoch {epoch}.")
            break

    writer.flush()
    writer.close()

    total_epochs = len(history["train_loss"])
    logger.info("=" * 60)
    logger.info("Training Summary")
    logger.info("=" * 60)
    logger.info(f"  Total epochs trained : {total_epochs}")
    logger.info(f"  Final train loss     : {history['train_loss'][-1]:.4f}")
    logger.info(f"  Final val loss       : {history['val_loss'][-1]:.4f}")
    logger.info(f"  Final train acc      : {history['train_acc'][-1]:.4f}")
    logger.info(f"  Final val acc        : {history['val_acc'][-1]:.4f}")
    logger.info(f"  Best val_acc         : {best_val_acc:.4f}")
    logger.info("=" * 60)

    return history


# ------------------------------------------------------------------ #
# Training curves visualization
# ------------------------------------------------------------------ #
def plot_training_curves(history: Dict[str, List[float]],
                         save_dir: str = "outputs/plots"):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=16, fontweight="bold")

    for ax, metric, ylabel in zip(
        axes,
        [("train_loss", "val_loss"), ("train_acc", "val_acc")],
        ["Cross-Entropy Loss", "Accuracy"],
    ):
        ax.plot(epochs, history[metric[0]], "b-o", ms=3, label="Train")
        ax.plot(epochs, history[metric[1]], "r-o", ms=3, label="Val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = f"{save_dir}/training_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Training curves saved → {path}")
