"""
============================================================
src/utils.py
Utility helpers: seed, device, config loading, logging
============================================================
"""

import os
import random
import yaml
import logging
import numpy as np
import torch
from pathlib import Path
from rich.logging import RichHandler

# ------------------------------------------------------------------ #
# Logger
# ------------------------------------------------------------------ #
def get_logger(name: str = "XAI-Med") -> logging.Logger:
    """Returns a rich-formatted logger."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    return logging.getLogger(name)

logger = get_logger()


# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {path}")
    return cfg


# ------------------------------------------------------------------ #
# Reproducibility
# ------------------------------------------------------------------ #
def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {seed}")


# ------------------------------------------------------------------ #
# Device
# ------------------------------------------------------------------ #
def get_device(preference: str = "auto") -> torch.device:
    """
    Returns the best available device.
    preference: "auto" | "cuda" | "mps" | "cpu"
    """
    if preference == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(preference)
    logger.info(f"Using device: {device}")
    return device


# ------------------------------------------------------------------ #
# Directory helpers
# ------------------------------------------------------------------ #
def make_dirs(cfg: dict) -> None:
    """Create all output directories specified in config."""
    paths = cfg.get("paths", {})
    for key, path in paths.items():
        if key.endswith("_dir"):
            Path(path).mkdir(parents=True, exist_ok=True)
    logger.info("Output directories verified.")


# ------------------------------------------------------------------ #
# Model checkpoint helpers
# ------------------------------------------------------------------ #
def save_checkpoint(state: dict, path: str) -> None:
    """Save training checkpoint."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)
    logger.info(f"Checkpoint saved → {path}")


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer=None, device: torch.device = None):
    """Load checkpoint into model (and optionally optimizer), aligning state_dict keys if needed."""
    if device is None:
        device = torch.device("cpu")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Extract state_dict
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # 1. Clean 'module.' prefix (from DataParallel / DDP)
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k.replace("module.", "", 1)] = v
        else:
            cleaned_state_dict[k] = v
            
    # 2. Get model's state_dict keys
    model_state_dict = model.state_dict()
    model_keys = set(model_state_dict.keys())
    
    # 3. Align keys (e.g. handle 'backbone.' prefix mismatch)
    final_state_dict = {}
    aligned_count = 0
    for k, v in cleaned_state_dict.items():
        if k in model_keys:
            final_state_dict[k] = v
        elif f"backbone.{k}" in model_keys:
            final_state_dict[f"backbone.{k}"] = v
            aligned_count += 1
        elif k.startswith("backbone.") and k.replace("backbone.", "", 1) in model_keys:
            final_state_dict[k.replace("backbone.", "", 1)] = v
            aligned_count += 1
        else:
            final_state_dict[k] = v
            
    if aligned_count > 0:
        logger.info(f"Automatically aligned {aligned_count} state_dict keys (e.g. adding/removing 'backbone.' prefix).")
        
    # 4. Load state dict using strict=False to be robust against minor architecture variances
    missing_keys, unexpected_keys = model.load_state_dict(final_state_dict, strict=False)
    if missing_keys:
        logger.warning(f"Missing keys when loading state_dict: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys when loading state_dict: {unexpected_keys}")
        
    # Load optimizer state if available
    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception as e:
            logger.warning(f"Could not load optimizer state_dict: {e}")
            
    val_acc = checkpoint.get('val_acc', None)
    val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, (int, float)) else "?"
    logger.info(f"Checkpoint loaded from {path} "
                f"(epoch {checkpoint.get('epoch', '?')}, "
                f"val_acc={val_acc_str})")
    return checkpoint



# ------------------------------------------------------------------ #
# Image denormalization helper
# ------------------------------------------------------------------ #
def denormalize(tensor: torch.Tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """
    Reverses ImageNet normalization and returns an (H, W, 3) uint8 array.
    """
    t = tensor.clone().detach().cpu()
    if t.dim() == 4:          # (B, C, H, W) → (C, H, W)
        t = t.squeeze(0)
    for ch, m, s in zip(t, mean, std):
        ch.mul_(s).add_(m)
    t = torch.clamp(t, 0, 1)
    img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return img
