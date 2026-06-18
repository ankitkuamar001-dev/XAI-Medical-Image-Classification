"""
============================================================
src/shap_explain.py
SHAP-based feature attribution for CNN models
============================================================

THEORY
------
SHAP (SHapley Additive exPlanations) — Lundberg & Lee, 2017.

Core idea: from game theory (Shapley values), fairly distribute the
"payout" (model prediction) among "players" (input features/pixels).

For an input x with baseline (background) E[f(x)]:
  f(x) = φ₀ + Σᵢ φᵢ
where φ₀ = expected prediction over background,
      φᵢ = SHAP value of feature i (signed contribution).

Properties:
  • Efficiency:   Σφᵢ = f(x) - E[f(x)]  (values sum to prediction gap)
  • Symmetry:     Equal contributors get equal values.
  • Dummy:        Unused features → φ = 0.
  • Additivity:   Sum of per-model SHAPs = ensemble SHAP.

For CNNs we use DeepExplainer (PyTorch backend):
  - Backpropagates SHAP values through the network efficiently.
  - Background = small set of training images (≈50).
  - Output: per-pixel SHAP map, one channel per output class.

Grad-CAM vs SHAP:
  Grad-CAM  → class-discriminative, high spatial fidelity, fast.
  SHAP      → theoretically grounded, signed, pixel-level attribution.
"""

from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



from src.utils import get_logger, denormalize

logger = get_logger("shap")


# ------------------------------------------------------------------ #
# SHAP explanation
# ------------------------------------------------------------------ #
def compute_shap_values(
    model: nn.Module,
    background_loader,        # DataLoader for background (≈50 images)
    test_images: torch.Tensor,
    device: torch.device,
    num_background: int = 50,
) -> np.ndarray:
    """
    Compute SHAP values using DeepExplainer.

    Args:
        model            : trained PyTorch model (in eval mode).
        background_loader: DataLoader supplying background images.
        test_images      : (N, C, H, W) tensor of images to explain.
        device           : torch device.
        num_background   : number of background samples (keep small!).

    Returns:
        shap_values : list of arrays, one per class.
                      Each array: (N, H, W, C) for image data.
    """
    model.eval()

    import shap

    # Collect background images
    bg_imgs = []
    count = 0
    for imgs, _ in background_loader:
        bg_imgs.append(imgs)
        count += imgs.shape[0]
        if count >= num_background:
            break
    background = torch.cat(bg_imgs)[:num_background].to(device)

    logger.info(
        f"SHAP background: {background.shape} | "
        f"Test samples: {test_images.shape}"
    )

    # Build DeepExplainer
    explainer = shap.DeepExplainer(model, background)

    test_tensor = test_images.to(device)
    shap_vals   = explainer.shap_values(test_tensor)
    # shap_vals: list[C] of (N, C_img, H, W) numpy arrays

    # Normalize format: some SHAP versions return a single ndarray
    # with classes along the last axis instead of a list of arrays.
    if isinstance(shap_vals, np.ndarray):
        shap_vals = [shap_vals[..., i] for i in range(shap_vals.shape[-1])]

    logger.info(
        f"SHAP values computed. "
        f"Shape per class: {shap_vals[0].shape}"
    )
    return shap_vals


# ------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------ #
def visualize_shap(
    shap_values: list,          # list[num_classes] of (N, C, H, W)
    test_images: torch.Tensor,  # (N, C, H, W) normalized
    class_names: List[str],
    cfg: dict,
    num_samples: int = 5,
):
    """
    Save SHAP visualizations:
      - SHAP summary map (mean |SHAP| across channels)
      - Red-blue SHAP overlay on original image
    """
    save_dir = Path(cfg["paths"]["shap_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    n = min(num_samples, test_images.shape[0])
    num_classes = len(class_names)

    # shap_values[class][sample] has shape (C_img, H, W)
    # Convert to (N, H, W, C_img) for SHAP plotting conventions
    shap_arr = [np.transpose(sv[:n], (0, 2, 3, 1)) for sv in shap_values]

    # Denormalized test images: (N, H, W, 3) float [0,1]
    orig_imgs = np.stack([
        denormalize(test_images[i]).astype(np.float32) / 255.0
        for i in range(n)
    ])

    for sample_idx in range(n):
        fig, axes = plt.subplots(1, num_classes + 1, figsize=(5 * (num_classes + 1), 5))

        # Original image
        axes[0].imshow(orig_imgs[sample_idx])
        axes[0].set_title("Original Image", fontsize=11)
        axes[0].axis("off")

        for cls_idx, cls_name in enumerate(class_names):
            # Mean absolute SHAP across RGB channels → saliency map
            sv_sample = shap_arr[cls_idx][sample_idx]  # (H, W, 3)
            saliency  = np.abs(sv_sample).mean(axis=-1)  # (H, W)

            # Normalize for display
            s_min, s_max = saliency.min(), saliency.max()
            saliency_norm = (saliency - s_min) / (s_max - s_min + 1e-8)

            im = axes[cls_idx + 1].imshow(
                saliency_norm, cmap="RdBu_r", vmin=0, vmax=1
            )
            axes[cls_idx + 1].set_title(
                f"SHAP for class:\n{cls_name}", fontsize=11
            )
            axes[cls_idx + 1].axis("off")
            plt.colorbar(im, ax=axes[cls_idx + 1], fraction=0.046, pad=0.04)

        plt.suptitle(f"SHAP Explanation — Sample {sample_idx}",
                     fontsize=14, fontweight="bold")
        plt.tight_layout()
        path = save_dir / f"shap_sample_{sample_idx:03d}.png"
        plt.savefig(str(path), dpi=150, bbox_inches="tight")
        plt.close()

    logger.info(f"SHAP visualizations saved → {save_dir}")


# ------------------------------------------------------------------ #
# SHAP summary bar (mean |SHAP| per superpixel region)
# ------------------------------------------------------------------ #
def shap_summary_bar(
    shap_values: list,
    class_names: List[str],
    save_dir: str = "outputs/shap",
):
    """
    Plot mean |SHAP| value per class as a bar chart.
    Provides a quick view of which class the model focuses on.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    mean_abs = [np.abs(sv).mean() for sv in shap_values]

    plt.figure(figsize=(8, 4))
    colors = plt.cm.viridis(np.linspace(0.3, 0.8, len(class_names)))
    bars   = plt.bar(class_names, mean_abs, color=colors)
    plt.bar_label(bars, fmt="%.4f", padding=3)
    plt.ylabel("Mean |SHAP|")
    plt.title("Mean Absolute SHAP per Class")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = f"{save_dir}/shap_summary_bar.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info(f"SHAP summary bar saved → {path}")
