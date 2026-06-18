"""
============================================================
src/uncertainty.py — Monte Carlo Dropout Uncertainty Estimation
============================================================

THEORY
------
Monte Carlo (MC) Dropout (Gal & Ghahramani, 2016) estimates
epistemic uncertainty (model uncertainty) by:

  1. Keeping dropout ENABLED at inference time.
  2. Running N stochastic forward passes on the same input.
  3. Computing mean and standard deviation of predictions.

Interpretation:
  - High mean confidence + low std → Model is certain.
  - High mean confidence + high std → Model is overconfident (uncertain).
  - Low confidence + high std → Model genuinely doesn't know.

Why this matters in medical AI:
  - A confident-but-wrong prediction is dangerous.
  - Uncertainty flags help clinicians triage ambiguous cases.
  - Regulators (FDA, CE Mark) increasingly require uncertainty estimates.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple

from src.utils import get_logger

logger = get_logger("uncertainty")


def enable_mc_dropout(model: nn.Module) -> None:
    """
    Enable dropout layers during inference for MC Dropout.
    Only affects nn.Dropout modules — leaves BatchNorm in eval mode.
    """
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()


def mc_dropout_predict(
    model: nn.Module,
    img_tensor: torch.Tensor,
    device: torch.device,
    n_forward: int = 30,
    class_names: list = None,
) -> Dict:
    """
    Perform Monte Carlo Dropout inference.

    Args:
        model       : trained model (with dropout layers).
        img_tensor  : single image tensor (C, H, W) or (1, C, H, W).
        device      : torch device.
        n_forward   : number of stochastic forward passes (default: 30).
        class_names : list of class names (optional, for labeling).

    Returns:
        dict with keys:
            mean_probs    : (C,) mean class probabilities
            std_probs     : (C,) std dev of class probabilities
            pred_class    : int, predicted class index
            confidence    : float, mean confidence of predicted class
            uncertainty   : float, std dev of predicted class
            all_probs     : (n_forward, C) all probability vectors
            entropy       : float, predictive entropy (bits)
    """
    model.eval()
    enable_mc_dropout(model)

    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    img_tensor = img_tensor.to(device)

    all_probs = []

    with torch.no_grad():
        for _ in range(n_forward):
            logits = model(img_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    all_probs = np.stack(all_probs)  # (n_forward, C)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    pred_class = int(mean_probs.argmax())
    confidence = float(mean_probs[pred_class])
    uncertainty = float(std_probs[pred_class])

    # Predictive entropy (bits) — higher = more uncertain
    entropy = float(-np.sum(mean_probs * np.log2(mean_probs + 1e-10)))

    # Restore model to normal eval mode
    model.eval()

    result = {
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "pred_class": pred_class,
        "confidence": confidence,
        "uncertainty": uncertainty,
        "all_probs": all_probs,
        "entropy": entropy,
    }

    if class_names:
        logger.info(
            f"MC Dropout ({n_forward} passes) → "
            f"Prediction: {class_names[pred_class]} | "
            f"Confidence: {confidence:.4f} ± {uncertainty:.4f} | "
            f"Entropy: {entropy:.4f} bits"
        )

    return result


def plot_uncertainty(
    result: Dict,
    class_names: list,
    save_path: str = None,
):
    """
    Visualize MC Dropout uncertainty as box plots of predicted probabilities.
    """
    import matplotlib.pyplot as plt

    all_probs = result["all_probs"]  # (n_forward, C)
    n_classes = len(class_names)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Monte Carlo Dropout Uncertainty", fontsize=15, fontweight="bold")

    # Box plot of probability distributions per class
    ax = axes[0]
    bp = ax.boxplot(
        [all_probs[:, i] for i in range(n_classes)],
        labels=class_names,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2),
    )
    colors = ["#3fb950", "#f85149", "#58a6ff", "#bc8cff"][:n_classes]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Probability Distribution per Class")
    ax.grid(True, alpha=0.3)

    # Bar chart: mean ± std
    ax = axes[1]
    x = np.arange(n_classes)
    bars = ax.bar(
        x, result["mean_probs"], yerr=result["std_probs"],
        capsize=8, color=colors, alpha=0.8, edgecolor="white",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Probability")
    ax.set_title(f"Mean ± Std (Entropy: {result['entropy']:.3f} bits)")
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3)

    for bar, mean, std in zip(bars, result["mean_probs"], result["std_probs"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 0.03,
            f"{mean:.3f}±{std:.3f}", ha="center", va="bottom", fontsize=9,
        )

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Uncertainty plot saved → {save_path}")

    plt.close()
    return fig

